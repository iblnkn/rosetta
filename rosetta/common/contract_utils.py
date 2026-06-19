# Copyright 2025 Isaac Blankenau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Contract utilities: spec iteration, resampling buffers, and data helpers.

This module provides runtime utilities for working with contracts:
- Spec iteration: convert contract specs to runtime stream specs
- Feature building for LeRobot datasets
- StreamBuffer: thread-safe online resampler
- Zero-fill utilities for missing data
"""

from __future__ import annotations

import threading
from typing import Any, Iterable, TYPE_CHECKING, TypeVar

import numpy as np

from .contract import (
    ActionSpec,
    ActionStreamSpec,
    AlignSpec,
    Contract,
    ContractValidationError,
    DEPTH_ENCODINGS,
    ObservationSpec,
    ObservationStreamSpec,
    ResamplePolicy,
    StreamSpec,
)
from .converters import DTYPES, ENCODERS, get_decoder_dtype
from .ops import build_op, OpContext

if TYPE_CHECKING:
    pass  # All types imported above


# =============================================================================
# Feature Building
# =============================================================================

# All supported image encodings are normalized to 3-channel uint8 RGB by the
# decoders (3ch color as-is, 4ch drops alpha, mono->rgb); depth/RGBD encodings
# are explicitly rejected at decode (see decoders.DepthEncodingNotSupported).
# The decoded output channel count is therefore always 3, independent of the
# source encoding's channel count (spec.image_channels). This is the single
# source of truth for the feature shape and the zero-fill shape so they agree.
DECODED_IMAGE_CHANNELS = 3


def build_feature(spec: ObservationStreamSpec | ActionStreamSpec) -> dict[str, Any]:
    """
    Build LeRobot feature dict from a spec.

    Returns a dict with:
    - dtype: LeRobot dtype string
    - shape: tuple of dimensions
    - names: axis names (or None)
    """
    dtype = getattr(spec, 'dtype', None)
    if dtype is None:
        raise ValueError(f"Spec '{spec.key}' has no dtype")

    if dtype in ('video', 'image'):
        if not spec.image_resize:
            raise ValueError(f"Image spec '{spec.key}' must have image_resize")
        h, w = spec.image_resize
        return {
            'dtype': 'video',
            'shape': (h, w, DECODED_IMAGE_CHANNELS),
            'names': ['height', 'width', 'channels'],
        }

    if dtype == 'string':
        return {'dtype': 'string', 'shape': (1,), 'names': None}

    # Numeric types (float32, float64, int32, int64, bool)
    n = len(spec.names) if spec.names else 1
    names = list(spec.names) if spec.names else None
    return {'dtype': dtype, 'shape': (n,), 'names': names}


# =============================================================================
# Zero-fill for Missing Data
# =============================================================================


def zeros_for_spec(spec: ObservationStreamSpec) -> np.ndarray:
    """
    Create a zero-filled array for missing data.

    Args
    ----
        spec: Observation stream spec defining shape/type.

    Returns
    -------
        Zero-filled numpy array:
        - Images: (H, W, DECODED_IMAGE_CHANNELS) uint8
        - Vectors: (N,) with dtype from spec

    """
    if spec.is_image:
        h, w = spec.image_resize
        # Use the decoded channel count (always 3, see DECODED_IMAGE_CHANNELS),
        # NOT spec.image_channels (the source-encoding count, which may be 1/4).
        # A zero-filled missing frame must match the shape of a real decoded
        # frame and the feature declared by build_feature.
        return np.zeros((h, w, DECODED_IMAGE_CHANNELS), dtype=np.uint8)

    dtype_map = {
        'float32': np.float32,
        'float64': np.float64,
        'int32': np.int32,
        'int64': np.int64,
        'bool': bool,
    }
    np_dtype = dtype_map.get(spec.dtype, np.float32)
    return np.zeros(len(spec.names), dtype=np_dtype)


# =============================================================================
# Stream Buffer
# =============================================================================


class StreamBuffer:
    """
    Thread-safe, constant-memory online resampler.

    Policies:
    - "hold": Always return the last value (default)
    - "asof": Return last value only if within tolerance window
    - "drop": Return last value only if it arrived within the step window
    """

    def __init__(self, policy: str, step_ns: int, tol_ns: int = 0):
        self.policy = policy
        self.step_ns = int(step_ns)
        self.tol_ns = int(tol_ns)
        self.last_ts: int | None = None
        self.last_val: Any | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_spec(cls, spec: ObservationStreamSpec) -> 'StreamBuffer':
        """Create a StreamBuffer from an ObservationStreamSpec."""
        step_ns = int(1e9 / spec.fps) if spec.fps > 0 else int(1e9 / 30)
        tol_ns = spec.asof_tol_ms * 1_000_000
        return cls(policy=spec.resample_policy, step_ns=step_ns, tol_ns=tol_ns)

    def push(self, ts_ns: int, val: Any) -> None:
        """Insert a sample (keeps the newest by timestamp)."""
        with self._lock:
            if self.last_ts is None or ts_ns >= self.last_ts:
                self.last_ts, self.last_val = ts_ns, val

    def sample(self, tick_ns: int) -> Any | None:
        """Sample according to policy at a given tick."""
        with self._lock:
            if self.last_ts is None:
                return None

            # Handle clock resets (e.g., simulation restart)
            # If buffered timestamp is in the future, clock was reset - clear stale data
            if self.last_ts > tick_ns:
                self._clear_unsafe()  # Already holding lock
                return None

            if self.policy == ResamplePolicy.DROP.value:
                return self.last_val if (self.last_ts > tick_ns - self.step_ns) else None
            if self.policy == ResamplePolicy.ASOF.value:
                return self.last_val if (tick_ns - self.last_ts <= self.tol_ns) else None
            return self.last_val  # hold is default

    def _clear_unsafe(self) -> None:
        """Clear buffered data without acquiring lock (internal use only)."""
        self.last_ts = None
        self.last_val = None

    def reset(self) -> None:  # noqa: F811
        """Clear buffered data (e.g., between episodes or after sim reset)."""
        with self._lock:
            self._clear_unsafe()


# =============================================================================
# Image Encoding Validation
# =============================================================================

_ENCODING_CHANNELS = {
    'mono8': 1,
    '8uc1': 1,
    'bgr8': 3,
    'rgb8': 3,
    'bgr16': 3,
    'rgb16': 3,
    'bgra8': 4,
    'rgba8': 4,
}


def _validate_image_encoding(encoding: str) -> int:
    """
    Validate encoding and return channel count.

    Raises
    ------
        ContractValidationError: If encoding is unsupported or is depth format.

    """
    enc = encoding.lower()

    if enc in DEPTH_ENCODINGS:
        raise ContractValidationError(
            f"Depth image encoding '{encoding}' is not supported. "
            f'LeRobot does not currently have proper depth image handling.'
        )

    channels = _ENCODING_CHANNELS.get(enc)
    if channels is None:
        raise ContractValidationError(
            f"Unsupported image encoding '{encoding}'. "
            f'Supported: {", ".join(sorted(_ENCODING_CHANNELS.keys()))}'
        )
    return channels


# =============================================================================
# Namespace Derivation
# =============================================================================


def _derive_namespaces(topics: list[str]) -> dict[str, str]:
    """
    Derive unique namespace for each topic from path segments.

    Finds the first segment that uniquely identifies each topic.

    Examples
    --------
        ['/arm/state', '/base/state'] -> {'...': 'arm', '...': 'base'}
        ['/arm/pos', '/arm/vel'] -> {'...': 'pos', '...': 'vel'}

    """
    if len(topics) <= 1:
        return dict.fromkeys(topics, '')

    parts_list = [[p for p in t.split('/') if p] for t in topics]
    max_depth = max(len(p) for p in parts_list)

    # Find first segment where all topics differ
    for i in range(max_depth):
        segments = [p[i] if i < len(p) else '' for p in parts_list]
        if len(set(segments)) == len(topics):
            return dict(zip(topics, segments, strict=False))

    # No single segment unique - build compound namespace
    for depth in range(2, max_depth + 1):
        namespaces = ['.'.join(p[:depth]) for p in parts_list]
        if len(set(namespaces)) == len(namespaces):
            return dict(zip(topics, namespaces, strict=False))

    # Fallback: full path
    return {t: '.'.join([p for p in t.split('/') if p]) for t in topics}


def get_namespaced_names(spec: StreamSpec) -> list[str]:
    """Get selector names with namespace prefix applied."""
    if spec.namespace:
        return [f'{spec.namespace}.{n}' for n in spec.names]
    return list(spec.names)


# =============================================================================
# Op Pipeline Resolution
# =============================================================================


def _native_dtype(msg_type: str, fallback: str = 'float64') -> str:
    """The codec's declared dtype for a message type, or a fallback."""
    return get_decoder_dtype(msg_type) if msg_type in DTYPES else fallback


def _build_ops(
    apply: list[tuple[str, Any]] | None,
    *,
    is_image: bool = False,
    image_channels: int | None = None,
) -> tuple:
    """Resolve a parsed ``apply`` list into a tuple of runtime op instances."""
    ctx = OpContext(is_image=is_image, image_channels=image_channels)
    return tuple(build_op(name, args, ctx) for name, args in (apply or []))


def _resize_from_ops(ops: tuple) -> tuple[int, int] | None:
    """Output (h, w) from a ``resize`` op in the pipeline, if present."""
    for op in ops:
        if op.name == 'resize':
            return (op.height, op.width)
    return None


# =============================================================================
# Generic Spec Builder
# =============================================================================

T = TypeVar('T')


def _apply_namespaces(
    items: list[tuple[str, dict, T]],  # (topic, kwargs, original)
    key_getter: callable,
    spec_class: type,
    forbid_image_aggregation: bool = False,
) -> Iterable[T]:
    """
    Apply namespace derivation and yield final specs.

    Args:
    ----
    items: List of (topic, spec_kwargs, original_item) tuples
    key_getter: Function to get the key from original_item
    spec_class: The spec class to instantiate
    forbid_image_aggregation: If True, raise error on multi-topic image keys

    """
    # Group by key
    by_key: dict[str, list[tuple[str, dict, T]]] = {}
    for topic, kwargs, original in items:
        key = key_getter(original)
        by_key.setdefault(key, []).append((topic, kwargs, original))

    # Compute namespaces for multi-topic groups
    topic_to_namespace: dict[str, str] = {}
    for key, group in by_key.items():
        if len(group) > 1:
            if forbid_image_aggregation and group[0][1].get('is_image'):
                raise ContractValidationError(
                    f"Cannot aggregate multiple image topics under key '{key}'. "
                    f'Each image must have a unique key.'
                )
            topics = [topic for topic, _, _ in group]
            topic_to_namespace.update(_derive_namespaces(topics))

    # Yield specs with namespace
    for topic, kwargs, _ in items:
        ns = topic_to_namespace.get(topic, '')
        yield spec_class(**kwargs, namespace=ns if ns else None)


# =============================================================================
# Observation Spec Iteration
# =============================================================================


def iter_observation_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """
    Yield observation stream specs from a contract.

    Resolves dtypes, validates images, and computes namespaces for
    multi-topic aggregation under the same key.
    """
    items: list[tuple[str, dict, ObservationSpec]] = []

    for o in contract.observations:
        is_image = o.key.startswith('observation.images.')

        # Reject depth images
        if is_image and ('depth' in o.topic.lower() or 'depth' in o.key.lower()):
            raise ContractValidationError(
                f"Depth image observation '{o.key}' (topic: {o.topic}) is not supported. "
                f'LeRobot does not currently have proper depth image handling.'
            )

        # Image encoding/channels (resize is an op in `apply`, not here)
        encoding = 'bgr8'
        if o.image and 'encoding' in o.image:
            encoding = str(o.image['encoding']).lower()
        channels = _validate_image_encoding(encoding)
        if o.image and 'channels' in o.image:
            channels = int(o.image['channels'])

        # Resolve declared (feature) dtype: explicit > video > custom-decoder > native
        native = _native_dtype(o.type) if o.type in DTYPES else None
        if o.dtype:
            dtype = o.dtype
        elif is_image:
            dtype = 'video'
        elif o.decoder:
            dtype = 'float64'  # Custom decoder - default to float64 if not specified
        elif native:
            dtype = native
        else:
            raise ContractValidationError(
                f"No decoder registered for '{o.type}'. "
                f'Add a decoder in decoders.py, specify dtype '
                f'explicitly, or provide a custom decoder.'
            )

        ops = _build_ops(o.apply, is_image=is_image, image_channels=channels)
        resize = _resize_from_ops(ops)
        if is_image and resize is None:
            raise ContractValidationError(
                f"Image observation '{o.key}' must specify a resize op "
                f'(apply: [resize: [h, w]])'
            )

        al = o.align or AlignSpec()
        names = list(o.select or [])

        kwargs = {
            'key': o.key,
            'topic': o.topic,
            'msg_type': o.type,
            'names': names,
            'fps': contract.fps,
            'is_image': is_image,
            'image_resize': resize,
            'image_encoding': encoding,
            'image_channels': channels,
            'resample_policy': al.strategy,
            'asof_tol_ms': int(al.tol_ms),
            'stamp_src': al.stamp,
            'dtype': dtype,
            'qos': o.qos,
            'decoder': o.decoder,
            'ops': ops,
        }
        items.append((o.topic, kwargs, o))

    yield from _apply_namespaces(
        items, lambda o: o.key, ObservationStreamSpec, forbid_image_aggregation=True
    )


def iter_action_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """Yield action stream specs from a contract."""
    items: list[tuple[str, dict, ActionSpec]] = []

    for a in contract.actions:
        # Only require registered encoder if no custom encoder provided
        if not a.encoder and a.type not in ENCODERS:
            raise ContractValidationError(
                f"No encoder registered for '{a.type}' in action '{a.key}'. "
                f'Add an encoder in encoders.py or provide a custom encoder.'
            )

        names = list(a.select or [])
        ops = _build_ops(a.apply)

        kwargs = {
            'key': a.key,
            'topic': a.topic,
            'msg_type': a.type,
            'names': names,
            'fps': contract.fps,
            'stamp_src': contract.timestamp_source,
            'safety_behavior': (a.safety_behavior or 'none').lower(),
            'qos': a.qos,
            'decoder': a.decoder,
            'encoder': a.encoder,
            'ops': ops,
        }
        items.append((a.topic, kwargs, a))

    yield from _apply_namespaces(items, lambda a: a.key, ActionStreamSpec)


def iter_reward_as_action_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """
    Yield action stream specs derived from the contract's reward section.

    Used when is_classifier=True so that a reward classifier's policy output
    publishes to reward topics instead of action topics.
    """
    items: list[tuple[str, dict, ObservationSpec]] = []

    for o in contract.rewards:
        if o.decoder is None and o.type not in ENCODERS:
            raise ContractValidationError(
                f"No encoder registered for '{o.type}' in reward '{o.key}'. "
                f'Add an encoder in encoders.py or provide a custom encoder.'
            )

        names = list(o.select or [])
        if not names:
            names = ['data']

        # A reward used as an action must be invertible (it publishes).
        ops = _build_ops(o.apply)
        for op in ops:
            if not op.invertible:
                raise ContractValidationError(
                    f"Reward '{o.key}' used as an action has non-invertible op "
                    f"'{op.name}' in apply; remove it."
                )

        kwargs = {
            'key': 'action',
            'topic': o.topic,
            'msg_type': o.type,
            'names': names,
            'fps': contract.fps,
            'stamp_src': contract.timestamp_source,
            'safety_behavior': 'none',
            'qos': o.qos,
            'decoder': o.decoder,
            'encoder': None,
            'ops': ops,
        }
        items.append((o.topic, kwargs, o))

    yield from _apply_namespaces(items, lambda o: 'action', ActionStreamSpec)


def iter_extended_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """Yield specs from extended categories (rewards, signals, info, complementary_data)."""
    extended = [
        contract.rewards,
        contract.signals,
        contract.info,
        contract.complementary_data,
    ]

    for obs_list in extended:
        for o in obs_list:
            al = o.align or AlignSpec()
            names = list(o.select or [])
            ops = _build_ops(o.apply)

            yield ObservationStreamSpec(
                key=o.key,
                topic=o.topic,
                msg_type=o.type,
                names=names,
                fps=contract.fps,
                is_image=False,
                image_resize=None,
                image_encoding='',
                image_channels=0,
                resample_policy=al.strategy,
                asof_tol_ms=int(al.tol_ms),
                stamp_src=al.stamp,
                dtype=o.dtype,
                qos=o.qos,
                namespace=None,
                decoder=o.decoder,
                ops=ops,
            )


def iter_specs(contract: Contract) -> Iterable[ObservationStreamSpec | ActionStreamSpec]:
    """Yield all stream specs (observations, actions, extended)."""
    yield from iter_observation_specs(contract)
    yield from iter_action_specs(contract)
    yield from iter_extended_specs(contract)


# =============================================================================
# Teleop Spec Iteration
# =============================================================================


def iter_teleop_input_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """Yield teleop input stream specs."""
    if not contract.teleop:
        return

    items: list[tuple[str, dict, ObservationSpec]] = []

    for o in contract.teleop.inputs:
        # Resolve dtype
        if o.dtype:
            dtype = o.dtype
        elif o.decoder:
            dtype = 'float64'  # Default for custom decoder
        else:
            if o.type not in DTYPES:
                raise ContractValidationError(
                    f"No decoder registered for '{o.type}' in teleop input '{o.key}'. "
                    f'Provide a custom decoder or add one in decoders.py.'
                )
            dtype = get_decoder_dtype(o.type)

        al = o.align or AlignSpec()
        names = list(o.select or [])
        ops = _build_ops(o.apply)

        kwargs = {
            'key': o.key,
            'topic': o.topic,
            'msg_type': o.type,
            'names': names,
            'fps': contract.fps,
            'is_image': False,
            'image_resize': None,
            'image_encoding': '',
            'image_channels': 0,
            'resample_policy': al.strategy,
            'asof_tol_ms': int(al.tol_ms),
            'stamp_src': al.stamp,
            'dtype': dtype,
            'qos': o.qos,
            'decoder': o.decoder,
            'ops': ops,
        }
        items.append((o.topic, kwargs, o))

    yield from _apply_namespaces(items, lambda o: o.key, ObservationStreamSpec)


def iter_teleop_feedback_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """Yield teleop feedback stream specs."""
    if not contract.teleop:
        return

    items: list[tuple[str, dict, ActionSpec]] = []

    for a in contract.teleop.feedback:
        names = list(a.select or [])
        ops = _build_ops(a.apply)

        kwargs = {
            'key': a.key,
            'topic': a.topic,
            'msg_type': a.type,
            'names': names,
            'fps': contract.fps,
            'stamp_src': contract.timestamp_source,
            'safety_behavior': 'none',
            'qos': a.qos,
            'decoder': a.decoder,
            'encoder': a.encoder,
            'ops': ops,
        }
        items.append((a.topic, kwargs, a))

    yield from _apply_namespaces(items, lambda a: a.key, ActionStreamSpec)
