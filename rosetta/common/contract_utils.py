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
from typing import TYPE_CHECKING, Any, Iterable, TypeVar

import numpy as np

from .contract import (DEPTH_ENCODINGS, ActionSpec, ActionStreamSpec,
                       AlignSpec, Contract, ContractValidationError,
                       ObservationSpec, ObservationStreamSpec, ResamplePolicy,
                       StreamSpec)
from .converters import DTYPES, ENCODERS, get_decoder_dtype

if TYPE_CHECKING:
    pass  # All types imported above


# =============================================================================
# Feature Building
# =============================================================================


def build_feature(spec: ObservationStreamSpec | ActionStreamSpec) -> dict[str, Any]:
    """Build LeRobot feature dict from a spec.

    Returns a dict with:
    - dtype: LeRobot dtype string
    - shape: tuple of dimensions
    - names: axis names (or None)
    """
    dtype = getattr(spec, "dtype", None)
    if dtype is None:
        raise ValueError(f"Spec '{spec.key}' has no dtype")

    if dtype in ("video", "image"):
        if not spec.image_shape:
            raise ValueError(f"Image spec '{spec.key}' must have image_shape")
        h, w = spec.image_shape
        return {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }

    if dtype == "string":
        return {"dtype": "string", "shape": (1,), "names": None}

    # Numeric types (float32, float64, int32, int64, bool)
    n = len(spec.names) if spec.names else 1
    names = list(spec.names) if spec.names else None
    return {"dtype": dtype, "shape": (n,), "names": names}


def build_features(specs: Iterable[StreamSpec]) -> dict[str, dict[str, Any]]:
    """Build the LeRobot feature dict for a list of contract stream specs.

    Specs sharing the same ``key`` are aggregated: numeric vectors concatenate
    their namespaced names (in spec order); images/strings take the first spec.
    This is the single source of the contract -> dataset-feature mapping, reused
    by ``port_bags`` (dataset writing) and the contract validators (so a config's
    derived layout is identical to what porting produces). Frame-boundary markers
    (``is_first``/``is_last``/``is_terminal``) are added by the dataset writer,
    not here, since they are not contract-derived.
    """
    by_key: dict[str, list[StreamSpec]] = {}
    for spec in specs:
        by_key.setdefault(spec.key, []).append(spec)

    features: dict[str, dict[str, Any]] = {}
    for key, key_specs in by_key.items():
        first = key_specs[0]
        dtype = DTYPES[first.msg_type]

        if dtype in ("video", "image", "string"):
            # No aggregation for images/strings.
            features[key] = build_feature(first)
        else:
            # Numeric: aggregate names from all specs sharing this key.
            all_names: list[str] = []
            for spec in key_specs:
                all_names.extend(get_namespaced_names(spec))
            n = len(all_names) or 1
            features[key] = {
                "dtype": dtype,
                "shape": (n,),
                "names": all_names if all_names else None,
            }

    return features


def _aggregate_namespaced_names(
    items: list[tuple[str, str, list[str]]]
) -> dict[str, list[str]]:
    """Aggregate selector names per key, namespaced exactly as the dataset
    writer does it (shared :func:`_topic_namespace_map` policy): namespaces
    derive from multi-topic keys but are applied globally **by topic**, so a
    single-topic key whose topic also feeds a multi-topic key is prefixed
    here just as it is in the written dataset. ``items`` is a list of
    ``(key, topic, names)`` tuples.
    """
    by_key: dict[str, list[tuple[str, list[str]]]] = {}
    for key, topic, names in items:
        by_key.setdefault(key, []).append((topic, names))

    ns_map = _topic_namespace_map(
        [[topic for topic, _ in group] for group in by_key.values()]
    )

    out: dict[str, list[str]] = {}
    for key, group in by_key.items():
        agg: list[str] = []
        for topic, names in group:
            ns = ns_map.get(topic, "")
            agg.extend(f"{ns}.{n}" if ns else n for n in names)
        out[key] = agg
    return out


def contract_interface(contract: Contract) -> dict[str, Any]:
    """ROS-free extraction of a contract's data interface.

    Unlike ``build_features(iter_specs(...))`` this does **not** resolve dtypes
    (which requires the ROS decoder registry / rclpy), so it is usable in the
    training environment and CI where ROS is absent. It returns enough to
    validate a contract against a LeRobot ``dataset.meta`` or a checkpoint
    ``config.json``: image keys+shapes and the aggregated, namespaced vector
    names (hence dims) for the state/action keys.

    Returns a mapping::

        {
          "robot_type": str,
          "fps": int,
          "images": {"<obs key>": [H, W], ...},
          "state":   {"<obs key>": {"names": [...], "dim": N}, ...},
          "actions": {"<action key>": {"names": [...], "dim": N}, ...},
        }
    """
    images: dict[str, list[int]] = {}
    numeric_items: list[tuple[str, str, list[str]]] = []
    for o in contract.observations:
        if o.key.startswith("observation.images."):
            shape = (o.image or {}).get("shape") or (o.image or {}).get("resize")
            if not shape or len(shape) != 2:
                raise ContractValidationError(
                    f"Image observation '{o.key}' must specify image.shape [height, width]"
                )
            images[o.key] = [int(shape[0]), int(shape[1])]
        else:
            numeric_items.append(
                (o.key, o.topic, list((o.selector or {}).get("names", [])))
            )

    action_items = [
        (a.key, a.publish_topic, list((a.selector or {}).get("names", [])))
        for a in contract.actions
    ]

    state = {
        k: {"names": v, "dim": len(v)}
        for k, v in _aggregate_namespaced_names(numeric_items).items()
    }
    actions = {
        k: {"names": v, "dim": len(v)}
        for k, v in _aggregate_namespaced_names(action_items).items()
    }

    return {
        "robot_type": contract.robot_type,
        "fps": contract.fps,
        "images": images,
        "state": state,
        "actions": actions,
    }


# =============================================================================
# Zero-fill for Missing Data
# =============================================================================


def zeros_for_spec(spec: ObservationStreamSpec) -> np.ndarray:
    """Create a zero-filled array for missing data.

    Args:
        spec: Observation stream spec defining shape/type.

    Returns:
        Zero-filled numpy array:
        - Images: (H, W, C) uint8
        - Vectors: (N,) with dtype from spec
    """
    if spec.is_image:
        h, w = spec.image_shape
        return np.zeros((h, w, spec.image_channels), dtype=np.uint8)

    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "bool": bool,
    }
    np_dtype = dtype_map.get(spec.dtype, np.float32)
    return np.zeros(len(spec.names), dtype=np_dtype)


# =============================================================================
# Stream Buffer
# =============================================================================


class StreamBuffer:
    """Thread-safe, constant-memory online resampler.

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
    def from_spec(cls, spec: ObservationStreamSpec) -> "StreamBuffer":
        """Create a StreamBuffer from an ObservationStreamSpec."""
        step_ns = int(1e9 / spec.fps) if spec.fps > 0 else int(1e9 / 30)
        tol_ns = spec.asof_tol_ms * 1_000_000
        return cls(policy=spec.resample_policy, step_ns=step_ns, tol_ns=tol_ns)

    def push(self, ts_ns: int, val: Any) -> None:
        """Insert a sample (keeps the newest by timestamp)."""
        with self._lock:
            if self.last_ts is None or ts_ns >= self.last_ts:
                self.last_ts, self.last_val = ts_ns, val

    def reset(self) -> None:
        """Clear buffered data (e.g., between episodes)."""
        with self._lock:
            self.last_ts = None
            self.last_val = None

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
                return (
                    self.last_val if (self.last_ts > tick_ns - self.step_ns) else None
                )
            if self.policy == ResamplePolicy.ASOF.value:
                return (
                    self.last_val if (tick_ns - self.last_ts <= self.tol_ns) else None
                )
            return self.last_val  # hold is default

    def _clear_unsafe(self) -> None:
        """Clear buffered data without acquiring lock (internal use only)."""
        self.last_ts = None
        self.last_val = None

    def reset(self) -> None:
        """Clear buffered data (e.g., between episodes or after sim reset)."""
        with self._lock:
            self._clear_unsafe()


# =============================================================================
# Image Encoding Validation
# =============================================================================

_ENCODING_CHANNELS = {
    "mono8": 1,
    "8uc1": 1,
    "bgr8": 3,
    "rgb8": 3,
    "bgr16": 3,
    "rgb16": 3,
    "bgra8": 4,
    "rgba8": 4,
}


def _validate_image_encoding(encoding: str) -> int:
    """Validate encoding and return channel count.

    Raises:
        ContractValidationError: If encoding is unsupported or is depth format.
    """
    enc = encoding.lower()

    if enc in DEPTH_ENCODINGS:
        raise ContractValidationError(
            f"Depth image encoding '{encoding}' is not supported. "
            f"LeRobot does not currently have proper depth image handling."
        )

    channels = _ENCODING_CHANNELS.get(enc)
    if channels is None:
        raise ContractValidationError(
            f"Unsupported image encoding '{encoding}'. "
            f"Supported: {', '.join(sorted(_ENCODING_CHANNELS.keys()))}"
        )
    return channels


# =============================================================================
# Namespace Derivation
# =============================================================================


def _derive_namespaces(topics: list[str]) -> dict[str, str]:
    """Derive unique namespace for each topic from path segments.

    Finds the first segment that uniquely identifies each topic.

    Examples:
        ['/arm/state', '/base/state'] -> {'...': 'arm', '...': 'base'}
        ['/arm/pos', '/arm/vel'] -> {'...': 'pos', '...': 'vel'}
    """
    if len(topics) <= 1:
        return {t: "" for t in topics}

    parts_list = [[p for p in t.split("/") if p] for t in topics]
    max_depth = max(len(p) for p in parts_list)

    # Find first segment where all topics differ
    for i in range(max_depth):
        segments = [p[i] if i < len(p) else "" for p in parts_list]
        if len(set(segments)) == len(topics):
            return dict(zip(topics, segments))

    # No single segment unique - build compound namespace
    for depth in range(2, max_depth + 1):
        namespaces = [".".join(p[:depth]) for p in parts_list]
        if len(set(namespaces)) == len(namespaces):
            return dict(zip(topics, namespaces))

    # Fallback: full path
    return {t: ".".join([p for p in t.split("/") if p]) for t in topics}


def _topic_namespace_map(groups: list[list[str]]) -> dict[str, str]:
    """Build the global ``topic -> namespace`` map the dataset writer applies.

    ``groups`` holds each key's topic list. Only multi-topic groups derive
    namespaces, but the merged map is applied **by topic** to every spec —
    so a topic that also feeds a single-topic key is prefixed there too.
    This is the single source of the namespacing policy, shared by the
    dataset writer (:func:`_apply_namespaces`) and the ROS-free validator
    path (:func:`_aggregate_namespaced_names`); the two must agree or the
    validators flag drift on datasets their own contract produced.

    A topic appearing in two multi-topic groups whose derivations disagree
    would make the merged map depend on key order (the old silent-overwrite
    behavior); that ambiguity is rejected instead.
    """
    out: dict[str, str] = {}
    for topics in groups:
        if len(topics) <= 1:
            continue
        derived = _derive_namespaces(topics)
        for topic, ns in derived.items():
            if topic in out and out[topic] != ns:
                raise ContractValidationError(
                    f"Topic '{topic}' is aggregated under multiple keys with "
                    f"conflicting derived namespaces ('{out[topic]}' vs "
                    f"'{ns}'); dataset feature names would depend on key "
                    f"order. Rename the topics or keys to disambiguate."
                )
        out.update(derived)
    return out


def get_namespaced_names(spec: StreamSpec) -> list[str]:
    """Get selector names with namespace prefix applied."""
    if spec.namespace:
        return [f"{spec.namespace}.{n}" for n in spec.names]
    return list(spec.names)


# =============================================================================
# Generic Spec Builder
# =============================================================================

T = TypeVar("T")


def _apply_namespaces(
    items: list[tuple[str, dict, T]],  # (topic, kwargs, original)
    key_getter: callable,
    spec_class: type,
    forbid_image_aggregation: bool = False,
) -> Iterable[T]:
    """Apply namespace derivation and yield final specs.

    Args:
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

    for key, group in by_key.items():
        if len(group) > 1 and forbid_image_aggregation and group[0][1].get("is_image"):
            raise ContractValidationError(
                f"Cannot aggregate multiple image topics under key '{key}'. "
                f"Each image must have a unique key."
            )

    # Namespaces derive from multi-topic groups but apply globally by topic
    # (shared policy with the validator path — see _topic_namespace_map).
    topic_to_namespace = _topic_namespace_map(
        [[topic for topic, _, _ in group] for group in by_key.values()]
    )

    # Yield specs with namespace
    for topic, kwargs, _ in items:
        ns = topic_to_namespace.get(topic, "")
        yield spec_class(**kwargs, namespace=ns if ns else None)


# =============================================================================
# Observation Spec Iteration
# =============================================================================


def iter_observation_specs(contract: Contract) -> Iterable[ObservationStreamSpec]:
    """Yield observation stream specs from a contract.

    Resolves dtypes, validates images, and computes namespaces for
    multi-topic aggregation under the same key.
    """
    items: list[tuple[str, dict, ObservationSpec]] = []

    for o in contract.observations:
        is_image = o.key.startswith("observation.images.")

        # Reject depth images
        if is_image and ("depth" in o.topic.lower() or "depth" in o.key.lower()):
            raise ContractValidationError(
                f"Depth image observation '{o.key}' (topic: {o.topic}) is not supported. "
                f"LeRobot does not currently have proper depth image handling."
            )

        # Parse image config
        img_shape = None
        encoding = "bgr8"
        if o.image:
            # "shape" is the canonical key; "resize" is accepted for backward compat
            r = o.image.get("shape") or o.image.get("resize")
            if r and len(r) == 2:
                img_shape = (int(r[0]), int(r[1]))
            if "encoding" in o.image:
                encoding = str(o.image["encoding"]).lower()

        if is_image and img_shape is None:
            raise ContractValidationError(
                f"Image observation '{o.key}' must specify image.shape [height, width]"
            )

        channels = _validate_image_encoding(encoding)
        if o.image and "channels" in o.image:
            channels = int(o.image["channels"])

        # Resolve dtype
        if o.dtype:
            dtype = o.dtype
        elif is_image:
            dtype = "video"
        elif o.decoder:
            # Custom decoder - default to float64 if not specified
            dtype = "float64"
        else:
            if o.type not in DTYPES:
                raise ContractValidationError(
                    f"No decoder registered for '{o.type}'. "
                    f"Add a decoder in decoders.py, specify dtype explicitly, or provide a custom decoder."
                )
            dtype = get_decoder_dtype(o.type)

        al = o.align or AlignSpec()
        names = list((o.selector or {}).get("names", []))

        kwargs = dict(
            key=o.key,
            topic=o.topic,
            msg_type=o.type,
            names=names,
            fps=contract.fps,
            is_image=is_image,
            image_shape=img_shape,
            image_encoding=encoding,
            image_channels=channels,
            resample_policy=al.strategy,
            asof_tol_ms=int(al.tol_ms),
            stamp_src=al.stamp,
            dtype=dtype,
            qos=o.qos,
            decoder=o.decoder,
            unit_conversion=o.unit_conversion,
        )
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
                f"Add an encoder in encoders.py or provide a custom encoder."
            )

        names = list((a.selector or {}).get("names", []))
        clamp = None
        if a.from_tensor and "clamp" in a.from_tensor:
            lo, hi = a.from_tensor["clamp"]
            clamp = (float(lo), float(hi))

        kwargs = dict(
            key=a.key,
            topic=a.publish_topic,
            msg_type=a.type,
            names=names,
            fps=contract.fps,
            stamp_src=contract.timestamp_source,
            clamp=clamp,
            safety_behavior=(a.safety_behavior or "none").lower(),
            qos=a.publish_qos,
            decoder=a.decoder,
            encoder=a.encoder,
            unit_conversion=a.unit_conversion,
        )
        items.append((a.publish_topic, kwargs, a))

    yield from _apply_namespaces(items, lambda a: a.key, ActionStreamSpec)


def iter_reward_as_action_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """Yield action stream specs derived from the contract's reward section.

    Used when is_classifier=True so that a reward classifier's policy output
    publishes to reward topics instead of action topics.
    """
    items: list[tuple[str, dict, ObservationSpec]] = []

    for o in contract.rewards:
        if o.decoder is None and o.type not in ENCODERS:
            raise ContractValidationError(
                f"No encoder registered for '{o.type}' in reward '{o.key}'. "
                f"Add an encoder in encoders.py or provide a custom encoder."
            )

        names = list((o.selector or {}).get("names", []))
        if not names:
            names = ["data"]

        kwargs = dict(
            key="action",
            topic=o.topic,
            msg_type=o.type,
            names=names,
            fps=contract.fps,
            stamp_src=contract.timestamp_source,
            clamp=None,
            safety_behavior="none",
            qos=o.qos,
            decoder=o.decoder,
            encoder=None,
            unit_conversion=o.unit_conversion,
        )
        items.append((o.topic, kwargs, o))

    yield from _apply_namespaces(items, lambda o: "action", ActionStreamSpec)


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
            names = list((o.selector or {}).get("names", []))

            yield ObservationStreamSpec(
                key=o.key,
                topic=o.topic,
                msg_type=o.type,
                names=names,
                fps=contract.fps,
                is_image=False,
                image_shape=None,
                image_encoding="",
                image_channels=0,
                resample_policy=al.strategy,
                asof_tol_ms=int(al.tol_ms),
                stamp_src=al.stamp,
                dtype=o.dtype,
                qos=o.qos,
                namespace=None,
                decoder=o.decoder,
                unit_conversion=o.unit_conversion,
            )


def iter_specs(
    contract: Contract,
) -> Iterable[ObservationStreamSpec | ActionStreamSpec]:
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
            dtype = "float64"  # Default for custom decoder
        else:
            if o.type not in DTYPES:
                raise ContractValidationError(
                    f"No decoder registered for '{o.type}' in teleop input '{o.key}'. "
                    f"Provide a custom decoder or add one in decoders.py."
                )
            dtype = get_decoder_dtype(o.type)

        al = o.align or AlignSpec()
        names = list((o.selector or {}).get("names", []))

        kwargs = dict(
            key=o.key,
            topic=o.topic,
            msg_type=o.type,
            names=names,
            fps=contract.fps,
            is_image=False,
            image_shape=None,
            image_encoding="",
            image_channels=0,
            resample_policy=al.strategy,
            asof_tol_ms=int(al.tol_ms),
            stamp_src=al.stamp,
            dtype=dtype,
            qos=o.qos,
            decoder=o.decoder,
            unit_conversion=o.unit_conversion,
        )
        items.append((o.topic, kwargs, o))

    yield from _apply_namespaces(items, lambda o: o.key, ObservationStreamSpec)


def iter_teleop_feedback_specs(contract: Contract) -> Iterable[ActionStreamSpec]:
    """Yield teleop feedback stream specs."""
    if not contract.teleop:
        return

    items: list[tuple[str, dict, ActionSpec]] = []

    for a in contract.teleop.feedback:
        names = list((a.selector or {}).get("names", []))
        clamp = None
        if a.from_tensor and "clamp" in a.from_tensor:
            lo, hi = a.from_tensor["clamp"]
            clamp = (float(lo), float(hi))

        kwargs = dict(
            key=a.key,
            topic=a.publish_topic,
            msg_type=a.type,
            names=names,
            fps=contract.fps,
            stamp_src=contract.timestamp_source,
            clamp=clamp,
            safety_behavior="none",
            qos=a.publish_qos,
            decoder=a.decoder,
            encoder=a.encoder,
            unit_conversion=a.unit_conversion,
        )
        items.append((a.publish_topic, kwargs, a))

    yield from _apply_namespaces(items, lambda a: a.key, ActionStreamSpec)
