# rosetta/common/contract_utils.py
# -----------------------------------------------------------------------------
# Contract schema + loader + runtime processing.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import hashlib
import json
import numpy as np
import yaml
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

# ---------- Contract datamodel ----------


@dataclass(frozen=True, slots=True)
class AlignSpec:
    """Timestamp alignment/selection behavior for an observation stream.

    strategy:  "hold" | "asof" | "drop"
    tol_ms:    as-of tolerance in ms for 'asof' (ignored for others)
    stamp:     "receive" | "header"
    """

    strategy: str = "hold"
    tol_ms: int = 0
    stamp: str = "receive"


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Observation stream description (image/vector), driven by AlignSpec."""

    key: str
    topic: str
    type: str
    selector: Optional[Dict[str, Any]] = None  # {names: [...]}
    image: Optional[Dict[str, Any]] = (
        None  # {resize:[H,W], encoding:'rgb8'|'bgr8'|'mono8'...}
    )
    align: Optional[AlignSpec] = None
    # {reliability, history, depth, durability}
    qos: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Action stream description (publisher settings + mapping).

    YAML (example):
      actions:
        - key: action
          publish:
            topic: /cmd_vel
            type: geometry_msgs/msg/Twist
            qos: { reliability: best_effort, history: keep_last, depth: 10 }
            strategy: { mode: nearest, tolerance_ms: 500 }
          selector: { names: [linear.x, angular.z] }
          from_tensor: { clamp: [-2.0, 2.0] }
          safety_behavior: hold
    """

    key: str
    publish_topic: str
    type: str
    selector: Optional[Dict[str, Any]] = None
    from_tensor: Optional[Dict[str, Any]] = None
    publish_qos: Optional[Dict[str, Any]] = None
    publish_strategy: Optional[Dict[str, Any]] = None
    safety_behavior: str = "zeros"  # "zeros" | "hold"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Optional 'task' channels (e.g., prompts)."""

    key: str
    topic: str
    type: str
    qos: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class Contract:
    """Top-level contract describing a policy's ROS 2 I/O surface."""

    name: str
    version: int
    rate_hz: float
    max_duration_s: float
    observations: List[ObservationSpec]
    actions: List[ActionSpec]
    tasks: List[TaskSpec]
    recording: Dict[str, Any]
    robot_type: Optional[str] = None
    timestamp_source: str = "receive"


def _as_align(d: Optional[Dict[str, Any]]) -> Optional[AlignSpec]:
    if not d:
        return None
    return AlignSpec(
        strategy=str(d.get("strategy", "hold")).lower(),
        tol_ms=int(d.get("tol_ms", 0)),
        stamp=str(d.get("stamp", "receive")).lower(),
    )


def load_contract(path: Path | str) -> Contract:
    """Load + normalize contract YAML into dataclasses."""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    def _obs(it: Dict[str, Any]) -> ObservationSpec:
        return ObservationSpec(
            key=it["key"],
            topic=it["topic"],
            type=it["type"],
            selector=it.get("selector"),
            image=it.get("image"),
            align=_as_align(it.get("align")),
            qos=it.get("qos"),
        )

    def _act(it: Dict[str, Any]) -> ActionSpec:
        pub = it["publish"]
        # new: read top-level safety_behavior (default "zeros")
        sb = str(it.get("safety_behavior", "zeros")).lower().strip()
        if sb not in ("zeros", "hold"):
            sb = "zeros"
        return ActionSpec(
            key=it["key"],
            publish_topic=pub["topic"],
            type=pub["type"],
            selector=it.get("selector"),
            from_tensor=it.get("from_tensor"),
            publish_qos=pub.get("qos"),
            publish_strategy=pub.get("strategy"),
            safety_behavior=sb,
        )

    def _task(it: Dict[str, Any]) -> TaskSpec:
        return TaskSpec(
            key=it.get("key", it["topic"]),
            topic=it["topic"],
            type=it["type"],
            qos=it.get("qos"),
        )

    obs = [_obs(it) for it in (d.get("observations") or [])]
    acts = [_act(it) for it in (d.get("actions") or [])]
    tks = [_task(it) for it in (d.get("tasks") or [])]
    rec = d.get("recording") or {}

    return Contract(
        name=d.get("name", "contract"),
        version=int(d.get("version", 1)),
        rate_hz=float(d.get("rate_hz", d.get("fps", 20.0))),
        max_duration_s=float(d.get("max_duration_s", 30.0)),
        observations=obs,
        actions=acts,
        tasks=tks,
        recording=rec,
        robot_type=d.get("robot_type"),
        timestamp_source=str(d.get("timestamp_source", "receive")).lower(),
    )


# ---------- QoS mapping (shared) ----------
# Note: qos_profile_from_dict moved to processing_utils.py


# ---------- Unified SpecView (runtime) ----------


@dataclass(frozen=True, slots=True)
class SpecView:
    """Normalized runtime view of a single stream (observation or action)."""

    key: str
    topic: str
    ros_type: str
    is_action: bool
    names: List[str]
    image_resize: Optional[Tuple[int, int]]
    image_encoding: str
    image_channels: int  # 1, 3, or 4
    resample_policy: str  # obs: align.strategy; actions: "hold"
    asof_tol_ms: int
    stamp_src: str
    clamp: Optional[Tuple[float, float]]  # actions only
    safety_behavior: Optional[str]  # actions only: "zeros" | "hold"
    qos: Optional[Dict[str, Any]] = None  # observation subscription QoS
    publish_qos: Optional[Dict[str, Any]] = None  # action publisher QoS


def _num_channels_from_encoding(encoding: str) -> int:
    """Infer channel count from ROS image encoding (based on sensor_msgs/image_encodings.h)."""
    enc = encoding.lower()
    
    # Common-case encodings
    if enc in ("mono8", "mono16"):
        return 1
    if enc in ("bgr8", "rgb8", "bgr16", "rgb16"):
        return 3
    if enc in ("bgra8", "rgba8", "bgra16", "rgba16"):
        return 4
    
    # Bayer encodings (all single channel raw mosaics)
    if enc in ("bayer_rggb8", "bayer_bggr8", "bayer_gbrg8", "bayer_grbg8",
               "bayer_rggb16", "bayer_bggr16", "bayer_gbrg16", "bayer_grbg16"):
        return 1
    
    # Generic content encodings (8UC1, 16UC3, 32FC1, etc.)
    abstract_prefixes = ["8uc", "8sc", "16uc", "16sc", "32sc", "32fc", "64fc"]
    for prefix in abstract_prefixes:
        if enc.startswith(prefix):
            if len(enc) == len(prefix):
                return 1  # e.g., 8UC -> 1
            try:
                # Extract channel count from suffix (e.g., 8UC3 -> 3)
                channel_str = enc[len(prefix):]
                if channel_str.isdigit():
                    return int(channel_str)
            except (ValueError, IndexError):
                pass
    
    # Special cases
    if enc == "yuv422":
        return 2
    
    # Default fallback for unknown encodings
    return 3  # assume RGB for unknown encodings


def iter_specs(contract: Contract) -> Iterable[SpecView]:
    """Yield normalized runtime specs for observations and actions."""
    # Observations
    for o in contract.observations:
        resize = None
        # Smart default encoding based on topic name and key
        default_enc = "bgr8"  # default for color images
        
        # Heuristic: if topic/key contains "depth", assume depth encoding
        if "depth" in o.topic.lower() or "depth" in o.key.lower():
            default_enc = "32fc1"  # common depth encoding
        
        if o.image:
            r = o.image.get("resize")
            if r and len(r) == 2:
                resize = (int(r[0]), int(r[1]))
            # Only use contract encoding if explicitly specified (for shape inference)
            if "encoding" in o.image:
                default_enc = str(o.image.get("encoding")).lower()
        
        # Infer channels from default encoding (for dataset shape)
        # At runtime, decoder will read actual msg.encoding
        channels = _num_channels_from_encoding(default_enc)
        
        # Special case: depth images are always replicated to 3 channels by the decoder
        if "depth" in o.topic.lower() or "depth" in o.key.lower():
            channels = 3
            
        if o.image and "channels" in o.image:
            channels = int(o.image["channels"])  # explicit override

        names = list((o.selector or {}).get("names", []))
        al = o.align or AlignSpec()  # default: hold/0ms/receive

        yield SpecView(
            key=o.key,
            topic=o.topic,
            ros_type=o.type,
            is_action=False,
            names=names,
            image_resize=resize,
            image_encoding=default_enc,  # Used for shape inference only
            image_channels=channels,
            resample_policy=al.strategy,
            asof_tol_ms=int(al.tol_ms),
            stamp_src=al.stamp,
            clamp=None,
            safety_behavior=None,
            qos=o.qos,  # per-spec QoS for subscription
            publish_qos=None,  # observations don't publish
        )

    # Actions
    for a in contract.actions:
        names = list((a.selector or {}).get("names", []))
        clamp: Optional[Tuple[float, float]] = None
        if a.from_tensor and "clamp" in a.from_tensor:
            lo, hi = a.from_tensor["clamp"]
            clamp = (float(lo), float(hi))

        yield SpecView(
            key=a.key,
            topic=a.publish_topic,
            ros_type=a.type,
            is_action=True,
            names=names,
            image_resize=None,
            image_encoding="bgr8",
            image_channels=3,  # actions don't use images
            resample_policy="hold",  # actions are outputs
            asof_tol_ms=0,
            stamp_src=contract.timestamp_source,
            clamp=clamp,
            safety_behavior=(a.safety_behavior or "zeros").lower(),
            qos=None,  # actions don't subscribe
            publish_qos=a.publish_qos,  # per-spec QoS for publication
        )


# ---------- LeRobot feature helpers ----------


def feature_from_spec(
    spec: SpecView, use_videos: bool
) -> Tuple[str, Dict[str, Any], bool]:
    """Return (key, feature_meta, is_image)."""
    if spec.image_resize:
        h, w = int(spec.image_resize[0]), int(spec.image_resize[1])
        dtype = "video" if use_videos else "image"
        return (
            spec.key,
            {
                "dtype": dtype,
                "shape": (h, w, spec.image_channels),
                "names": ["height", "width", "channel"],
            },
            True,
        )

    if not spec.names:
        raise ValueError(
            f"{spec.key}: vector features must specify selector.names")

    return (
        spec.key,
        {"dtype": "float32", "shape": (
            len(spec.names),), "names": list(spec.names)},
        False,
    )


# ---------- Decoder/Encoder registries ----------

DecoderFn = Callable[[Any, Any], Any]
DECODERS: Dict[str, DecoderFn] = {}


def register_decoder(type_str: str):
    """Decorator to register a decoder for a given ROS type string."""

    def _wrap(fn: DecoderFn):
        DECODERS[type_str] = fn
        return fn

    return _wrap


EncoderFn = Callable[[list[str], Any, Any], Any]
ENCODERS: Dict[str, EncoderFn] = {}


def register_encoder(type_str: str):
    """Decorator to register an encoder for a given ROS type string."""

    def _wrap(fn: EncoderFn):
        ENCODERS[type_str] = fn
        return fn

    return _wrap


# ---------- Time helpers ----------


def stamp_from_header_ns(msg) -> Optional[int]:
    """Extract a monotonic-ish nanosecond timestamp from a ROS Header."""
    try:
        st = msg.header.stamp
        return int(st.sec) * 1_000_000_000 + int(st.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None


# ---------- Decoders (ROS -> numpy/str) ----------


def decode_value(ros_type: str, msg, spec) -> Any:
    """Decode a ROS message using a registered decoder."""
    fn = DECODERS.get(ros_type)
    if not fn:
        raise ValueError(f"No decoder registered for ROS type: {ros_type}")
    return fn(msg, spec)


# ---------- Resampling (offline) ----------


def resample_hold(
    ts_ns: np.ndarray, vals: List[Any], ticks_ns: np.ndarray
) -> List[Any]:
    """Classic last-value-hold: for each tick, carry forward the last sample."""
    out: List[Any] = []
    j, last = 0, None

    # If we have data but the first tick is before the first data, use the first data for early ticks
    if len(ts_ns) > 0 and len(ticks_ns) > 0 and ticks_ns[0] < ts_ns[0]:
        last = vals[0]  # Use first available data

    for t in ticks_ns:
        while j + 1 < len(ts_ns) and ts_ns[j + 1] <= t:
            j += 1
        if j < len(vals) and ts_ns[j] <= t:
            last = vals[j]
        out.append(last)
    return out


def resample_asof(
    ts_ns: np.ndarray, vals: List[Any], ticks_ns: np.ndarray, tol_ns: int
) -> List[Optional[Any]]:
    """As-of: last value only if not older than tol_ns at the tick; else None."""
    if tol_ns <= 0:
        return resample_hold(ts_ns, vals, ticks_ns)
    out: List[Optional[Any]] = []
    j = 0
    for t in ticks_ns:
        while j + 1 < len(ts_ns) and ts_ns[j + 1] <= t:
            j += 1
        ok = j < len(vals) and ts_ns[j] <= t and (t - ts_ns[j]) <= tol_ns
        out.append(vals[j] if ok else None)
    return out


def resample_drop(
    ts_ns: np.ndarray, vals: List[Any], ticks_ns: np.ndarray, step_ns: int
) -> List[Optional[Any]]:
    """Drop: only return a value if it arrived within (t - step_ns, t]."""
    out: List[Optional[Any]] = []
    j, n = -1, len(ts_ns)
    for t in ticks_ns:
        while j + 1 < n and ts_ns[j + 1] <= t:
            j += 1
        out.append(vals[j] if (j >= 0 and ts_ns[j] > t - step_ns) else None)
    return out


def resample(
    policy: str,
    ts_ns: np.ndarray,
    vals: List[Any],
    ticks_ns: np.ndarray,
    step_ns: int,
    tol_ms: int,
) -> List[Any]:
    """Dispatch resampling policy: 'hold' | 'asof' | 'drop'."""
    if policy == "drop":
        return resample_drop(ts_ns, vals, ticks_ns, step_ns)
    if policy == "asof":
        return resample_asof(ts_ns, vals, ticks_ns, max(0, int(tol_ms)) * 1_000_000)
    return resample_hold(ts_ns, vals, ticks_ns)


# ---------- Live resampling (online) ----------


class StreamBuffer:
    """Constant-memory online resampler."""

    def __init__(self, policy: str, step_ns: int, tol_ns: int = 0):
        self.policy = policy
        self.step_ns = int(step_ns)
        self.tol_ns = int(tol_ns)
        self.last_ts: Optional[int] = None
        self.last_val: Optional[Any] = None

    def push(self, ts_ns: int, val: Any) -> None:
        """Insert a sample; monotonicity is not required (we keep the newest)."""
        if self.last_ts is None or ts_ns >= self.last_ts:
            self.last_ts, self.last_val = ts_ns, val

    def sample(self, tick_ns: int):
        """Sample according to policy at a given tick."""
        if self.last_ts is None:
            return None
        if self.policy == "drop":
            return self.last_val if (self.last_ts > tick_ns - self.step_ns) else None
        if self.policy == "asof":
            return self.last_val if (tick_ns - self.last_ts <= self.tol_ns) else None
        if self.policy == "hold":
            return self.last_val
        else:
            return None


# ---------- Encoders (numpy -> ROS) ----------


def encode_value(
    ros_type: str,
    names: List[str],
    action_vec: Sequence[float],
    clamp: Optional[Tuple[float, float]] = None,
):
    """Encode a flat action vector into a ROS message using a registered encoder."""
    fn = ENCODERS.get(ros_type)
    if not fn:
        raise ValueError(f"No encoder registered for ROS type: {ros_type}")
    return fn(names, action_vec, clamp)


# ---------- QoS utilities ----------


def qos_profile_from_dict(d: Optional[Dict[str, Any]]) -> Optional[QoSProfile]:
    """Convert a dictionary to a ROS QoS profile."""
    if not d:
        return None
    rel = str(d.get("reliability", "reliable")).lower()
    hist = str(d.get("history", "keep_last")).lower()
    dur = str(d.get("durability", "volatile")).lower()
    depth = int(d.get("depth", 10))
    return QoSProfile(
        reliability=(
            ReliabilityPolicy.BEST_EFFORT
            if rel == "best_effort"
            else ReliabilityPolicy.RELIABLE
        ),
        history=(
            HistoryPolicy.KEEP_ALL if hist == "keep_all" else HistoryPolicy.KEEP_LAST
        ),
        depth=depth,
        durability=(
            DurabilityPolicy.TRANSIENT_LOCAL
            if dur == "transient_local"
            else DurabilityPolicy.VOLATILE
        ),
    )


# ---------- Data processing utilities ----------


def zero_pad(feature_meta: Dict[str, Any]) -> Any:
    """Allocate a zero/empty default for the given feature meta description."""
    dtype = feature_meta["dtype"]
    shape = tuple(feature_meta.get("shape") or ())

    if dtype in ("video", "image"):
        return np.zeros(shape, dtype=np.float32)
    if dtype == "float32":
        return np.zeros(shape, dtype=np.float32)
    if dtype == "float64":
        return np.zeros(shape, dtype=np.float64)
    if dtype == "string":
        return ""
    return None


def concatenate_action_specs(
    action_specs: List[SpecView],
    samples: Dict[str, Optional[np.ndarray]],
    zero_pad_map: Optional[Dict[str, Any]] = None,
    logger: Optional[Any] = None,
) -> np.ndarray:
    """Concatenate multiple action specs with the same key into a single ordered vector.
    
    Given a list of action specs (for the same action key) and a dictionary of per-topic
    samples, extracts values for each spec in contract order, handles missing values with
    zero padding.
    
    Args:
        action_specs: List of SpecView for action specs with the same key (in contract order)
        samples: Dict mapping dict_key -> sample value (or None if missing)
            Keys should be "<action_key>_<topic>" format
        zero_pad_map: Optional dict mapping dict_key -> zero-padded default
            If None, creates zeros from spec.names length
        logger: Optional logger for warnings (if None, warnings are silenced)
            
    Returns:
        Concatenated action vector as numpy array (float32)
    """
    if not action_specs:
        return np.zeros((0,), dtype=np.float32)
    
    if zero_pad_map is None:
        zero_pad_map = {}
    
    action_parts = []
    for sv in action_specs:
        # Create unique key for action specs (always use topic suffix for consistency)
        dict_key = f"{sv.key}_{sv.topic.replace('/', '_')}"
        
        # Get sample value
        v = samples.get(dict_key)
        
        if v is None:
            # Use zero padding
            zp = zero_pad_map.get(dict_key)
            if zp is None:
                # Create default zeros from spec names length
                zp = np.zeros((len(sv.names) if sv.names else 0,), dtype=np.float32)
            v = zp.copy() if isinstance(zp, np.ndarray) else zp
            if logger:
                logger.warning(f"Action {dict_key} is None, zero padding")
        else:
            # Ensure it's a numpy array
            if not isinstance(v, np.ndarray):
                v = np.asarray(v, dtype=np.float32)
            # Flatten to 1D if needed
            v = v.ravel()
        
        action_parts.append(v)
    
    # Concatenate all action parts in contract order
    if action_parts:
        concatenated = np.concatenate(action_parts, axis=0)
    else:
        concatenated = np.zeros((0,), dtype=np.float32)
    
    return concatenated.astype(np.float32)


def concatenate_state_specs(
    state_specs: List[SpecView],
    samples: Dict[str, Optional[np.ndarray]],
    zero_pad_map: Optional[Dict[str, Any]] = None,
    permutation_map: Optional[np.ndarray] = None,
    logger: Optional[Any] = None,
) -> np.ndarray:
    """Concatenate multiple observation.state specs into a single ordered vector.
    
    Given a list of state specs and a dictionary of per-topic samples, extracts
    values for each spec in contract order, handles missing values with zero padding,
    and optionally applies a permutation to match policy training order.
    
    Args:
        state_specs: List of SpecView for observation.state specs (in contract order)
        samples: Dict mapping dict_key -> sample value (or None if missing)
            Keys should be either "observation.state" or "observation.state_<topic>"
        zero_pad_map: Optional dict mapping dict_key -> zero-padded default
            If None, creates zeros from spec.names length
        permutation_map: Optional numpy array for reordering state dimensions
            (e.g., to match policy training order)
        logger: Optional logger for warnings (if None, warnings are silenced)
            
    Returns:
        Concatenated state vector as numpy array (float32)
    """
    if not state_specs:
        return np.zeros((0,), dtype=np.float32)
    
    if zero_pad_map is None:
        zero_pad_map = {}
    
    state_parts = []
    for sv in state_specs:
        # Create unique key for observation.state specs (always use topic suffix for consistency)
        # This matches the key format used in bag_to_lerobot and PolicyBridge
        dict_key = f"{sv.key}_{sv.topic.replace('/', '_')}"
        
        # Get sample value
        v = samples.get(dict_key)
        
        if v is None:
            # Use zero padding
            zp = zero_pad_map.get(dict_key)
            if zp is None:
                # Create default zeros from spec names length
                zp = np.zeros((len(sv.names) if sv.names else 0,), dtype=np.float32)
            v = zp.copy() if isinstance(zp, np.ndarray) else zp
            if logger:
                logger.warning(f"Observation {dict_key} is None, zero padding")
        else:
            # Ensure it's a numpy array
            if not isinstance(v, np.ndarray):
                v = np.asarray(v, dtype=np.float32)
            # Flatten to 1D if needed
            v = v.ravel()
        
        state_parts.append(v)
    
    # Concatenate all state parts in contract order
    if state_parts:
        concatenated = np.concatenate(state_parts, axis=0)
    else:
        concatenated = np.zeros((0,), dtype=np.float32)
    
    # Apply permutation if provided and vector is 1D
    if permutation_map is not None and concatenated.ndim == 1:
        concatenated = concatenated[permutation_map]
    
    return concatenated.astype(np.float32)


def contract_fingerprint(contract) -> str:
    """Generate a deterministic fingerprint for a contract to detect changes.
    
    This fingerprint captures the essential structure of the contract that affects
    data processing, allowing training to validate that the policy was trained
    with the same contract structure as the serving environment.
    
    The fingerprint is computed from SpecView (which is what's actually used
    for training/serving), ensuring it captures all meaningful contract changes.
    
    Args:
        contract: Contract dataclass instance
        
    Returns:
        A hex string representing the contract fingerprint.
    """
    # Compute fingerprint from SpecView list (what's actually used)
    specs = list(iter_specs(contract))
    
    fingerprint_data = {
        "rate_hz": float(contract.rate_hz),
        "specs": []
    }
    
    # Extract meaningful fields from each SpecView
    for spec in specs:
        spec_data = {
            "key": spec.key,
            "topic": spec.topic,
            "ros_type": spec.ros_type,
            "is_action": spec.is_action,
            "names": list(spec.names) if spec.names else [],
            "image_resize": list(spec.image_resize) if spec.image_resize else None,
            "image_channels": spec.image_channels,
            "resample_policy": spec.resample_policy,
            "asof_tol_ms": spec.asof_tol_ms,
            "stamp_src": spec.stamp_src,
        }
        
        # Action-specific fields
        if spec.is_action:
            spec_data["clamp"] = list(spec.clamp) if spec.clamp else None
            spec_data["safety_behavior"] = spec.safety_behavior
            spec_data["publish_qos"] = spec.publish_qos
        else:
            spec_data["qos"] = spec.qos
        
        fingerprint_data["specs"].append(spec_data)
    
    # Create deterministic JSON string and hash it
    json_str = json.dumps(fingerprint_data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]


# Note: decoders and encoders are imported separately when needed to avoid circular imports
