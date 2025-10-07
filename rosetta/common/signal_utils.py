# rosetta/common/signal_utils.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import time
import numpy as np
from rosidl_runtime_py.utilities import get_message

# ---------- Time helpers ----------

def now_ns() -> int:
    """Monotonic nanoseconds (steady clock)."""
    return time.monotonic_ns()

def stamp_from_header_ns(msg) -> Optional[int]:
    """Extract ROS Header stamp nanoseconds if present; else None."""
    try:
        st = msg.header.stamp
        return int(st.sec) * 1_000_000_000 + int(st.nanosec)
    except Exception:
        return None


# ---------- Dot selectors ----------

def dot_get(obj, path: str):
    """
    Resolve dotted attribute path.

    Special case:
      JointState 'field.joint_name' pattern, e.g. 'position.elbow_joint'
      will locate index by name from msg.name and return field[index].
    """
    parts = path.split(".")
    if len(parts) == 2 and hasattr(obj, "name") and hasattr(obj, parts[0]):
        field, key = parts
        try:
            idx = list(obj.name).index(key)
            return getattr(obj, field)[idx]
        except Exception:
            # Let upstream decide how to treat missing entries (usually -> NaN)
            raise
    cur = obj
    for p in parts:
        cur = getattr(cur, p)
    return cur

def dot_set(obj, path: str, value: float):
    """Set dotted attribute path; creates no new objects (ROS msgs are pre-shaped)."""
    cur = obj
    parts = path.split(".")
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], float(value))


# ---------- Decoders (ROS -> numpy/str) ----------

DecoderFn = Callable[[Any, Any], Any]
DECODERS: Dict[str, DecoderFn] = {}

def register_decoder(type_str: str):
    def _wrap(fn: DecoderFn):
        DECODERS[type_str] = fn
        return fn
    return _wrap

def _nearest_resize_rgb(img: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Fast nearest-neighbor resize for HxWxC arrays (no deps on cv2)."""
    if img.shape[0] == rh and img.shape[1] == rw:
        return img
    y = np.clip(np.linspace(0, img.shape[0] - 1, rh), 0, img.shape[0] - 1).astype(np.int64)
    x = np.clip(np.linspace(0, img.shape[1] - 1, rw), 0, img.shape[1] - 1).astype(np.int64)
    return img[y][:, x]


# ---- Image decoding ----------------------------------------------------------

def _as_rgb_u8_from_encoding(
    data: np.ndarray,
    h: int,
    w: int,
    step: int,
    encoding: str,
) -> np.ndarray:
    """
    Convert raw Image.data buffer into HxWx3 uint8 RGB given the *actual* encoding.
    Supports: rgb8, bgr8, rgba8, bgra8, mono8, mono16, 8UC1, 16UC1.
    Falls back to ValueError for unknown encodings.
    """
    enc = (encoding or "").lower()

    # Normalize common synonyms
    if enc == "bayer_rggb8" or enc.startswith("yuv"):
        # Not supported without cv2; fail fast with clear message
        raise ValueError(f"Unsupported image encoding '{encoding}' (bayer/yuv not supported).")

    # Single-channel 8-bit
    if enc in ("mono8", "8uc1", "uint8"):
        # Step may be >= w; slice each row to w bytes
        if data.size < step * h:
            raise ValueError(f"Image data too small for {h}x{w} step={step}")
        arr = data.reshape(h, step)[:, :w]
        gray = arr.reshape(h, w)
        # gray -> RGB
        return np.repeat(gray[..., None], 3, axis=2)

    # Single-channel 16-bit (mono16 / 16UC1) -> normalize to uint8
    if enc in ("mono16", "16uc1", "uint16"):
        if data.dtype != np.uint16:
            data = data.view(np.uint16)
        if data.size < (step // 2) * h:
            raise ValueError(f"Image data too small for {h}x{w} step={step}")
        row_elems = step // 2
        arr = data.reshape(h, row_elems)[:, :w]
        gray16 = arr.reshape(h, w)
        # Simple 16->8 conversion by scaling down (preserve contrast)
        # Avoid division by zero if image is all zeros
        maxv = int(gray16.max()) if gray16.size else 0
        if maxv <= 0:
            gray8 = np.zeros_like(gray16, dtype=np.uint8)
        else:
            gray8 = (gray16.astype(np.float32) * (255.0 / maxv) + 0.5).astype(np.uint8)
        return np.repeat(gray8[..., None], 3, axis=2)

    # 3-channel 8-bit (RGB/BGR)
    if enc in ("rgb8", "bgr8"):
        ch = 3
        if data.size < step * h:
            raise ValueError(f"Image data too small for {h}x{w}x{ch} step={step}")
        row = data.reshape(h, step)[:, : w * ch]
        arr = row.reshape(h, w, ch)
        if enc == "bgr8":
            arr = arr[..., ::-1]  # BGR -> RGB
        return arr

    # 4-channel 8-bit (RGBA/BGRA) -> drop alpha
    if enc in ("rgba8", "bgra8"):
        ch = 4
        if data.size < step * h:
            raise ValueError(f"Image data too small for {h}x{w}x{ch} step={step}")
        row = data.reshape(h, step)[:, : w * ch]
        arr = row.reshape(h, w, ch)
        rgb = arr[..., :3]
        if enc == "bgra8":
            rgb = rgb[..., ::-1]
        return rgb

    # Unknown encoding
    raise ValueError(
        f"Unsupported image encoding '{encoding}'. "
        "Supported: rgb8, bgr8, rgba8, bgra8, mono8, mono16, 8UC1, 16UC1."
    )

def _coerce_to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Ensure uint8 RGB HxWx3 without resizing."""
    if arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    if arr.ndim == 2:  # gray → RGB
        arr = np.repeat(arr[..., None], 3, axis=2)
    # Float images in [0,1] or [0,255]
    if arr.dtype.kind == "f":
        # Heuristic: if max <= 1.0 assume 0..1; else clip 0..255
        mx = float(arr.max()) if arr.size else 0.0
        if mx <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr

def _image_to_rgb_u8(
    msg,
    expected_encoding: Optional[str],
    resize_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    """
    Decode sensor_msgs/Image to HxWx3 uint8 RGB.

    Precedence:
      1) Use msg.encoding if set/known.
      2) Otherwise use expected_encoding (from contract image.encoding).
      3) For legacy cases, assume 'bgr8' as last resort.
    """
    h, w = int(msg.height), int(msg.width)
    step = int(getattr(msg, "step", 0))
    # Fallback “packed 3-channel” stride to avoid 0
    if step <= 0:
        step = max(w * 3, 1)

    # Interpret underlying buffer
    # NOTE: For 16-bit encodings we will re-view as uint16 inside converter.
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    # Choose encoding
    enc = str(getattr(msg, "encoding", "") or "").strip()
    if not enc:
        enc = (expected_encoding or "bgr8")

    # Fast-path: handle known encodings
    try:
        rgb = _as_rgb_u8_from_encoding(raw, h, w, step, enc)
    except ValueError:
        # If actual msg.encoding failed, try expected encoding from contract
        if enc != (expected_encoding or ""):
            rgb = _as_rgb_u8_from_encoding(raw, h, w, step, (expected_encoding or "bgr8"))
        else:
            raise

    rgb = _coerce_to_uint8_rgb(rgb)

    if resize_hw:
        rh, rw = int(resize_hw[0]), int(resize_hw[1])
        rgb = _nearest_resize_rgb(rgb, rh, rw)
    return rgb


@register_decoder("sensor_msgs/msg/Image")
def _dec_image(msg, spec):
    return _image_to_rgb_u8(msg, spec.image_encoding, spec.image_resize)


@register_decoder("std_msgs/msg/Float32MultiArray")
def _dec_f32(msg, spec):
    return np.asarray(msg.data, dtype=np.float32)

@register_decoder("std_msgs/msg/Int32MultiArray")
def _dec_i32(msg, spec):
    return np.asarray(msg.data, dtype=np.int32)

@register_decoder("std_msgs/msg/String")
def _dec_str(msg, spec):
    return str(getattr(msg, "data", ""))


def _decode_via_names(msg, names: List[str]) -> Optional[np.ndarray]:
    """Fallback decoder using dotted selectors; returns float32 vector or None."""
    if not names:
        return None
    out: List[float] = []
    for name in names:
        try:
            out.append(float(dot_get(msg, name)))
        except Exception:
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float32)

def decode_value(ros_type: str, msg, spec) -> Any:
    """Decode a ROS message to numpy/str using registered decoders or dotted names."""
    fn = DECODERS.get(ros_type)
    return fn(msg, spec) if fn else _decode_via_names(msg, spec.names)


# ---------- Resampling (offline) ----------

def resample_hold(ts_ns: np.ndarray, vals: List[Any], ticks_ns: np.ndarray) -> List[Any]:
    """Last-value-hold resampling onto ticks."""
    out: List[Any] = []
    j, last = 0, None
    for t in ticks_ns:
        while j + 1 < len(ts_ns) and ts_ns[j + 1] <= t:
            j += 1
        if j < len(vals) and ts_ns[j] <= t:
            last = vals[j]
        out.append(last)
    return out

def resample_asof(ts_ns: np.ndarray, vals: List[Any], ticks_ns: np.ndarray, tol_ns: int) -> List[Optional[Any]]:
    """As-of resampling: use last value only if not older than tol_ns; else None."""
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

def resample_drop(ts_ns: np.ndarray, vals: List[Any], ticks_ns: np.ndarray, step_ns: int) -> List[Optional[Any]]:
    """Drop resampling: only keep a value if it arrived within the current step window."""
    out: List[Optional[Any]] = []
    j, n = -1, len(ts_ns)
    for t in ticks_ns:
        while j + 1 < n and ts_ns[j + 1] <= t:
            j += 1
        out.append(vals[j] if (j >= 0 and ts_ns[j] > t - step_ns) else None)
    return out

def resample(policy: str, ts_ns: np.ndarray, vals: List[Any], ticks_ns: np.ndarray, step_ns: int, tol_ms: int) -> List[Any]:
    """Dispatch resampling policy: 'hold' | 'asof' | 'drop'."""
    if policy == "drop":
        return resample_drop(ts_ns, vals, ticks_ns, step_ns)
    if policy == "asof":
        return resample_asof(ts_ns, vals, ticks_ns, max(0, int(tol_ms)) * 1_000_000)
    return resample_hold(ts_ns, vals, ticks_ns)


# ---------- Live resampling (online) ----------

class StreamBuffer:
    """Constant-memory online resampler matching hold/asof/drop semantics."""
    def __init__(self, policy: str, step_ns: int, tol_ns: int = 0):
        self.policy = policy
        self.step_ns = int(step_ns)
        self.tol_ns = int(tol_ns)
        self.last_ts: Optional[int] = None
        self.last_val: Optional[Any] = None

    def push(self, ts_ns: int, val: Any) -> None:
        # Keep only the newest sample
        if self.last_ts is None or ts_ns >= self.last_ts:
            self.last_ts, self.last_val = ts_ns, val

    def sample(self, tick_ns: int):
        if self.last_ts is None:
            return None
        if self.policy == "drop":
            return self.last_val if (self.last_ts > tick_ns - self.step_ns) else None
        if self.policy == "asof":
            return self.last_val if (tick_ns - self.last_ts <= self.tol_ns) else None
        return self.last_val  # hold


# ---------- Action encoding (numpy -> ROS) ----------

def encode_action_to_ros(
    ros_type: str,
    names: List[str],
    action_vec: Sequence[float],
    clamp: Optional[Tuple[float, float]] = None,
):
    """
    Map a flat action vector into a ROS message using dot paths in `names`,
    or a sensible default mapping for common types when `names` is empty.

    Supported defaults (when `names` is empty):
      - geometry_msgs/msg/Twist:
          len>=6 → [lin.x,y,z, ang.x,y,z]
          len>=2 → [lin.x, ang.z]
          len=1  → [lin.x]
      - std_msgs/msg/Float32MultiArray / Int32MultiArray:
          fills .data
    """
    msg_cls = get_message(ros_type)
    msg = msg_cls()
    arr = np.asarray(action_vec, dtype=np.float32).reshape(-1)

    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])

    if names:
        for i, path in enumerate(names):
            v = float(arr[i]) if i < arr.size else 0.0
            dot_set(msg, path, v)
        return msg

    if ros_type.endswith('geometry_msgs/msg/Twist'):
        if arr.size >= 6:
            msg.linear.x = float(arr[0]); msg.linear.y = float(arr[1]); msg.linear.z = float(arr[2])
            msg.angular.x = float(arr[3]); msg.angular.y = float(arr[4]); msg.angular.z = float(arr[5])
        elif arr.size >= 2:
            msg.linear.x = float(arr[0]); msg.angular.z = float(arr[1])
        elif arr.size == 1:
            msg.linear.x = float(arr[0])
        return msg

    if ros_type.endswith('Float32MultiArray'):
        msg.data = [float(x) for x in arr.tolist()]
        return msg
    if ros_type.endswith('Int32MultiArray'):
        msg.data = [int(round(x)) for x in arr.tolist()]
        return msg

    raise ValueError(f"encode_action_to_ros: unsupported ros_type '{ros_type}' without names")
