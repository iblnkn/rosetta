# rosetta/common/signal_utils.py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import time
import numpy as np
from rosidl_runtime_py.utilities import get_message

# ---------- Time helpers ----------

def now_ns() -> int:
    return time.monotonic_ns()

def stamp_from_header_ns(msg) -> Optional[int]:
    try:
        st = msg.header.stamp
        return int(st.sec) * 1_000_000_000 + int(st.nanosec)
    except Exception:
        return None

# ---------- Dot selectors ----------

def dot_get(obj, path: str):
    parts = path.split(".")
    # JointState 'position.joint_name' pattern
    if len(parts) == 2 and hasattr(obj, "name") and hasattr(obj, parts[0]):
        field, key = parts
        idx = list(obj.name).index(key)
        return getattr(obj, field)[idx]
    cur = obj
    for p in parts:
        cur = getattr(cur, p)
    return cur

def dot_set(obj, path: str, value: float):
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
    if img.shape[0] == rh and img.shape[1] == rw:
        return img
    y = np.clip(np.linspace(0, img.shape[0]-1, rh), 0, img.shape[0]-1).astype(np.int64)
    x = np.clip(np.linspace(0, img.shape[1]-1, rw), 0, img.shape[1]-1).astype(np.int64)
    return img[y][:, x]

def _image_to_rgb_u8(msg, expected_encoding: str, resize_hw: Optional[Tuple[int,int]]) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    step = int(getattr(msg, "step", 0)) or (w * 3)
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if step == w * 3 and data.size == h * w * 3:
        arr = data.reshape(h, w, 3)
    else:
        arr = data.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
    enc = (expected_encoding or "bgr8").lower()
    if enc == "bgr8":
        arr = arr[..., ::-1]
    elif enc != "rgb8":
        raise ValueError(f"Unsupported image encoding '{expected_encoding}' (use 'rgb8' or 'bgr8')")
    if resize_hw:
        arr = _nearest_resize_rgb(arr, int(resize_hw[0]), int(resize_hw[1]))
    return arr

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
    if not names:
        return None
    out = []
    for name in names:
        try:
            out.append(float(dot_get(msg, name)))
        except Exception:
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float32)

def decode_value(ros_type: str, msg, spec) -> Any:
    fn = DECODERS.get(ros_type)
    return fn(msg, spec) if fn else _decode_via_names(msg, spec.names)

# ---------- Resampling (offline) ----------

def resample_hold(ts_ns: np.ndarray, vals: List[Any], ticks_ns: np.ndarray) -> List[Any]:
    out, j, last = [], 0, None
    for t in ticks_ns:
        while j + 1 < len(ts_ns) and ts_ns[j + 1] <= t:
            j += 1
        if j < len(vals) and ts_ns[j] <= t:
            last = vals[j]
        out.append(last)
    return out

def resample_asof(ts_ns: np.ndarray, vals: List[Any], ticks_ns: np.ndarray, tol_ns: int) -> List[Optional[Any]]:
    if tol_ns <= 0:
        return resample_hold(ts_ns, vals, ticks_ns)
    out, j = [], 0
    for t in ticks_ns:
        while j + 1 < len(ts_ns) and ts_ns[j + 1] <= t:
            j += 1
        ok = j < len(vals) and ts_ns[j] <= t and (t - ts_ns[j]) <= tol_ns
        out.append(vals[j] if ok else None)
    return out

def resample_drop(ts_ns: np.ndarray, vals: List[Any], ticks_ns: np.ndarray, step_ns: int) -> List[Optional[Any]]:
    out, j, n = [], -1, len(ts_ns)
    for t in ticks_ns:
        while j + 1 < n and ts_ns[j + 1] <= t:
            j += 1
        out.append(vals[j] if (j >= 0 and ts_ns[j] > t - step_ns) else None)
    return out

def resample(policy: str, ts_ns: np.ndarray, vals: List[Any], ticks_ns: np.ndarray, step_ns: int, tol_ms: int) -> List[Any]:
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
    clamp: Optional[Tuple[float,float]] = None,
):
    """
    Map a flat action vector into a ROS message using dot paths in `names`,
    or a sensible default mapping for common types when `names` is empty.

    Supported defaults (when `names` is empty):
      - geometry_msgs/msg/Twist:
          len>=6 → [lin.x,y,z, ang.x,y,z]
          len>=2 → [lin.x, ang.z]
          len=1  → [lin.x]
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
