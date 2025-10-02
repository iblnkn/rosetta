from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import yaml
from rosidl_runtime_py.utilities import get_message

# ---------------- Datamodel ----------------

@dataclass
class ObservationSpec:
    key: str
    topic: str
    type: str
    resample: str = "hold"                    # "drop" | "hold" (kept for backward compat; not used by recorder logic)
    image: Optional[Dict[str, Any]] = None   # {encoding: "bgr8"|"rgb8", resize: [H,W]}
    selector: Optional[Dict[str, Any]] = None  # {names: [...], field: "position"|"velocity"|"effort"}
    qos: Optional[Dict[str, Any]] = None     # {reliability, history, depth, durability}
    align: Optional[Dict[str, Any]] = None   # {strategy: hold|asof|drop, tol_ms: int, stamp: header|receive}

@dataclass
class TaskSpec:
    key: str
    topic: str
    type: str
    qos: Optional[Dict[str, Any]] = None
    align: Optional[Dict[str, Any]] = None   # same schema as above

@dataclass
class ActionPublishSpec:
    topic: str
    type: str
    layout: str = "flat"
    qos: Optional[Dict[str, Any]] = None

@dataclass
class ActionSpec:
    key: str
    publish: ActionPublishSpec
    from_tensor: Optional[Dict[str, Any]] = None  # e.g. {"clamp":[min,max]}

@dataclass
class Contract:
    name: str
    version: int
    rate_hz: float
    max_duration_s: float
    observations: List[ObservationSpec]
    tasks: List[TaskSpec]
    actions: List[ActionSpec]
    recording: Dict[str, Any]                  # at least {"storage": "mcap"}
    robot_type: Optional[str] = None

# --------------- Loader --------------------

def load_contract(path: str) -> Contract:
    with open(path, "r") as f:
        d = yaml.safe_load(f)

    obs: List[ObservationSpec] = []
    for it in d.get("observations", []) or []:
        obs.append(ObservationSpec(
            key=it["key"],
            topic=it["topic"],
            type=it["type"],
            resample=it.get("resample", "hold"),
            image=it.get("image"),
            selector=it.get("selector"),
            qos=it.get("qos"),
            align=it.get("align"),
        ))

    tks: List[TaskSpec] = []
    for it in d.get("tasks", []) or []:
        tks.append(TaskSpec(
            key=it["key"],
            topic=it["topic"],
            type=it["type"],
            qos=it.get("qos"),
            align=it.get("align"),
        ))

    acts: List[ActionSpec] = []
    for it in d.get("actions", []) or []:
        pub = it["publish"]
        acts.append(ActionSpec(
            key=it["key"],
            publish=ActionPublishSpec(
                topic=pub["topic"],
                type=pub["type"],
                layout=pub.get("layout", "flat"),
                qos=pub.get("qos"),
            ),
            from_tensor=it.get("from_tensor"),
        ))

    rec = d.get("recording", {}) or {}
    if "storage" not in rec:
        rec["storage"] = "mcap"
    # Enable uniform sampling to rate_hz by default
    rate_hz = float(d.get("rate_hz", d.get("fps", 20.0)))
    max_duration_s = float(d.get("max_duration_s", 30.0))

    return Contract(
        name=d.get("name", "contract"),
        version=int(d.get("version", 1)),
        rate_hz=rate_hz,
        max_duration_s=max_duration_s,
        observations=obs,
        tasks=tks,
        actions=acts,
        recording=rec,
        robot_type=d.get("robot_type"),
    )

# --------------- QoS helper ----------------

def qos_profile_from_dict(q: Optional[Dict[str, Any]]):
    """
    Build an rclpy QoSProfile from a minimal YAML dict like:
      {reliability: reliable|best_effort, history: keep_last|keep_all, depth: 10, durability: volatile|transient_local}
    Returns None if q is falsy.
    """
    if not q:
        return None

    from rclpy.qos import (
        QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
    )

    history_map = {
        "keep_last": QoSHistoryPolicy.KEEP_LAST,
        "keep_all": QoSHistoryPolicy.KEEP_ALL,
        "system_default": QoSHistoryPolicy.SYSTEM_DEFAULT,
    }
    rel_map = {
        "reliable": QoSReliabilityPolicy.RELIABLE,
        "best_effort": QoSReliabilityPolicy.BEST_EFFORT,
        "system_default": QoSReliabilityPolicy.SYSTEM_DEFAULT,
    }
    dur_map = {
        "volatile": QoSDurabilityPolicy.VOLATILE,
        "transient_local": QoSDurabilityPolicy.TRANSIENT_LOCAL,
        "system_default": QoSDurabilityPolicy.SYSTEM_DEFAULT,
    }

    history = history_map.get(str(q.get("history", "keep_last")).lower(), QoSHistoryPolicy.KEEP_LAST)
    depth = int(q.get("depth", 10))
    reliability = rel_map.get(str(q.get("reliability", "reliable")).lower(), QoSReliabilityPolicy.RELIABLE)
    durability = dur_map.get(str(q.get("durability", "volatile")).lower(), QoSDurabilityPolicy.VOLATILE)

    return QoSProfile(
        history=history,
        depth=depth,
        reliability=reliability,
        durability=durability,
    )

# --------------- Minimal ROS <-> NumPy helpers ----------------

def _resize_nn(img: np.ndarray, resize_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    """Nearest-neighbor resize without external deps. img: HxWxC uint8."""
    if not resize_hw:
        return img
    h, w = img.shape[:2]
    rh, rw = int(resize_hw[0]), int(resize_hw[1])
    if (h, w) == (rh, rw):
        return img
    y_idx = (np.linspace(0, h - 1, rh)).astype(np.int64)
    x_idx = (np.linspace(0, w - 1, rw)).astype(np.int64)
    return img[y_idx][:, x_idx]

def _image_msg_to_numpy_rgb_u8(msg, expected_encoding: str = "bgr8", resize_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """sensor_msgs/Image -> np.uint8[H,W,3] RGB."""
    h, w = int(msg.height), int(msg.width)
    data = np.frombuffer(msg.data, dtype=np.uint8)
    # Step might include padding; handle common packed 3-channel case.
    step = int(getattr(msg, "step", 0)) or (w * 3)
    if step == w * 3 and data.size == h * w * 3:
        arr = data.reshape(h, w, 3)
    else:
        # Fallback for padded rows
        arr = data.reshape(h, step)[:, : w * 3].reshape(h, w, 3)

    enc = (expected_encoding or "bgr8").lower()
    if enc not in ("bgr8", "rgb8"):
        raise ValueError(f"Unsupported image encoding '{expected_encoding}', expected 'bgr8' or 'rgb8'")
    if enc == "bgr8":
        arr = arr[..., ::-1]  # BGR->RGB

    arr = _resize_nn(arr, resize_hw)
    return arr

def np_rgb_to_image(
    img: np.ndarray,
    type_str: str = "sensor_msgs/msg/Image",
    encoding: str = "rgb8",
) -> object:
    """
    np.uint8[H,W,3] RGB -> sensor_msgs/Image.
    If encoding='bgr8', data will be converted accordingly.
    """
    msg_cls = get_message(type_str)
    msg = msg_cls()
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("np_rgb_to_image expects HxWx3 uint8 array")

    h, w = img.shape[:2]
    if encoding.lower() == "bgr8":
        data = img[..., ::-1].tobytes()
    elif encoding.lower() == "rgb8":
        data = img.tobytes()
    else:
        raise ValueError(f"Unsupported encoding '{encoding}', expected 'rgb8' or 'bgr8'")

    msg.height = int(h)
    msg.width = int(w)
    msg.encoding = encoding
    msg.step = int(w * 3)
    msg.data = data
    return msg

def _joint_state_to_vector(msg, names: List[str], field: str) -> np.ndarray:
    """sensor_msgs/JointState subset by names -> np.float32[len(names)] from field."""
    idx = {n: i for i, n in enumerate(msg.name)}
    src = getattr(msg, field)
    out = [float(src[idx[n]]) if (n in idx and idx[n] < len(src)) else np.nan for n in names]
    return np.asarray(out, dtype=np.float32)

def float32_multiarray_to_np(msg) -> np.ndarray:
    return np.asarray(msg.data, dtype=np.float32)

def int32_multiarray_to_np(msg) -> np.ndarray:
    return np.asarray(msg.data, dtype=np.int32)

def np_to_float32_multiarray(arr, type_str: str = "std_msgs/msg/Float32MultiArray") -> object:
    msg_cls = get_message(type_str)
    msg = msg_cls()
    msg.data = [float(x) for x in np.asarray(arr, dtype=np.float32).ravel()]
    return msg

def np_to_int32_multiarray(arr, type_str: str = "std_msgs/msg/Int32MultiArray") -> object:
    msg_cls = get_message(type_str)
    msg = msg_cls()
    msg.data = [int(x) for x in np.asarray(arr, dtype=np.int32).ravel()]
    return msg

def twist_to_np(msg) -> np.ndarray:
    return np.asarray([msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float32)

def decode_observation_msg(spec: ObservationSpec, msg: object):
    """
    Minimal unified decoder driven by ObservationSpec.
    Returns one of:
      - np.uint8[H,W,3] RGB (for images)
      - np.float32[...] (for vectors/arrays)
      - str (for String)
      - None (unsupported type)
    """
    rtype = spec.type

    # Images
    if rtype.endswith("/Image") and spec.image is not None:
        enc = (spec.image.get("encoding") if spec.image else None) or "bgr8"
        resize = tuple(spec.image.get("resize")) if (spec.image and spec.image.get("resize")) else None
        return _image_msg_to_numpy_rgb_u8(msg, expected_encoding=enc, resize_hw=resize)

    # JointState with selector
    if rtype.endswith("/JointState") and spec.selector:
        names = list(spec.selector.get("names", []))
        field = str(spec.selector.get("field", "position"))
        return _joint_state_to_vector(msg, names, field)

    # Generic arrays / strings
    if rtype.endswith("/Float32MultiArray"):
        return float32_multiarray_to_np(msg)
    if rtype.endswith("/Int32MultiArray"):
        return int32_multiarray_to_np(msg)
    if rtype.endswith("/String"):
        return str(getattr(msg, "data", ""))
    if rtype.endswith("/Twist"):
        return twist_to_np(msg)


    # Unsupported here (intentionally minimal)
    return None
