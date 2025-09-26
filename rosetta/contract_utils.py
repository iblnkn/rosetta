from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import yaml

# ---------------- Datamodel ----------------

@dataclass
class ObservationSpec:
    key: str
    topic: str
    type: str
    resample: str = "hold"                    # "drop" | "hold" (kept for backward compat; not used by recorder logic)
    image: Optional[Dict[str, Any]] = None   # {encoding, to_rgb, resize}
    selector: Optional[Dict[str, Any]] = None
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
    # Optional: enable uniform sampling to rate_hz
    rec["sample_to_rate"] = bool(rec.get("sample_to_rate", False))

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
