# rosetta/common/contract_utils.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

# ROS 2 QoS helpers (used by recorder & any nodes that want contract-driven QoS)
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

# ---------- Contract datamodel ----------

@dataclass(frozen=True, slots=True)
class AlignSpec:
    strategy: str = "hold"   # "hold" | "asof" | "drop"
    tol_ms: int = 0
    stamp: str = "receive"   # "receive" | "header"

@dataclass(frozen=True, slots=True)
class ObservationSpec:
    key: str
    topic: str
    type: str
    selector: Optional[Dict[str, Any]] = None   # {names: [...]}
    image: Optional[Dict[str, Any]] = None      # {resize:[H,W], encoding:'rgb8'|'bgr8'|'mono8'}
    align: Optional[AlignSpec] = None
    resample: Optional[str] = None              # optional legacy/alternate spelling
    qos: Optional[Dict[str, Any]] = None        # QoS dict (reliability/history/depth/... )

@dataclass(frozen=True, slots=True)
class ActionSpec:
    key: str
    publish_topic: str
    type: str
    selector: Optional[Dict[str, Any]] = None       # {names: [...]}
    from_tensor: Optional[Dict[str, Any]] = None    # {clamp:[min,max]}
    publish_qos: Optional[Dict[str, Any]] = None    # QoS for publisher

@dataclass(frozen=True, slots=True)
class TaskSpec:
    key: str
    topic: str
    type: str
    qos: Optional[Dict[str, Any]] = None

@dataclass(frozen=True, slots=True)
class Contract:
    name: str
    version: int
    rate_hz: float
    max_duration_s: float
    observations: List[ObservationSpec]
    actions: List[ActionSpec]
    # Optional sections / metadata
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
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    def _obs(it) -> ObservationSpec:
        return ObservationSpec(
            key=it["key"],
            topic=it["topic"],
            type=it["type"],
            selector=it.get("selector"),
            image=it.get("image"),
            align=_as_align(it.get("align")),
            resample=(str(it.get("resample")).lower() if it.get("resample") else None),
            qos=it.get("qos"),
        )

    def _act(it) -> ActionSpec:
        pub = it["publish"]
        return ActionSpec(
            key=it["key"],
            publish_topic=pub["topic"],
            type=pub["type"],
            selector=it.get("selector"),
            from_tensor=it.get("from_tensor"),
            publish_qos=pub.get("qos"),
        )

    def _task(it) -> TaskSpec:
        # Optional block. If present, we record it verbatim just like obs.
        return TaskSpec(
            key=it.get("key", it["topic"]),
            topic=it["topic"],
            type=it["type"],
            qos=it.get("qos"),
        )

    obs = [_obs(it) for it in (d.get("observations") or [])]
    acts = [_act(it) for it in (d.get("actions") or [])]
    tks  = [_task(it) for it in (d.get("tasks") or [])]
    rec  = d.get("recording") or {}

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

def qos_profile_from_dict(d: Optional[Dict[str, Any]]) -> Optional[QoSProfile]:
    """
    Translate a simple dict into a QoSProfile. Supported keys:
      reliability: reliable|best_effort
      history: keep_last|keep_all
      depth: int
      durability: volatile|transient_local
    Unknown/missing -> sensible defaults.
    """
    if not d:
        return None
    rel = str(d.get("reliability", "reliable")).lower()
    hist = str(d.get("history", "keep_last")).lower()
    dur = str(d.get("durability", "volatile")).lower()
    depth = int(d.get("depth", 10))

    return QoSProfile(
        reliability=(
            ReliabilityPolicy.BEST_EFFORT
            if rel == "best_effort" else ReliabilityPolicy.RELIABLE
        ),
        history=(HistoryPolicy.KEEP_ALL if hist == "keep_all" else HistoryPolicy.KEEP_LAST),
        depth=depth,
        durability=(DurabilityPolicy.TRANSIENT_LOCAL if dur == "transient_local" else DurabilityPolicy.VOLATILE),
    )

# ---------- Unified SpecView (for runtime/offline processing) ----------

@dataclass(frozen=True, slots=True)
class SpecView:
    key: str
    topic: str
    ros_type: str
    is_action: bool
    names: List[str]
    image_resize: Optional[Tuple[int,int]]
    image_encoding: str
    resample_policy: str
    asof_tol_ms: int
    stamp_src: str
    clamp: Optional[Tuple[float, float]]  # actions only

def iter_specs(contract: Contract) -> Iterable[SpecView]:
    # Observations
    for o in contract.observations:
        resize = None
        enc = "bgr8"
        if o.image:
            r = o.image.get("resize")
            if r and len(r) == 2:
                resize = (int(r[0]), int(r[1]))
            enc = str(o.image.get("encoding", "bgr8")).lower()
        names = list((o.selector or {}).get("names", []))
        # Prefer explicit align; else fall back to legacy "resample"
        al = o.align or AlignSpec(strategy=(o.resample or "hold"))
        yield SpecView(
            key=o.key,
            topic=o.topic,
            ros_type=o.type,
            is_action=False,
            names=names,
            image_resize=resize,
            image_encoding=enc,
            resample_policy=al.strategy.lower(),
            asof_tol_ms=int(al.tol_ms),
            stamp_src=(al.stamp or contract.timestamp_source).lower(),
            clamp=None,
        )
    # Actions
    for a in contract.actions:
        names = list((a.selector or {}).get("names", []))
        clamp = None
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
            resample_policy="hold",
            asof_tol_ms=0,
            stamp_src=contract.timestamp_source,
            clamp=clamp,
        )

# ---------- LeRobot feature helpers ----------

def feature_from_spec(spec: SpecView, use_videos: bool) -> tuple[str, Dict[str, Any], bool]:
    """
    Returns (key, feature_meta, is_image).
      feature_meta: {dtype, shape, names}
      dtype ∈ {"video","image","float32","float64","string"}
    """
    if spec.image_resize:
        h, w = int(spec.image_resize[0]), int(spec.image_resize[1])
        dtype = "video" if use_videos else "image"
        return spec.key, {"dtype": dtype, "shape": (h, w, 3), "names": ["height","width","channel"]}, True
    if not spec.names:
        raise ValueError(f"{spec.key}: vector features must specify selector.names")
    return spec.key, {"dtype": "float32", "shape": (len(spec.names),), "names": list(spec.names)}, False

def zero_pad(feature_meta: Dict[str, Any]) -> Any:
    dt = feature_meta["dtype"]; shape = tuple(feature_meta.get("shape") or ())
    if dt in ("video","image"):
        return np.zeros(shape, dtype=np.uint8)
    if dt == "float32":
        return np.zeros(shape, dtype=np.float32)
    if dt == "float64":
        return np.zeros(shape, dtype=np.float64)
    if dt == "string":
        return ""
    return None
