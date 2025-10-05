# rosetta/common/contract_utils.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import yaml
import numpy as np

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
    image: Optional[Dict[str, Any]] = None      # {resize:[H,W], encoding:'rgb8'|'bgr8'}
    align: Optional[AlignSpec] = None

@dataclass(frozen=True, slots=True)
class ActionSpec:
    key: str
    publish_topic: str
    type: str
    selector: Optional[Dict[str, Any]] = None   # {names: [...]}
    from_tensor: Optional[Dict[str, Any]] = None  # {clamp:[min,max]}

@dataclass(frozen=True, slots=True)
class Contract:
    name: str
    version: int
    rate_hz: float
    max_duration_s: float
    observations: List[ObservationSpec]
    actions: List[ActionSpec]
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
        )

    def _act(it) -> ActionSpec:
        pub = it["publish"]
        return ActionSpec(
            key=it["key"],
            publish_topic=pub["topic"],
            type=pub["type"],
            selector=it.get("selector"),
            from_tensor=it.get("from_tensor"),
        )

    obs = [_obs(it) for it in (d.get("observations") or [])]
    acts = [_act(it) for it in (d.get("actions") or [])]

    return Contract(
        name=d.get("name", "contract"),
        version=int(d.get("version", 1)),
        rate_hz=float(d.get("rate_hz", d.get("fps", 20.0))),
        max_duration_s=float(d.get("max_duration_s", 30.0)),
        observations=obs,
        actions=acts,
        robot_type=d.get("robot_type"),
        timestamp_source=str(d.get("timestamp_source", "receive")).lower(),
    )

# ---------- Unified SpecView ----------

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
        al = o.align or AlignSpec()
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
