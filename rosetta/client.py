# client.py
# -*- coding: utf-8 -*-
"""
ROS client for PolicyBridge:
- Builds all ROS 2 subscriptions/publishers directly from the contract.
- Resamples observations via StreamBuffer and assembles a frame on demand.
- Splits & publishes actions across repeated specs per action key.
- Exposes timing helpers and safety metadata used by the node.

Drop-in for: `from client import RosClient, TimingPolicy`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rosidl_runtime_py.utilities import get_message

from rosetta.common.contract_utils import (
    SpecView,
    concatenate_state_specs,
    decode_value,
    encode_value,
    feature_from_spec,
    iter_specs,
    load_contract,
    qos_profile_from_dict,
    stamp_from_header_ns,
    StreamBuffer,
    zero_pad,
)

# ---------- Data containers ----------

@dataclass(frozen=True, slots=True)
class ActionView:
    key: str
    topic: str
    ros_type: str
    names: List[str]
    clamp: Optional[Tuple[float, float]]

@dataclass(slots=True)
class SubState:
    spec: SpecView
    buf: StreamBuffer

@dataclass(slots=True)
class TimingPolicy:
    """Timing policy shared with the node."""
    step_ns: int
    publish_tolerance_ms: Optional[float]
    use_header_time: bool
    header_skew_ms: float

    @property
    def publish_tolerance_ns(self) -> int:
        # Default tolerance: one frame
        if self.publish_tolerance_ms is None:
            return self.step_ns
        return int(float(self.publish_tolerance_ms) * 1e6)

    @property
    def header_skew_ns(self) -> int:
        return int(float(self.header_skew_ms) * 1e6)

# ---------- Action routing ----------

class ActionRouter:
    """
    Splits a packet {action_key -> flat_vector} across repeated specs for each key
    and publishes to the corresponding ROS topics in contract order.
    """

    def __init__(self, groups: Dict[str, List[ActionView]], pubs_by_topic: Dict[str, Any], logger: Optional[Any] = None):
        self._groups = groups
        self._pubs = pubs_by_topic
        self._logger = logger

    def publish_packet(self, packet: Dict[str, np.ndarray], context: str = "action") -> int:
        total = 0
        for key, vec in packet.items():
            if key not in self._groups:
                raise KeyError(f"{context}: unknown action key '{key}'")
            if vec is None:
                continue
            if not isinstance(vec, np.ndarray):
                vec = np.asarray(vec, dtype=np.float32)
            if vec.ndim != 1:
                raise ValueError(f"{context}: vector for key='{key}' must be 1-D, got shape={vec.shape}")

            idx = 0
            for av in self._groups[key]:
                n = len(av.names or [])
                j = idx + n
                if j > vec.size:
                    raise ValueError(
                        f"{context}: vec too short for {av.topic} "
                        f"(need {n}, have {vec.size - idx})"
                    )
                slice_vec = vec[idx:j]
                idx = j

                msg = encode_value(
                    ros_type=av.ros_type,
                    names=av.names,
                    action_vec=slice_vec,
                    clamp=av.clamp,
                )
                pub = self._pubs.get(av.topic)
                if pub is None:
                    raise RuntimeError(f"{context}: no publisher for {av.topic}")
                pub.publish(msg)

            if idx != vec.size:
                raise ValueError(
                    f"{context}: extra values in vector for key='{key}' "
                    f"(used {idx} of {vec.size})"
                )
            total += 1
        return total

# ---------- Main client ----------

class RosClient:
    """
    Owns all ROS 2 I/O per the contract:
      - Subscribes to observations with appropriate QoS and buffering.
      - Samples a consistent observation frame at a requested timestamp.
      - Creates publishers for actions and routes packets per action key.

    Exposes:
      - fps, step_sec, step_ns
      - sample_frame(sample_t_ns, prompt, permutation_map)
      - publish_packet(packet)
      - recent_image_timestamp_ns()
      - action_key_dims()
      - safety_behavior_by_key()
    """

    def __init__(self, node: Node, contract_path: str, rate_hz: Optional[int] = None) -> None:
        self._n = node
        self._contract = load_contract(contract_path)

        self.fps: int = int(self._contract.rate_hz)
        if self.fps <= 0:
            raise ValueError("Contract rate_hz must be >= 1")
        self.step_ns: int = int(round(1e9 / self.fps))
        self.step_sec: float = 1.0 / self.fps

        # Parse specs
        self._specs: List[SpecView] = list(iter_specs(self._contract))
        self._obs_specs: List[SpecView] = [s for s in self._specs if not s.is_action]
        self._act_specs: List[SpecView] = [s for s in self._specs if s.is_action]

        # --- Observation subscriptions ---
        self._obs_zero: Dict[str, Any] = {}
        self._subs: Dict[str, SubState] = {}
        self._ros_sub_handles: List[Any] = []
        self._state_groups: Dict[str, List[SpecView]] = {}  # observation.state*

        # Only group the exact base key "observation.state"
        for s in self._obs_specs:
            if s.key == "observation.state":
                self._state_groups.setdefault(s.key, []).append(s)

        for s in self._obs_specs:
            _k, meta, _ = feature_from_spec(s, use_videos=False)

            # Unique dict key: disambiguate repeated state groups by topic suffix
            if s.key == "observation.state":
                dict_key = f"{s.key}_{s.topic.replace('/', '_')}"
            else:
                dict_key = s.key

            self._obs_zero[dict_key] = zero_pad(meta)

            msg_cls = get_message(s.ros_type)
            obs_qos_dict = s.qos or {}
            sub_qos = qos_profile_from_dict(obs_qos_dict) or QoSProfile(depth=10)
            sub = self._n.create_subscription(
                msg_cls,
                s.topic,
                lambda m, sv=s: self._obs_cb(m, sv),
                sub_qos,
            )
            self._ros_sub_handles.append(sub)

            tol_ns = int(max(0, float(getattr(s, "asof_tol_ms", 0.0)))) * 1_000_000
            self._subs[dict_key] = SubState(
                spec=s,
                buf=StreamBuffer(policy=s.resample_policy, step_ns=self.step_ns, tol_ns=tol_ns),
            )

        # --- Action publishers + router ---
        pubs_by_topic: Dict[str, Any] = {}
        groups: Dict[str, List[ActionView]] = {}
        for s in self._act_specs:
            pub_qos = qos_profile_from_dict(s.publish_qos) or QoSProfile(depth=10)
            pub = self._n.create_publisher(get_message(s.ros_type), s.topic, pub_qos)
            pubs_by_topic[s.topic] = pub
            groups.setdefault(s.key, []).append(
                ActionView(
                    key=s.key,
                    topic=s.topic,
                    ros_type=s.ros_type,
                    names=list(s.names or []),
                    clamp=getattr(s, "clamp", None),
                )
            )

        self._router = ActionRouter(groups, pubs_by_topic, logger=self._n.get_logger())

        # Precompute per-key dims and safety behaviors for convenience
        self._action_dims_by_key: Dict[str, int] = {
            k: sum(len(av.names or []) for av in v) for k, v in groups.items()
        }
        self._safety_behavior_by_key: Dict[str, str] = {}
        for k, v in groups.items():
            behavior = getattr(v[0], "safety_behavior", None)
            if behavior is None:
                src = next((sv for sv in self._act_specs if sv.key == k), None)
                behavior = getattr(src, "safety_behavior", "zeros") if src else "zeros"
            behavior = str(behavior).lower()
            if behavior not in ("zeros", "hold"):
                behavior = "zeros"
            self._safety_behavior_by_key[k] = behavior

        self._n.get_logger().info(
            f"ROS client ready: "
            f"{len(self._subs)} observation streams, "
            f"{sum(len(v) for v in groups.values())} action topics, "
            f"rate={self.fps} Hz"
        )

    # ---------- Observation callbacks & sampling ----------

    def _obs_cb(self, msg, sv: SpecView) -> None:
        # Prefer header time if requested by the spec; else receive time.
        ts_ns = stamp_from_header_ns(msg) if sv.stamp_src == "header" else None
        if ts_ns is None:
            ts_ns = self._n.get_clock().now().nanoseconds

        if sv.key == "observation.state":
            dict_key = f"{sv.key}_{sv.topic.replace('/', '_')}"
        else:
            dict_key = sv.key

        val = decode_value(sv.ros_type, msg, sv)
        if val is not None:
            self._subs[dict_key].buf.push(int(ts_ns), val)

    def sample_frame(
        self,
        sample_t_ns: int,
        prompt: str,
        permutation_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Sample a consistent observation frame at sample_t_ns.
        Concatenates every observation.state* group (with optional permutation for the base group),
        and zero-pads missing signals using contract metadata.
        """
        frame: Dict[str, Any] = {}

        # 1) Concatenate grouped state specs (exact base key)
        for group_key, group_specs in self._state_groups.items():
            samples: Dict[str, Any] = {}
            for sv in group_specs:
                dict_key = f"{sv.key}_{sv.topic.replace('/', '_')}"
                if dict_key in self._subs:
                    samples[dict_key] = self._subs[dict_key].buf.sample(sample_t_ns)
            perm = permutation_map if group_key == "observation.state" else None
            frame[group_key] = concatenate_state_specs(
                state_specs=group_specs,
                samples=samples,
                zero_pad_map=self._obs_zero,
                permutation_map=perm,
                logger=self._n.get_logger(),
            )

        # 2) All other observations
        for key, st in self._subs.items():
            if key.startswith("observation.state_"):
                continue
            v = st.buf.sample(sample_t_ns)
            if v is None:
                zp = self._obs_zero[key]
                frame[key] = zp.copy() if isinstance(zp, np.ndarray) else zp
            else:
                frame[key] = v

        # 3) Inject task/prompt string expected by many LeRobot policies
        frame["task"] = prompt or ""
        return frame

    # ---------- Publishing ----------

    def publish_packet(self, packet: Dict[str, np.ndarray]) -> None:
        """Publish a full packet {key -> vec} across all action topics."""
        self._router.publish_packet(packet, context="action")

    # ---------- Safety & timing helpers used by the node ----------

    def recent_image_timestamp_ns(self) -> Optional[int]:
        """
        Return timestamp of the most recent Image observation (from buffers),
        or None if no image streams are present or populated.
        """
        latest: Optional[int] = None
        for key, st in self._subs.items():
            ros_type = getattr(st.spec, "ros_type", "")
            if ros_type == "sensor_msgs/msg/Image":
                ts = getattr(st.buf, "last_ts", None)
                if ts is not None and (latest is None or ts > latest):
                    latest = int(ts)
        return latest

    def action_key_dims(self) -> Dict[str, int]:
        """Total dimensionality per action key (sum over repeated specs)."""
        return dict(self._action_dims_by_key)

    def safety_behavior_by_key(self) -> Dict[str, str]:
        """Safety behavior per action key ('zeros' | 'hold')."""
        return dict(self._safety_behavior_by_key)
