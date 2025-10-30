#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PolicyBridge: contract-true live policy inference.

"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from std_srvs.srv import Trigger
from rosidl_runtime_py.utilities import get_message
from rcl_interfaces.msg import SetParametersResult
import torch


from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from rosetta.common.contract_utils import (
    load_contract,
    iter_specs,
    SpecView,
    feature_from_spec,
    zero_pad,
    qos_profile_from_dict,
    contract_fingerprint,
    decode_value,
    StreamBuffer,
    stamp_from_header_ns,
    encode_value,
    concatenate_state_specs,
)

from rosetta_interfaces.action import RunPolicy

_ACTION_NAME = "run_policy"
_FEEDBACK_PERIOD_S = 0.5


# ---------------- Action routing helpers ----------------
@dataclass(frozen=True, slots=True)
class _ActionSpecView:
    key: str
    topic: str
    ros_type: str
    names: List[str]
    clamp: Optional[Tuple[float, float]]


class ActionRouter:
    """
    Routes model outputs to ROS topics per contract action specs.
    - Supports multiple action KEYS (e.g., 'action', 'action.arm', 'action.gripper').
    - Supports repeated KEYS (multiple specs under the same key, different topics).
    - Accepts either a dict[str, np.ndarray] keyed by action key, or a single vector for legacy 'action'.
    """
    def __init__(self, specs_by_key: Dict[str, List[SpecView]], pubs_by_topic: Dict[str, Any], logger):
        self._logger = logger
        self._groups: Dict[str, List[_ActionSpecView]] = {}
        self._pubs = pubs_by_topic
        for k, specs in specs_by_key.items():
            group: List[_ActionSpecView] = []
            for s in specs:
                n = len(s.names or [])
                if n <= 0:
                    raise ValueError(f"Action spec {s.topic} has no names (len=0) – unsupported.")
                group.append(
                    _ActionSpecView(
                        key=k,
                        topic=s.topic,
                        ros_type=s.ros_type,
                        names=list(s.names or []),
                        clamp=getattr(s, "clamp", None),
                    )
                )
            self._groups[k] = group

    def _split_and_publish(self, key: str, vec: np.ndarray, context="action") -> int:
        """Split vec across specs of 'key' in contract order and publish."""
        if key not in self._groups:
            raise KeyError(f"{context}: unknown action key '{key}'")
        idx = 0
        published = 0
        for sv in self._groups[key]:
            n = len(sv.names)
            j = idx + n
            if j > len(vec):
                raise ValueError(
                    f"{context}: vec too short for {sv.topic} (need {n}, have {len(vec) - idx})"
                )
            slice_vec = vec[idx:j]
            idx = j
            msg = encode_value(
                ros_type=sv.ros_type,
                names=sv.names,
                action_vec=slice_vec,
                clamp=sv.clamp,
            )
            pub = self._pubs.get(sv.topic)
            if pub is None:
                raise RuntimeError(f"{context}: no publisher for {sv.topic}")
            pub.publish(msg)
            published += 1
        if idx != len(vec):
            raise ValueError(
                f"{context}: extra values in vector for key='{key}' (used {idx} of {len(vec)})"
            )
        return published

    def publish_packet(self, packet: Dict[str, np.ndarray], context="action") -> int:
        """Publish a packet of {action_key -> vec}."""
        total = 0
        for key, vec in packet.items():
            total += self._split_and_publish(key, vec, context=context)
        return total


@dataclass(slots=True)
class _SubState:
    spec: SpecView
    msg_type: Any
    buf: StreamBuffer
    stamp_src: str  # 'receive' or 'header'


@dataclass(slots=True)
class _RuntimeParams:
    use_action_chunks: bool
    actions_per_chunk: int
    chunk_size_threshold: float
    use_header_time: bool
    use_autocast: bool
    max_queue_actions: int = 512
    header_skew_ms: float = 500.0
    publish_tolerance_ms: Optional[float] = None


@dataclass(slots=True)
class TimingPolicy:
    """Consolidates timing-related configuration and logic.
    
    Centralizes tolerance, anchor, skew, and safety strategy configuration
    to simplify PolicyBridge timing logic.
    """
    step_ns: int
    use_header_time: bool
    header_skew_ms: float
    publish_tolerance_ms: Optional[float]
    
    @property
    def header_skew_ns(self) -> int:
        """Header skew tolerance in nanoseconds."""
        return int(self.header_skew_ms * 1e6)
    
    @property
    def publish_tolerance_ns(self) -> int:
        """Publish tolerance in nanoseconds. Defaults to one frame if None."""
        if self.publish_tolerance_ms is None:
            return self.step_ns
        return int(float(self.publish_tolerance_ms) * 1e6)
    
    def compute_sample_time(self, clock, anchor_timestamp_fn) -> int:
        """Compute sampling time using configured anchor strategy.
        
        Args:
            clock: ROS clock for current time
            anchor_timestamp_fn: Function returning anchor timestamp or None
            
        Returns:
            Sampling timestamp in nanoseconds
        """
        sample_t_ns = None
        if self.use_header_time:
            ts = anchor_timestamp_fn()
            if ts is not None:
                skew = clock.now().nanoseconds - ts
                if 0 <= skew <= self.header_skew_ns:
                    sample_t_ns = ts
        if sample_t_ns is None:
            sample_t_ns = clock.now().nanoseconds
        return sample_t_ns
    
    def is_header_time_valid(self, header_ts_ns: int, now_ns: int) -> bool:
        """Check if header timestamp is within valid skew window."""
        if not self.use_header_time:
            return False
        skew = now_ns - header_ts_ns
        return 0 <= skew <= self.header_skew_ns


def _device_from_param(requested: Optional[str] = None) -> torch.device:
    r = (requested or "auto").lower().strip()

    def mps_available() -> bool:
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if r == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Explicit CUDA (supports 'cuda' and 'cuda:N')
    if r.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(r)  # 'cuda' or 'cuda:N'

    # Explicit MPS (or 'metal' alias)
    if r in {"mps", "metal"}:
        if not mps_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    # Anything else: try to parse ('cpu', 'xpu', etc.), otherwise fallback
    try:
        return torch.device(r)
    except (TypeError, ValueError, RuntimeError):
        # Invalid device requested, fallback to CPU
        return torch.device("cpu")


class PolicyBridge(Node):
    """Contract-true live inference node with persistent timers and action control."""

    def __init__(self) -> None:
        super().__init__("policy_bridge")

        # ---------------- Parameters ----------------
        self.declare_parameter("contract_path", "")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("policy_device", "auto")
        # Canonical LeRobot-style async knobs
        self.declare_parameter("use_action_chunks", True)
        self.declare_parameter("actions_per_chunk", 25)
        self.declare_parameter("chunk_size_threshold", 0.5)
        self.declare_parameter("max_queue_actions", 512)
        self.declare_parameter("use_header_time", True)
        self.declare_parameter("use_autocast", False)
        self.declare_parameter("aggregate_fn_name", "weighted_average")  # LeRobot parity
        self.declare_parameter("header_skew_ms", 500.0)
        self.declare_parameter("publish_tolerance_ms", None)  # if None, defaults to one frame

        self._params = self._read_params()
        
        # Aggregation registry (same semantics as LeRobot)
        self._AGG_FNS = {
            "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
            "latest_only": lambda old, new: new,
            "average": lambda old, new: 0.5 * old + 0.5 * new,
            "conservative": lambda old, new: 0.7 * old + 0.3 * new,
        }
        agg_name = str(self.get_parameter("aggregate_fn_name").value or "weighted_average")
        if agg_name not in self._AGG_FNS:
            raise ValueError(
                f"Unknown aggregate_fn_name='{agg_name}'. "
                f"Options: {list(self._AGG_FNS.keys())}"
            )
        self._agg_name = agg_name
        
        # Timestep bookkeeping for LeRobot parity
        self._timestep_cursor: int = 0  # next timestep to assign on production
        self._latest_executed_timestep: int = -1
        self.add_on_set_parameters_callback(self._on_params)

        # ---------------- Contract ----------------
        contract_path = str(self.get_parameter("contract_path").value or "")
        if not contract_path:
            raise RuntimeError("policy_bridge: 'contract_path' is required")
        self._contract = load_contract(Path(contract_path))

        self._obs_qos_by_key: Dict[str, Optional[Dict[str, Any]]] = {
            o.key: o.qos for o in (self._contract.observations or [])
        }
        self._act_qos_by_key: Dict[str, Optional[Dict[str, Any]]] = {
            a.key: a.publish_qos for a in (self._contract.actions or [])
        }

        # ---------------- Policy load ----------------
        policy_path = str(self.get_parameter("policy_path").value or "")
        if not policy_path:
            raise RuntimeError("policy_bridge: 'policy_path' is required")

        # Check if policy_path is a Hugging Face repo ID (contains '/')
        is_hf_repo = '/' in policy_path and not os.path.exists(policy_path)
    
        cfg_type = ""  # Default value
        if is_hf_repo:
            # For Hugging Face repos, we'll let from_pretrained handle the download
            # and get the config type from the loaded policy
            self.get_logger().info(f"Detected Hugging Face repo: {policy_path}")
        else:
            # For local paths, try to read config.json
            cfg_json = os.path.join(policy_path, "config.json")
            try:
                if os.path.exists(cfg_json):
                    with open(cfg_json, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        cfg_type = str(cfg.get("type", "")).lower()
            except (OSError, json.JSONDecodeError, KeyError) as e:
                self.get_logger().warning(
                    f"Could not read policy config.json: {e!r}"
                )

        req = str(self.get_parameter("policy_device").value)
        self.device = _device_from_param(req)
        r = req.strip().lower()
        r = "mps" if r == "metal" else r
        if r not in {"auto", ""} and str(torch.device(r)) != str(self.device):
            self.get_logger().warning(f"policy_device='{req}' requested; using '{self.device}' instead.")
        self.get_logger().info(f"Using device: {self.device}")

        if is_hf_repo:
            # For Hugging Face repos, we need to load the policy first to get the config type
            # We'll use a temporary approach: try common policy types
            policy_types_to_try = ["act", "diffusion", "pi0", "pi05", "smolvla"]
            policy_loaded = False
            
            for policy_type in policy_types_to_try:
                try:
                    self.get_logger().info(f"Trying to load as {policy_type} policy...")
                    PolicyCls = get_policy_class(policy_type)
                    self.policy = PolicyCls.from_pretrained(policy_path)
                    self.policy.to(self.device)
                    self.policy.eval()
                    policy_loaded = True
                    cfg_type = policy_type
                    self.get_logger().info(f"Successfully loaded {policy_type} policy from {policy_path}")
                    break
                except Exception as e:
                    self.get_logger().debug(f"Failed to load as {policy_type}: {e}")
                    continue
            
            if not policy_loaded:
                raise RuntimeError(f"Could not load policy from {policy_path} with any known policy type")
        else:
            # For local paths, use the config type we read earlier
            if not cfg_type:
                raise RuntimeError(f"Could not determine policy type from {policy_path}")
            PolicyCls = get_policy_class(cfg_type)
            self.policy = PolicyCls.from_pretrained(policy_path)
            self.policy.to(self.device)
            self.policy.eval()

        # Load dataset stats from the policy artifact if present
        ds_stats = None
        for cand in ("dataset_stats.json", "stats.json", "meta/stats.json"):
            p = Path(policy_path) / cand
            if p.exists():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        ds_stats = json.load(f)
                    self.get_logger().info(f"Loaded dataset stats from {p}")
                    break
                except Exception as e:
                    self.get_logger().warning(f"Failed to read {p}: {e!r}")

        # Validate contract fingerprint if available
        try:
            current_fp = contract_fingerprint(self._contract)
            self.get_logger().info(f"Contract fingerprint: {current_fp}")
            
            # Check if policy has a stored fingerprint
            policy_fp_path = Path(policy_path) / "contract_fingerprint.txt"
            if policy_fp_path.exists():
                with policy_fp_path.open("r") as f:
                    stored_fp = f.read().strip()
                if stored_fp != current_fp:
                    self.get_logger().warning(
                        f"Contract fingerprint mismatch! Policy: {stored_fp}, Current: {current_fp}"
                    )
                else:
                    self.get_logger().info("Contract fingerprint matches policy")
        except Exception as e:
            self.get_logger().warning(f"Contract fingerprint validation failed: {e!r}")

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=policy_path,
            dataset_stats=ds_stats,  # <-- critical for parity
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)}},
            postprocessor_overrides={
                "device_processor": {"device": str(self.device)}},
        )
        
        # After policy is loaded, clamp actions_per_chunk to policy limits (if exposed)
        try:
            policy_max = None
            if hasattr(self.policy.config, "n_action_steps"):
                policy_max = int(self.policy.config.n_action_steps)
            elif hasattr(self.policy.config, "chunk_size"):
                policy_max = int(self.policy.config.chunk_size)
            if policy_max is not None and self._params.actions_per_chunk > policy_max:
                self.get_logger().info(
                    f"Clamping actions_per_chunk from {self._params.actions_per_chunk} to policy max {policy_max}"
                )
                self._params = _RuntimeParams(
                    use_action_chunks=self._params.use_action_chunks,
                    actions_per_chunk=policy_max,
                    chunk_size_threshold=self._params.chunk_size_threshold,
                    use_header_time=self._params.use_header_time,
                    use_autocast=self._params.use_autocast,
                    max_queue_actions=self._params.max_queue_actions,
                    header_skew_ms=self._params.header_skew_ms,
                    publish_tolerance_ms=self._params.publish_tolerance_ms,
                )
        except Exception as e:
            self.get_logger().debug(f"Could not clamp actions_per_chunk from policy config: {e!r}")

        # ---------------- Specs & rate ----------------
        self._specs: List[SpecView] = list(iter_specs(self._contract))
        self._obs_specs = [s for s in self._specs if not s.is_action]
        self._act_specs = [s for s in self._specs if s.is_action]
        #TODO: process _task_specs

        # Group action specs by key
        self._action_specs_by_key: Dict[str, List[SpecView]] = {}
        for spec in self._act_specs:
            self._action_specs_by_key.setdefault(spec.key, []).append(spec)
        
        if len(self._act_specs) == 0:
            raise ValueError("Contract must declare at least one action spec.")
        if len(self._action_specs_by_key) > 1:
            self.get_logger().info(
                f"Detected multiple action keys: {list(self._action_specs_by_key.keys())}"
            )
        
        # Keep action_keys for reference
        self._action_keys = list(self._action_specs_by_key.keys())

        self.fps = int(self._contract.rate_hz)
        if self.fps <= 0:
            raise ValueError("Contract rate_hz must be >= 1")
        self.step_ns = int(round(1e9 / self.fps))
        self.step_sec = 1.0 / self.fps

        # ---------------- Timing Policy ----------------
        # Consolidate timing configuration and logic
        self._timing_policy = TimingPolicy(
            step_ns=self.step_ns,
            use_header_time=self._params.use_header_time,
            header_skew_ms=self._params.header_skew_ms,
            publish_tolerance_ms=self._params.publish_tolerance_ms,
        )

        self._cbg = ReentrantCallbackGroup()
        self._obs_zero, self._subs, self._ros_sub_handles = {}, {}, []
        # Group any observation.state* (e.g., observation.state, observation.state.arm)
        self._state_groups: Dict[str, List[SpecView]] = {}
        for s in self._obs_specs:
            if s.key.startswith("observation.state"):
                self._state_groups.setdefault(s.key, []).append(s)
        
        # Keep permutation audit ONLY for the base 'observation.state' group (if present)
        self._contract_state_names = []
        for sv in self._state_groups.get("observation.state", []):
            self._contract_state_names.extend(list(sv.names or []))
        
        # State layout audit and permutation mapping (critical for correct policy inputs)
        self._state_index_map = None
        expected = None
        try:
            # Try a few common locations/keys
            if isinstance(ds_stats, dict):
                if "observation.state" in ds_stats and isinstance(ds_stats["observation.state"], dict):
                    expected = ds_stats["observation.state"].get("names")
                if expected is None:
                    expected = ds_stats.get("state_names")
            if expected is None and hasattr(self.policy.config, "state_names"):
                expected = list(self.policy.config.state_names)
        except Exception as e:
            self.get_logger().warning(f"Could not read expected state names: {e!r}")
        
        if expected:
            cur = self._contract_state_names
            if set(cur) != set(expected):
                raise RuntimeError(
                    "State feature set mismatch between contract and policy. "
                    f"Contract-only={sorted(set(cur)-set(expected))}, Policy-only={sorted(set(expected)-set(cur))}"
                )
            if cur != expected:
                # map from current index -> expected index
                pos = {name: i for i, name in enumerate(cur)}
                self._state_index_map = np.asarray([pos[name] for name in expected], dtype=np.int64)
                self.get_logger().warning(
                    "Observation state order differs from policy; applying runtime permutation to match policy."
                )

        for s in self._obs_specs:
            k, meta, _ = feature_from_spec(s, use_videos=False)
            
            # Create unique key for observation.state specs (always use topic suffix for consistency)
            # This matches bag_to_lerobot and concatenate_state_specs helper
            if s.key.startswith("observation.state"):
                dict_key = f"{s.key}_{s.topic.replace('/', '_')}"
            else:
                dict_key = s.key
                
            self._obs_zero[dict_key] = zero_pad(meta)

            msg_cls = get_message(s.ros_type)
            # Prefer per-spec QoS, then fall back to key-based QoS
            obs_qos_dict = s.qos or self._obs_qos_by_key.get(s.key)
            sub_qos = qos_profile_from_dict(obs_qos_dict) or QoSProfile(depth=10)
            sub = self.create_subscription(
                msg_cls, s.topic, lambda m, sv=s: self._obs_cb(m, sv),
                sub_qos,
                callback_group=self._cbg,
            )
            self._ros_sub_handles.append(sub)

            tol_ns = int(max(0, s.asof_tol_ms)) * 1_000_000
            self._subs[dict_key] = _SubState(
                spec=s,
                msg_type=msg_cls,
                buf=StreamBuffer(policy=s.resample_policy, step_ns=self.step_ns, tol_ns=tol_ns),
                stamp_src=s.stamp_src,
            )

        self.get_logger().info(
            f"Subscribed to {len(self._subs)} observation streams.")

        # ---------------- Publishers ----------------
        self._act_pubs: Dict[str, Any] = {}
        for spec in self._act_specs:
            # Prefer per-spec QoS, then fall back to key-based QoS
            act_qos_dict = spec.publish_qos or self._act_qos_by_key.get(spec.key)
            pub_qos = qos_profile_from_dict(act_qos_dict) or QoSProfile(depth=10)
            pub = self.create_publisher(
                get_message(spec.ros_type), spec.topic, pub_qos
            )
            self._act_pubs[spec.topic] = pub
            self.get_logger().info(f"Created publisher for {spec.topic} ({spec.ros_type})")
        
        # Initialize action router
        self._router = ActionRouter(self._action_specs_by_key, self._act_pubs, self.get_logger())

        self._cancel_srv = self.create_service(
            Trigger, f"{_ACTION_NAME}/cancel", self._cancel_service_cb,
            callback_group=self._cbg
        )

        # ---------------- Action server ----------------
        self._active_handle: Optional[Any] = None
        self._running_event = threading.Event()
        self._stop_requested = threading.Event()
        self._done_event = threading.Event()
        self._finishing = threading.Event()
        self._prompt = ""
        self._pub_count = 0
        self._terminal: Optional[Tuple[str, str]] = None

        self._action_server = ActionServer(
            self,
            RunPolicy,
            _ACTION_NAME,
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._cbg,
        )

        # ---------------- Deadline state ----------------
        self._max_duration_s = float(
            getattr(self._contract, "max_duration_s", 1000.0))
        self._deadline_active = False
        self._deadline_end_ns: Optional[int] = None


        # ---------------- Safety behavior ----------------
        # Store safety behavior per action key (different keys may have different behaviors)
        self._safety_behavior_by_key: Dict[str, str] = {}
        for key, specs in self._action_specs_by_key.items():
            # Get safety behavior from first spec of this key (specs for same key should share behavior)
            first_spec = specs[0] if specs else None
            behavior = getattr(first_spec, "safety_behavior", "zeros").lower() if first_spec else "zeros"
            if behavior not in ("zeros", "hold"):
                self.get_logger().warning(
                    f"Unknown safety_behavior '{behavior}' for key '{key}', defaulting to 'zeros'"
                )
                behavior = "zeros"
            self._safety_behavior_by_key[key] = behavior
            self.get_logger().info(
                f"Safety for '{key}': {'hold last action' if behavior == 'hold' else 'zeros'} on stop."
            )

        # ---------------- Timing strategy ----------------
        # Timing policy already handles publish_tolerance_ns via property
        
        # Starvation guard (must-produce after queue empties)
        self._starved_since_ns: Optional[int] = None
        
        # Warning throttle
        self._last_warn_ns = 0

        # ---------------- Async producer/executor ----------------
        # Queue holds (publish_time_ns, timestep, packet)
        self._queue: Deque[Tuple[int, int, Dict[str, np.ndarray]]] = deque(
            maxlen=self._params.max_queue_actions
        )
        # Ordered mapping by timestep for O(1) merge lookups (avoids quadratic rebuild)
        self._queue_by_step: Dict[int, Tuple[int, Dict[str, np.ndarray]]] = {}
        self._queue_lock = threading.Lock()
        self._last_packet_lock = threading.Lock()
        self._last_packet: Optional[Dict[str, np.ndarray]] = None
        self._producer_buffer: List[Tuple[int, int, Dict[str, np.ndarray]]] = []

        self._cbg_timers = ReentrantCallbackGroup()
        self._producer_timer = self.create_timer(
            self.step_sec, self._producer_tick, callback_group=self._cbg_timers
        )
        self._executor_timer = self.create_timer(
            self.step_sec, self._executor_tick, callback_group=self._cbg_timers
        )
        self._feedback_timer = self.create_timer(
            _FEEDBACK_PERIOD_S, self._feedback_tick, callback_group=self._cbg_timers
        )
        self._deadline_timer = self.create_timer(
            0.2, self._deadline_tick, callback_group=self._cbg_timers
        )  # poll 5 Hz

        self.get_logger().info(
            f"PolicyBridge ready at {self.fps:.1f} Hz on device={self.device}."
        )

    # ---------------- Parameter handling ----------------
    def _read_params(self) -> _RuntimeParams:
        return _RuntimeParams(
            use_action_chunks=bool(self.get_parameter("use_action_chunks").value),
            actions_per_chunk=int(self.get_parameter("actions_per_chunk").value),
            chunk_size_threshold=float(self.get_parameter("chunk_size_threshold").value),
            use_header_time=bool(self.get_parameter("use_header_time").value),
            use_autocast=bool(self.get_parameter("use_autocast").value),
            max_queue_actions=self.get_parameter("max_queue_actions").value,
            header_skew_ms=float(self.get_parameter("header_skew_ms").value),
            publish_tolerance_ms=self.get_parameter("publish_tolerance_ms").value,
        )

    def _on_params(self, _params: List[Parameter]) -> SetParametersResult:
        new_params = self._read_params()
        if self._queue.maxlen != new_params.max_queue_actions:
            with self._queue_lock:
                self._queue = deque(self._queue, maxlen=new_params.max_queue_actions)
        self._params = new_params
        # Update timing policy with new parameters
        self._timing_policy = TimingPolicy(
            step_ns=self.step_ns,
            use_header_time=self._params.use_header_time,
            header_skew_ms=self._params.header_skew_ms,
            publish_tolerance_ms=self._params.publish_tolerance_ms,
        )
        return SetParametersResult(successful=True)

    def _next_exec_tick_ns(self, now_ns: int) -> int:
        return ((now_ns + self.step_ns - 1) // self.step_ns) * self.step_ns

    # ---------------- Timers (persistent) ----------------
    def _feedback_tick(self) -> None:
        if (
            self._active_handle is None
            or not self._running_event.is_set()
            or self._finishing.is_set()
        ):
            return
        rem_s: Optional[int] = None
        if self._deadline_end_ns is not None:
            now_ns = self.get_clock().now().nanoseconds
            rem_s = max(0, (self._deadline_end_ns - now_ns) // 1_000_000_000)
        try:
            fb = RunPolicy.Feedback()
            if hasattr(fb, "published_actions"):
                fb.published_actions = int(self._pub_count)
            if hasattr(fb, "queue_depth"):
                with self._queue_lock:
                    fb.queue_depth = int(len(self._queue))
            if hasattr(fb, "status"):
                fb.status = "executing"
            if rem_s is not None and hasattr(fb, "seconds_remaining"):
                fb.seconds_remaining = int(rem_s)
            self._active_handle.publish_feedback(fb)
        except (RuntimeError, AttributeError) as e:
            self.get_logger().warning(f"Feedback timer publish failed: {e!r}")

    def _deadline_tick(self) -> None:
        if (
            not self._deadline_active
            or self._active_handle is None
            or self._finishing.is_set()
        ):
            return
        now_ns = self.get_clock().now().nanoseconds
        if self._deadline_end_ns is not None and now_ns >= self._deadline_end_ns:
            self.get_logger().warning(
                f"Policy run timed out after {self._max_duration_s:.1f}s."
            )
            self._finish_run(timeout=True)

    # ---------------- Action callbacks ----------------
    def _goal_cb(self, _req) -> GoalResponse:
        if self._active_handle is not None:
            self.get_logger().info("Goal request: REJECT (already running)")
            return GoalResponse.REJECT
        self.get_logger().info("Goal request: ACCEPT")
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle) -> CancelResponse:
        if self._active_handle is None or goal_handle != self._active_handle:
            return CancelResponse.REJECT
        self.get_logger().info("Action cancel requested")
        self._stop_requested.set()
        return CancelResponse.ACCEPT

    def _cancel_service_cb(self, _req, resp):
        self.get_logger().info("Cancel service called")
        self._stop_requested.set()
        self._done_event.set()  # Wake executor immediately
        resp.success = True
        resp.message = "Policy run cancellation requested"
        return resp

    # ---- Execute lifecycle ---------------------------------------------------
    def _execute_cb(self, goal_handle) -> RunPolicy.Result:
        if self._active_handle is not None:
            goal_handle.abort()
            res = RunPolicy.Result()
            res.success = False
            res.message = "Already running"
            return res

        self._active_handle = goal_handle
        self._stop_requested.clear()
        self._done_event.clear()
        self._finishing.clear()
        self._running_event.set()
        self._pub_count = 0
        with self._queue_lock:
            self._queue.clear()
            self._queue_by_step.clear()
        
        # Reset run-local cursors / lasts (critical for correct aggregation)
        self._timestep_cursor = 0
        self._latest_executed_timestep = -1
        with self._last_packet_lock:
            self._last_packet = None
        self._starved_since_ns = None
        self._last_warn_ns = 0

        task = getattr(goal_handle.request, "task", None)
        prompt = getattr(goal_handle.request, "prompt", None)
        self._prompt = task or prompt or ""

        if hasattr(self.policy, "reset"):
            try:
                self.policy.reset()
            except (RuntimeError, AttributeError) as e:
                self.get_logger().warning(f"policy.reset() failed: {e!r}")

        self.get_logger().info(
            f"{_ACTION_NAME}: started (task='{self._prompt}')")

        # Arm deadline
        now_ns = self.get_clock().now().nanoseconds
        self._deadline_end_ns = now_ns + int(self._max_duration_s * 1e9)
        self._deadline_active = True

        self._done_event.wait()

        # Send terminal status from execute callback thread (canonical ROS 2 pattern)
        status, msg = (self._terminal or ("aborted", "No active goal"))
        was_action_cancel = goal_handle.is_cancel_requested  # only true for real action cancels

        try:
            if was_action_cancel:
                goal_handle.canceled()            # valid: EXECUTING -> CANCELING -> CANCELED
            elif status == "timeout":
                goal_handle.abort()               # timeouts are typically 'aborted'
            elif status == "canceled":            # cancel via Trigger service → no action cancel
                status, msg = "aborted", "Cancelled via service"
                goal_handle.abort()               # valid from EXECUTING
            elif status == "succeeded":
                goal_handle.succeed()
            else:
                goal_handle.abort()
        except (RuntimeError, AttributeError) as e:
            self.get_logger().warning(f"Result send failed: {e!r}")

        # Build the Result payload to match the status
        ok = status == "succeeded"
        # Cleanup for next goal
        self._active_handle = None
        self._terminal = None

        return self._mk_result(ok, msg)
    def _mk_result(self, success: bool, message: str) -> RunPolicy.Result:
        res = RunPolicy.Result()
        res.success = bool(success)
        res.message = str(message)
        return res

    def _finish_run(self, timeout: bool = False) -> None:
        if self._finishing.is_set():
            return
        self._finishing.set()
        self._running_event.clear()
        self._deadline_active = False

        # Decide outcome only - don't send status from worker thread
        if timeout:
            self._terminal = ("timeout", f"Timed out after {self._max_duration_s:.1f}s")
            self.get_logger().info(f"{_ACTION_NAME}: completed (timeout)")
        elif self._stop_requested.is_set():
            self._terminal = ("canceled", "Cancelled")
            self.get_logger().info(f"{_ACTION_NAME}: stopped (cancelled)")
        else:
            self._terminal = ("succeeded", "Policy run ended")
            self.get_logger().info(f"{_ACTION_NAME}: stopped (succeeded)")

        self._publish_safety_command(increment_count=False, log_message="Published safety command on stop.")
        with self._queue_lock:
            self._queue.clear()
            self._queue_by_step.clear()
        self._prompt = ""
        self._done_event.set()

    def _create_safety_packet(self) -> Dict[str, np.ndarray]:
        """Create safety packet for all action keys using per-key safety behavior."""
        packet: Dict[str, np.ndarray] = {}
        # Thread-safe read of _last_packet
        with self._last_packet_lock:
            last_packet = self._last_packet
        
        for key, specs in self._action_specs_by_key.items():
            behavior = self._safety_behavior_by_key.get(key, "zeros")
            total = sum(len(s.names or []) for s in specs)
            if total == 0:
                raise ValueError(f"No dims for action key '{key}' in contract")
            
            if behavior == "hold" and last_packet is not None and key in last_packet:
                # Hold last action for this key
                packet[key] = last_packet[key].copy()
            else:
                # Zeros for this key
                packet[key] = np.zeros((total,), dtype=np.float32)
        return packet

    def _publish_action_packet(self, packet: Dict[str, np.ndarray],
                               increment_count: bool = True,
                               log_message: Optional[str] = None,
                               error_context: str = "action") -> None:
        """Publish an action packet with consistent error handling."""
        try:
            self._validate_packet(packet)
            published_topics = self._router.publish_packet(packet, context=error_context)
            if published_topics > 0 and increment_count:
                self._pub_count += 1  # once per packet/timestep (not per topic)
            if log_message:
                self.get_logger().info(log_message)
        except Exception as e:
            self.get_logger().error(f"{error_context} publish error: {e}")
            raise

    def _publish_safety_command(self, increment_count: bool = True, log_message: str = None) -> None:
        """Publish safety command (zeros or hold last) across all keys/specs."""
        pkt = self._create_safety_packet()
        self._publish_action_packet(pkt, increment_count, log_message, "safety")

    # ---------- Aggregation helpers ----------
    def _shape_str(self, x) -> str:
        """Safely format shape for logging (handles tensors, dicts, numpy arrays)."""
        try:
            return str({k: tuple(v.shape) for k, v in x.items()}) if isinstance(x, dict) else str(tuple(x.shape))
        except Exception:
            return "<unknown>"

    def _aggregate_packets(self, old_pkt: Dict[str, np.ndarray],
                           new_pkt: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Elementwise aggregate old/new packets (numpy), key-wise."""
        agg = self._AGG_FNS[self._agg_name]
        out: Dict[str, np.ndarray] = {}
        keys = set(old_pkt.keys()) | set(new_pkt.keys())
        for k in keys:
            a = old_pkt.get(k)
            b = new_pkt.get(k)
            if a is None:
                out[k] = b
                continue
            if b is None:
                out[k] = a
                continue
            if a.shape != b.shape:
                raise ValueError(f"Aggregate shape mismatch for key='{k}': {a.shape} vs {b.shape}")
            out[k] = agg(a, b)
        return out

    def _merge_step_packets(self, step: int, old_pkt: Dict[str, np.ndarray], 
                            new_pkt: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Merge packets for the same timestep using configured aggregation function."""
        mode = self._agg_name
        if mode == "latest_only":
            return new_pkt
        elif mode in {"average", "conservative", "weighted_average"}:
            return self._aggregate_packets(old_pkt, new_pkt)
        else:
            # default fallback: latest_only
            return new_pkt

    def _warn_throttled(self, msg: str, period_ms: float = 500.0) -> None:
        """Emit a warning, but only if period_ms has elapsed since last warning."""
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_warn_ns >= int(period_ms * 1e6):
            self._last_warn_ns = now_ns
            self.get_logger().warning(msg)

    def _enqueue_actions(self, items: List[Tuple[int, int, Dict[str, np.ndarray]]]) -> None:
        """
        Merge items into queue with LeRobot semantics:
        - skip timesteps <= latest executed
        - if timestep exists in queue, aggregate its packet
        - keep queue sorted by publish time
        - drop stale (past) items first
        - if overflow, drop farthest-future items
        """
        with self._queue_lock:
            # Sync dict from queue if needed (shouldn't happen, but safety)
            if len(self._queue_by_step) != len(self._queue):
                self._queue_by_step = {step: (t_ns, pkt) for t_ns, step, pkt in self._queue}

            # Merge incoming items (O(m) where m = len(items))
            for t_ns, step, pkt in items:
                if step <= self._latest_executed_timestep:
                    continue  # too old, skip
                if step in self._queue_by_step:
                    old_t_ns, old_pkt = self._queue_by_step[step]
                    merged = self._merge_step_packets(step, old_pkt, pkt)
                    # keep the earlier publish time for stability
                    self._queue_by_step[step] = (min(old_t_ns, t_ns), merged)
                else:
                    self._queue_by_step[step] = (t_ns, pkt)

            # Rebuild list and sort by publish time ascending
            merged_list = sorted(
                [(t_ns, step, pkt) for step, (t_ns, pkt) in self._queue_by_step.items()],
                key=lambda x: x[0]
            )

            # 1) Drop stale (publish time before now - tolerance)
            now_ns = self.get_clock().now().nanoseconds
            tol_ns = self._timing_policy.publish_tolerance_ns
            fresh = [(t_ns, step, pkt) for (t_ns, step, pkt) in merged_list if t_ns >= now_ns - tol_ns]

            # Keep dict in sync (remove truly stale entries)
            if len(fresh) != len(merged_list):
                stale_steps = {step for (t_ns, step, pkt) in merged_list if t_ns < now_ns - tol_ns}
                for s in stale_steps:
                    self._queue_by_step.pop(s, None)
            merged_list = fresh

            self._queue.clear()
            # 2) Enforce capacity: keep earliest due items; drop far-future overflow
            capacity = self._queue.maxlen
            if capacity is None or len(merged_list) <= capacity:
                kept_items = merged_list
                dropped = 0
            else:
                kept_items = merged_list[:capacity]        # earliest items
                dropped = len(merged_list) - capacity      # how many far-future we dropped

            for it in kept_items:
                self._queue.append(it)
            # Sync dict to match queue (for items that were dropped)
            self._queue_by_step = {step: (t_ns, pkt) for t_ns, step, pkt in kept_items}

            # Optional visibility
            if dropped:
                head_dt_ms = (kept_items[0][0] - now_ns) / 1e6 if kept_items else 0.0
                tail_dt_ms = (kept_items[-1][0] - now_ns) / 1e6 if kept_items else 0.0
                self._warn_throttled(
                    f"Queue overflow: dropped {dropped} far-future items; "
                    f"kept window [{head_dt_ms:.1f}ms .. {tail_dt_ms:.1f}ms]"
                )

    def _validate_packet(self, pkt: Dict[str, np.ndarray]) -> None:
        """Ensure every key is known and vector lengths match concatenated spec."""
        for key, vec in pkt.items():
            if key not in self._action_specs_by_key:
                raise KeyError(f"Unknown action key in packet: '{key}'")
            expected = sum(len(s.names or []) for s in self._action_specs_by_key[key])
            if vec.ndim != 1 or vec.size != expected:
                raise ValueError(
                    f"Packet key '{key}' size mismatch: got {vec.shape}, expected ({expected},)"
                )

    # ---------------- Sub callback ----------------
    def _obs_cb(self, msg, spec: SpecView) -> None:
        # Honor per-spec stamp_src when pushing to buffer (no global override here)
        # The global use_header_time is only used for anchor selection in compute_sample_time
        use_header = (spec.stamp_src == "header")
        ts = stamp_from_header_ns(msg) if use_header else None
        ts_ns = int(
            ts) if ts is not None else self.get_clock().now().nanoseconds
        val = decode_value(spec.ros_type, msg, spec)
        if val is not None:
            # Mirror the subscription key used at construction time
            if spec.key.startswith("observation.state"):
                dict_key = f"{spec.key}_{spec.topic.replace('/', '_')}"
            else:
                dict_key = spec.key
            self._subs[dict_key].buf.push(ts_ns, val)

    # ---------------- Producer: timer tick (persistent) ----------------
    def _producer_tick(self) -> None:
        if not self._running_event.is_set() or self._finishing.is_set():
            return
        if self._stop_requested.is_set():
            try:
                self._finish_run(timeout=False)
            except (RuntimeError, AttributeError) as e:
                self.get_logger().error(
                    f"finish on cancel (producer) failed: {e!r}")
            return
        try:
            self._produce_actions()
        except (RuntimeError, ValueError, TypeError) as e:
            self.get_logger().error(f"producer tick failed: {e!r}")

    def _produce_actions(self) -> int:
        """Run policy inference and enqueue actions. Returns number produced."""
        use_chunks = self._params.use_action_chunks
        k = int(self._params.actions_per_chunk)
        thr = float(self._params.chunk_size_threshold)

        produced = 0
        
        # Starvation guard: if we published safety, force production
        starved = self._starved_since_ns is not None
        
        with self._queue_lock:
            queue_length = len(self._queue)
            need_chunk = use_chunks and (
                starved or queue_length == 0 or (k > 0 and (queue_length / max(1, k)) <= thr)
            )
        if use_chunks and not need_chunk:
            return 0

        self._producer_buffer.clear()
        use_autocast = self._params.use_autocast and (
            hasattr(torch.amp, "autocast_mode")
            and torch.amp.autocast_mode.is_autocast_available(self.device.type)
        )

        # 1) Choose a sampling time using timing policy (handles anchor + skew)
        sample_t_ns = self._timing_policy.compute_sample_time(
            self.get_clock(),
            self._get_most_recent_image_timestamp
        )

        obs_frame = self._sample_obs_frame(sample_t_ns)
        batch = self._prepare(obs_frame)
        batch = self.preprocessor(batch)

        with torch.inference_mode():
            cm = (
                torch.autocast(self.device.type, enabled=use_autocast)
            )
            with cm:
                if use_chunks:
                    try:
                        chunk = self.policy.predict_action_chunk(batch)  # [B, T, ...]
                        self.get_logger().debug(f"Generated action chunk shape: {self._shape_str(chunk)}")
                        
                        # Batch postprocessing (process entire chunk at once, keep batch dim)
                        post = self._postprocess_actions(chunk)  # same structure, same dims
                        
                        self.get_logger().debug(f"Postprocessed action chunk shape: {self._shape_str(post)}")
                        
                        # Diagnostics: log action dims on first chunk (fail-fast on mismatch)
                        if not hasattr(self, "_logged_layout_once"):
                            self._logged_layout_once = True
                            sizes = {k: v.shape[-1] for k, v in (post.items() if isinstance(post, dict) else {"action": post}).items()}
                            expected = {k: sum(len(s.names or []) for s in self._action_specs_by_key.get(k, [])) for k in sizes}
                            self.get_logger().debug(f"Action dims (policy vs contract): {sizes} vs {expected}")
                            if sizes != expected:
                                raise RuntimeError(f"Policy/contract action dimension mismatch: {sizes} vs {expected}")
                    except Exception as e:
                        self.get_logger().error(f"Error generating actions: {e}")
                        import traceback
                        self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                        return 0

                    # De-batch to numpy (post is [B, T, ...] where B=1 typically)
                    if isinstance(post, dict):
                        # Extract batch dimension [B=0] then convert to numpy
                        arrs = {k: v[0].detach().cpu().numpy().astype(np.float32) for k, v in post.items()}
                        m = min(k, min(a.shape[0] for a in arrs.values()))
                    else:
                        # Tensor case: extract batch dimension then convert
                        arr = post[0].detach().cpu().numpy().astype(np.float32)  # [T, D]
                        m = min(k, arr.shape[0])
                    
                    if m <= 0:
                        raise RuntimeError("Postprocessed chunk is empty")

                    # 2) Always schedule publishes on the node clock,
                    #    aligned to the next execution tick and in the future.
                    now_wall = self.get_clock().now().nanoseconds
                    base_t = self._next_exec_tick_ns(now_wall + self.step_ns)
                    i0 = self._timestep_cursor
                    for i in range(m):
                        t_i = base_t + i * self.step_ns
                        step_i = i0 + i
                        # De-batch to individual packets (slicing from numpy is cheap)
                        if isinstance(post, dict):
                            pkt = {key: arrs[key][i].ravel() for key in arrs.keys()}
                        else:
                            # If multiple keys exist but model returns a single 'action' vector,
                            # put it under 'action' and let router split across repeated specs.
                            pkt = {"action": arr[i].ravel()}
                        self._producer_buffer.append((t_i, step_i, pkt))
                    self._timestep_cursor += m
                    produced = m
                else:
                    try:
                        a = self.policy.select_action(batch)
                        self.get_logger().debug(f"Generated single action shape: {self._shape_str(a)}")
                        post = self._postprocess_actions(a)  # could be tensor or dict, keeps batch dim
                        self.get_logger().debug(f"Postprocessed single action shape: {self._shape_str(post)}")
                    except Exception as e:
                        self.get_logger().error(f"Error generating single action: {e}")
                        import traceback
                        self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                        return 0
                    now_wall = self.get_clock().now().nanoseconds
                    t0 = self._next_exec_tick_ns(now_wall + self.step_ns)
                    i0 = self._timestep_cursor
                    # Extract batch dimension [B=0] and convert to numpy
                    if isinstance(post, dict):
                        pkt = {k: post[k][0].detach().cpu().numpy().astype(np.float32).ravel() for k in post}
                    else:
                        pkt = {"action": post[0].detach().cpu().numpy().astype(np.float32).ravel()}
                    self._producer_buffer.append((t0, i0, pkt))
                    self._timestep_cursor += 1
                    produced = 1

        if self._producer_buffer:
            # Aggregate into queue by timestep (LeRobot parity)
            self._enqueue_actions(self._producer_buffer)
            # Clear starvation flag if we successfully produced
            if produced > 0:
                self._starved_since_ns = None

        return produced

    # ---------------- Executor: timer tick (persistent) ----------------
    def _executor_tick(self) -> None:
        if not self._running_event.is_set() or self._finishing.is_set():
            return

        now_ns = self.get_clock().now().nanoseconds

        # Warmup: widen tolerance x4 and avoid noisy warnings
        tol_ns = self._timing_policy.publish_tolerance_ns

        packet: Optional[Dict[str, np.ndarray]] = None
        selected_step: Optional[int] = None
        with self._queue_lock:
            # Drop clearly stale actions so we can catch up
            removed_steps = set()
            while self._queue and (self._queue[0][0] < now_ns - tol_ns):
                _, step, _ = self._queue.popleft()
                removed_steps.add(step)
            # Sync dict after pops
            if removed_steps:
                for step in removed_steps:
                    self._queue_by_step.pop(step, None)
            
            # Check if queue is empty (either initially or after cleanup)
            if not self._queue:
                self._warn_throttled(
                    "Executor tick: queue empty, publishing safety command"
                )
                # Publish safety command instead of skipping
                if self._starved_since_ns is None:
                    self._starved_since_ns = now_ns
                self._publish_safety_command()
                return

            # Find best action after cleanup
            best_idx = -1
            best_abs = None
            for idx, (t_ns, _, _) in enumerate(self._queue):
                d = abs(t_ns - now_ns)
                if best_abs is None or d < best_abs:
                    best_abs, best_idx = d, idx

            if best_abs is None or best_abs > tol_ns:
                head_dt_ms = (
                    (self._queue[0][0] - now_ns) /
                    1e6 if self._queue else 0
                )
                self._warn_throttled(
                    f"No action within ±{tol_ns/1e6:.1f}ms of now "
                    f"(head Δ={head_dt_ms:.1f}ms, size={len(self._queue)})"
                )
                if self._starved_since_ns is None:
                    self._starved_since_ns = now_ns
                self._publish_safety_command()
                return

            # Remove all actions before the best one and sync dict
            removed_steps = set()
            for _ in range(best_idx):
                _, step, _ = self._queue.popleft()
                removed_steps.add(step)
            _t_sel, selected_step, packet = self._queue.popleft()
            removed_steps.add(selected_step)
            # Sync dict after pops
            for step in removed_steps:
                self._queue_by_step.pop(step, None)

        if selected_step is not None:
            self._latest_executed_timestep = max(self._latest_executed_timestep, selected_step)
        # Thread-safe write of _last_packet
        with self._last_packet_lock:
            self._last_packet = packet
        try:
            self._publish_action_packet(packet)
        except Exception as e:
            self._warn_throttled(f"Publish failed: {e}")
            return


    def _get_most_recent_image_timestamp(self) -> Optional[int]:
        """Get the timestamp of the most recent image observation.
        
        Uses image observations (sensor_msgs/msg/Image) as the timing anchor
        for synchronizing other observations. Falls back to None if no images.
        """
        # Look for image observations by ROS message type
        image_keys = []
        for key, sub_state in self._subs.items():
            ros_type = getattr(sub_state.spec, 'ros_type', '')
            if ros_type == 'sensor_msgs/msg/Image':
                image_keys.append(key)
        
        if not image_keys:
            return None
            
        # Get the most recent timestamp from image observations
        most_recent_ts = None
        for key in image_keys:
            latest_ts = getattr(self._subs[key].buf, 'last_ts', None)
            if latest_ts is not None:
                if most_recent_ts is None or latest_ts > most_recent_ts:
                    most_recent_ts = latest_ts
        
        # Optional: log clock skew for debugging
        if most_recent_ts is not None:
            skew_ms = (self.get_clock().now().nanoseconds - most_recent_ts) / 1e6
            # noisy while running; keep at debug
            self.get_logger().debug(f"obs-header skew: {skew_ms:.1f} ms")
                    
        return most_recent_ts

    # ---------------- Observation sampling ----------------
    def _sample_obs_frame(self, sample_t_ns: int) -> Dict[str, Any]:
        obs_frame: Dict[str, Any] = {}
        
        # Concatenate every observation.state* group
        for group_key, group_specs in self._state_groups.items():
            samples = {}
            for sv in group_specs:
                dict_key = f"{sv.key}_{sv.topic.replace('/', '_')}"
                if dict_key in self._subs:
                    samples[dict_key] = self._subs[dict_key].buf.sample(sample_t_ns)
            
            # Only the base group uses the computed permutation (if any)
            perm = self._state_index_map if group_key == "observation.state" else None
            
            obs_frame[group_key] = concatenate_state_specs(
                state_specs=group_specs,
                samples=samples,
                zero_pad_map=self._obs_zero,
                permutation_map=perm,
                logger=self.get_logger(),
            )
        
        # Handle all other observations
        for key, st in self._subs.items():
            # Skip individual state keys (already handled by concatenate_state_specs above)
            if key.startswith("observation.state_"):
                continue
                
            v = st.buf.sample(sample_t_ns)
            if v is None:
                zp = self._obs_zero[key]
                obs_frame[key] = zp.copy() if isinstance(
                    zp, np.ndarray) else zp
                self.get_logger().warning(f"Observation {key} is None, zero padding")
            else:
                obs_frame[key] = v
        
        # Always inject the current goal's prompt as 'task'
        # If there are additional streams like 'task.local', 'task.plan', they are already in obs_frame.
        obs_frame["task"] = self._prompt
        return obs_frame

    # ---------------- Batch preparation ----------------
    def _prepare(self, obs_frame: Dict[str, Any]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        for k, v in obs_frame.items():
            if v is None:
                continue
            if isinstance(v, str):
                batch[k] = v
                continue
            if isinstance(v, np.ndarray):
                t = torch.from_numpy(v)
                if t.ndim == 3 and t.shape[2] in (1, 3, 4):
                    t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
                    if np.issubdtype(v.dtype, np.integer):
                        max_val = float(np.iinfo(v.dtype).max)
                        t = t.to(self.device, dtype=torch.float32) / max_val
                    else:
                        t = t.to(self.device, dtype=torch.float32)
                    batch[k] = t
                    continue
                batch[k] = torch.as_tensor(
                    v, dtype=torch.float32, device=self.device)
                continue
            if torch.is_tensor(v):
                t = v
                if t.ndim == 3 and t.shape[2] in (1, 3, 4):
                    t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
                batch[k] = t.to(self.device, dtype=torch.float32)
                continue
            try:
                batch[k] = torch.as_tensor(
                    v, dtype=torch.float32, device=self.device)
            except (ValueError, TypeError, RuntimeError):
                pass
        return batch

    # ---------------- Postprocess wrapper ----------------
    def _postprocess_actions(self, x):
        """Postprocess action outputs: accepts torch.Tensor or dict[str, torch.Tensor].
        
        Postprocessor pipeline expects the full output structure (tensor or dict),
        not each key separately. Keeps batch dimensions intact.
        """
        return self.postprocessor(x)


def main() -> None:
    """Main function to run the policy bridge node."""
    try:
        rclpy.init()
        node = PolicyBridge()
        exe = MultiThreadedExecutor(num_threads=4)
        exe.add_node(node)
        exe.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
