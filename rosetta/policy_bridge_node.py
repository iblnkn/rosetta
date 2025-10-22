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
from contextlib import nullcontext
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
    qos_profile_from_dict,    contract_fingerprint,
    decode_value,
    StreamBuffer,
    stamp_from_header_ns,
    encode_value,
)

from rosetta_interfaces.action import RunPolicy

_ACTION_NAME = "run_policy"
_FEEDBACK_PERIOD_S = 0.5


@dataclass(slots=True)
class _SubState:
    spec: SpecView
    msg_type: Any
    buf: StreamBuffer
    stamp_src: str  # 'receive' or 'header'


@dataclass(slots=True)
class _RuntimeParams:
    use_chunks: bool
    actions_per_chunk: int
    chunk_size_threshold: float
    use_header_time: bool
    use_autocast: bool
    max_queue_actions: int = 512


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
        self.declare_parameter("use_chunks", True)
        self.declare_parameter("actions_per_chunk", 25)
        self.declare_parameter("chunk_size_threshold", 0.5)
        self.declare_parameter("max_queue_actions", 512)
        self.declare_parameter("use_header_time", True)
        self.declare_parameter("use_autocast", False)

        self._params = self._read_params()
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

        # ---------------- Specs & rate ----------------
        self._specs: List[SpecView] = list(iter_specs(self._contract))
        self._obs_specs = [s for s in self._specs if not s.is_action]
        self._act_specs = [s for s in self._specs if s.is_action]
        #TODO: process _task_specs

        #TODO: Multiple action specs are not supported yet. We want to support multiple 
        #possible action topic output at the very least.
        if len(self._act_specs) != 1:
            raise ValueError(
                "This bridge expects exactly one action spec in the contract."
            )
        self._act_spec = self._act_specs[0]

        self.fps = int(self._contract.rate_hz)
        if self.fps <= 0:
            raise ValueError("Contract rate_hz must be >= 1")
        self.step_ns = int(round(1e9 / self.fps))
        self.step_sec = 1.0 / self.fps

        self._cbg = ReentrantCallbackGroup()
        self._obs_zero, self._subs, self._ros_sub_handles = {}, {}, []
        self._state_specs = [s for s in self._obs_specs if s.key == "observation.state"]

        for s in self._obs_specs:
            k, meta, _ = feature_from_spec(s, use_videos=False)
            
            # Create unique key for multiple observation.state specs (mirror bag_to_lerobot logic)
            if s.key == "observation.state" and len(self._state_specs) > 1:
                dict_key = f"{s.key}_{s.topic.replace('/', '_')}"
            else:
                dict_key = s.key
                
            self._obs_zero[dict_key] = zero_pad(meta)

            msg_cls = get_message(s.ros_type)
            sub = self.create_subscription(
                msg_cls, s.topic, lambda m, sv=s: self._obs_cb(m, sv),
                qos_profile_from_dict(self._obs_qos_by_key.get(s.key)),
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

        # ---------------- Publisher ----------------
        act_qos_dict = self._act_qos_by_key.get(self._act_spec.key)
        pub_qos = qos_profile_from_dict(act_qos_dict) or QoSProfile(depth=10)
        self._act_pub = self.create_publisher(
            get_message(self._act_spec.ros_type), self._act_spec.topic, pub_qos
        )

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
        self._safety_behavior = getattr(
            self._act_spec, "safety_behavior", "zeros"
        ).lower()
        if self._safety_behavior not in ("zeros", "hold"):
            self.get_logger().warning(
                f"Unknown safety_behavior '{self._safety_behavior}', defaulting to 'zeros'"
            )
            self._safety_behavior = "zeros"
        self.get_logger().info(
            "Safety: zeros on stop."
            if self._safety_behavior == "zeros"
            else "Safety: hold last action on stop."
        )

        # ---------------- Timing strategy ----------------
        raw_action_spec = self._contract.actions[0] if self._contract.actions else None
        strategy = getattr(raw_action_spec, "publish_strategy", None) or {}
        self._publish_mode = strategy.get("mode", "nearest")
        # Be a bit lenient on timing jitter (network + scheduling)
        self._publish_tolerance_ns = int(self.step_ns)

        # ---------------- Async producer/executor ----------------
        self._queue: Deque[Tuple[int, np.ndarray]] = deque(
            maxlen=self._params.max_queue_actions
        )
        self._queue_lock = threading.Lock()
        self._last_action: Optional[np.ndarray] = None
        self._producer_buffer: List[Tuple[int, np.ndarray]] = []

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
            use_chunks=bool(self.get_parameter("use_chunks").value),
            actions_per_chunk=self.get_parameter("actions_per_chunk").value, #TODO: this should come from the model/policy config. This nomenclature is confusing and not consistent with LeRobot. 
            chunk_size_threshold=float(
                self.get_parameter("chunk_size_threshold").value or 0.5 #TODO: also inconsistent with LeRobot naming.
            ),
            use_header_time=bool(self.get_parameter("use_header_time").value),
            use_autocast=bool(self.get_parameter("use_autocast").value),
            max_queue_actions=self.get_parameter("max_queue_actions").value,
        )

    def _on_params(self, _params: List[Parameter]) -> SetParametersResult:
        new_params = self._read_params()
        if self._queue.maxlen != new_params.max_queue_actions:
            with self._queue_lock:
                self._queue = deque(self._queue, maxlen=new_params.max_queue_actions)
        self._params = new_params
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
            self._terminal = ("timeout", f"Completed successfully after {self._max_duration_s:.1f}s")
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
        self._prompt = ""
        self._done_event.set()

    def _create_safety_vector(self) -> np.ndarray:
        """Create safety action vector (zeros or hold last action)."""
        if self._safety_behavior == "hold" and self._last_action is not None:
            return self._last_action.copy()
        else:
            n = max(len(self._act_spec.names or []), 2)
            return np.zeros((n,), dtype=np.float32)

    def _publish_action_vector(self, action_vec: np.ndarray, increment_count: bool = True, 
                              log_message: str = None, error_context: str = "action") -> None:
        """Publish an action vector with consistent error handling.
        
        Args:
            action_vec: The action vector to publish
            increment_count: Whether to increment the publish counter
            log_message: Optional message to log after successful publish
            error_context: Context string for error messages
        """
        try:
            msg = encode_value(
                ros_type=self._act_spec.ros_type,
                names=self._act_spec.names,
                action_vec=action_vec,
                clamp=getattr(self._act_spec, "clamp", None),
            )
            self._act_pub.publish(msg)

            if increment_count:
                self._pub_count += 1
            if log_message:
                self.get_logger().info(log_message)

        except (RuntimeError, ValueError, TypeError) as e:
            action_type = getattr(self._act_spec, "ros_type", "unknown")
            action_names = getattr(self._act_spec, "names", None)
            vec_len = len(action_vec) if action_vec is not None else "unknown"
            names_count = len(action_names) if action_names else 0
            self.get_logger().error(
                f"{error_context} publish error: {e} "
                f"(type={action_type}, names_count={names_count}, vec_len={vec_len})"
            )

    def _publish_safety_command(self, increment_count: bool = True, log_message: str = None) -> None:
        """Publish safety command (zeros or hold last action)."""
        safety_vec = self._create_safety_vector()
        self._publish_action_vector(safety_vec, increment_count, log_message, "safety")

    # ---------------- Sub callback ----------------
    def _obs_cb(self, msg, spec: SpecView) -> None:
        use_header = (spec.stamp_src ==
                      "header") or self._params.use_header_time
        ts = stamp_from_header_ns(msg) if use_header else None
        ts_ns = int(
            ts) if ts is not None else self.get_clock().now().nanoseconds
        val = decode_value(spec.ros_type, msg, spec)
        if val is not None:
            # Mirror the subscription key used at construction time
            if spec.key == "observation.state" and len(self._state_specs) > 1:
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
        use_chunks = self._params.use_chunks
        k = self._params.actions_per_chunk
        thr = float(self._params.chunk_size_threshold)

        produced = 0
        with self._queue_lock:
            queue_length = len(self._queue)
            need_chunk = use_chunks and (
                queue_length == 0 or (k > 0 and (queue_length / max(1, k)) <= thr)
            )
        if use_chunks and not need_chunk:
            return 0

        self._producer_buffer.clear()
        use_autocast = self._params.use_autocast and (
            hasattr(torch.amp, "autocast_mode")
            and torch.amp.autocast_mode.is_autocast_available(self.device.type)
        )

        # 1) Choose a sampling time (can be header time) for observations
        sample_t_ns = None
        if self._params.use_header_time:
            ts = self._get_most_recent_image_timestamp()
            # Guard: if header time is too stale relative to node clock, ignore it
            if ts is not None:
                skew = self.get_clock().now().nanoseconds - ts
                if 0 <= skew <= int(500e6):  # <= 500 ms stale is OK
                    sample_t_ns = ts
        if sample_t_ns is None:
            sample_t_ns = self.get_clock().now().nanoseconds

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
                        chunk = self.policy.predict_action_chunk(batch)
                        self.get_logger().info(f"Generated action chunk shape: {chunk.shape}")
                        chunk = chunk.squeeze(0)
                        chunk = self._postprocess_actions(chunk)
                        self.get_logger().info(f"Postprocessed action chunk shape: {chunk.shape}")
                    except Exception as e:
                        self.get_logger().error(f"Error generating actions: {e}")
                        import traceback
                        self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                        return 0

                    # 2) Always schedule publishes on the node clock,
                    #    aligned to the next execution tick and in the future.
                    now_wall = self.get_clock().now().nanoseconds
                    base_t = self._next_exec_tick_ns(now_wall + self.step_ns)
                    for i in range(k):
                        t_i = base_t + i * self.step_ns
                        self._producer_buffer.append(
                            (
                                t_i,
                                np.asarray(
                                    chunk[i].detach().cpu().numpy(),
                                    dtype=np.float32, #TODO: We might want to have the option to use other dtypes
                                ).ravel(),
                            )
                        )
                        produced = k
                else:
                    try:
                        a = self.policy.select_action(batch)
                        self.get_logger().info(f"Generated single action shape: {a.shape}")
                        a = self._postprocess_actions(a)
                        self.get_logger().info(f"Postprocessed single action shape: {a.shape}")
                    except Exception as e:
                        self.get_logger().error(f"Error generating single action: {e}")
                        import traceback
                        self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                        return 0
                    now_wall = self.get_clock().now().nanoseconds
                    t0 = self._next_exec_tick_ns(now_wall + self.step_ns)
                    self._producer_buffer.append(
                        (
                            t0,
                            np.asarray(
                                a[0].detach().cpu().numpy(), dtype=np.float32
                            ).ravel(),
                        )
                    )
                    produced = 1

        if self._producer_buffer:
            with self._queue_lock:
                self._queue.extend(self._producer_buffer)

        return produced

    # ---------------- Executor: timer tick (persistent) ----------------
    def _executor_tick(self) -> None:
        if not self._running_event.is_set() or self._finishing.is_set():
            return

        now_ns = self.get_clock().now().nanoseconds

        # Warmup: widen tolerance x4 and avoid noisy warnings
        tol_ns = self._publish_tolerance_ns

        act_vec: Optional[np.ndarray] = None
        with self._queue_lock:
            # Drop clearly stale actions so we can catch up
            while self._queue and (self._queue[0][0] < now_ns - tol_ns):
                self._queue.popleft()
            
            # Check if queue is empty (either initially or after cleanup)
            if not self._queue:
                self.get_logger().warning(
                    "Executor tick: queue empty, publishing safety command"
                )
                # Publish safety command instead of skipping
                self._publish_safety_command()
                return

            # Find best action after cleanup
            best_idx = -1
            best_abs = None
            for idx, (t_ns, _) in enumerate(self._queue):
                d = abs(t_ns - now_ns)
                if best_abs is None or d < best_abs:
                    best_abs, best_idx = d, idx

            if best_abs is None or best_abs > tol_ns:
                head_dt_ms = (
                    (self._queue[0][0] - now_ns) /
                    1e6 if self._queue else 0
                )
                self.get_logger().warning(
                    f"No action within ±{tol_ns/1e6:.1f}ms of now "
                    f"(head Δ={head_dt_ms:.1f}ms, size={len(self._queue)})"
                )
                return

            # Remove all actions before the best one
            for _ in range(best_idx):
                self._queue.popleft()
            _t_sel, act_vec = self._queue.popleft()

        self._last_action = act_vec
        self._publish_action_vector(act_vec)


    def _get_most_recent_image_timestamp(self) -> Optional[int]:
        """Get the timestamp of the most recent primary image observation."""
        # Look for image observations by checking if the spec has image_resize set
        image_keys = []
        for key, sub_state in self._subs.items():
            if hasattr(sub_state.spec, 'image_resize') and sub_state.spec.image_resize is not None:
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
            self.get_logger().info(f"obs-header skew: {skew_ms:.1f} ms")
                    
        return most_recent_ts

    # ---------------- Observation sampling ----------------
    def _sample_obs_frame(self, sample_t_ns: int) -> Dict[str, Any]:
        obs_frame: Dict[str, Any] = {}
        
        # Handle multiple observation.state specs by consolidating them
        if len(self._state_specs) > 1:
            state_parts = []
            for sv in self._state_specs:
                dict_key = f"{sv.key}_{sv.topic.replace('/', '_')}"
                if dict_key in self._subs:
                    v = self._subs[dict_key].buf.sample(sample_t_ns)
                    if v is None:
                        zp = self._obs_zero[dict_key]
                        v = zp.copy() if isinstance(zp, np.ndarray) else zp
                        self.get_logger().warning(f"Observation {dict_key} is None, zero padding")
                    state_parts.append(v)
                else:
                    # Fallback to zero padding if subscription missing
                    zp = self._obs_zero.get(dict_key, np.zeros((len(sv.names),), dtype=np.float32))
                    state_parts.append(zp.copy() if isinstance(zp, np.ndarray) else zp)
            
            # Concatenate all state parts in contract order
            if state_parts:
                obs_frame["observation.state"] = np.concatenate(state_parts, axis=0)
            else:
                obs_frame["observation.state"] = np.zeros((0,), dtype=np.float32)
        
        # Handle all other observations
        for key, st in self._subs.items():
            # Skip individual state keys if we have multiple state specs (already handled above)
            if key.startswith("observation.state_") and len(self._state_specs) > 1:
                continue
                
            v = st.buf.sample(sample_t_ns)
            if v is None:
                zp = self._obs_zero[key]
                obs_frame[key] = zp.copy() if isinstance(
                    zp, np.ndarray) else zp
                self.get_logger().warning(f"Observation {key} is None, zero padding")
            else:
                obs_frame[key] = v
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
            x = x.to(self.device)
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
