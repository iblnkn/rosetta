#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PolicyBridge (slim): orchestrates ROS I/O (client) and inference (server).

- ROS I/O lives in client.py (subs/pubs, resampling, routing).
- Inference backends live in server.py (local LeRobot or LeRobot gRPC).
- This node owns timers, queue policy, action server, feedback, and safety.

Params (subset):
  contract_path (str)  — required
  policy_path   (str)  — required
  policy_device (str)  — "auto"/"cuda"/"cuda:N"/"mps"/"cpu"
  server_backend (str) — "local" (default) | "lerobot_grpc"
  lerobot_host   (str) — host for gRPC backend
  lerobot_port   (int) — port for gRPC backend
  use_action_chunks (bool)
  actions_per_chunk (int)
  chunk_size_threshold (float)
  max_queue_actions (int)
  use_header_time (bool)
  header_skew_ms (float)
  publish_tolerance_ms (float|None)
  aggregate_fn_name (str) — "weighted_average"|"latest_only"|"average"|"conservative"
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import rclpy
import torch
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, ParameterType
from std_srvs.srv import Trigger

# Split modules
from .client import RosClient, TimingPolicy
from .server import (
    ServerConfig,
    InferenceServer,
    LocalPolicyServer,
    LerobotGrpcServer,
)

from rosetta_interfaces.action import RunPolicy

_ACTION_NAME = "run_policy"
_FEEDBACK_PERIOD_S = 0.5


@dataclass(slots=True)
class RuntimeParams:
    use_action_chunks: bool
    actions_per_chunk: int
    chunk_size_threshold: float
    use_header_time: bool
    use_autocast: bool
    max_queue_actions: int
    header_skew_ms: float
    publish_tolerance_ms: Optional[float]
    max_future_ms: float


def _torch_autocast_ok(device_type: str) -> bool:
    return hasattr(torch.amp, "autocast_mode") and torch.amp.autocast_mode.is_autocast_available(device_type)


class PolicyBridge(Node):
    """Slim orchestrator: timers, queue, action interface, safety."""

    def __init__(self) -> None:
        super().__init__("policy_bridge")

        # -------- Parameters --------
        self.declare_parameter("contract_path", "")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("policy_device", "auto")

        self.declare_parameter("use_action_chunks", True)
        self.declare_parameter("actions_per_chunk", 100)
        self.declare_parameter("chunk_size_threshold", 0.5)
        self.declare_parameter("max_queue_actions", 512)
        self.declare_parameter("use_header_time", True)
        self.declare_parameter("use_autocast", False)
        self.declare_parameter("aggregate_fn_name", "weighted_average")
        self.declare_parameter("header_skew_ms", 500.0)
        desc = ParameterDescriptor()
        desc.type = ParameterType.PARAMETER_DOUBLE
        desc.dynamic_typing = True
        self.declare_parameter("publish_tolerance_ms", None, desc)
        self.declare_parameter("max_future_ms", 250.0)  # cap how far-ahead we keep

        # Backend selection (local vs LeRobot gRPC)
        self.declare_parameter("server_backend", "local")  # "local" | "lerobot_grpc"
        self.declare_parameter("lerobot_host", "127.0.0.1")
        self.declare_parameter("lerobot_port", 8080)

        # Validate required params
        contract_path = str(self.get_parameter("contract_path").value or "")
        if not contract_path:
            raise RuntimeError("'contract_path' is required")
        policy_path = str(self.get_parameter("policy_path").value or "")
        if not policy_path:
            raise RuntimeError("'policy_path' is required")

        # Read runtime params
        self._params = self._read_params()
        self.add_on_set_parameters_callback(self._on_params)

        # --- ROS client (subs/pubs per contract) ---
        self.client = RosClient(self, contract_path)
        # Timing policy for sampling/publish tolerance
        self.timing = TimingPolicy(
            step_ns=self.client.step_ns,
            publish_tolerance_ms=self._params.publish_tolerance_ms,
            use_header_time=self._params.use_header_time,
            header_skew_ms=self._params.header_skew_ms,
        )

        # --- Aggregation modes (parity with your node) ---
        self._AGG = {
            "weighted_average": lambda a, b: 0.3 * a + 0.7 * b,
            "latest_only": lambda a, b: b,
            "average": lambda a, b: 0.5 * a + 0.5 * b,
            "conservative": lambda a, b: 0.7 * a + 0.3 * b,
        }
        agg_name = str(self.get_parameter("aggregate_fn_name").value or "weighted_average")
        if agg_name not in self._AGG:
            raise ValueError(f"Unknown aggregate_fn_name='{agg_name}'")
        self._agg_name = agg_name

        # --- Inference backend ---
        backend = str(self.get_parameter("server_backend").value or "local")
        if backend == "lerobot_grpc":
            host = str(self.get_parameter("lerobot_host").value)
            port = int(self.get_parameter("lerobot_port").value)
            self.server: InferenceServer = LerobotGrpcServer(host, port)
            self.get_logger().info(f"Backend: lerobot_grpc @ {host}:{port}")
        else:
            self.server = LocalPolicyServer()
            self.get_logger().info("Backend: local (in-process)")

        # Optional dataset stats passthrough (parity with your preprocessing)
        self._ds_stats = None
        for cand in ("dataset_stats.json", "stats.json", "meta/stats.json"):
            p = Path(policy_path) / cand
            if p.exists():
                try:
                    self._ds_stats = json.loads(p.read_text(encoding="utf-8"))
                    self.get_logger().info(f"Loaded dataset stats from {p}")
                    break
                except Exception as e:
                    self.get_logger().warning(f"Failed to read {p}: {e!r}")

        # Load server
        self.server.load(
            ServerConfig(
                policy_path=policy_path,
                device=str(self.get_parameter("policy_device").value),
                actions_per_chunk=int(self.get_parameter("actions_per_chunk").value),
                dataset_stats=self._ds_stats,
                rename_map=None,  # can supply if you have renames
            )
        )

        # ---- Build state permutation to match training order ----
        self._state_index_map = None
        try:
            expected = None
            if isinstance(self._ds_stats, dict):
                if isinstance(self._ds_stats.get("observation.state"), dict):
                    expected = self._ds_stats["observation.state"].get("names")
                if expected is None:
                    expected = self._ds_stats.get("state_names")
            # Fallback if your LeRobot policy exposes names in config (optional)
            if expected is None and hasattr(getattr(self.server, "policy", None), "config"):
                cfg = self.server.policy.config  # type: ignore[attr-defined]
                if hasattr(cfg, "state_names"):
                    expected = list(cfg.state_names)
            if expected:
                # Current contract order from client (base group only)
                contract_names = []
                for sv in self.client._state_groups.get("observation.state", []):  # type: ignore[attr-defined]
                    contract_names.extend(list(sv.names or []))
                if set(contract_names) != set(expected):
                    self.get_logger().error(
                        "State feature set mismatch between contract and policy.\n"
                        f"Contract-only={sorted(set(contract_names)-set(expected))}\n"
                        f"Policy-only={sorted(set(expected)-set(contract_names))}"
                    )
                else:
                    pos = {name: i for i, name in enumerate(contract_names)}
                    self._state_index_map = np.asarray([pos[name] for name in expected], dtype=np.int64)
                    if contract_names != list(expected):
                        self.get_logger().warning(
                            "Observation.state order differs from policy; applying runtime permutation."
                        )
        except Exception as e:
            self.get_logger().warning(f"Could not build state permutation: {e!r}")

        # --- Action interface ---
        self._cbg = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            RunPolicy,
            _ACTION_NAME,
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._cbg,
        )
        self._cancel_srv = self.create_service(
            Trigger, f"{_ACTION_NAME}/cancel", self._cancel_service_cb, callback_group=self._cbg
        )

        # --- State for run ---
        self._active_handle: Optional[Any] = None
        self._running = threading.Event()
        self._stop_requested = threading.Event()
        self._done_evt = threading.Event()
        self._finishing = threading.Event()
        self._prompt = ""
        self._pub_count = 0

        # Timesteps + queue
        self._timestep_cursor: int = 0
        self._latest_executed_timestep: int = -1
        self._queue: Deque[Tuple[int, int, Dict[str, np.ndarray]]] = deque(maxlen=self._params.max_queue_actions)
        self._queue_by_step: Dict[int, Tuple[int, Dict[str, np.ndarray]]] = {}
        self._queue_lock = threading.Lock()

        # Safety state
        self._last_packet: Optional[Dict[str, np.ndarray]] = None
        self._last_packet_lock = threading.Lock()
        self._starved_since_ns: Optional[int] = None

        # Timers
        self._cbg_timers = ReentrantCallbackGroup()
        self._producer_timer = self.create_timer(self.client.step_sec, self._producer_tick, callback_group=self._cbg_timers)
        self._executor_timer = self.create_timer(self.client.step_sec, self._executor_tick, callback_group=self._cbg_timers)
        self._feedback_timer = self.create_timer(_FEEDBACK_PERIOD_S, self._feedback_tick, callback_group=self._cbg_timers)

        self.get_logger().info(f"PolicyBridge ready at {self.client.fps:.1f} Hz.")

    # ---------------- Param handling ----------------
    def _read_params(self) -> RuntimeParams:
        return RuntimeParams(
            use_action_chunks=bool(self.get_parameter("use_action_chunks").value),
            actions_per_chunk=int(self.get_parameter("actions_per_chunk").value),
            chunk_size_threshold=float(self.get_parameter("chunk_size_threshold").value),
            use_header_time=bool(self.get_parameter("use_header_time").value),
            use_autocast=bool(self.get_parameter("use_autocast").value),
            max_queue_actions=int(self.get_parameter("max_queue_actions").value),
            header_skew_ms=float(self.get_parameter("header_skew_ms").value),
            publish_tolerance_ms=self.get_parameter("publish_tolerance_ms").value,
            max_future_ms=float(self.get_parameter("max_future_ms").value),
        )

    def _on_params(self, _params: List[Parameter]) -> SetParametersResult:
        newp = self._read_params()
        if self._queue.maxlen != newp.max_queue_actions:
            with self._queue_lock:
                self._queue = deque(self._queue, maxlen=newp.max_queue_actions)
        self._params = newp
        self.timing = TimingPolicy(
            step_ns=self.client.step_ns,
            publish_tolerance_ms=newp.publish_tolerance_ms,
            use_header_time=newp.use_header_time,
            header_skew_ms=newp.header_skew_ms,
        )
        return SetParametersResult(successful=True)

    def _future_window_ns(self) -> int:
        """Convenience helper: max_future_ms in nanoseconds."""
        return int(float(self._params.max_future_ms) * 1e6)

    # ---------------- Action lifecycle ----------------
    def _goal_cb(self, _req) -> GoalResponse:
        return GoalResponse.REJECT if self._active_handle is not None else GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle) -> CancelResponse:
        if self._active_handle is None or goal_handle != self._active_handle:
            return CancelResponse.REJECT
        self._stop_requested.set()
        return CancelResponse.ACCEPT

    def _cancel_service_cb(self, _req, resp):
        self._stop_requested.set()
        self._done_evt.set()
        resp.success = True
        resp.message = "Policy run cancellation requested"
        return resp

    def _mk_result(self, success: bool, message: str) -> RunPolicy.Result:
        res = RunPolicy.Result()
        res.success = bool(success)
        res.message = str(message)
        return res

    def _execute_cb(self, goal_handle) -> RunPolicy.Result:
        if self._active_handle is not None:
            goal_handle.abort()
            return self._mk_result(False, "Already running")

        # start
        self._active_handle = goal_handle
        self._running.set()
        self._stop_requested.clear()
        self._done_evt.clear()
        self._finishing.clear()
        self._pub_count = 0
        self._timestep_cursor = 0
        self._latest_executed_timestep = -1
        with self._queue_lock:
            self._queue.clear()
            self._queue_by_step.clear()
        with self._last_packet_lock:
            self._last_packet = None
        self._starved_since_ns = None

        # prompt/task
        self._prompt = getattr(goal_handle.request, "task", None) or getattr(goal_handle.request, "prompt", "") or ""
        try:
            self.server.reset()
        except Exception:
            pass

        self.get_logger().info(f"{_ACTION_NAME}: started (task='{self._prompt}')")
        self._done_evt.wait()

        ok, msg = True, "Policy run ended"
        if self._stop_requested.is_set():
            ok, msg = False, "Cancelled"

        try:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            elif ok:
                goal_handle.succeed()
            else:
                goal_handle.abort()
        except Exception:
            pass

        self._active_handle = None
        return self._mk_result(ok, msg)

    # ---------------- Timers ----------------
    def _feedback_tick(self) -> None:
        if not (self._active_handle and self._running.is_set() and not self._finishing.is_set()):
            return
        try:
            fb = RunPolicy.Feedback()
            if hasattr(fb, "published_actions"):
                fb.published_actions = int(self._pub_count)
            with self._queue_lock:
                if hasattr(fb, "queue_depth"):
                    fb.queue_depth = int(len(self._queue))
            if hasattr(fb, "status"):
                fb.status = "executing"
            self._active_handle.publish_feedback(fb)
        except Exception as e:
            self.get_logger().warning(f"feedback publish failed: {e!r}")

    def _producer_tick(self) -> None:
        if not self._running.is_set() or self._finishing.is_set():
            return
        if self._stop_requested.is_set():
            self._finish_run()
            return
        try:
            # --- gating like the monolith ---
            if self._params.use_action_chunks:
                k = max(1, int(self._params.actions_per_chunk))
                thr = float(self._params.chunk_size_threshold)
                with self._queue_lock:
                    qlen = len(self._queue)
                    head_dt_ns = None
                    if qlen > 0:
                        head_dt_ns = self._queue[0][0] - self.get_clock().now().nanoseconds
                need_chunk = (
                    self._starved_since_ns is not None
                    or qlen == 0
                    or (qlen / k) <= thr
                )
                # If starved because head is *ahead* of now, producing more is harmful.
                if (
                    self._starved_since_ns is not None
                    and head_dt_ns is not None
                    and head_dt_ns > self.timing.publish_tolerance_ns
                ):
                    # Aggressively trim to the future window to let executor catch up
                    if head_dt_ns > self._future_window_ns():
                        cutoff = self.get_clock().now().nanoseconds + self._future_window_ns()
                        with self._queue_lock:
                            kept = [(t, s, p) for (t, s, p) in self._queue if t <= cutoff]
                            self._queue.clear()
                            for it in kept:
                                self._queue.append(it)
                            self._queue_by_step = {s: (t, p) for (t, s, p) in kept}
                    self.get_logger().debug("starved-but-ahead: suppressing production to let executor catch up")
                    return
                if not need_chunk:
                    return
            self._produce()
        except Exception as e:
            self.get_logger().error(f"producer tick failed: {e!r}")

    def _executor_tick(self) -> None:
        if not self._running.is_set() or self._finishing.is_set():
            return

        now_ns = self.get_clock().now().nanoseconds
        tol_ns = self.timing.publish_tolerance_ns

        packet = None
        selected_step = None
        with self._queue_lock:
            # Drop stale first
            removed = set()
            while self._queue and (self._queue[0][0] < now_ns - tol_ns):
                _, step, _ = self._queue.popleft()
                removed.add(step)
            for s in removed:
                self._queue_by_step.pop(s, None)

            # Telemetry: head lead
            head_dt_ms = (self._queue[0][0] - now_ns) / 1e6 if self._queue else None
            if head_dt_ms is not None:
                self.get_logger().debug(f"head_dt={head_dt_ms:.1f}ms q={len(self._queue)} tol={tol_ns/1e6:.1f}ms")

            if not self._queue:
                if self._starved_since_ns is None:
                    self._starved_since_ns = now_ns
                self._publish_safety()
                return

            # Choose closest item to now
            best_idx, best_abs = -1, None
            for idx, (t_ns, _, _) in enumerate(self._queue):
                d = abs(t_ns - now_ns)
                if best_abs is None or d < best_abs:
                    best_abs, best_idx = d, idx
            if best_abs is None or best_abs > tol_ns:
                if self._starved_since_ns is None:
                    self._starved_since_ns = now_ns
                self._publish_safety()
                return

            # Pop up to best
            removed = set()
            for _ in range(best_idx):
                _, step, _ = self._queue.popleft()
                removed.add(step)
            _t_sel, selected_step, packet = self._queue.popleft()
            removed.add(selected_step)
            for s in removed:
                self._queue_by_step.pop(s, None)

        if selected_step is not None:
            self._latest_executed_timestep = max(self._latest_executed_timestep, selected_step)
            # Optional telemetry: log if action is stale
            sel_ms = (now_ns - _t_sel) / 1e6
            if sel_ms > 5.0:
                with self._queue_lock:
                    qlen = len(self._queue)
                self.get_logger().debug(f"Selected action was {sel_ms:.1f}ms old; q={qlen}")

        with self._last_packet_lock:
            self._last_packet = packet
        try:
            self.client.publish_packet(packet)
            self._pub_count += 1
        except Exception as e:
            self.get_logger().warning(f"publish failed: {e!r}")

    # ---------------- Produce path ----------------
    def _next_exec_tick_ns(self, now_ns: int) -> int:
        s = self.client.step_ns
        return ((now_ns + s - 1) // s) * s

    def _prepare_tensor_batch(self, obs_frame: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy/strings to torch tensors on the server device."""
        dev = getattr(self.server, "device", torch.device("cpu"))
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
                        t = t.to(dev, dtype=torch.float32) / float(np.iinfo(v.dtype).max)
                    else:
                        t = t.to(dev, dtype=torch.float32)
                else:
                    t = torch.as_tensor(v, dtype=torch.float32, device=dev)
                batch[k] = t
                continue
            try:
                batch[k] = torch.as_tensor(v, dtype=torch.float32, device=dev)
            except Exception:
                pass
        return batch

    def _produce(self) -> None:
        # Anchor using image header when possible (client may implement recent_image_timestamp_ns())
        get_anchor = getattr(self.client, "recent_image_timestamp_ns", None)
        anchor_ts = get_anchor() if callable(get_anchor) and self._params.use_header_time else None
        if anchor_ts is not None:
            skew = self.get_clock().now().nanoseconds - anchor_ts
            if skew < 0 or skew > int(self._params.header_skew_ms * 1e6):
                anchor_ts = None
        sample_t_ns = anchor_ts if anchor_ts is not None else self.get_clock().now().nanoseconds

        # Sample frame and build tensor batch
        obs_frame = self.client.sample_frame(sample_t_ns, self._prompt, permutation_map=self._state_index_map)
        batch = self._prepare_tensor_batch(obs_frame)

        # Inference (chunks or single)
        use_autocast = self._params.use_autocast and _torch_autocast_ok(getattr(self.server, "device", torch.device("cpu")).type)
        with torch.inference_mode(), torch.autocast(getattr(self.server, "device", torch.device("cpu")).type, enabled=use_autocast):
            out = self.server.predict_chunk(batch)
            if not self._params.use_action_chunks:
                if isinstance(out, dict):
                    out = {k: v[:, :1, ...] for k, v in out.items()}
                else:
                    out = out[:, :1, ...]

        # Convert to numpy packets and enqueue
        if isinstance(out, dict):
            arrs = {k: v[0].detach().cpu().numpy().astype(np.float32) for k, v in out.items()}  # [T, Dk]
            T = min(a.shape[0] for a in arrs.values())
        else:
            arr = out[0].detach().cpu().numpy().astype(np.float32)  # [T, D]
            T = arr.shape[0]

        # --- clamp horizon to actions_per_chunk like the monolith ---
        k = int(self._params.actions_per_chunk)
        if k > 0:
            T = min(T, k)
            if isinstance(out, dict):
                arrs = {key: arrs[key][:T] for key in arrs.keys()}
            else:
                arr = arr[:T]

        now_ns = self.get_clock().now().nanoseconds
        base_t = self._next_exec_tick_ns(now_ns + self.client.step_ns)
        i0 = self._timestep_cursor
        items: List[Tuple[int, int, Dict[str, np.ndarray]]] = []
        for i in range(T):
            t_i = base_t + i * self.client.step_ns
            step_i = i0 + i
            if isinstance(out, dict):
                pkt = {k: arrs[k][i].ravel() for k in arrs.keys()}
            else:
                pkt = {"action": arr[i].ravel()}
            items.append((t_i, step_i, pkt))

        self._enqueue(items)
        self._timestep_cursor += T
        if T > 0:
            self._starved_since_ns = None

    # ---------------- Queue policy ----------------
    def _merge_packets(self, old_pkt: Dict[str, np.ndarray], new_pkt: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        mode = self._agg_name
        if mode == "latest_only":
            return new_pkt
        agg = self._AGG.get(mode) or self._AGG["latest_only"]
        keys = set(old_pkt.keys()) | set(new_pkt.keys())
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            a, b = old_pkt.get(k), new_pkt.get(k)
            if a is None:
                out[k] = b
            elif b is None:
                out[k] = a
            else:
                if a.shape != b.shape:
                    raise ValueError(f"Aggregate shape mismatch for key='{k}': {a.shape} vs {b.shape}")
                out[k] = agg(a, b)
        return out

    def _enqueue(self, items: List[Tuple[int, int, Dict[str, np.ndarray]]]) -> None:
        """Drop stale, then enforce capacity (drop farthest-future)."""
        with self._queue_lock:
            if len(self._queue_by_step) != len(self._queue):
                self._queue_by_step = {step: (t_ns, pkt) for t_ns, step, pkt in self._queue}

            for t_ns, step, pkt in items:
                if step <= self._latest_executed_timestep:
                    continue
                if step in self._queue_by_step:
                    old_t, old_pkt = self._queue_by_step[step]
                    merged = self._merge_packets(old_pkt, pkt)
                    self._queue_by_step[step] = (min(old_t, t_ns), merged)
                else:
                    self._queue_by_step[step] = (t_ns, pkt)

            merged_list = sorted([(t, s, p) for s, (t, p) in self._queue_by_step.items()], key=lambda x: x[0])

            # 1) Drop stale (t < now - tol)
            now_ns = self.get_clock().now().nanoseconds
            tol_ns = self.timing.publish_tolerance_ns
            fresh = [(t, s, p) for (t, s, p) in merged_list if t >= now_ns - tol_ns]
            if len(fresh) != len(merged_list):
                stale_steps = {s for (t, s, _p) in merged_list if t < now_ns - tol_ns}
                for s in stale_steps:
                    self._queue_by_step.pop(s, None)
            merged_list = fresh

            # 1b) Drop too-far future (t > now + future_window)
            upper = now_ns + self._future_window_ns()
            in_window = [(t, s, p) for (t, s, p) in merged_list if t <= upper]
            if len(in_window) != len(merged_list):
                too_far_steps = {s for (t, s, _p) in merged_list if t > upper}
                for s in too_far_steps:
                    self._queue_by_step.pop(s, None)
            merged_list = in_window

            # 2) Capacity: contextual keep
            cap = self._queue.maxlen
            if cap is None or len(merged_list) <= cap:
                kept = merged_list
            else:
                # If we are starved *and* the head is ahead of now (executor is waiting),
                # keep earliest items to pull the head back toward "now".
                head_ahead = False
                if merged_list:
                    head_dt = merged_list[0][0] - now_ns
                    head_ahead = head_dt > tol_ns
                if self._starved_since_ns is not None and head_ahead:
                    kept = merged_list[:cap]   # keep earliest
                else:
                    kept = merged_list[-cap:]  # keep latest (normal case)

            self._queue.clear()
            for it in kept:
                self._queue.append(it)
            self._queue_by_step = {s: (t, p) for t, s, p in kept}

    # ---------------- Safety ----------------
    def _create_safety_packet(self) -> Dict[str, np.ndarray]:
        """Hold last packet if available; otherwise zeros.
        If client exposes per-key dims or behaviors, use them."""
        # Optional richer safety if client exposes these helpers
        dims_fn = getattr(self.client, "action_key_dims", None)
        beh_fn = getattr(self.client, "safety_behavior_by_key", None)
        dims = dims_fn() if callable(dims_fn) else {}
        behaviors = beh_fn() if callable(beh_fn) else {}

        with self._last_packet_lock:
            last = self._last_packet

        pkt: Dict[str, np.ndarray] = {}
        if dims:
            for key, dim in dims.items():
                behavior = (behaviors.get(key) or "zeros").lower()
                if behavior == "hold" and last is not None and key in last:
                    pkt[key] = last[key].copy()
                else:
                    pkt[key] = np.zeros((int(dim),), dtype=np.float32)
            return pkt

        # Fallback (no dims available)
        if last is not None:
            return {k: v.copy() for k, v in last.items()}
        return {"action": np.zeros((1,), dtype=np.float32)}

    def _publish_safety(self) -> None:
        try:
            self.client.publish_packet(self._create_safety_packet())
            self._pub_count += 1
        except Exception as e:
            self.get_logger().warning(f"safety publish failed: {e!r}")

    def _finish_run(self) -> None:
        if self._finishing.is_set():
            return
        self._finishing.set()
        self._running.clear()
        try:
            self._publish_safety()
        except Exception:
            pass
        with self._queue_lock:
            self._queue.clear()
            self._queue_by_step.clear()
        self._prompt = ""
        self._done_evt.set()


def main() -> None:
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
