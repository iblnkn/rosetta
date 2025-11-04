#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PolicyBridge (bin-queued): orchestrates ROS I/O (client) and inference (server).

- Actions are time-stamped samples on a fixed step grid (step_ns).
- Each chunk's i-th action is assigned to bin: base_bin(sample_t_ns) + i.
- Overlaps across chunks are merged ONLY when they land in the same bin.
- Executor publishes exactly the current bin (or safety if missing).

Compatibility:
- Keeps the same parameters and action interface for drop-in use.
- 'publish_tolerance_ms', 'max_future_ms', 'header_skew_ms', 'max_queue_actions'
  are accepted but unused (we log lookahead telemetry instead).
"""

from __future__ import annotations

import json
import heapq
from pathlib import Path
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

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
    latency_compensation_ns: int
    actions_per_chunk: int
    chunk_size_threshold: float
    use_header_time: bool
    use_autocast: bool
    # Declared but unused (kept for drop-in compatibility)
    max_queue_actions: int
    header_skew_ms: float
    publish_tolerance_ms: Optional[float]
    max_future_ms: float


def _torch_autocast_ok(device_type: str) -> bool:
    return hasattr(torch.amp, "autocast_mode") and torch.amp.autocast_mode.is_autocast_available(device_type)


class PolicyBridge(Node):
    """Slim orchestrator: timers, bin-queue, action interface, safety."""

    def __init__(self) -> None:
        super().__init__("policy_bridge")

        # -------- Parameters --------
        self.declare_parameter("contract_path", "")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("policy_device", "auto")
        self.declare_parameter("latency_compensation_ns", 0)
        self.declare_parameter("use_action_chunks", True)
        self.declare_parameter("actions_per_chunk", 100)
        self.declare_parameter("chunk_size_threshold", 0.9)
        # Unused (accepted for compatibility)
        self.declare_parameter("max_queue_actions", 512)
        self.declare_parameter("use_header_time", True)
        self.declare_parameter("use_autocast", False)
        self.declare_parameter("aggregate_fn_name", "weighted_average")
        self.declare_parameter("header_skew_ms", 500.0)
        desc = ParameterDescriptor()
        desc.type = ParameterType.PARAMETER_DOUBLE
        desc.dynamic_typing = True
        self.declare_parameter("publish_tolerance_ms", None, desc)
        self.declare_parameter("max_future_ms", 250.0)

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

        # TimingPolicy kept for compatibility (node no longer uses its tolerances)
        self.timing = TimingPolicy(
            step_ns=self.client.step_ns,
            publish_tolerance_ms=self._params.publish_tolerance_ms,  # unused here
            use_header_time=self._params.use_header_time,
            header_skew_ms=self._params.header_skew_ms,              # unused here
        )

        # --- Aggregation modes ---
        self._AGG = {
            "weighted_average": lambda a, b: 0.3 * a + 0.7 * b,
            "latest_only": lambda a, b: b,
            "average": lambda a, b: 0.5 * a + 0.5 * b,
            "conservative": lambda a, b: 0.7 * a + 0.3 * b,
        }
        agg_name = str(self.get_parameter("aggregate_fn_name").value)
        if agg_name not in self._AGG:
            raise ValueError(f"Unknown aggregate_fn_name='{agg_name}'")
        self._agg_name = agg_name

        # --- Inference backend ---
        backend = str(self.get_parameter("server_backend").value)
        if backend == "lerobot_grpc":
            host = str(self.get_parameter("lerobot_host").value)
            port = int(self.get_parameter("lerobot_port").value)
            self.server: InferenceServer = LerobotGrpcServer(host, port)
            self.get_logger().info(f"Backend: lerobot_grpc @ {host}:{port}")
        else:
            self.server = LocalPolicyServer()
            self.get_logger().info("Backend: local")

        # Optional dataset stats (to help with state permutation)
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
                rename_map=None,
            )
        )

        # ---- Build state permutation to match training order (optional) ----
        self._state_index_map = None
        try:
            expected = None
            if isinstance(self._ds_stats, dict):
                if isinstance(self._ds_stats.get("observation.state"), dict):
                    expected = self._ds_stats["observation.state"].get("names")
                if expected is None:
                    expected = self._ds_stats.get("state_names")
            if expected is None and hasattr(getattr(self.server, "policy", None), "config"):
                cfg = self.server.policy.config  # type: ignore[attr-defined]
                if hasattr(cfg, "state_names"):
                    expected = list(cfg.state_names)
            if expected:
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

        # --- Run state ---
        self._active_handle: Optional[Any] = None
        self._running = threading.Event()
        self._stop_requested = threading.Event()
        self._done_evt = threading.Event()
        self._finishing = threading.Event()
        self._prompt = ""
        self._pub_count = 0

        # --- Bin queue (time-grid) ---
        self._step_ns = self.client.step_ns
        # _bins: {bin_idx -> packet}, _bin_heap: min-heap of bin_idx
        self._bins: Dict[int, Dict[str, np.ndarray]] = {}
        self._bin_heap: List[int] = []
        self._queue_lock = threading.Lock()

        # Throttled lookahead logging
        self._future_log_next_ns = 0

        # Safety
        self._last_packet: Optional[Dict[str, np.ndarray]] = None
        self._last_packet_lock = threading.Lock()

        # Timers
        self._cbg_timers = ReentrantCallbackGroup()
        self._producer_timer = self.create_timer(self.client.step_sec, self._producer_tick, callback_group=self._cbg_timers)
        self._executor_timer = self.create_timer(self.client.step_sec, self._executor_tick, callback_group=self._cbg_timers)
        self._feedback_timer = self.create_timer(_FEEDBACK_PERIOD_S, self._feedback_tick, callback_group=self._cbg_timers)

        self.get_logger().info("PolicyBridge ready (bin-queued).")

    # ---------------- Param handling ----------------
    def _read_params(self) -> RuntimeParams:
        return RuntimeParams(
            latency_compensation_ns=int(self.get_parameter("latency_compensation_ns").value),
            use_action_chunks=bool(self.get_parameter("use_action_chunks").value),
            actions_per_chunk=int(self.get_parameter("actions_per_chunk").value),
            chunk_size_threshold=float(self.get_parameter("chunk_size_threshold").value),
            use_header_time=bool(self.get_parameter("use_header_time").value),
            use_autocast=bool(self.get_parameter("use_autocast").value),
            # Unused (compat only)
            max_queue_actions=int(self.get_parameter("max_queue_actions").value),
            header_skew_ms=float(self.get_parameter("header_skew_ms").value),
            publish_tolerance_ms=self.get_parameter("publish_tolerance_ms").value,
            max_future_ms=float(self.get_parameter("max_future_ms").value),
        )

    def _on_params(self, _params: List[Parameter]) -> SetParametersResult:
        self._params = self._read_params()
        self.timing = TimingPolicy(
            step_ns=self.client.step_ns,
            publish_tolerance_ms=self._params.publish_tolerance_ms,
            use_header_time=self._params.use_header_time,
            header_skew_ms=self._params.header_skew_ms,
        )
        return SetParametersResult(successful=True)

    # ---------------- Bin helpers ----------------
    def _bin_index(self, t_ns: int) -> int:
        """Quantize timestamp to bin **start** (floor), not nearest."""
        s = self._step_ns
        return int(t_ns // s)

    def _time_from_bin(self, b: int) -> int:
        return b * self._step_ns

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

        self._active_handle = goal_handle
        self._running.set()
        self._stop_requested.clear()
        self._done_evt.clear()
        self._finishing.clear()
        self._pub_count = 0
        with self._queue_lock:
            self._bins.clear()
            self._bin_heap.clear()
        with self._last_packet_lock:
            self._last_packet = None

        self._prompt = getattr(goal_handle.request, "prompt", "")
        try:
            self.server.reset()
        except Exception as e:
            self.get_logger().warning(f"Error resetting server: {e!r}")

        self.get_logger().info(f"{_ACTION_NAME}: started (prompt='{self._prompt}')")
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
                depth = len(self._bins)
            if hasattr(fb, "queue_depth"):
                fb.queue_depth = int(depth)
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
            if not self._params.use_action_chunks:
                self._produce()
                return

            k = int(self._params.actions_per_chunk)
            low_water = int(max(1, self._params.chunk_size_threshold * k))

            with self._queue_lock:
                if self._bins:
                    now_bin = self._bin_index(self.get_clock().now().nanoseconds)
                    max_bin = max(self._bins.keys())
                    lookahead_bins = max(0, max_bin - now_bin)
                else:
                    lookahead_bins = 0

            if lookahead_bins > 0:
                now_ns = self.get_clock().now().nanoseconds
                if now_ns >= self._future_log_next_ns:
                    future_ms = lookahead_bins * (self._step_ns / 1e6)
                    if future_ms >= 1000.0:
                        self.get_logger().info(f"lookahead ≈ {future_ms:.1f} ms ({lookahead_bins} bins ahead)")
                    self._future_log_next_ns = now_ns + int(2e9)

            if lookahead_bins < low_water:
                self._produce()

        except Exception as e:
            self.get_logger().error(f"producer tick failed: {e!r}")

    def _executor_tick(self) -> None:
        if not self._running.is_set() or self._finishing.is_set():
            return

        now_ns = self.get_clock().now().nanoseconds
        now_bin = self._bin_index(now_ns)

        packet = None
        dropped = 0

        with self._queue_lock:
            # Drop strictly past bins
            while self._bin_heap and self._bin_heap[0] < now_bin:
                b = heapq.heappop(self._bin_heap)
                if self._bins.pop(b, None) is not None:
                    dropped += 1

            # Publish exactly the current bin if present
            if self._bin_heap and self._bin_heap[0] == now_bin:
                _ = heapq.heappop(self._bin_heap)
                packet = self._bins.pop(now_bin, None)
            else:
                packet = self._bins.pop(now_bin, None)

            # Tiny diagnostic (debug only) when we miss
            if packet is None and self._bin_heap:
                head = self._bin_heap[0]
                tail = max(self._bin_heap) if self._bin_heap else head
                self.get_logger().debug(
                    f"no action for now_bin={now_bin}; queue range: [{head}, {tail}] "
                    f"(gap={head - now_bin} bins)"
                )

        if dropped:
            self.get_logger().debug(f"dropped {dropped} stale bins before publish (real-time)")

        if packet is None:
            self._publish_safety()
            return

        try:
            with self._last_packet_lock:
                self._last_packet = packet
            self.client.publish_packet(packet)
            self._pub_count += 1
        except Exception as e:
            self.get_logger().warning(f"publish failed: {e!r}")

    # ---------------- Produce path ----------------
    def _prepare_tensor_batch(self, obs_frame: Dict[str, Any]) -> Dict[str, Any]:
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

    def _extract_chunk_array(self, out: Union[torch.Tensor, Dict[str, torch.Tensor], List, Tuple]) -> np.ndarray:
        tensor: Optional[torch.Tensor] = None
        if isinstance(out, dict):
            if "action" not in out:
                raise ValueError("Policy output dict missing 'action' key")
            tensor = out["action"]
        elif isinstance(out, (list, tuple)):
            if not out:
                raise ValueError("Empty sequence from policy output")
            tensor = out[0] if isinstance(out[0], torch.Tensor) else None
        elif isinstance(out, torch.Tensor):
            tensor = out
        else:
            raise TypeError(f"Unsupported policy output type: {type(out).__name__}")
        if tensor is None:
            raise ValueError("Could not locate Tensor in policy output")
        if tensor.ndim == 3:   # [B, T, D]
            tensor = tensor[0]
        elif tensor.ndim != 2: # [T, D]
            raise ValueError(f"Unexpected policy tensor shape: {tuple(tensor.shape)}")
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError(f"Policy produced invalid chunk shape: {arr.shape}")
        return arr  # [T, D]

    def _produce(self) -> None:
        sample_t_ns = self.get_clock().now().nanoseconds - self._params.latency_compensation_ns

        obs_frame = self.client.sample_frame(sample_t_ns, self._prompt, permutation_map=self._state_index_map)
        batch = self._prepare_tensor_batch(obs_frame)

        if not any(isinstance(v, torch.Tensor) for v in batch.values()):
            return

        use_autocast = self._params.use_autocast and _torch_autocast_ok(getattr(self.server, "device").type)
        try:
            with torch.inference_mode(), torch.autocast(getattr(self.server, "device").type, enabled=use_autocast):
                out = self.server.predict_chunk(batch)
                if not self._params.use_action_chunks:
                    if isinstance(out, dict):
                        out = {k: v[:, :1, ...] for k, v in out.items()}
                    elif isinstance(out, torch.Tensor):
                        out = out[:, :1, ...]
        except (RuntimeError, KeyError) as e:
            error_msg = str(e)
            policy_info = ""
            if hasattr(self.server, "policy") and hasattr(self.server.policy, "config"):
                cfg = self.server.policy.config
                if hasattr(cfg, "image_features"):
                    policy_info = f", expects image_features={cfg.image_features}"
                if hasattr(cfg, "robot_state_feature"):
                    policy_info += f", robot_state_feature={cfg.robot_state_feature}"
                if hasattr(cfg, "env_state_feature"):
                    policy_info += f", env_state_feature={cfg.env_state_feature}"
            self.get_logger().warning(
                f"Inference failed due to observation mismatch. "
                f"Batch keys: {list(batch.keys())}{policy_info}. "
                f"Error: {error_msg}. Waiting for observations to match."
            )
            return

        arr = self._extract_chunk_array(out)  # [T, D]
        T = int(arr.shape[0])
        if T == 0:
            return

        base_bin = self._bin_index(sample_t_ns)  # floor-anchored
        items: List[Tuple[int, Dict[str, np.ndarray]]] = []
        for i in range(T):
            bin_i = base_bin + i
            pkt = {"action": arr[i].ravel()}
            items.append((bin_i, pkt))

        self._enqueue_bins(items)

    # ---------------- Queue policy (bin-based) ----------------
    def _merge_packets(self, a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._agg_name == "latest_only":
            return b
        agg = self._AGG.get(self._agg_name) or self._AGG["latest_only"]
        keys = set(a.keys()) | set(b.keys())
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            va, vb = a.get(k), b.get(k)
            if va is None:
                out[k] = vb
                continue
            if vb is None:
                out[k] = va
                continue
            if va.shape != vb.shape:
                raise ValueError(f"Aggregate shape mismatch for key='{k}': {va.shape} vs {vb.shape}")
            out[k] = agg(va, vb)
        return out

    def _enqueue_bins(self, items: List[Tuple[int, Dict[str, np.ndarray]]]) -> None:
        merges = 0
        with self._queue_lock:
            for bin_i, pkt in items:
                if bin_i in self._bins:
                    self._bins[bin_i] = self._merge_packets(self._bins[bin_i], pkt)
                    merges += 1
                else:
                    self._bins[bin_i] = pkt
                    heapq.heappush(self._bin_heap, bin_i)
        if merges:
            self.get_logger().debug(f"merged {merges} bins (agg='{self._agg_name}')")

    # ---------------- Safety ----------------
    def _create_safety_packet(self) -> Dict[str, np.ndarray]:
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
            self._bins.clear()
            self._bin_heap.clear()
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
