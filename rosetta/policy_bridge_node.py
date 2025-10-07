#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rosidl_runtime_py.utilities import get_message
from std_srvs.srv import Trigger

import torch
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from rosetta_interfaces.action import RecordEpisode

# ---- shared core ------------------------------------------------------------
from rosetta.common.contract_utils import (
    load_contract, iter_specs, SpecView, feature_from_spec,
    zero_pad, qos_profile_from_dict,
)
from rosetta.common.signal_utils import (
    decode_value, StreamBuffer, now_ns, stamp_from_header_ns, encode_action_to_ros,
)
# ---------------------------------------------------------------------------

@dataclass
class _SubState:
    spec: SpecView
    msg_type: Any
    buf: StreamBuffer
    stamp_src: str  # 'receive' | 'header'


class PolicyBridge(Node):
    """
    Contract-true live inference with async producer/consumer.
    IMPORTANT: We DO NOT spin inside action callbacks. The executor threads
    handle other callbacks; the execute loop waits on Events/timeouts only.
    """

    def __init__(self):
        super().__init__('policy_bridge')

        # ---------------- Parameters ----------------
        self.declare_parameter('contract_path', '')
        self.declare_parameter('policy_path', '')
        self.declare_parameter('policy_device', 'cpu')         # cpu|cuda|mps

        # Chunking/execution
        self.declare_parameter('use_chunks', True)
        self.declare_parameter('actions_per_chunk', 20)
        self.declare_parameter('chunk_size_threshold', 0.5)

        # Transport & timing
        self.declare_parameter('image_qos_depth', 10)          # fallback if spec lacks QoS
        self.declare_parameter('use_header_timestamps', False)

        # Executor idle behavior
        # We will NOT publish periodically when idle; only once on stop.
        self.declare_parameter('publish_when_idle', False)     # ignored in this variant

        # Inference
        self.declare_parameter('use_autocast', False)          # optional CUDA AMP

        # Debug
        self.declare_parameter('debug_log_every_n', 0)
        self.declare_parameter('debug_dump_inputs', False)

        # ---------------- Contract ----------------
        contract_path = self.get_parameter('contract_path').get_parameter_value().string_value
        if not contract_path:
            raise RuntimeError("policy_bridge: 'contract_path' is required")
        self._contract = load_contract(Path(contract_path))

        # ---------------- Policy load ----------------
        policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        if not policy_path:
            raise RuntimeError("policy_bridge: 'policy_path' is required")

        import json
        cfg_type = 'act'
        cfg_json = os.path.join(policy_path, 'config.json')
        try:
            if os.path.exists(cfg_json):
                with open(cfg_json, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                    cfg_type = str(cfg.get('type', 'act')).lower()
        except Exception as e:
            self.get_logger().warning(f"Could not read policy config.json, defaulting to 'act': {e!r}")

        # Device selection
        requested = (self.get_parameter('policy_device').get_parameter_value().string_value or 'cpu').lower()
        if requested == 'cuda':
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device_str == 'cpu':
                self.get_logger().warning("policy_device='cuda' requested but CUDA not available; using CPU.")
        elif requested in ('mps', 'metal'):
            if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                device_str = 'mps'
            else:
                self.get_logger().warning("policy_device='mps' requested but MPS not available; using CPU.")
                device_str = 'cpu'
        else:
            device_str = 'cuda' if torch.cuda.is_available() else requested
            if device_str == 'cuda' and requested != 'cuda':
                self.get_logger().info("Using CUDA because it is available (override with policy_device).")
            if device_str == 'cpu' and requested not in ('cpu', 'mps', 'metal'):
                self.get_logger().warning(f"Unknown policy_device='{requested}', using CPU.")
        self.device = torch.device(device_str)

        PolicyCls = get_policy_class(cfg_type)
        self.policy = PolicyCls.from_pretrained(policy_path)
        self.policy.to(self.device)
        self.policy.eval()

        # --------- OFFICIAL pre/post processors -----
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        # ---------------- Specs & rate ----------------
        self._specs: List[SpecView] = list(iter_specs(self._contract))
        self._obs_specs = [s for s in self._specs if not s.is_action]
        self._act_specs = [s for s in self._specs if s.is_action]
        if len(self._act_specs) != 1:
            raise ValueError('This bridge expects exactly one action spec in the contract.')
        self._act_spec = self._act_specs[0]

        self.fps = float(self._contract.rate_hz)
        if self.fps <= 0:
            raise ValueError('Contract rate_hz must be > 0')
        self.step_ns = int(round(1e9 / self.fps))

        # Precompute zero pads for observations
        self._obs_zero: Dict[str, Any] = {}
        for s in self._obs_specs:
            k, meta, _ = feature_from_spec(s, use_videos=False)
            self._obs_zero[k] = zero_pad(meta)

        # ---------------- Subscriptions ----------------
        self._cbg = ReentrantCallbackGroup()
        self._subs: Dict[str, _SubState] = {}
        self._ros_sub_handles = []

        fallback_qos = QoSProfile(depth=int(self.get_parameter('image_qos_depth').value or 10))
        for s in self._obs_specs:
            msg_cls = get_message(s.ros_type)
            sub_qos = qos_profile_from_dict(getattr(s, 'qos', None)) or fallback_qos
            sub = self.create_subscription(
                msg_cls, s.topic, lambda m, sv=s: self._obs_cb(m, sv), sub_qos, callback_group=self._cbg
            )
            self._ros_sub_handles.append(sub)

            tol_ns = int(max(0, s.asof_tol_ms)) * 1_000_000
            buf = StreamBuffer(policy=s.resample_policy, step_ns=self.step_ns, tol_ns=tol_ns)
            self._subs[s.key] = _SubState(spec=s, msg_type=msg_cls, buf=buf, stamp_src=s.stamp_src)

        self.get_logger().info(f'Subscribed to {len(self._subs)} observation streams.')

        # ---------------- Publisher ----------------
        pub_qos = qos_profile_from_dict(getattr(self._act_spec, 'publish_qos', None)) or QoSProfile(depth=10)
        self._act_pub = self.create_publisher(get_message(self._act_spec.ros_type), self._act_spec.topic, pub_qos)

        # ---------------- Action server ----------------
        self._active_handle = None
        self._running_event = threading.Event()
        self._stop_requested = threading.Event()   # set by cancel or timeout
        self._timeout_event = threading.Event()    # latched when timeout is hit

        self._action_server = ActionServer(
            self,
            RecordEpisode,
            'run_policy',
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._cbg,
        )

        # ---------------- Cancel service ----------------
        self._cancel_service = self.create_service(
            Trigger,
            'run_policy/cancel',
            self._cancel_service_cb,
            callback_group=self._cbg,
        )

        # ---------------- Contract timeout ----------------
        self._max_duration_s = float(getattr(self._contract, 'max_duration_s', 300.0))

        # ---------------- Safety behavior ----------------
        self._safety_behavior = getattr(self._act_spec, 'safety_behavior', 'zeros').lower()
        if self._safety_behavior not in ('zeros', 'hold'):
            self.get_logger().warning(f"Unknown safety_behavior '{self._safety_behavior}', defaulting to 'zeros'")
            self._safety_behavior = 'zeros'
        self.get_logger().info('Safety behavior: zeros when policy is idle/stopped.' if self._safety_behavior == 'zeros'
                               else 'Safety behavior: hold last action when policy is idle/stopped.')

        # ---------------- Async producer/executor ----------------
        self._queue: deque[np.ndarray] = deque()
        self._queue_lock = threading.Lock()
        self._last_action: np.ndarray | None = None

        self._producer = threading.Thread(target=self._producer_loop, daemon=True)
        self._executor = threading.Thread(target=self._executor_loop, daemon=True)
        self._producer.start()
        self._executor.start()

        self.get_logger().info(f'PolicyBridge ready at {self.fps:.1f} Hz on device={self.device}.')

    # ---------------- Action callbacks ----------------
    def _goal_cb(self, _req) -> GoalResponse:
        if self._active_handle is not None:
            self.get_logger().info('Goal request: REJECT (already running)')
            return GoalResponse.REJECT
        self.get_logger().info('Goal request: ACCEPT')
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle) -> CancelResponse:
        # Accept cancel if it's the active goal
        if self._active_handle is None or goal_handle != self._active_handle:
            return CancelResponse.REJECT
        self.get_logger().info('Action cancel requested')
        self._stop_requested.set()
        return CancelResponse.ACCEPT

    def _cancel_service_cb(self, _req, resp):
        self.get_logger().info('Cancel service called')
        self._stop_requested.set()
        resp.success = True
        resp.message = 'Policy run cancellation requested'
        return resp

    def _execute_cb(self, goal_handle) -> RecordEpisode.Result:
        # Guard: only one active goal
        if self._active_handle is not None:
            goal_handle.abort()
            return RecordEpisode.Result(success=False, message='Already running')

        # ---- Start run
        self._active_handle = goal_handle
        self._stop_requested.clear()
        self._timeout_event.clear()
        self._running_event.set()

        prompt = getattr(goal_handle.request, 'prompt', '') or ''
        self._prompt = prompt

        # Reset policy state if available
        if hasattr(self.policy, "reset"):
            try:
                self.policy.reset()
            except Exception as e:
                self.get_logger().warning(f'policy.reset() failed: {e!r}')
        self.get_logger().info(f"run_policy: started (task='{self._prompt}')")

        # Compute deadline using ROS time
        now_time = self.get_clock().now()
        deadline = now_time + Duration(seconds=self._max_duration_s)

        # Wait loop — NO spinning here; only wait on events/timeouts
        # Allow cancel/timeout to break the loop.
        while rclpy.ok() and self._running_event.is_set():
            # Timeout?
            if self.get_clock().now() >= deadline:
                self.get_logger().warn(f'Policy run timed out after {self._max_duration_s:.1f}s (signaling).')
                self._timeout_event.set()
                break
            # Check stop request (from cancel service or action cancel)
            if self._stop_requested.is_set():
                break
            # Small wait to yield CPU; executor threads handle other callbacks.
            self._stop_requested.wait(timeout=0.05)

        # ---- Stop & cleanup
        timed_out = self._timeout_event.is_set()
        self._stop_running()

        # Publish one immediate safety command on stop if requested
        self._publish_safety_once()

        if timed_out:
            # Timeout → ABORTED
            goal_handle.abort()
            self.get_logger().info('run_policy: stopped (timeout)')
            return RecordEpisode.Result(success=False, message=f'Timed out at {self._max_duration_s:.1f}s')
        elif self._stop_requested.is_set():
            # Client cancel → CANCELED
            goal_handle.canceled()
            self.get_logger().info('run_policy: stopped (cancelled)')
            return RecordEpisode.Result(success=False, message='Cancelled')
        else:
            # Normal stop path (not used in this node, but keep for completeness)
            goal_handle.succeed()
            self.get_logger().info('run_policy: stopped (succeeded)')
            return RecordEpisode.Result(success=True, message='Policy run ended')

    def _stop_running(self):
        # Reset run flags/state so next goal can be accepted immediately
        self._running_event.clear()
        self._stop_requested.clear()
        self._timeout_event.clear()
        self._prompt = ''
        with self._queue_lock:
            self._queue.clear()
        self._active_handle = None

    def _publish_safety_once(self):
        """Publish a single safety command when the policy stops."""
        try:
            if self._safety_behavior == 'hold' and self._last_action is not None:
                vec = self._last_action.copy()
            else:
                n = max(len(self._act_spec.names or []), 2)
                vec = np.zeros((n,), dtype=np.float32)
            msg = encode_action_to_ros(
                ros_type=self._act_spec.ros_type,
                names=self._act_spec.names,
                action_vec=vec,
                clamp=getattr(self._act_spec, 'clamp', None),
            )
            self._act_pub.publish(msg)
            self.get_logger().info('Published immediate safety command on stop.')
        except Exception as e:
            self.get_logger().error(f'safety publish error: {e}')

    # ---------------- Sub callback ----------------
    def _obs_cb(self, msg, spec: SpecView):
        use_header = (spec.stamp_src == 'header') or bool(self.get_parameter('use_header_timestamps').value)
        ts = stamp_from_header_ns(msg) if use_header else None
        ts_ns = int(ts) if ts is not None else now_ns()
        val = decode_value(spec.ros_type, msg, spec)
        if val is not None:
            self._subs[spec.key].buf.push(ts_ns, val)

    # ---------------- Helper: sample obs at a given tick ----------------
    def _sample_obs_frame(self, sample_t_ns: int) -> Dict[str, Any]:
        obs_frame: Dict[str, Any] = {}
        for key, st in self._subs.items():
            v = st.buf.sample(sample_t_ns)
            if v is None:
                zp = self._obs_zero[key]
                obs_frame[key] = zp.copy() if isinstance(zp, np.ndarray) else zp
            else:
                obs_frame[key] = v
        obs_frame['task'] = self._prompt  # preprocessor will tokenize
        return obs_frame

    # ---------------- Helper: robust postprocess wrapper ----------------
    def _postprocess_actions(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=self.device)
        try:
            out = self.postprocessor({"action": x})
            if isinstance(out, dict) and "action" in out:
                return out["action"]
            return out
        except Exception:
            return self.postprocessor(x)

    # ---------------- Producer: compute chunks asynchronously ----------------
    def _producer_loop(self):
        use_autocast = bool(self.get_parameter('use_autocast').value) and (self.device.type == 'cuda')
        while rclpy.ok():
            if not self._running_event.is_set():
                time.sleep(0.01); continue

            use_chunks = bool(self.get_parameter('use_chunks').value)
            k   = int(self.get_parameter('actions_per_chunk').value or 20)
            thr = float(self.get_parameter('chunk_size_threshold').value or 0.5)

            with self._queue_lock:
                qlen = len(self._queue)
            need_chunk = use_chunks and (qlen == 0 or (k > 0 and (qlen / k) <= thr))
            if not need_chunk:
                time.sleep(0.001); continue

            actions: List[np.ndarray] = []
            try:
                obs_frame = self._sample_obs_frame(now_ns())
                batch = self._prepare(obs_frame)
                batch = self.preprocessor(batch)

                if bool(self.get_parameter('debug_dump_inputs').value):
                    self.get_logger().info(f"Policy input keys: {list(batch.keys())}")
                    for k_, v_ in batch.items():
                        if isinstance(v_, torch.Tensor):
                            self.get_logger().info(f"  {k_}: {tuple(v_.shape)} {v_.dtype} {v_.device}")
                        else:
                            self.get_logger().info(f"  {k_}: {type(v_)} -> {v_}")

                with torch.inference_mode():
                    if self.device.type == 'cuda':
                        cm = torch.amp.autocast('cuda', enabled=use_autocast)  # type: ignore[attr-defined]
                    else:
                        cm = _nullcontext()
                    with cm:  # type: ignore
                        if use_chunks:
                            chunk = self.policy.predict_action_chunk(batch)
                            if torch.is_tensor(chunk) and chunk.ndim == 3:
                                chunk = chunk.squeeze(0)          # -> (K, A)
                            chunk = self._postprocess_actions(chunk)
                            if not torch.is_tensor(chunk):
                                chunk = torch.as_tensor(chunk, device=self.device)
                            k_eff = min(k, chunk.shape[0]) if k > 0 else chunk.shape[0]
                            actions = [np.asarray(chunk[i].detach().cpu().numpy()).ravel()
                                       for i in range(k_eff)]
                        else:
                            a = self.policy.select_action(batch)      # (1, A)
                            a = self._postprocess_actions(a)
                            if not torch.is_tensor(a):
                                a = torch.as_tensor(a, device=self.device)
                            actions = [np.asarray(a[0].detach().cpu().numpy()).ravel()]
            except Exception as e:
                import traceback
                self.get_logger().error(f'producer inference error: {e}')
                self.get_logger().error(traceback.format_exc())
                actions = []

            if actions:
                with self._queue_lock:
                    self._queue.extend(actions)

    # ---------------- Executor: publish at fixed FPS (only when running) -----
    def _executor_loop(self):
        next_t = now_ns()
        log_every = int(self.get_parameter('debug_log_every_n').value or 0)
        publish_count = 0

        while rclpy.ok():
            t_ns = now_ns()
            if t_ns < next_t:
                time.sleep(min(0.002, (next_t - t_ns) / 1e9))
                continue

            if self._running_event.is_set():
                act_vec: np.ndarray | None = None
                with self._queue_lock:
                    if self._queue:
                        act_vec = self._queue.popleft()
                if act_vec is not None:
                    self._last_action = act_vec
                    try:
                        msg = encode_action_to_ros(
                            ros_type=self._act_spec.ros_type,
                            names=self._act_spec.names,
                            action_vec=act_vec,
                            clamp=getattr(self._act_spec, 'clamp', None),
                        )
                        self._act_pub.publish(msg)
                        publish_count += 1
                        if log_every and (publish_count % log_every == 0):
                            with self._queue_lock:
                                qlen = len(self._queue)
                            head = ', '.join([f'{x:.3f}' for x in act_vec[:min(4, act_vec.shape[0])]])
                            self.get_logger().info(
                                f'pub #{publish_count}: {self._act_spec.topic} ← [{head} …] | queue={qlen}'
                            )
                    except Exception as e:
                        self.get_logger().error(f'publish error: {e}')
            # No idle publishing here.

            # Catch-up to avoid long-term drift after overruns
            next_t = max(next_t + self.step_ns, now_ns())

    # ---------------- Minimal local preprocessor (shapes only, no extra normalization) ---------------
    def _prepare(self, obs_frame: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw obs to tensors with correct SHAPES for the official LeRobot processors.
    
          - images: accept HWC/CHW np/torch; output BCHW float32 on self.device
            * If dtype is uint8 or dynamic range looks like 0..255, rescale to 0..1
            * Otherwise, DO NOT normalize again (stats-based normalization is done later)
          - vectors: 1D -> (1, D) float32 on self.device
          - 'task': pass through as str (tokenized by the preprocessor)
        """
        batch: Dict[str, Any] = {}
    
        def _to_bchw_float01(t: torch.Tensor) -> torch.Tensor:
            # t: HWC or CHW or HW, any dtype; returns BCHW float32 on device with values in [0,1]
            if t.ndim == 2:  # HW -> HWC (gray)
                t = t.unsqueeze(-1).repeat(1, 1, 3)
            if t.ndim == 3 and t.shape[-1] in (1, 3):            # HWC
                t = t.permute(2, 0, 1)                            # -> CHW
            elif t.ndim == 3 and t.shape[0] in (1, 3):            # CHW
                pass
            else:
                # Not an image-like tensor; return as-is and let caller handle
                return t
    
            # Now CHW. Add batch
            if t.ndim == 3:
                t = t.unsqueeze(0)                                # -> BCHW
    
            # Convert to float32 on device
            t = t.to(self.device, dtype=torch.float32)
    
            # Bring into [0,1] if needed:
            # - If original looked like uint8, scale by 1/255.
            # - If float but clearly 0..255-ish, scale as well.
            # NOTE: tiny reduction cost here is acceptable and avoids double-normalization bugs.
            # We check range on a small subsample to stay lightweight.
            with torch.no_grad():
                # take a small sample to estimate range
                sample = t.flatten()
                sample = sample[:: max(1, sample.numel() // 8192)]  # ~8k elems
                vmin = torch.min(sample)
                vmax = torch.max(sample)
            if t.dtype == torch.float32:
                # Heuristic: if range suggests 0..255, scale to 0..1
                if vmax > 1.5 or vmin < -0.001:
                    t = torch.clamp(t, 0.0, 255.0) / 255.0
            # (If it came in as uint8 we already casted to float, but range check above catches it too)
            return t
    
        for k, v in obs_frame.items():
            if v is None:
                continue
    
            # Strings (e.g., 'task')
            if isinstance(v, str):
                batch[k] = v
                continue
    
            # Torch tensors
            if torch.is_tensor(v):
                t = v
                # Try image path first; if _to_bchw_float01 decides it's not image-like, we fallback to vector handling
                t_img = _to_bchw_float01(t)
                if t_img.ndim == 4 and t_img.shape[1] in (1, 3):
                    batch[k] = t_img
                    continue
                # Vector / generic: make (1, D)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                batch[k] = t.to(self.device, dtype=torch.float32)
                continue
    
            # Numpy arrays
            if isinstance(v, np.ndarray):
                t = torch.from_numpy(v)
                t_img = _to_bchw_float01(t)
                if torch.is_tensor(t_img) and t_img.ndim == 4 and t_img.shape[1] in (1, 3):
                    batch[k] = t_img
                    continue
                # Vector / generic: make (1, D)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                batch[k] = t.to(self.device, dtype=torch.float32)
                continue
    
            # Anything else: ignore quietly
            # (You could log a warning here if you expect only tensors/arrays/strings.)
    
        # Ensure 'task' exists (empty ok)
        batch['task'] = str(obs_frame.get('task', ''))
        return batch


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False


def main():
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


if __name__ == '__main__':
    main()
