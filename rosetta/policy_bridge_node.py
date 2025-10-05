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
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rosidl_runtime_py.utilities import get_message

import torch
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from rosetta_interfaces.action import RecordEpisode

# ---- shared core ------------------------------------------------------------
from rosetta.common.contract_utils import (
    load_contract, iter_specs, SpecView, feature_from_spec,
    zero_pad as feature_zero_pad,
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
    Contract-true live inference with local async, using LeRobot's official
    pre/post processors (adds language tokens, normalizes, de-normalizes, etc.).
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
        self.declare_parameter('image_qos_depth', 10)
        self.declare_parameter('use_header_timestamps', False)

        # Executor idle behavior
        self.declare_parameter('publish_when_idle', True)
        self.declare_parameter('idle_behavior', 'hold')        # 'hold' | 'zero'

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

        # Device selection: prefer CUDA; warn if requested but unavailable
        requested = (self.get_parameter('policy_device').get_parameter_value().string_value or 'cpu').lower()
        if requested == 'cuda':
            if torch.cuda.is_available():
                device_str = 'cuda'
            else:
                self.get_logger().warning("policy_device='cuda' requested but CUDA is not available; falling back to CPU.")
                device_str = 'cpu'
        elif requested in ('mps', 'metal'):
            if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                device_str = 'mps'
            else:
                self.get_logger().warning("policy_device='mps' requested but MPS is not available; falling back to CPU.")
                device_str = 'cpu'
        else:
            # If user didn't request cuda, still prefer it if available
            device_str = 'cuda' if torch.cuda.is_available() else requested
            if device_str == 'cuda' and requested != 'cuda':
                self.get_logger().info("Using CUDA because it is available (override by setting policy_device explicitly).")
            if device_str == 'cpu' and requested != 'cpu' and requested not in ('mps', 'metal'):
                self.get_logger().warning(f"Unknown policy_device='{requested}', using CPU.")

        self.device = torch.device(device_str)

        PolicyCls = get_policy_class(cfg_type)
        self.policy = PolicyCls.from_pretrained(policy_path)
        self.policy.to(self.device)
        self.policy.eval()

        # --------- OFFICIAL pre/post processors (adds lang tokens & de-normalizes outputs) -----
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
            self._obs_zero[k] = feature_zero_pad(meta)

        # ---------------- Subscriptions ----------------
        self._cbg = ReentrantCallbackGroup()
        self._subs: Dict[str, _SubState] = {}
        self._ros_sub_handles = []

        qos = QoSProfile(depth=int(self.get_parameter('image_qos_depth').value or 10))
        for s in self._obs_specs:
            msg_cls = get_message(s.ros_type)
            sub = self.create_subscription(
                msg_cls, s.topic, lambda m, sv=s: self._obs_cb(m, sv), qos, callback_group=self._cbg
            )
            self._ros_sub_handles.append(sub)

            tol_ns = int(max(0, s.asof_tol_ms)) * 1_000_000
            buf = StreamBuffer(policy=s.resample_policy, step_ns=self.step_ns, tol_ns=tol_ns)
            self._subs[s.key] = _SubState(spec=s, msg_type=msg_cls, buf=buf, stamp_src=s.stamp_src)

        self.get_logger().info(f'Subscribed to {len(self._subs)} observation streams.')

        # ---------------- Publisher ----------------
        self._act_pub = self.create_publisher(get_message(self._act_spec.ros_type), self._act_spec.topic, 10)

        # ---------------- Action server ----------------
        self._running = False
        self._running_event = threading.Event()
        self._prompt = ''
        self._goal_handle = None
        self._action_server = ActionServer(
            self,
            RecordEpisode,
            'run_policy',
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._cbg,
        )

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
        return GoalResponse.REJECT if self._running else GoalResponse.ACCEPT

    def _cancel_cb(self, _goal) -> CancelResponse:
        self._running = False
        self._running_event.clear()
        return CancelResponse.ACCEPT

    def _execute_cb(self, goal_handle) -> RecordEpisode.Result:
        self._running = True
        self._running_event.set()
        self._prompt = getattr(goal_handle.request, 'prompt', '') or ''
        # Reset policy state (if the policy supports it)
        if hasattr(self.policy, "reset"):
            try:
                self.policy.reset()
            except Exception as e:
                self.get_logger().warning(f'policy.reset() failed: {e!r}')
        self.get_logger().info(f"run_policy: started (task='{self._prompt}')")

        try:
            while rclpy.ok() and self._running:
                rclpy.spin_once(self, timeout_sec=0.1)
        finally:
            self._running = False
            self._running_event.clear()
            self.get_logger().info('run_policy: stopped')

        goal_handle.succeed()
        return RecordEpisode.Result(success=True, message='Policy run ended')

    # ---------------- Sub callback ----------------
    def _obs_cb(self, msg, spec: SpecView):
        use_header = bool(self.get_parameter('use_header_timestamps').value)
        if use_header and spec.stamp_src == 'header':
            ts = stamp_from_header_ns(msg)
            ts_ns = int(ts) if ts is not None else now_ns()
        else:
            ts_ns = now_ns()

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
        obs_frame['task'] = self._prompt  # free-form text; preprocessor will tokenize
        return obs_frame

    # ---------------- Producer: compute chunks asynchronously ----------------
    def _producer_loop(self):
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
                batch = self._prepare(obs_frame)             # shapes only
                batch = self.preprocessor(batch)             # official preprocessing (tokens, norm, etc.)

                if bool(self.get_parameter('debug_dump_inputs').value):
                    self.get_logger().info(f"Policy input keys: {list(batch.keys())}")
                    for k_, v_ in batch.items():
                        if isinstance(v_, torch.Tensor):
                            self.get_logger().info(f"  {k_}: {tuple(v_.shape)} {v_.dtype} {v_.device}")
                        else:
                            self.get_logger().info(f"  {k_}: {type(v_)} -> {v_}")

                with torch.inference_mode():
                    if use_chunks:
                        # Predict (1,K,A) or (K,A)
                        chunk = self.policy.predict_action_chunk(batch)
                        if chunk.ndim == 3:
                            chunk = chunk.squeeze(0)              # -> (K, A)

                        # Apply official POSTPROCESSOR to de-normalize/map to robot space
                        # Expecting shape (N, A)
                        if not torch.is_tensor(chunk):
                            chunk = torch.as_tensor(chunk, device=self.device)
                        chunk = self.postprocessor(chunk)         # still (K, A)

                        # To CPU numpy list
                        k_eff = min(k, chunk.shape[0]) if k > 0 else chunk.shape[0]
                        actions = [
                            np.asarray(chunk[i].detach().cpu().numpy()).ravel()
                            for i in range(k_eff)
                        ]
                    else:
                        a = self.policy.select_action(batch)      # (1, A)
                        a = self.postprocessor(a)                 # (1, A) in robot units
                        actions = [np.asarray(a[0].detach().cpu().numpy()).ravel()]
            except Exception as e:
                import traceback
                self.get_logger().error(f'producer inference error: {e}')
                self.get_logger().error(traceback.format_exc())
                actions = []

            if actions:
                with self._queue_lock:
                    self._queue.extend(actions)

    # ---------------- Executor: publish at fixed FPS ----------------
    def _executor_loop(self):
        next_t = now_ns()
        pub_idle = bool(self.get_parameter('publish_when_idle').value)
        idle_behavior = str(self.get_parameter('idle_behavior').value or 'hold').lower()
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

                if act_vec is None and pub_idle:
                    if idle_behavior == 'hold' and self._last_action is not None:
                        act_vec = self._last_action.copy()
                    else:
                        n = max(len(self._act_spec.names or []), 2)
                        act_vec = np.zeros((n,), dtype=np.float32)

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

            next_t += self.step_ns

    # ---------------- Minimal local preprocessor (shapes only) ---------------
    def _prepare(self, obs_frame: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw obs to tensors with correct SHAPES for the official
        preprocessor/SmolVLA:

          - images accepted in HWC/BGR/RGB as np.ndarray or torch.Tensor
            → (1, C, H, W) float32 (no /255 here; let LeRobot do it)
          - vectors 1D → (1, D) float32
          - 'task' stays as str (preprocessor will create lang tokens)
        """
        batch: Dict[str, Any] = {}
        for k, v in obs_frame.items():
            if v is None:
                continue

            # Torch tensors
            if torch.is_tensor(v):
                t = v
                if t.ndim == 3 and t.shape[-1] in (1, 3):  # HWC
                    t = t.permute(2, 0, 1)                 # -> CHW
                if t.ndim == 3 and t.shape[0] in (1, 3):   # CHW
                    t = t.unsqueeze(0)                     # -> BCHW
                if t.ndim == 1:                            # D -> (1, D)
                    t = t.unsqueeze(0)
                batch[k] = t.to(self.device, dtype=torch.float32)
                continue

            # Numpy arrays
            if isinstance(v, np.ndarray):
                if v.ndim == 3 and v.shape[-1] in (1, 3):   # HWC
                    t = torch.from_numpy(v).permute(2, 0, 1).unsqueeze(0)
                    batch[k] = t.to(self.device, dtype=torch.float32)  # -> BCHW
                elif v.ndim == 3 and v.shape[0] in (1, 3): # CHW
                    t = torch.from_numpy(v).unsqueeze(0)
                    batch[k] = t.to(self.device, dtype=torch.float32)  # -> BCHW
                elif v.ndim == 1:
                    t = torch.from_numpy(v).reshape(1, -1)
                    batch[k] = t.to(self.device, dtype=torch.float32)  # -> (1, D)
                else:
                    # Fallback: flatten anything else to (1, -1)
                    t = torch.from_numpy(v).reshape(1, -1)
                    batch[k] = t.to(self.device, dtype=torch.float32)
                continue

            # Strings (task prompt)
            if isinstance(v, str):
                batch[k] = v

        # Ensure 'task' is present (empty ok)
        batch['task'] = str(obs_frame.get('task', ''))
        return batch


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
