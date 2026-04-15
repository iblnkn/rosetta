"""RTC-aware policy server for async inference.

Subclasses LeRobot's ``PolicyServer`` to integrate ``ActionQueue`` on the
server side, enabling Real-Time Chunking guidance without modifying
lerobot source code.

The key additions:
1. An ``ActionQueue`` that tracks original (model-space) and processed
   (post-processed) actions, providing ``get_left_over()`` for RTC guidance.
2. ``inference_delay`` computed from measured inference latency.
3. Both are passed as kwargs to ``policy.predict_action_chunk()``.
4. Estimated client-side consumption to keep ``ActionQueue.last_index``
   in sync despite the server and client being in separate processes.

In the reference RTC deployment (``eval_with_real_robot.py``), a single
``ActionQueue`` is shared between the robot thread (which calls ``.get()``
to advance ``last_index``) and the inference thread (which calls
``.get_left_over()``). In the async gRPC pipeline the client and server
are separate processes, so the server must **estimate** how many actions
the client consumed between inference calls based on elapsed wall time
and the configured FPS.

Usage (standalone):
    python -m rosetta.common.rtc_policy_server --host=0.0.0.0 --port=8080
"""

import logging
import math
import time
from dataclasses import asdict
from pprint import pformat

import draccus
import grpc
import torch
from torch import Tensor
from concurrent import futures

# Register SNSDiffusionConfig before anything loads a checkpoint
import lerobot_policy_sns_diffusion  # noqa: F401

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.helpers import TimedAction, TimedObservation, get_logger
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)

from rosetta.common.obs_history import TimedObservationWithHistory

logger = logging.getLogger(__name__)


class RTCPolicyServer(PolicyServer):
    """PolicyServer with server-side ActionQueue for RTC.

    Overrides ``_predict_action_chunk`` to:
    - Retrieve ``prev_chunk_left_over`` from the ``ActionQueue``
    - Compute ``inference_delay`` from measured inference latency
    - Pass both to ``policy.predict_action_chunk()``
    - Merge original + postprocessed actions back into the ``ActionQueue``
    """

    prefix = "rtc_policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        super().__init__(config)
        self._action_queue: ActionQueue | None = None
        self._last_inference_time: float = 0.0
        self._last_chunk_sent_at: float = 0.0  # wall time when last chunk was sent

    def _reset_server(self) -> None:
        """Flush server state when new client connects."""
        super()._reset_server()
        # ActionQueue will be re-created after policy is loaded (we need RTCConfig)
        self._action_queue = None
        self._last_inference_time = 0.0
        self._last_chunk_sent_at = 0.0

    def _ensure_action_queue(self) -> None:
        """Lazily initialize ActionQueue from the policy's RTCConfig."""
        if self._action_queue is not None:
            return

        rtc_config = getattr(self.policy, "config", None)
        rtc_cfg = getattr(rtc_config, "rtc_config", None) if rtc_config else None

        if rtc_cfg is not None and isinstance(rtc_cfg, RTCConfig) and rtc_cfg.enabled:
            self._action_queue = ActionQueue(rtc_cfg)
            self.logger.info(
                f"ActionQueue initialized (RTC enabled, "
                f"execution_horizon={rtc_cfg.execution_horizon}, "
                f"max_guidance_weight={rtc_cfg.max_guidance_weight})"
            )
        else:
            # Create a disabled ActionQueue (non-RTC fallback, uses append mode)
            fallback_cfg = RTCConfig(enabled=False)
            self._action_queue = ActionQueue(fallback_cfg)
            self.logger.info("ActionQueue initialized (RTC disabled, append mode)")

    def _rtc_enabled(self) -> bool:
        """Check if the loaded policy has RTC enabled."""
        rtc_config = getattr(self.policy, "config", None)
        rtc_cfg = getattr(rtc_config, "rtc_config", None) if rtc_config else None
        return rtc_cfg is not None and getattr(rtc_cfg, "enabled", False)

    def _predict_action_chunk(
        self, observation_t: TimedObservation | TimedObservationWithHistory
    ) -> list[TimedAction]:
        """Predict an action chunk with RTC support.

        Pipeline:
        1. Build ``(B, n_obs_steps, ...)`` observations from client-sent obs history
        2. Build RTC kwargs (prev_chunk_left_over, inference_delay)
        3. Run policy inference with RTC kwargs
        4. Merge into ActionQueue (original + postprocessed)
        5. Apply postprocessor
        6. Convert to TimedAction list
        """
        self._ensure_action_queue()

        """1. Build stacked observations from client history.

        The client attaches a rolling window of raw obs captured at control
        rate. We convert + preprocess each, stack per-camera images, then
        stack along dim=1 into a (B, n_obs_steps, ...) tensor per key — the
        shape the diffusion model was trained on. This bypasses the policy's
        internal deque entirely (which would otherwise carry ~1 s-spaced
        inter-chunk obs, out of training distribution).
        """
        start_prepare = time.perf_counter()
        observations = self._build_stacked_observations(observation_t)
        self.last_processed_obs: TimedObservation = observation_t
        prepare_time = time.perf_counter() - start_prepare

        """2. Build RTC kwargs"""
        rtc_kwargs = {}
        action_index_before = None

        if self._rtc_enabled() and self._action_queue is not None:
            # Simulate client-side consumption.
            #
            # In eval_with_real_robot.py the robot thread calls
            # action_queue.get() which advances last_index.  Here the
            # client is in a separate process so we estimate how many
            # actions it consumed since we last sent a chunk.
            self._simulate_client_consumption()

            prev_chunk_left_over = self._action_queue.get_left_over()
            rtc_kwargs["prev_chunk_left_over"] = prev_chunk_left_over

            # Compute inference_delay from the last measured inference time
            inference_delay = (
                math.ceil(self._last_inference_time / self.config.environment_dt)
                if self._last_inference_time > 0
                else 0
            )
            rtc_kwargs["inference_delay"] = inference_delay

            # Optionally pass execution_horizon (falls back to config default in RTCProcessor)
            action_index_before = self._action_queue.get_action_index()

            self.logger.debug(
                f"RTC kwargs: inference_delay={inference_delay}, "
                f"prev_chunk_left_over={'None' if prev_chunk_left_over is None else prev_chunk_left_over.shape}, "
                f"action_index_before={action_index_before}"
            )

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk_with_kwargs(observations, **rtc_kwargs)
        inference_time = time.perf_counter() - start_inference
        self._last_inference_time = inference_time
        self.logger.info(
            f"Inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        # Keep original actions (model-space) before postprocessing for ActionQueue
        original_actions = action_tensor.squeeze(0).clone()

        """4. Apply postprocessor"""
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        processed_actions = []
        for i in range(chunk_size):
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        # Postprocessed actions for ActionQueue
        processed_actions_2d = action_tensor.clone()

        """5. Merge into ActionQueue"""
        if self._action_queue is not None:
            inference_delay = rtc_kwargs.get("inference_delay", 0)
            # Upstream `eval_with_real_robot.py` has a robot thread popping
            # actions in parallel with inference, which naturally advances
            # last_index by ~real_delay. Here the client is remote, so we
            # simulate that consumption explicitly before merging — otherwise
            # ActionQueue._check_and_resolve_delays warns with
            # "indexes_diff=0, real_delay=N".
            if self._rtc_enabled() and inference_delay > 0:
                steps_to_consume = min(inference_delay, self._action_queue.qsize())
                for _ in range(steps_to_consume):
                    self._action_queue.get()
            self._action_queue.merge(
                original_actions=original_actions,
                processed_actions=processed_actions_2d,
                real_delay=inference_delay,
                action_index_before_inference=action_index_before,
            )
            self.logger.debug(f"ActionQueue merged: qsize={self._action_queue.qsize()}")

        # Record when this chunk was produced so the next cycle can
        # estimate how many actions the client consumed in the meantime.
        self._last_chunk_sent_at = time.perf_counter()

        action_tensor = action_tensor.detach().cpu()

        """6. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(),
            list(action_tensor),
            observation_t.get_timestep(),
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def _get_action_chunk_with_kwargs(
        self, observations: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """Get an action chunk, passing RTC kwargs through to predict_action_chunk."""
        chunk = self.policy.predict_action_chunk(observations, **kwargs)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        return chunk[:, : self.actions_per_chunk, :]

    def _build_stacked_observations(
        self,
        observation_t: TimedObservation | TimedObservationWithHistory,
    ) -> dict[str, Tensor]:
        """Build ``(B, n_obs_steps, ...)`` observations from client-sent history.

        Steps per history entry:
          1. ``raw_observation_to_observation`` (feature selection + tensorize).
          2. ``self.preprocessor`` (normalize, batch-dim, device move).
          3. Stack per-camera images into ``OBS_IMAGES``.

        Then stack all ``n_obs_steps`` entries along ``dim=1`` — this is the
        shape the diffusion model was trained on (33 ms-spaced history). If
        the client sent fewer than ``n_obs_steps`` entries (ramp-up), left-pad
        by repeating the oldest entry.
        """
        from lerobot.async_inference.helpers import raw_observation_to_observation
        from lerobot.utils.constants import OBS_IMAGES

        n_obs_steps = getattr(self.policy.config, "n_obs_steps", 1)
        image_features = getattr(self.policy.config, "image_features", None)

        history = getattr(observation_t, "history", None) or [
            observation_t.get_observation()
        ]
        recent = list(history[-n_obs_steps:])
        if len(recent) < n_obs_steps:
            recent = [recent[0]] * (n_obs_steps - len(recent)) + recent

        per_step: list[dict[str, Tensor]] = []
        for raw_obs in recent:
            obs_i = raw_observation_to_observation(
                raw_obs, self.lerobot_features, self.policy_image_features
            )
            obs_i = self.preprocessor(obs_i)
            if image_features:
                obs_i = dict(obs_i)
                obs_i[OBS_IMAGES] = torch.stack(
                    [obs_i[k] for k in image_features], dim=-4
                )
            per_step.append(obs_i)

        stack_keys = {"observation.state", "observation.environment_state", OBS_IMAGES}
        stacked: dict[str, Tensor] = {
            k: torch.stack([step[k] for step in per_step], dim=1)
            for k in per_step[-1]
            if k in stack_keys
        }

        self.logger.info(
            f"obs history: hist_len_sent={len(history)}, "
            f"n_obs_steps={n_obs_steps}, used={len(recent)}, "
            f"stacked_keys={sorted(stacked.keys())}"
        )
        return stacked

    def _simulate_client_consumption(self) -> None:
        """Advance ``ActionQueue.last_index`` to match estimated client consumption.

        In ``eval_with_real_robot.py`` the robot thread calls
        ``action_queue.get()`` which advances ``last_index`` so that
        ``get_left_over()`` returns only the unconsumed tail.  In the
        async gRPC pipeline the client lives in a separate process and
        the server has no direct signal of how many actions were popped.

        We estimate consumption from wall-clock time::

            consumed ≈ elapsed_since_last_chunk / environment_dt

        and pop that many entries (via ``.get()``) to keep
        ``last_index`` consistent.
        """
        if self._action_queue is None or self._action_queue.empty():
            return
        if self._last_chunk_sent_at <= 0:
            return  # first inference, nothing to simulate

        elapsed = time.perf_counter() - self._last_chunk_sent_at
        estimated_consumed = min(
            int(elapsed / self.config.environment_dt),
            self._action_queue.qsize(),
        )

        for _ in range(estimated_consumed):
            self._action_queue.get()  # advances last_index

        self.logger.debug(
            f"Simulated client consumption: {estimated_consumed} actions "
            f"(elapsed={elapsed:.4f}s, remaining={self._action_queue.qsize()})"
        )


def _patch_supported_policies() -> None:
    """Add 'sns_diffusion' to SUPPORTED_POLICIES if not already present.

    We mutate the list in-place so the check in
    ``PolicyServer.SendPolicyInstructions`` passes.
    """
    if "sns_diffusion" not in SUPPORTED_POLICIES:
        SUPPORTED_POLICIES.append("sns_diffusion")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the RTCPolicyServer."""
    _patch_supported_policies()

    logging.info(pformat(asdict(cfg)))

    policy_server = RTCPolicyServer(cfg)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"RTCPolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
