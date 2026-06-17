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
from lerobot.async_inference.helpers import (
    TimedAction,
    TimedObservation,
    get_logger,
)
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.transport import (
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

    # Registered policy types whose predict_action_chunk consumes server-built
    # (B, n_obs_steps, ...) observations stacked from the client's raw obs
    # history. Everything else — ACT, plain diffusion, and the VLAs
    # (pi0/pi05/pi0fast/smolvla, which also declare rtc_config but self-prepare
    # from the latest per-step observation) — takes the plain upstream path.
    STACKED_HISTORY_POLICY_TYPES = frozenset({"sns_diffusion"})

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
        """Lazily initialize the ActionQueue from the policy's RTCConfig.

        Only invoked on the RTC-enabled path (see ``_predict_stacked_action_chunk``),
        so it always builds an enabled queue. Non-RTC policies never create a
        queue, so it cannot grow unbounded in append mode.
        """
        if self._action_queue is not None:
            return
        rtc_cfg = getattr(getattr(self.policy, "config", None), "rtc_config", None)
        self._action_queue = ActionQueue(rtc_cfg)
        self.logger.info(
            f"ActionQueue initialized (RTC enabled, "
            f"execution_horizon={rtc_cfg.execution_horizon}, "
            f"max_guidance_weight={rtc_cfg.max_guidance_weight})"
        )

    def _rtc_enabled(self) -> bool:
        """Whether the loaded policy has RTC guidance enabled."""
        rtc_cfg = getattr(getattr(self.policy, "config", None), "rtc_config", None)
        return isinstance(rtc_cfg, RTCConfig) and rtc_cfg.enabled

    def _needs_server_history(self) -> bool:
        """Whether the loaded policy consumes server-stacked
        ``(B, n_obs_steps, ...)`` observations built from the client's raw obs
        history.

        Keyed on the registered policy ``type`` (see
        ``STACKED_HISTORY_POLICY_TYPES``), NOT on the presence of an
        ``rtc_config`` field: the VLA configs (pi0/pi05/pi0fast/smolvla) also
        declare ``rtc_config`` but expect the standard per-step observation
        (per-camera image keys + ``task``), so they must take the upstream path
        like ACT. Only sns_diffusion is built for the server-side stacking
        protocol.

        A more explicit per-policy capability flag could replace this set, but
        that would require a change in each such policy package.
        """
        cfg = getattr(self.policy, "config", None)
        return getattr(cfg, "type", None) in self.STACKED_HISTORY_POLICY_TYPES

    def _predict_action_chunk(
        self, observation_t: TimedObservation | TimedObservationWithHistory
    ) -> list[TimedAction]:
        """Dispatch by policy capability.

        Vanilla policies (ACT, upstream diffusion, smolvla, pi0, ...) take the
        unmodified upstream path: per-camera image keys preserved,
        ``observation.state`` as ``(B, D)``, no temporal dim, no ActionQueue.
        The sns_diffusion family takes the stacked-history path below, which
        layers RTC guidance on top only when the policy has it enabled.
        """
        if not self._needs_server_history():
            return super()._predict_action_chunk(observation_t)
        return self._predict_stacked_action_chunk(observation_t)

    def _predict_stacked_action_chunk(
        self, observation_t: TimedObservation | TimedObservationWithHistory
    ) -> list[TimedAction]:
        """Inference for policies that consume server-built
        ``(B, n_obs_steps, ...)`` observations (the sns_diffusion family).

        Pipeline:
        1. Build stacked observations from the client's raw obs history.
        2. (RTC only) build kwargs: prev_chunk_left_over, inference_delay.
        3. Run policy inference.
        4. Apply the postprocessor.
        5. (RTC only) merge original + postprocessed actions into the ActionQueue.
        6. Convert to a TimedAction list.
        """
        rtc_enabled = self._rtc_enabled()
        if rtc_enabled:
            self._ensure_action_queue()

        # 1. Build stacked observations from client history. The client attaches
        # a rolling window of raw obs captured at control rate; stacking them
        # here bypasses the policy's internal deque (which would otherwise carry
        # ~1 s-spaced inter-chunk obs, out of training distribution).
        start_prepare = time.perf_counter()
        observations = self._build_stacked_observations(observation_t)
        self.last_processed_obs: TimedObservation = observation_t
        prepare_time = time.perf_counter() - start_prepare

        # 2. RTC kwargs (only when enabled).
        if rtc_enabled:
            rtc_kwargs, action_index_before = self._build_rtc_kwargs()
        else:
            rtc_kwargs, action_index_before = {}, None

        # 3. Inference.
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk_with_kwargs(observations, **rtc_kwargs)
        inference_time = time.perf_counter() - start_inference
        self._last_inference_time = inference_time
        self.logger.info(
            f"Inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        # Original (model-space) actions are only needed to feed the ActionQueue.
        original_actions = action_tensor.squeeze(0).clone() if rtc_enabled else None

        # 4. Apply postprocessor (per-step; matches upstream).
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape
        processed_actions = [
            self.postprocessor(action_tensor[:, i, :]) for i in range(chunk_size)
        ]
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        # 5. Merge into the ActionQueue (RTC only).
        if rtc_enabled:
            self._merge_into_action_queue(
                original_actions=original_actions,
                processed_actions=action_tensor.clone(),
                inference_delay=rtc_kwargs.get("inference_delay", 0),
                action_index_before=action_index_before,
            )
            # Record when this chunk was produced so the next cycle can estimate
            # how many actions the client consumed in the meantime.
            self._last_chunk_sent_at = time.perf_counter()

        action_tensor = action_tensor.detach().cpu()

        # 6. Convert to TimedAction list.
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

    def _build_rtc_kwargs(self) -> tuple[dict, int | None]:
        """Build RTC kwargs and capture the pre-inference action index.

        Advances the queue for estimated client consumption, then reads the
        leftover prefix and computes ``inference_delay`` from the last measured
        inference latency. Assumes RTC is enabled and the queue exists.
        """
        # In eval_with_real_robot.py the robot thread calls action_queue.get()
        # (advancing last_index). Here the client is a separate process, so
        # estimate how many actions it consumed since the last chunk was sent.
        self._simulate_client_consumption()

        prev_chunk_left_over = self._action_queue.get_left_over()
        inference_delay = (
            math.ceil(self._last_inference_time / self.config.environment_dt)
            if self._last_inference_time > 0
            else 0
        )
        action_index_before = self._action_queue.get_action_index()

        self.logger.debug(
            f"RTC kwargs: inference_delay={inference_delay}, "
            f"prev_chunk_left_over="
            f"{'None' if prev_chunk_left_over is None else prev_chunk_left_over.shape}, "
            f"action_index_before={action_index_before}"
        )
        return (
            {
                "prev_chunk_left_over": prev_chunk_left_over,
                "inference_delay": inference_delay,
            },
            action_index_before,
        )

    def _merge_into_action_queue(
        self,
        *,
        original_actions: Tensor,
        processed_actions: Tensor,
        inference_delay: int,
        action_index_before: int | None,
    ) -> None:
        """Advance the queue for the chunk just executed, then merge the new
        chunk. Assumes RTC is enabled and the queue exists.
        """
        # Upstream eval_with_real_robot.py has a robot thread popping actions in
        # parallel with inference, advancing last_index by ~real_delay. The
        # client is remote here, so simulate that consumption explicitly before
        # merging — otherwise ActionQueue._check_and_resolve_delays warns with
        # "indexes_diff=0, real_delay=N".
        if inference_delay > 0:
            steps_to_consume = min(inference_delay, self._action_queue.qsize())
            for _ in range(steps_to_consume):
                self._action_queue.get()
        self._action_queue.merge(
            original_actions=original_actions,
            processed_actions=processed_actions,
            real_delay=inference_delay,
            action_index_before_inference=action_index_before,
        )
        self.logger.debug(f"ActionQueue merged: qsize={self._action_queue.qsize()}")

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

        # State-only policies declare no image_features. The client still
        # publishes camera frames because the contract has them, and upstream
        # ``prepare_raw_observation`` would then KeyError trying to look those
        # keys up in the empty ``policy_image_features``. Strip image entries
        # from lerobot_features for the conversion in that case so only state
        # features round-trip.
        lerobot_features = self.lerobot_features
        if not image_features:
            lerobot_features = {
                k: v
                for k, v in self.lerobot_features.items()
                if not k.startswith(OBS_IMAGES)
            }

        history = getattr(observation_t, "history", None) or [
            observation_t.get_observation()
        ]
        recent = list(history[-n_obs_steps:])
        if len(recent) < n_obs_steps:
            recent = [recent[0]] * (n_obs_steps - len(recent)) + recent

        per_step: list[dict[str, Tensor]] = []
        for raw_obs in recent:
            obs_i = raw_observation_to_observation(
                raw_obs, lerobot_features, self.policy_image_features
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

        # Spacing / freshness diagnostics. Each client tick stamps
        # raw_obs["_capture_time"] = time.time(); missing entries (e.g. legacy
        # clients) show up as NaN.
        capture_times = [raw.get("_capture_time", float("nan")) for raw in recent]
        deltas_ms = [
            (capture_times[i] - capture_times[i - 1]) * 1000.0
            for i in range(1, len(capture_times))
        ]
        latest_age_ms = (
            (time.time() - capture_times[-1]) * 1000.0
            if capture_times and capture_times[-1] == capture_times[-1]  # non-NaN
            else float("nan")
        )

        self.logger.debug(
            f"obs history: hist_len_sent={len(history)}, "
            f"n_obs_steps={n_obs_steps}, used={len(recent)}, "
            f"stacked_keys={sorted(stacked.keys())}, "
            f"deltas_ms={[round(d, 2) for d in deltas_ms]}, "
            f"newest_age_ms={latest_age_ms:.1f}"
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
