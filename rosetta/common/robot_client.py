"""Patched LeRobot robot client for Rosetta.

Monkey-patches ``RobotClient`` to:
* Add an ``_observation_pending`` event that prevents sending observations
  back-to-back while waiting for actions.
* Rewrite ``_aggregate_action_queues`` to hold the lock for the entire merge and
  filter by ``latest_action`` correctly.
* Adjust ``_ready_to_send_observation`` to respect ``_observation_pending``.
* Adjust ``control_loop_observation`` must-go logic for threshold-based refill.
"""

import collections
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from queue import Queue
from typing import Any

import grpc
import torch

from lerobot.async_inference.robot_client import RobotClient
from lerobot.async_inference.helpers import (
    RawObservation,
    TimedAction,
)
from lerobot.robots import openarm_follower  # noqa: F401  — register plugin
from lerobot.transport import (
    services_pb2,  # type: ignore
)

from rosetta.common.obs_history import TimedObservationWithHistory

# Rolling obs-history window length on the client. Must be >= any deployed
# policy's n_obs_steps. The server trims to policy.config.n_obs_steps.
_OBS_HISTORY_MAXLEN = 2

# ---------------------------------------------------------------------------
# 1. Patch RobotClient.__init__ — add _observation_pending event
# ---------------------------------------------------------------------------
_original_init = RobotClient.__init__


def _patched_init(self, config):
    _original_init(self, config)
    # Prevent sending observations back-to-back while waiting for actions
    self._observation_pending = threading.Event()
    # Rolling window of raw observations captured at control rate (30 Hz).
    # Sent along with each observation so the server can rebuild a 33 ms-spaced
    # n_obs_steps history instead of relying on its stale inter-chunk deque.
    self._obs_history: collections.deque[RawObservation] = collections.deque(
        maxlen=_OBS_HISTORY_MAXLEN
    )


RobotClient.__init__ = _patched_init

# ---------------------------------------------------------------------------
# 2. Patch _aggregate_action_queues
# ---------------------------------------------------------------------------


def _patched_aggregate_action_queues(
    self,
    incoming_actions: list[TimedAction],
    aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
):
    """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
    if aggregate_fn is None:

        def aggregate_fn(x1, x2):
            return x2

    with self.latest_action_lock:
        latest_action = self.latest_action

    # Hold action_queue_lock for the entire merge to prevent race with control_loop_action
    with self.action_queue_lock:
        # Build a dict of existing queue actions keyed by timestep (only future ones)
        current_action_queue = {
            action.get_timestep(): action
            for action in self.action_queue.queue
            if action.get_timestep() > latest_action
        }

        # Filter incoming actions to only future timesteps
        incoming_by_timestep = {
            action.get_timestep(): action
            for action in incoming_actions
            if action.get_timestep() > latest_action
        }

        # Start with existing queue items
        merged: dict[int, TimedAction] = dict(current_action_queue)

        # Merge or add incoming actions
        for ts, new_action in incoming_by_timestep.items():
            if ts in merged:
                merged[ts] = TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=ts,
                    action=aggregate_fn(
                        merged[ts].get_action(), new_action.get_action()
                    ),
                )
            else:
                merged[ts] = new_action

        # Rebuild queue in timestep order
        future_action_queue = Queue()
        for ts in sorted(merged.keys()):
            future_action_queue.put(merged[ts])

        self.action_queue = future_action_queue

        self.logger.debug(
            f"Queue after merge: {future_action_queue.qsize()} actions | "
            f"latest_action={latest_action} | "
            f"existing={len(current_action_queue)} | "
            f"incoming_valid={len(incoming_by_timestep)} | "
            f"timesteps={sorted(merged.keys())[0] if merged else 'empty'}:"
            f"{sorted(merged.keys())[-1] if merged else 'empty'}"
        )


RobotClient._aggregate_action_queues = _patched_aggregate_action_queues

# ---------------------------------------------------------------------------
# 2. Patch receive_actions
# ---------------------------------------------------------------------------


def _patched_receive_actions(self, verbose: bool = False):
    """Receive actions from the policy server"""
    # Wait at barrier for synchronized start
    self.start_barrier.wait()
    self.logger.info("Action receiving thread starting")

    while self.running:
        try:
            actions_chunk = self.stub.GetActions(services_pb2.Empty())
            if len(actions_chunk.data) == 0:
                continue

            receive_time = time.time()

            # Deserialize bytes back into list[TimedAction]
            deserialize_start = time.perf_counter()
            timed_actions = pickle.loads(actions_chunk.data)  # nosec
            deserialize_time = time.perf_counter() - deserialize_start

            # Log device type of received actions
            if len(timed_actions) > 0:
                received_device = timed_actions[0].get_action().device.type
                self.logger.debug(f"Received actions on device: {received_device}")

            # Move actions to client_device (e.g., for downstream planners that need GPU)
            client_device = self.config.client_device
            if client_device != "cpu":
                for timed_action in timed_actions:
                    if timed_action.get_action().device.type != client_device:
                        timed_action.action = timed_action.get_action().to(
                            client_device
                        )
                self.logger.debug(f"Converted actions to device: {client_device}")
            else:
                self.logger.debug(f"Actions kept on device: {client_device}")

            self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

            # Calculate network latency if we have matching observations
            if len(timed_actions) > 0 and verbose:
                with self.latest_action_lock:
                    latest_action = self.latest_action

                self.logger.debug(f"Current latest action: {latest_action}")

                # Get queue state before changes
                old_size, old_timesteps = self._inspect_action_queue()
                if not old_timesteps:
                    old_timesteps = [latest_action]  # queue was empty

                # Log incoming actions
                incoming_timesteps = [a.get_timestep() for a in timed_actions]

                first_action_timestep = timed_actions[0].get_timestep()
                server_to_client_latency = (
                    receive_time - timed_actions[0].get_timestamp()
                ) * 1000

                self.logger.info(
                    f"Received action chunk for step #{first_action_timestep} | "
                    f"Latest action: #{latest_action} | "
                    f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                    f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                    f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                )

            # Update action queue
            start_time = time.perf_counter()
            self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
            queue_update_time = time.perf_counter() - start_time

            self._observation_pending.clear()  # allow sending a new observation
            self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

            if verbose:
                # Get queue state after changes
                new_size, new_timesteps = self._inspect_action_queue()

                with self.latest_action_lock:
                    latest_action = self.latest_action

                self.logger.info(
                    f"Latest action: {latest_action} | "
                    f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                    f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                    f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                )
                self.logger.debug(
                    f"Queue update complete ({queue_update_time:.6f}s) | "
                    f"Before: {old_size} items | "
                    f"After: {new_size} items | "
                )

        except grpc.RpcError as e:
            self.logger.error(f"Error receiving actions: {e}")


RobotClient.receive_actions = _patched_receive_actions

# ---------------------------------------------------------------------------
# 3. Patch control_loop_action
# ---------------------------------------------------------------------------


def _patched_control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
    """Reading and performing actions in local queue"""

    # Lock only for queue operations
    get_start = time.perf_counter()
    with self.action_queue_lock:
        self.action_queue_size.append(self.action_queue.qsize())
        # Get action from queue
        timed_action = self.action_queue.get_nowait()
    get_end = time.perf_counter() - get_start

    _performed_action = self.robot.send_action(
        self._action_tensor_to_action_dict(timed_action.get_action())
    )
    with self.latest_action_lock:
        self.latest_action = timed_action.get_timestep()

    if verbose:
        with self.action_queue_lock:
            current_queue_size = self.action_queue.qsize()

        self.logger.debug(
            f"Ts={timed_action.get_timestamp()} | "
            f"Action #{timed_action.get_timestep()} performed | "
            f"Queue size: {current_queue_size}"
        )

        self.logger.debug(
            f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
        )

    return _performed_action


RobotClient.control_loop_action = _patched_control_loop_action

# ---------------------------------------------------------------------------
# 4. Patch _ready_to_send_observation
# ---------------------------------------------------------------------------


def _patched_ready_to_send_observation(self):
    """Flags when the client is ready to send an observation"""
    if self._observation_pending.is_set():
        return False

    with self.action_queue_lock:
        queue_ratio = (
            self.action_queue.qsize() / self.action_chunk_size
            if self.action_chunk_size > 0
            else -1.0
        )

    return queue_ratio <= self._chunk_size_threshold


RobotClient._ready_to_send_observation = _patched_ready_to_send_observation

# ---------------------------------------------------------------------------
# 5. Patch control_loop_observation
# ---------------------------------------------------------------------------


def _patched_control_loop_observation(
    self, task: str, verbose: bool = False
) -> RawObservation:
    try:
        # Get serialized observation bytes from the function
        start_time = time.perf_counter()

        # Reuse the obs captured this tick by control_loop so the "current"
        # obs and the history tail are the same snapshot.
        assert self._obs_history, (
            "_obs_history is empty; control_loop_observation was called "
            "before control_loop captured this tick."
        )
        raw_observation = self._obs_history[-1]

        with self.latest_action_lock:
            latest_action = self.latest_action

        observation = TimedObservationWithHistory(
            timestamp=time.time(),
            observation=raw_observation,
            timestep=max(latest_action, 0),
            history=list(self._obs_history),  # oldest first; current obs last
        )

        obs_capture_time = time.perf_counter() - start_time

        # Observation must go through processing when:
        # 1. Queue is empty and must_go event is set (recovery path), OR
        # 2. Queue is at or below the refill threshold (we actively need actions)
        with self.action_queue_lock:
            queue_empty = self.action_queue.empty()
            current_queue_size = self.action_queue.qsize()
            queue_ratio = (
                current_queue_size / self.action_chunk_size
                if self.action_chunk_size > 0
                else -1.0
            )
            at_threshold = (not queue_empty) and (
                queue_ratio <= self._chunk_size_threshold
            )
            observation.must_go = (
                self.must_go.is_set() and queue_empty
            ) or at_threshold

        self._observation_pending.set()  # block further observations until actions arrive
        try:
            _ = self.send_observation(observation)
        except Exception:
            self._observation_pending.clear()
            raise

        self.logger.debug(
            f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})"
        )
        if observation.must_go:
            # must-go event will be set again after receiving actions
            self.must_go.clear()

        if verbose:
            # Calculate comprehensive FPS metrics
            fps_metrics = self.fps_tracker.calculate_fps_metrics(
                observation.get_timestamp()
            )

            self.logger.info(
                f"Obs #{observation.get_timestep()} | "
                f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                f"Target: {fps_metrics['target_fps']:.2f}"
            )

            self.logger.debug(
                f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
            )

        return raw_observation

    except Exception:
        self.logger.exception("Error in observation sender")
        raise


RobotClient.control_loop_observation = _patched_control_loop_observation

# ---------------------------------------------------------------------------
# 6. Patch control_loop
# ---------------------------------------------------------------------------


def _patched_control_loop(self, task: str, verbose: bool = False):
    """Combined function for executing actions and streaming observations"""
    # Wait at barrier for synchronized start
    self.start_barrier.wait()
    self.logger.info("Control loop thread starting")

    _performed_action = None
    _captured_observation = None

    while self.running:
        control_loop_start = time.perf_counter()

        # Capture one observation per tick into the rolling history buffer.
        # This preserves the 33 ms spacing that the diffusion policy was
        # trained with. Cheap: underlying ROS subscriptions already run at
        # 30 Hz, get_observation() just assembles the latest cached messages.
        tick_raw_obs = self.robot.get_observation()
        tick_raw_obs["task"] = task
        self._obs_history.append(tick_raw_obs)

        # Control loop: (1) Performing actions, when available
        if self.actions_available():
            _performed_action = self.control_loop_action(verbose)

        # Control loop: (2) Streaming observations to the remote policy server
        if self._ready_to_send_observation():
            _captured_observation = self.control_loop_observation(task, verbose)

        self.logger.debug(
            f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}"
        )
        # Dynamically adjust sleep time to maintain the desired control frequency
        time.sleep(
            max(
                0,
                self.config.environment_dt - (time.perf_counter() - control_loop_start),
            )
        )

    return _captured_observation, _performed_action


RobotClient.control_loop = _patched_control_loop
