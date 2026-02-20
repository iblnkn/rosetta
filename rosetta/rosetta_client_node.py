#!/usr/bin/env python
# Copyright 2025 Isaac Blankenau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RosettaClientNode: ROS2 action server wrapping LeRobot's RobotClient.

Usage:
    ros2 launch rosetta rosetta_client_launch.py

    ros2 action send_goal /rosetta_client/run_policy \
        rosetta_interfaces/action/RunPolicy "{prompt: 'pick up cube'}" --feedback
"""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rcl_interfaces.msg import ParameterDescriptor

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from rosetta_interfaces.action import RunPolicy

from lerobot_robot_rosetta import RosettaConfig
from lerobot_robot_rosetta.rosetta import _TopicBridge

from .common.ros2_utils import is_jazzy_or_newer

SERVER_STARTUP_TIMEOUT_SEC = 30.0
SERVER_STARTUP_POLL_SEC = 0.5
SERVER_STOP_TIMEOUT_SEC = 5.0
THREAD_JOIN_TIMEOUT_SEC = 2.0
FEEDBACK_THREAD_JOIN_TIMEOUT_SEC = 1.0


class RosettaClientNode(LifecycleNode):
    """ROS2 Lifecycle Action Server wrapping LeRobot's RobotClient for policy inference."""

    def __init__(self):
        # Initialize with enable_logger_service on Jazzy (not supported in Humble)
        # The logger service allows runtime configuration of log levels via
        # ros2 service call /node_name/set_logger_level ...
        # In Humble, logger services are always enabled by default.
        if is_jazzy_or_newer():
            super().__init__("rosetta_client", enable_logger_service=True)
        else:
            super().__init__("rosetta_client")

        # Parameters with descriptors for introspection (ros2 param describe)
        # Read-only parameters are set once at startup and cannot be changed
        self.declare_parameter(
            "contract_path", "",
            ParameterDescriptor(description="Path to contract YAML file", read_only=True)
        )
        self.declare_parameter(
            "pretrained_name_or_path", "",
            ParameterDescriptor(description="Path or HF repo ID of trained policy", read_only=True)
        )
        self.declare_parameter(
            "server_address", "127.0.0.1:8080",
            ParameterDescriptor(description="Policy server address (host:port)", read_only=True)
        )
        self.declare_parameter(
            "policy_type", "act",
            ParameterDescriptor(description="Policy architecture (act, diffusion, etc.)", read_only=True)
        )
        self.declare_parameter(
            "policy_device", "cuda",
            ParameterDescriptor(description="Device for policy inference (cuda, cpu)", read_only=True)
        )
        self.declare_parameter(
            "actions_per_chunk", 50,
            ParameterDescriptor(description="Number of actions to request per chunk")
        )
        self.declare_parameter(
            "chunk_size_threshold", 0.5,
            ParameterDescriptor(description="Queue threshold ratio to request new chunk (0.0-1.0)")
        )
        self.declare_parameter(
            "aggregate_fn_name", "weighted_average",
            ParameterDescriptor(description="Action aggregation function (weighted_average, etc.)")
        )
        self.declare_parameter(
            "feedback_rate_hz", 2.0,
            ParameterDescriptor(description="Rate for publishing action feedback")
        )
        self.declare_parameter(
            "launch_local_server", True,
            ParameterDescriptor(description="Launch local policy server subprocess", read_only=True)
        )
        self.declare_parameter(
            "obs_similarity_atol", 1.0,
            ParameterDescriptor(description="L2 norm tolerance for observation similarity (-1.0 to disable)")
        )
        self.declare_parameter(
            "is_classifier", False,
            ParameterDescriptor(
                description="Use reward section as action output (for reward classifiers)",
                read_only=True,
            )
        )
        self.declare_parameter(
            "sim_time_multiplier", 1.0,
            ParameterDescriptor(
                description="Multiplier for fps sent to LeRobot (contract_fps * sim_time_multiplier). "
                "Use values < 1.0 for slow sims (e.g., 0.5 for 0.5x speed sim) to maintain wall-time action rate. "
                "Set to 1.0 for real-time or when not using sim time."
            )
        )

        # Initialize state variables (resources created in lifecycle callbacks)
        self._contract_path: str | None = None
        self._pretrained: str | None = None
        self._server_process: subprocess.Popen | None = None
        self._client: RobotClient | None = None
        self._active_goal = None
        self._action_server: ActionServer | None = None
        self._accepting_goals = False

        # Topic bridge: manages observation subscriptions + action publishers
        self._rosetta_config: RosettaConfig | None = None
        self._bridge: _TopicBridge | None = None

        self.get_logger().info("Node created (unconfigured)")

    # -------------------- Lifecycle callbacks --------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Validate parameters and create action server."""
        self._contract_path = self.get_parameter("contract_path").value
        self._pretrained = self.get_parameter("pretrained_name_or_path").value

        if not self._contract_path:
            self.get_logger().error("contract_path parameter required")
            return TransitionCallbackReturn.FAILURE
        if not self._pretrained:
            self.get_logger().error("pretrained_name_or_path parameter required")
            return TransitionCallbackReturn.FAILURE
        # Distinguish local paths from HF repo IDs (e.g. "user/model").
        # Local paths start with /, ./, or ../
        _looks_local = self._pretrained.startswith(('/', './', '../'))
        if _looks_local and not os.path.isdir(self._pretrained):
            self.get_logger().error(
                f"Local model path does not exist: {self._pretrained}. "
                "Use a HuggingFace repo ID (e.g. 'user/model') or a valid local directory."
            )
            return TransitionCallbackReturn.FAILURE

        # Create topic bridge (observation subscriptions + lifecycle action publishers)
        # Bridge uses contract fps (unscaled) for ROS2 timing (watchdog, etc.)
        # ROS2 clock respects use_sim_time and /clock, so watchdog operates in sim time
        is_classifier = self.get_parameter("is_classifier").value
        self._rosetta_config = RosettaConfig(
            id="rosetta", config_path=self._contract_path,
            is_classifier=is_classifier,
        )
        self._bridge = _TopicBridge(self._rosetta_config)
        self._bridge.setup(self)

        # Create action server (can receive goals but rejects when not active)
        self._action_server = ActionServer(
            self,
            RunPolicy,
            "run_policy",
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info(
            f"Configured: contract={self._contract_path}, model={self._pretrained}"
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Start policy server and enable goal acceptance."""
        try:
            if self.get_parameter("launch_local_server").value:
                self._start_policy_server()

            self._accepting_goals = True
            self.get_logger().info("Activated and ready for policy execution")
            return super().on_activate(state)

        except Exception as e:
            self.get_logger().error(f"Activation failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop accepting goals, cancel in-progress execution, stop policy server."""
        self._accepting_goals = False

        # Send safety action while lifecycle publishers are still active
        if self._bridge is not None:
            self._bridge.send_safety_action()

        # Cancel any in-progress goal
        if self._client is not None:
            self.get_logger().info("Cancelling in-progress policy execution...")
            self._client.shutdown_event.set()

        # Wait for goal to complete (with timeout)
        timeout = 5.0
        start = time.time()
        while self._active_goal is not None and (time.time() - start) < timeout:
            time.sleep(0.1)

        if self._active_goal is not None:
            self.get_logger().warning("Goal did not complete within timeout")

        # Stop policy server
        self._stop_policy_server()

        self.get_logger().info("Deactivated")
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release resources and destroy action server."""
        self._stop_policy_server()  # Ensure stopped

        if self._bridge is not None:
            self._bridge.teardown()
            self._bridge = None
        self._rosetta_config = None

        if self._action_server is not None:
            self.destroy_action_server(self._action_server)
            self._action_server = None

        self._contract_path = None
        self._pretrained = None

        self.get_logger().info("Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Final cleanup before destruction."""
        self._accepting_goals = False
        self._stop_policy_server()

        if self._bridge is not None:
            self._bridge.teardown()
            self._bridge = None
        self._rosetta_config = None

        if self._action_server is not None:
            self.destroy_action_server(self._action_server)
            self._action_server = None

        self.get_logger().info("Shutdown complete")
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handle errors by cleaning up resources."""
        self.get_logger().error(f"Error occurred in state: {state.label}")

        try:
            self._accepting_goals = False
            if self._client is not None:
                self._client.shutdown_event.set()
            self._stop_policy_server()
            if self._bridge is not None:
                self._bridge.teardown()
                self._bridge = None
        except Exception as e:
            self.get_logger().error(f"Error during error handling: {e}")

        return TransitionCallbackReturn.SUCCESS

    # -------------------- Policy server management --------------------

    def _start_policy_server(self) -> None:
        """Launch local policy server subprocess."""
        server_address = self.get_parameter("server_address").value
        host, port = server_address.split(":")

        if self.get_parameter("is_classifier").value:
            module = "rosetta.common.classifier_server"
        else:
            module = "lerobot.async_inference.policy_server"

        self.get_logger().info(f"Launching {module} on {host}:{port}...")

        cmd = [
            sys.executable, "-m", module,
            f"--host={host}",
            f"--port={port}",
        ]
        self._server_process = subprocess.Popen(cmd, env=os.environ.copy())
        atexit.register(self._stop_policy_server)

        start_time = time.time()
        while time.time() - start_time < SERVER_STARTUP_TIMEOUT_SEC:
            if self._server_process.poll() is not None:
                raise RuntimeError(f"Policy server exited with code {self._server_process.returncode}")

            try:
                with socket.create_connection((host, int(port)), timeout=1.0):
                    self.get_logger().info(f"Policy server ready on {host}:{port}")
                    return
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(SERVER_STARTUP_POLL_SEC)

        raise RuntimeError(f"Policy server failed to start within {SERVER_STARTUP_TIMEOUT_SEC}s")

    def _stop_policy_server(self) -> None:
        """Terminate the policy server process."""
        if self._server_process is None or self._server_process.poll() is not None:
            return

        self.get_logger().info("Stopping local policy server...")
        self._server_process.terminate()
        try:
            self._server_process.wait(timeout=SERVER_STOP_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            self._server_process.kill()
        self._server_process = None

    def _on_goal(self, _goal_request) -> GoalResponse:
        """Accept or reject a client request to begin an action."""
        self.get_logger().info("Received goal request")
        if not self._accepting_goals:
            self.get_logger().warning("Rejected: node not active")
            return GoalResponse.REJECT
        if self._active_goal is not None:
            self.get_logger().warning("Rejected: already running")
            return GoalResponse.REJECT
        self.get_logger().info("Goal accepted")
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info("Received cancel request")
        if self._client is not None:
            self._client.shutdown_event.set()
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle) -> RunPolicy.Result:
        """Execute policy inference."""
        self._active_goal = goal_handle
        task = goal_handle.request.prompt
        result = RunPolicy.Result()

        self.get_logger().info(f"Starting: task='{task}'")

        try:
            config = self._build_config(task)
            client = RobotClient(config)
            self._client = client

            if not client.start():
                result.success = False
                result.message = "Failed to connect to policy server"
                return self._finish(goal_handle, result)

            receiver = threading.Thread(target=client.receive_actions, daemon=True)
            receiver.start()

            feedback_stop = threading.Event()
            feedback_thread = threading.Thread(
                target=self._feedback_loop,
                args=(goal_handle, client, feedback_stop),
                daemon=True,
            )
            feedback_thread.start()

            try:
                client.control_loop(task=task)
            finally:
                client.stop()
                receiver.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
                feedback_stop.set()
                feedback_thread.join(timeout=FEEDBACK_THREAD_JOIN_TIMEOUT_SEC)

            if goal_handle.is_cancel_requested:
                result.success = False
                result.message = "Cancelled"
            else:
                result.success = True
                result.message = "Completed"

        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            result.success = False
            result.message = str(e)

        return self._finish(goal_handle, result)

    def _feedback_loop(
        self,
        goal_handle,
        client: RobotClient,
        stop_event: threading.Event,
    ) -> None:
        """Publish feedback at configured rate."""
        interval = 1.0 / self.get_parameter("feedback_rate_hz").value
        while not stop_event.wait(interval):
            if goal_handle.is_cancel_requested:
                break

            feedback = RunPolicy.Feedback()
            with client.action_queue_lock:
                feedback.queue_depth = client.action_queue.qsize()
            with client.latest_action_lock:
                feedback.published_actions = max(0, client.latest_action)
            feedback.status = "executing"
            goal_handle.publish_feedback(feedback)

    def _build_config(self, task: str) -> RobotClientConfig:
        """Build RobotClientConfig from ROS2 parameters."""
        robot_config = RosettaConfig(
            id="rosetta",
            config_path=self._contract_path,
            is_classifier=self.get_parameter("is_classifier").value,
            # id is auto-populated from contract's robot_type (vortex_ctl)
            # use_sim_time=self.get_parameter("use_sim_time").value,
        )
        robot_config._external_bridge = self._bridge  # Inject pre-built bridge
        
        # Apply sim_time_multiplier to control loop fps
        # 
        # Architecture:
        #   - Contract fps (10Hz): Rate at which observations are resampled (sim time)
        #   - RobotClient fps: Rate at which control_loop() polls for observations (wall time)
        #   - PolicyServer fps: Only used for FPSTracker logging (not critical)
        #
        # In slow sims (0.5x speed):
        #   - Topics publish at sim-governed rates
        #   - StreamBuffer resamples to contract fps (10Hz sim time)
        #   - control_loop() polls at scaled fps (5Hz wall time) to match sim speed
        #   - This prevents spamming get_observation() faster than new data arrives
        #
        contract_fps = robot_config.fps
        sim_multiplier = self.get_parameter("sim_time_multiplier").value
        control_loop_fps = int(contract_fps * sim_multiplier)
        
        if sim_multiplier != 1.0:
            self.get_logger().info(
                f"Applied sim_time_multiplier={sim_multiplier:.2f}: "
                f"contract fps={contract_fps}Hz (sim time) → "
                f"control loop fps={control_loop_fps}Hz (wall time)"
            )

        config_kwargs = dict(
            robot=robot_config,
            server_address=self.get_parameter("server_address").value,
            policy_type=self.get_parameter("policy_type").value,
            pretrained_name_or_path=self._pretrained,
            policy_device=self.get_parameter("policy_device").value,
            task=task,
            fps=control_loop_fps,  # Wall-time polling rate for control_loop()
            actions_per_chunk=self.get_parameter("actions_per_chunk").value,
            chunk_size_threshold=self.get_parameter("chunk_size_threshold").value,
            aggregate_fn_name=self.get_parameter("aggregate_fn_name").value,
        )

        # obs_similarity_atol: controls observation filtering in the policy server.
        # LeRobot's default similarity filtering (atol=1.0) skips observations where
        # the L2 norm of state change is < 1.0. For many robots, joint states change
        # by much less than 1.0 between frames, causing most observations to be
        # filtered out and breaking inference. Set to -1.0 to disable filtering.
        obs_similarity_atol_param = self.get_parameter("obs_similarity_atol").value
        obs_similarity_atol = None if obs_similarity_atol_param < 0 else obs_similarity_atol_param

        # Check if this LeRobot version supports obs_similarity_atol
        from dataclasses import fields
        supported_fields = {f.name for f in fields(RobotClientConfig)}

        if "obs_similarity_atol" in supported_fields:
            config_kwargs["obs_similarity_atol"] = obs_similarity_atol
        elif obs_similarity_atol_param != 1.0:
            # User configured a non-default value but feature isn't available yet
            self.get_logger().warning(
                "obs_similarity_atol is not yet supported in this LeRobot version. "
                "The parameter will be ignored. If inference skips most observations, "
                "you may need a newer LeRobot version with this feature."
            )

        return RobotClientConfig(**config_kwargs)

    def _finish(self, goal_handle, result: RunPolicy.Result) -> RunPolicy.Result:
        """Clean up and set goal status."""
        self._client = None
        self._active_goal = None

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        self.get_logger().info(f"Finished: {result.message}")
        return result


def main(args=None):
    rclpy.init(args=args)
    node = RosettaClientNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Lifecycle callbacks handle cleanup; just destroy and shutdown
        node.destroy_node()
        rclpy.try_shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
