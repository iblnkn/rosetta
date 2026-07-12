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
Rosetta policy server: preload + cache wrapper over LeRobot's PolicyServer.

The stock server reloads the policy (from_pretrained + .to(device)) on every
client handshake, and only ever loads on a handshake — so the first RunPolicy
goal pays torch import, CUDA init, and the full weight load. This wrapper:

- preloads the policy before binding the gRPC port when --policy-type,
  --pretrained-name-or-path and --policy-device are given, so "port accepts
  connections" (what LeRobotPolicyRunner polls for) means "model loaded";
- caches the loaded policy across handshakes, skipping the reload when the
  requested (policy_type, pretrained_name_or_path, device) is unchanged.

Pre/post processors are still rebuilt on every handshake: they are cheap and
depend on the client-sent rename_map.

Usage:
    python -m lerobot_robot_rosetta.policy_server --host=127.0.0.1 --port=8080 \
        --policy-type=act --pretrained-name-or-path=user/repo --policy-device=cuda
"""

import argparse
import pickle  # nosec
import time
from concurrent import futures

import grpc
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
    SUPPORTED_POLICIES,
)
from lerobot.async_inference.helpers import RemotePolicyConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)


class RosettaPolicyServer(PolicyServer):
    """PolicyServer that loads the policy once and reuses it across handshakes."""

    def __init__(self, config: PolicyServerConfig):
        super().__init__(config)
        # (policy_type, pretrained_name_or_path, device) of the loaded policy.
        self._loaded_key: tuple[str, str, str] | None = None

    def preload(self, policy_type: str, pretrained_name_or_path: str, device: str) -> None:
        """Load the policy ahead of any client (call before binding the port)."""
        self._ensure_policy(policy_type, pretrained_name_or_path, device)

    def _ensure_policy(self, policy_type: str, pretrained_name_or_path: str, device: str) -> bool:
        """Load the policy iff it differs from what is loaded. Returns True on load."""
        if policy_type not in SUPPORTED_POLICIES:
            raise ValueError(f"Policy type {policy_type} not supported. Supported policies: {SUPPORTED_POLICIES}")

        key = (policy_type, pretrained_name_or_path, device)
        if self._loaded_key == key:
            self.logger.info(
                f"Reusing loaded policy (type={policy_type}, path={pretrained_name_or_path}, device={device})"
            )
            return False

        # Clear first: a failed load must not leave a stale key that makes the
        # next identical request look cached.
        self._loaded_key = None
        start = time.perf_counter()
        policy_class = get_policy_class(policy_type)
        policy = policy_class.from_pretrained(pretrained_name_or_path)
        policy.to(device)
        self.policy = policy
        self._loaded_key = key
        self.logger.info(
            f"Policy loaded on {device} in {time.perf_counter() - start:.2f}s "
            f"(type={policy_type}, path={pretrained_name_or_path})"
        )
        return True

    # Mirrors lerobot.async_inference.policy_server.PolicyServer.
    # SendPolicyInstructions (lerobot v0.6.0 pin, see repos/libs.repos) with the
    # load routed through the _ensure_policy cache — re-check on pin bumps.
    def SendPolicyInstructions(self, request, context):
        """Receive policy instructions; load the policy only if it changed."""
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        start = time.perf_counter()
        loaded = self._ensure_policy(
            policy_specs.policy_type,
            policy_specs.pretrained_name_or_path,
            policy_specs.device,
        )

        # Processors are rebuilt every handshake: cheap, and rename_map comes
        # from the client.
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        self.logger.info(
            f"Handshake done in {time.perf_counter() - start:.2f}s "
            f"({'loaded policy' if loaded else 'reused cached policy'})"
        )
        return services_pb2.Empty()


def main():
    parser = argparse.ArgumentParser(
        description="Rosetta policy server (preload + cache wrapper over LeRobot PolicyServer)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--inference-latency", type=float, default=DEFAULT_INFERENCE_LATENCY)
    parser.add_argument("--obs-queue-timeout", type=float, default=DEFAULT_OBS_QUEUE_TIMEOUT)
    parser.add_argument("--policy-type", default="", help="Preload: policy architecture")
    parser.add_argument("--pretrained-name-or-path", default="", help="Preload: weights path/repo")
    parser.add_argument("--policy-device", default="", help="Preload: inference device")
    args = parser.parse_args()

    config = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=args.inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
    )
    policy_server = RosettaPolicyServer(config)

    preload = (args.policy_type, args.pretrained_name_or_path, args.policy_device)
    if all(preload):
        policy_server.logger.info("Preloading policy before binding port...")
        policy_server.preload(*preload)
    elif any(preload):
        policy_server.logger.warning(
            "Preload needs --policy-type, --pretrained-name-or-path and --policy-device together; skipping preload."
        )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{args.host}:{args.port}")

    policy_server.logger.info(f"RosettaPolicyServer started on {args.host}:{args.port}")
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    main()
