# Copyright 2025 Brian Blankenau
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
Classifier gRPC server for reward classification.

Implements the same AsyncInference gRPC service as LeRobot's PolicyServer,
but calls predict_reward() instead of predict_action_chunk().  The existing
RobotClient connects to it unchanged.

Usage:
    python -m rosetta.classifier_server --host=127.0.0.1 --port=8081
"""

import argparse
import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from queue import Empty, Queue

import grpc
import torch
import torch.nn.functional as F

from lerobot.policies.factory import get_policy_class
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks

from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    raw_observation_to_observation,
    get_logger,
)

logger = get_logger("classifier_server")

OBS_QUEUE_TIMEOUT = 2.0


class ClassifierServer(services_pb2_grpc.AsyncInferenceServicer):
    """gRPC server for reward classifier inference.

    Speaks the same AsyncInference protocol as PolicyServer so the
    existing RobotClient can connect without modification.
    """

    def __init__(self):
        self.shutdown_event = threading.Event()
        self.observation_queue: Queue = Queue(maxsize=1)

        # Set by SendPolicyInstructions
        self.classifier = None
        self.device = None
        self.lerobot_features = None
        self._image_size = None  # (H, W) detected from model weights

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def _reset(self) -> None:
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

    # ----------------------------------------------------------------
    # gRPC RPCs (same interface as PolicyServer)
    # ----------------------------------------------------------------

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        logger.info(f"Client {client_id} connected")
        self._reset()
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            logger.warning("Server not running, ignoring policy instructions")
            return services_pb2.Empty()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(
                f"Expected RemotePolicyConfig, got {type(policy_specs)}"
            )

        self.device = policy_specs.device
        self.lerobot_features = policy_specs.lerobot_features

        logger.info(
            f"Loading classifier: type={policy_specs.policy_type}, "
            f"path={policy_specs.pretrained_name_or_path}, "
            f"device={policy_specs.device}"
        )

        start = time.perf_counter()
        policy_class = get_policy_class(policy_specs.policy_type)
        self.classifier = policy_class.from_pretrained(
            policy_specs.pretrained_name_or_path
        )
        self.classifier.to(self.device)
        self.classifier.eval()
        self._image_size = self._detect_image_size()
        elapsed = time.perf_counter() - start

        logger.info(
            f"Classifier loaded on {self.device} in {elapsed:.2f}s "
            f"(image_size={self._image_size})"
        )
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, logger
        )
        timed_obs = pickle.loads(received_bytes)  # nosec

        logger.debug(f"Received observation #{timed_obs.get_timestep()}")

        # Simple enqueue: always keep the latest observation
        if self.observation_queue.full():
            self.observation_queue.get_nowait()
        self.observation_queue.put(timed_obs)

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        try:
            obs = self.observation_queue.get(timeout=OBS_QUEUE_TIMEOUT)

            logger.debug(
                f"Classifying observation #{obs.get_timestep()}"
            )

            start = time.perf_counter()
            reward_actions = self._predict_reward(obs)
            elapsed = time.perf_counter() - start

            logger.info(
                f"Observation #{obs.get_timestep()} classified "
                f"(reward={reward_actions[0].action.item():.1f}) "
                f"in {elapsed * 1000:.1f}ms"
            )

            return services_pb2.Actions(data=pickle.dumps(reward_actions))

        except Empty:
            return services_pb2.Actions(data=b"")

        except Exception as e:
            logger.error(f"Error in GetActions: {e}", exc_info=True)
            return services_pb2.Actions(data=b"")

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------

    def _detect_image_size(self) -> tuple[int, int] | None:
        """Detect expected image size from SpatialLearnedEmbeddings kernel.

        The kernel shape is (channel, height, width, num_features) where
        height/width are the expected feature-map spatial dims.  For
        ResNet-18 the encoder downsamples by 32x, so the required input
        resolution is (height * 32, width * 32).
        """
        for name, param in self.classifier.named_parameters():
            if name.endswith(".kernel") and param.dim() == 4:
                _, h, w, _ = param.shape
                # ResNet-family downsampling factor
                size = (h * 32, w * 32)
                logger.info(
                    f"Detected SpatialLearnedEmbeddings kernel "
                    f"spatial dims ({h}, {w}) → image size {size}"
                )
                return size
        return None

    def _predict_reward(
        self, observation_t: TimedObservation
    ) -> list[TimedAction]:
        """Run classifier inference on an observation.

        Pipeline:
        1. Convert raw observation to tensor dict (reuses LeRobot's helper
           for key mapping, image resizing, and float32 [0,1] conversion).
        2. Move tensors to the inference device.
        3. Extract image tensors and resize to the model's expected
           spatial dimensions.
        4. Call predict() directly (bypasses predict_reward() which has
           broken normalize_inputs/normalize_targets calls from the
           pre-migration architecture).
        5. Threshold probabilities to get binary reward.
        6. Wrap the scalar reward as a single TimedAction so the
           RobotClient can process it through the normal action pipeline.
        """
        OBS_IMAGE = "observation.image"

        # 1. Raw observation → tensor dict
        observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.classifier.config.image_features,
        )

        # 2. Move to device
        batch = {
            k: v.to(self.device)
            for k, v in observation.items()
            if isinstance(v, torch.Tensor)
        }

        # 3. Extract image tensors (same key filtering as Classifier)
        images = [
            batch[key]
            for key in self.classifier.config.input_features
            if key.startswith(OBS_IMAGE)
        ]

        # Resize to the model's expected spatial dims if needed
        if self._image_size is not None:
            images = [
                F.interpolate(img, size=self._image_size, mode="bilinear",
                              align_corners=False)
                for img in images
            ]

        # 4. Run inference directly via predict()
        with torch.no_grad():
            output = self.classifier.predict(images)

        # 5. Binary threshold
        if self.classifier.config.num_classes == 2:
            reward = (output.probabilities > 0.5).float()
        else:
            reward = torch.argmax(
                output.probabilities, dim=1
            ).float()

        # 6. Wrap as TimedAction with shape (1,) to match action_features
        #    Use timestep+1 so the action is always newer than latest_action
        #    in RobotClient._aggregate_action_queues (which drops actions
        #    where timestep <= latest_action).  A regular PolicyServer avoids
        #    this because it returns multi-step action chunks that advance the
        #    timestep; the classifier returns a single scalar per observation.
        action_tensor = reward.detach().view(1).cpu()
        return [
            TimedAction(
                timestamp=observation_t.get_timestamp(),
                timestep=observation_t.get_timestep() + 1,
                action=action_tensor,
            )
        ]


def main():
    parser = argparse.ArgumentParser(
        description="Reward classifier gRPC server"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    classifier_server = ClassifierServer()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(
        classifier_server, server
    )
    server.add_insecure_port(f"{args.host}:{args.port}")

    logger.info(f"ClassifierServer starting on {args.host}:{args.port}")
    server.start()
    server.wait_for_termination()
    logger.info("Server terminated")


if __name__ == "__main__":
    main()
