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

Implements the same AsyncInference gRPC service as LeRobot's PolicyServer, but
calls predict_reward() instead of predict_action_chunk(). The existing
RobotClient connects to it unchanged.

This is lerobot-side inference tooling, not a rosetta adapter: it loads a
lerobot reward_classifier model and calls its own predict_reward() eval
method directly. It knows nothing about contracts or ROS; it lives in
lerobot_robot_rosetta purely so HIL deployments get it from the same install
as the robot adapter.

Usage:
    python -m lerobot_robot_rosetta.classifier_server --host=127.0.0.1 --port=8081 \
        [--policy-type=reward_classifier --pretrained-name-or-path=user/repo \
         --policy-device=cuda]

The optional preload flags load the model before the port is bound, so a
socket-level readiness check implies the model is ready; later handshakes
requesting the same (type, path, device) reuse the loaded model.
"""

import argparse
import pickle  # nosec
import threading
import time
from concurrent import futures
from queue import Empty, Queue

import grpc
import torch
import torch.nn.functional as F
from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    raw_observation_to_observation,
)
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class
from lerobot.rewards.factory import get_reward_model_class
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks

logger = get_logger("classifier_server")

OBS_QUEUE_TIMEOUT = 2.0

# Reward models moved out of the policy registry in lerobot 0.6.0
# (lerobot.rewards); "reward_classifier" is the only one this server
# supports. lerobot's other reward models (sarm, robometer, topreward) score
# a whole video against a task instruction and have no per-step, image-only
# inference method — they're offline dataset-labeling tools, not online
# classifiers, so they don't fit this gRPC-per-observation server at all.
# Anything else falls through to get_policy_class, for classifiers saved as a
# plain pre-0.6.0 PreTrainedPolicy.
REWARD_CLASSIFIER_TYPE = "reward_classifier"


class ClassifierServer(services_pb2_grpc.AsyncInferenceServicer):
    """
    gRPC server for reward classifier inference.

    Speaks the same AsyncInference protocol as PolicyServer so the existing
    RobotClient can connect without modification.
    """

    def __init__(self):
        self.shutdown_event = threading.Event()
        self.observation_queue: Queue = Queue(maxsize=1)

        # Set by SendPolicyInstructions (or preload)
        self.classifier = None
        self.device = None
        self.lerobot_features = None
        self._image_size = None  # (H, W), from config input_features (fallback: kernel heuristic)
        # (policy_type, pretrained_name_or_path, device) of the loaded model.
        self._loaded_key: tuple[str, str, str] | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def _reset(self) -> None:
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

    # ----------------------------------------------------------------
    # gRPC RPCs (same interface as PolicyServer)
    # ----------------------------------------------------------------

    def Ready(self, request, context):
        client_id = context.peer()
        logger.info(f"Client {client_id} connected")
        self._reset()
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):
        if not self.running:
            logger.warning("Server not running, ignoring policy instructions")
            return services_pb2.Empty()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Expected RemotePolicyConfig, got {type(policy_specs)}")

        self.device = policy_specs.device
        self.lerobot_features = policy_specs.lerobot_features

        self._ensure_classifier(
            policy_specs.policy_type,
            policy_specs.pretrained_name_or_path,
            policy_specs.device,
        )
        return services_pb2.Empty()

    def preload(self, policy_type: str, pretrained_name_or_path: str, device: str) -> None:
        """Load the model ahead of any client (call before binding the port)."""
        self.device = device
        self._ensure_classifier(policy_type, pretrained_name_or_path, device)

    def _ensure_classifier(self, policy_type: str, pretrained_name_or_path: str, device: str) -> bool:
        """Load the model iff it differs from what is loaded. Returns True on load."""
        key = (policy_type, pretrained_name_or_path, device)
        if self._loaded_key == key:
            logger.info(
                f"Reusing loaded classifier (type={policy_type}, path={pretrained_name_or_path}, device={device})"
            )
            return False

        # Clear first: a failed load must not leave a stale key that makes the
        # next identical request look cached.
        self._loaded_key = None
        logger.info(f"Loading classifier: type={policy_type}, path={pretrained_name_or_path}, device={device}")

        start = time.perf_counter()
        if policy_type == REWARD_CLASSIFIER_TYPE:
            policy_class = get_reward_model_class(policy_type)
        else:
            policy_class = get_policy_class(policy_type)
        self.classifier = policy_class.from_pretrained(pretrained_name_or_path)
        self.classifier.to(device)
        self.classifier.eval()
        self._image_size = self._detect_image_size()
        elapsed = time.perf_counter() - start
        self._loaded_key = key

        logger.info(f"Classifier loaded on {device} in {elapsed:.2f}s (image_size={self._image_size})")
        return True

    def SendObservations(self, request_iterator, context):
        received_bytes = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event, logger)
        timed_obs = pickle.loads(received_bytes)  # nosec

        logger.debug(f"Received observation #{timed_obs.get_timestep()}")

        # Keep only the latest observation
        if self.observation_queue.full():
            self.observation_queue.get_nowait()
        self.observation_queue.put(timed_obs)

        return services_pb2.Empty()

    def GetActions(self, request, context):
        try:
            obs = self.observation_queue.get(timeout=OBS_QUEUE_TIMEOUT)

            logger.debug(f"Classifying observation #{obs.get_timestep()}")

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

    def _image_features(self) -> dict:
        """Visual input features of the loaded model.

        Policy configs expose an ``image_features`` property; reward-model
        configs (lerobot >= 0.6.0) do not — derive it from ``input_features``.
        """
        cfg = self.classifier.config
        feats = getattr(cfg, "image_features", None)
        if feats is not None:
            return feats
        return {key: ft for key, ft in cfg.input_features.items() if ft.type is FeatureType.VISUAL}

    def _detect_image_size(self) -> tuple[int, int] | None:
        """Expected input image size (H, W), from config if possible.

        Primary: the visual PolicyFeature shapes in config.input_features —
        channel-first (C, H, W) per lerobot convention. Fallback: infer the
        feature-map dims from a SpatialLearnedEmbeddings kernel and assume a
        ResNet 32x downsample (loudly — it only holds for that architecture).
        None if neither source yields a size ("pass images through unresized").
        """
        sizes: set[tuple[int, int]] = set()
        for key, ft in self._image_features().items():
            shape = tuple(ft.shape)
            # A channel-first image shape is (C, H, W) with a small C; a big
            # first dim means a mis-conventioned (H, W, C) config — skip it
            # rather than interpolate to a garbage size.
            if len(shape) != 3 or shape[0] > 4:
                logger.warning(f"{key}: cannot read (C, H, W) from visual shape {shape}; ignoring")
                continue
            sizes.add((shape[1], shape[2]))
        if len(sizes) == 1:
            size = sizes.pop()
            logger.info(f"Image size {size} from config.input_features")
            return size
        if len(sizes) > 1:
            logger.warning(
                f"Visual features disagree on image size ({sorted(sizes)}); falling back to the kernel heuristic"
            )

        for name, param in self.classifier.named_parameters():
            if name.endswith(".kernel") and param.dim() == 4:
                _, h, w, _ = param.shape
                size = (h * 32, w * 32)
                logger.warning(
                    f"Image size {size} inferred from weight {name!r} — ASSUMES a "
                    f"SpatialLearnedEmbeddings kernel of shape (C, h, w, F) atop a "
                    f"ResNet with 32x downsampling. Set visual input_features in "
                    f"the model config to make the size explicit."
                )
                return size

        logger.warning("Could not determine expected image size; images pass through unresized")
        return None

    def _predict_reward(self, observation_t: TimedObservation) -> list[TimedAction]:
        """
        Run classifier inference on an observation.

        Pipeline:
        1. Convert raw observation to a tensor dict (LeRobot's helper handles key
           mapping, image resizing, and float32 [0,1] conversion).
        2. Move tensors to the inference device.
        3. Resize image tensors to the model's expected spatial dims.
        4. Call predict_reward() — the model's own eval method. It extracts
           images from the batch and thresholds (binary or multiclass)
           internally.
        5. Wrap the scalar reward as a single TimedAction so the RobotClient can
           process it through the normal action pipeline.
        """
        OBS_IMAGE = "observation.image"

        # 1. Raw observation to tensor dict
        observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self._image_features(),
        )

        # 2. Move to device
        batch = {k: v.to(self.device) for k, v in observation.items() if isinstance(v, torch.Tensor)}

        # 3. Resize image tensors to the model's expected spatial dims, in place
        if self._image_size is not None:
            image_keys = [key for key in self.classifier.config.input_features if key.startswith(OBS_IMAGE)]
            for key in image_keys:
                batch[key] = F.interpolate(batch[key], size=self._image_size, mode="bilinear", align_corners=False)

        # 4. predict_reward() extracts images from batch and thresholds internally.
        with torch.no_grad():
            reward = self.classifier.predict_reward(batch, threshold=0.5)

        # 5. Wrap as TimedAction with shape (1,) to match action_features.
        #    Use timestep+1 so the action is always newer than latest_action in
        #    RobotClient._aggregate_action_queues, which drops actions where
        #    timestep <= latest_action. A regular PolicyServer avoids this by
        #    returning multi-step action chunks that advance the timestep. The
        #    classifier returns a single scalar per observation.
        action_tensor = reward.detach().view(1).cpu()
        return [
            TimedAction(
                timestamp=observation_t.get_timestamp(),
                timestep=observation_t.get_timestep() + 1,
                action=action_tensor,
            )
        ]


def main():
    parser = argparse.ArgumentParser(description="Reward classifier gRPC server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--policy-type", default="", help="Preload: reward model type")
    parser.add_argument("--pretrained-name-or-path", default="", help="Preload: weights path/repo")
    parser.add_argument("--policy-device", default="", help="Preload: inference device")
    args = parser.parse_args()

    classifier_server = ClassifierServer()

    preload = (args.policy_type, args.pretrained_name_or_path, args.policy_device)
    if all(preload):
        logger.info("Preloading classifier before binding port...")
        classifier_server.preload(*preload)
    elif any(preload):
        logger.warning(
            "Preload needs --policy-type, --pretrained-name-or-path and --policy-device together; skipping preload."
        )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(classifier_server, server)
    server.add_insecure_port(f"{args.host}:{args.port}")

    logger.info(f"ClassifierServer starting on {args.host}:{args.port}")
    server.start()
    server.wait_for_termination()
    logger.info("Server terminated")


if __name__ == "__main__":
    main()
