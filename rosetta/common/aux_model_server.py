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
Auxiliary model gRPC server.

Implements the same AsyncInference gRPC service as LeRobot's PolicyServer,
but uses pluggable backends for model inference. The existing RobotClient
connects unchanged.

Usage:
    python -m rosetta.common.aux_model_server --host=127.0.0.1 --port=8081
"""

import argparse
import importlib
import pickle  # nosec
import re
import threading
import time
from concurrent import futures
from queue import Empty, Queue
from typing import Protocol

import grpc
import numpy as np
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
from .inference_profile import InferenceProfile, load_inference_profile

logger = get_logger("aux_model_server")

OBS_IMAGE_PREFIXES = ("observation.image", "observation.images.")


def _reshape_output_tensor(raw_value: torch.Tensor, profile: InferenceProfile) -> torch.Tensor:
    """Reshape/scaffold output tensor according to profile.output.tensor.shape."""
    target_shape = profile.output.tensor.shape
    action_tensor = raw_value.detach().float().reshape(-1).cpu()

    expected_numel = 1
    for dim in target_shape:
        expected_numel *= dim

    if action_tensor.numel() == 1 and expected_numel > 1:
        action_tensor = action_tensor.repeat(expected_numel)
    elif action_tensor.numel() != expected_numel:
        raise ValueError(
            f"Output shape mismatch: got {action_tensor.numel()} values, "
            f"expected {expected_numel} for shape {target_shape}"
        )

    return action_tensor.view(*target_shape)


def _build_timed_action(
    obs: TimedObservation,
    action_tensor: torch.Tensor,
    profile: InferenceProfile,
) -> TimedAction:
    timestep_offset = 1 if profile.output.timestep_policy == "obs_plus_one" else 0
    return TimedAction(
        timestamp=obs.get_timestamp(),
        timestep=obs.get_timestep() + timestep_offset,
        action=action_tensor,
    )


class ClassifierBackend(Protocol):
    """Backend interface for classifier inference."""

    def load(
        self,
        policy_specs: RemotePolicyConfig,
        profile: InferenceProfile,
    ) -> None:
        """Load model/resources for inference."""

    def predict_reward(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict reward-like action outputs for one observation."""


class LerobotClassifierBackend:
    """Legacy-compatible LeRobot classifier backend."""

    def __init__(self):
        self.classifier = None
        self.device = None
        self.lerobot_features = None
        self._image_size = None
        self._profile: InferenceProfile | None = None

    def load(
        self,
        policy_specs: RemotePolicyConfig,
        profile: InferenceProfile,
    ) -> None:
        self._profile = profile

        self.device = profile.backend.device or policy_specs.device
        self.lerobot_features = policy_specs.lerobot_features

        model_path = profile.backend.model_id or policy_specs.pretrained_name_or_path

        logger.info(
            f"Loading classifier backend={profile.backend.kind}, "
            f"type={policy_specs.policy_type}, "
            f"path={model_path}, "
            f"device={self.device}"
        )

        start = time.perf_counter()
        policy_class = get_policy_class(policy_specs.policy_type)
        self.classifier = policy_class.from_pretrained(model_path)
        self.classifier.to(self.device)
        self.classifier.eval()

        if profile.inputs.image_resize is not None:
            self._image_size = profile.inputs.image_resize
            logger.info(f"Using image_size override from profile: {self._image_size}")
        else:
            self._image_size = self._detect_image_size()

        elapsed = time.perf_counter() - start
        logger.info(
            f"Classifier loaded on {self.device} in {elapsed:.2f}s "
            f"(image_size={self._image_size})"
        )

    def _detect_image_size(self) -> tuple[int, int] | None:
        """Detect expected image size from SpatialLearnedEmbeddings kernel."""
        for name, param in self.classifier.named_parameters():
            if name.endswith(".kernel") and param.dim() == 4:
                _, h, w, _ = param.shape
                size = (h * 32, w * 32)
                logger.info(
                    f"Detected SpatialLearnedEmbeddings kernel "
                    f"spatial dims ({h}, {w}) → image size {size}"
                )
                return size
        return None

    def predict_reward(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Run classifier inference on an observation."""
        if self.classifier is None or self._profile is None:
            raise RuntimeError("Classifier backend is not loaded")

        observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.classifier.config.image_features,
        )

        batch = {
            k: v.to(self.device)
            for k, v in observation.items()
            if isinstance(v, torch.Tensor)
        }

        configured_image_keys = self._profile.inputs.image_keys
        if configured_image_keys:
            images = [batch[key] for key in configured_image_keys if key in batch]
        else:
            images = [
                batch[key]
                for key in self.classifier.config.input_features
                if key.startswith(OBS_IMAGE_PREFIXES)
            ]

        if self._profile.validation.require_nonempty_images and not images:
            raise ValueError("No image tensors available for classifier inference")

        if self._image_size is not None:
            images = [
                F.interpolate(
                    img,
                    size=self._image_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for img in images
            ]

        with torch.no_grad():
            output = self.classifier.predict(images)

        threshold = self._profile.postprocess.binary_threshold
        if self.classifier.config.num_classes == 2:
            reward = (output.probabilities > threshold).float()
        else:
            reward = torch.argmax(output.probabilities, dim=1).float()

        action_tensor = _reshape_output_tensor(reward, self._profile)
        return [
            _build_timed_action(observation_t, action_tensor, self._profile)
        ]


class TransformersVLMBackend:
    """Transformers VLM backend (first implementation).

    Supports models that can be loaded through AutoProcessor plus either
    AutoModelForImageTextToText or AutoModelForVision2Seq.
    """

    def __init__(self):
        self._profile: InferenceProfile | None = None
        self._processor = None
        self._model = None
        self._device = "cpu"
        self._task = ""

    def load(
        self,
        policy_specs: RemotePolicyConfig,
        profile: InferenceProfile,
    ) -> None:
        self._profile = profile
        self._task = str(getattr(policy_specs, "task", "") or "")

        model_id = profile.backend.model_id or policy_specs.pretrained_name_or_path
        if not model_id:
            raise ValueError(
                "backend.model_id is required for transformers_vlm profile"
            )

        self._device = profile.backend.device or policy_specs.device or "cpu"

        try:
            transformers = importlib.import_module("transformers")
        except ImportError as e:
            raise ImportError(
                "transformers_vlm backend requires 'transformers' to be installed"
            ) from e

        auto_processor = getattr(transformers, "AutoProcessor")
        self._processor = auto_processor.from_pretrained(
            model_id,
            trust_remote_code=profile.backend.trust_remote_code,
        )

        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            model_class = getattr(transformers, "AutoModelForVision2Seq", None)
        if model_class is None:
            raise RuntimeError(
                "No compatible transformers VLM auto-model class found in this version"
            )

        torch_dtype = None
        if profile.backend.dtype:
            torch_dtype = getattr(torch, profile.backend.dtype, None)
            if torch_dtype is None:
                raise ValueError(
                    f"Unsupported torch dtype '{profile.backend.dtype}' in profile"
                )

        load_kwargs = {
            "trust_remote_code": profile.backend.trust_remote_code,
        }
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        self._model = model_class.from_pretrained(model_id, **load_kwargs)
        self._model.to(self._device)
        self._model.eval()

        logger.info(
            f"Loaded transformers VLM model '{model_id}' on {self._device}"
        )

    def _to_pil_image(self, value):
        image_module = importlib.import_module("PIL.Image")
        image_cls = getattr(image_module, "Image")

        if isinstance(value, image_cls):
            img = value
        elif isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.dim() == 4:
                tensor = tensor[0]
            if tensor.dim() == 3 and tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            arr = tensor.numpy()
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255)
                arr = arr.astype(np.uint8)
            img = image_module.fromarray(arr)
        elif isinstance(value, np.ndarray):
            arr = value
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255)
                arr = arr.astype(np.uint8)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            img = image_module.fromarray(arr)
        else:
            raise TypeError(f"Unsupported image input type: {type(value)}")

        if self._profile and self._profile.inputs.image_resize is not None:
            h, w = self._profile.inputs.image_resize
            img = img.resize((w, h))
        return img.convert("RGB")

    def _extract_images(self, observation: dict) -> list:
        if self._profile is None:
            raise RuntimeError("Profile not loaded")

        image_keys = self._profile.inputs.image_keys
        if image_keys:
            values = [observation[k] for k in image_keys if k in observation]
        else:
            values = [
                v
                for k, v in observation.items()
                if isinstance(k, str) and k.startswith(OBS_IMAGE_PREFIXES)
            ]

        images = [self._to_pil_image(v) for v in values]
        if self._profile.validation.require_nonempty_images and not images:
            raise ValueError("No images found for transformers_vlm inference")
        return images

    def _build_prompt(self) -> str:
        if self._profile is None:
            raise RuntimeError("Profile not loaded")

        if self._profile.inputs.text_source == "literal":
            base_task = self._profile.inputs.text_literal
        else:
            base_task = self._task

        template = self._profile.inference.prompt_template.strip()
        if template:
            return template.format(task=base_task)
        return base_task

    def _parse_text_output(self, text: str) -> float:
        if self._profile is None:
            raise RuntimeError("Profile not loaded")

        parser = self._profile.postprocess.parser
        text_norm = text.strip().lower()

        if parser == "regex_float":
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            if match:
                return float(match.group(0))
            return float(self._profile.postprocess.default_value)

        if parser == "label_map":
            if self._profile.postprocess.label_map:
                for label, val in self._profile.postprocess.label_map.items():
                    if label.lower() in text_norm:
                        return float(val)
            return float(self._profile.postprocess.default_value)

        logger.warning(
            f"Unknown postprocess.parser '{parser}', using default value"
        )
        return float(self._profile.postprocess.default_value)

    def predict_reward(self, observation_t: TimedObservation) -> list[TimedAction]:
        if self._model is None or self._processor is None or self._profile is None:
            raise RuntimeError("transformers_vlm backend is not loaded")

        observation = observation_t.get_observation()
        if not isinstance(observation, dict):
            raise TypeError(
                f"Expected observation dict, got {type(observation)}"
            )

        images = self._extract_images(observation)
        prompt = self._build_prompt()

        generation_kwargs = dict(self._profile.inference.generation or {})

        if images:
            model_inputs = self._processor(
                images=images[0],
                text=prompt,
                return_tensors="pt",
            )
        else:
            model_inputs = self._processor(
                text=prompt,
                return_tensors="pt",
            )
        model_inputs = {
            k: (v.to(self._device) if hasattr(v, "to") else v)
            for k, v in model_inputs.items()
        }

        with torch.no_grad():
            output_ids = self._model.generate(**model_inputs, **generation_kwargs)

        decoded = self._processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0]
        score = self._parse_text_output(decoded)

        raw = torch.tensor([score], dtype=torch.float32)
        action_tensor = _reshape_output_tensor(raw, self._profile)
        return [
            _build_timed_action(observation_t, action_tensor, self._profile)
        ]


class AuxModelServer(services_pb2_grpc.AsyncInferenceServicer):
    """gRPC server for auxiliary model inference.

    Speaks the same AsyncInference protocol as PolicyServer so the
    existing RobotClient can connect without modification.
    """

    def __init__(self, inference_profile: InferenceProfile | None = None):
        self.inference_profile = inference_profile or InferenceProfile()
        self.shutdown_event = threading.Event()
        self.observation_queue: Queue = Queue(maxsize=self.inference_profile.server.queue.maxsize)
        self.backend: ClassifierBackend | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def _reset(self) -> None:
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=self.inference_profile.server.queue.maxsize)

    def _create_backend(self, kind: str) -> ClassifierBackend:
        if kind == "lerobot_classifier":
            return LerobotClassifierBackend()
        if kind == "transformers_vlm":
            return TransformersVLMBackend()
        raise NotImplementedError(
            f"Backend '{kind}' is not implemented yet in aux_model_server"
        )

    def _default_actions(self, obs: TimedObservation) -> list[TimedAction]:
        value = float(self.inference_profile.postprocess.default_value)
        tensor = torch.full(
            self.inference_profile.output.tensor.shape,
            value,
            dtype=torch.float32,
        )
        return [
            _build_timed_action(obs, tensor, self.inference_profile)
        ]

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

        self.backend = self._create_backend(self.inference_profile.backend.kind)
        self.backend.load(policy_specs, self.inference_profile)
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, logger
        )
        timed_obs = pickle.loads(received_bytes)  # nosec

        logger.debug(f"Received observation #{timed_obs.get_timestep()}")

        if self.inference_profile.server.queue.strategy == "latest_only":
            if self.observation_queue.full():
                self.observation_queue.get_nowait()
            self.observation_queue.put(timed_obs)
        else:
            if self.observation_queue.full():
                logger.warning("Observation queue full; dropping newest observation")
            else:
                self.observation_queue.put(timed_obs)

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        obs = None
        try:
            obs = self.observation_queue.get(timeout=self.inference_profile.server.timeout_s)

            if self.backend is None:
                logger.warning("No classifier backend loaded yet")
                return services_pb2.Actions(data=b"")

            logger.debug(
                f"Classifying observation #{obs.get_timestep()}"
            )

            start = time.perf_counter()
            reward_actions = self.backend.predict_reward(obs)
            elapsed = time.perf_counter() - start

            reward_preview = "n/a"
            if reward_actions and isinstance(reward_actions[0].action, torch.Tensor):
                reward_preview = f"{reward_actions[0].action.reshape(-1)[0].item():.3f}"

            logger.info(
                f"Observation #{obs.get_timestep()} classified "
                f"(reward={reward_preview}) "
                f"in {elapsed * 1000:.1f}ms"
            )

            return services_pb2.Actions(data=pickle.dumps(reward_actions))

        except Empty:
            return services_pb2.Actions(data=b"")

        except Exception as e:
            logger.error(f"Error in GetActions: {e}", exc_info=True)

            if (
                self.inference_profile.server.on_error == "emit_default"
                and obs is not None
            ):
                try:
                    return services_pb2.Actions(data=pickle.dumps(self._default_actions(obs)))
                except Exception as fallback_err:
                    logger.error(f"Failed to emit default action: {fallback_err}", exc_info=True)

            return services_pb2.Actions(data=b"")


def main():
    parser = argparse.ArgumentParser(
        description="Auxiliary model gRPC server"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--inference-profile",
        default="",
        help="Optional path to inference profile YAML",
    )
    args = parser.parse_args()

    profile = load_inference_profile(args.inference_profile or None)

    logger.info(
        f"AuxModelServer using backend='{profile.backend.kind}', "
        f"queue='{profile.server.queue.strategy}', "
        f"timeout={profile.server.timeout_s}s"
    )

    aux_model_server = AuxModelServer(inference_profile=profile)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(
        aux_model_server, server
    )
    server.add_insecure_port(f"{args.host}:{args.port}")

    logger.info(f"AuxModelServer starting on {args.host}:{args.port}")
    server.start()
    server.wait_for_termination()
    logger.info("Server terminated")


# Backward-compatible alias for older imports.
ClassifierServer = AuxModelServer


if __name__ == "__main__":
    main()
