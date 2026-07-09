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
Rosetta processor steps for RobotObservation / RobotAction dicts.

These steps operate on numpy arrays with short image keys (e.g. "front",
"left_wrist") as produced by Rosetta's get_observation() and
port_bags._sample_robot_dicts().

Unlike LeRobot's built-in ImageCropResizeProcessorStep (which requires
PyTorch tensors and keys containing "image"), these steps work directly
on the numpy uint8 (H, W, C) images that Rosetta produces.
"""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import (ObservationProcessorStep,
                                        ProcessorStepRegistry,
                                        RobotProcessorPipeline)

OBSERVATION_PROCESSOR_CONFIG_FILENAME = "robot_observation_processor.json"


def build_observation_processor(
    spec: dict[str, Any],
    *,
    to_transition: Callable | None = None,
    to_output: Callable | None = None,
) -> RobotProcessorPipeline:
    """Build a RobotProcessorPipeline from an in-memory processor spec.

    Mirrors ``RobotProcessorPipeline.from_pretrained`` for the case where the
    pipeline definition is inlined in a unified contract (the ``processor``
    block) rather than stored as a ``robot_observation_processor.json`` dir.
    The spec is materialised to a temp file and loaded through the normal
    ``from_pretrained`` path, so registry resolution, override/format
    validation, and converter defaults are identical to the on-disk path.

    Args:
        spec: ``{"steps": [{"registry_name": ..., "config": {...}}, ...]}``.
        to_transition / to_output: optional transition converters, forwarded
            to ``from_pretrained`` (defaults match the on-disk path).
    """
    if not isinstance(spec, dict) or "steps" not in spec:
        raise ValueError(
            "Observation processor spec must be a mapping with a 'steps' list; "
            f"got {type(spec).__name__}"
        )

    payload = dict(spec)
    payload.setdefault("name", "RobotObservationProcessor")

    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / OBSERVATION_PROCESSOR_CONFIG_FILENAME).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return RobotProcessorPipeline.from_pretrained(
            tmp_dir,
            config_filename=OBSERVATION_PROCESSOR_CONFIG_FILENAME,
            to_transition=to_transition,
            to_output=to_output,
        )


@ProcessorStepRegistry.register("numpy_image_crop_resize")
@dataclass
class NumpyImageCropResizeProcessorStep(ObservationProcessorStep):
    """Crop and/or resize numpy uint8 images in a RobotObservation dict.

    Designed for Rosetta's observation format where images are:
    - numpy arrays of shape (H, W, C), dtype uint8
    - keyed by short names: "front", "left_wrist", etc.

    Matching is done via ``image_keys``: an explicit list of keys to
    process.  If ``image_keys`` is empty, **all** keys whose values are
    3-D numpy arrays (H, W, C) are treated as images.

    Attributes:
        crop_params_dict: Maps image key → (top, left, height, width).
            - ``top``:    y-coordinate of the top-left corner of the crop box (pixels from top)
            - ``left``:   x-coordinate of the top-left corner of the crop box (pixels from left)
            - ``height``: height of the crop box in pixels
            - ``width``:  width of the crop box in pixels
            Example: ``{"front": [50, 30, 174, 164]}`` crops starting 50px from the
            top and 30px from the left, extracting a 174×164 region.
            Only keys present in this dict are cropped.
        resize_size: Target (height, width). Applied to every matched
            image after optional cropping.  ``None`` means no resize.
        image_keys: Explicit list of keys to process.  If empty, auto-
            detect by array shape.
        interpolation: OpenCV interpolation flag for resizing.
            Default ``cv2.INTER_LINEAR``.

    Note:
        ``resize_size`` MUST match the ``image.resize`` declared in the
        Rosetta contract (e.g. ``[224, 224]``).  The contract determines
        the LeRobot dataset feature shape; if the processor outputs a
        different size the dataset writer will raise a shape mismatch error.
    """

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None
    image_keys: list[str] = field(default_factory=list)
    interpolation: int = cv2.INTER_LINEAR

    # ---- ObservationProcessorStep interface ----

    def observation(self, observation: dict) -> dict:
        if self.resize_size is None and not self.crop_params_dict:
            return observation

        new_obs = dict(observation)

        for key in list(observation.keys()):
            if not self._is_image_key(key, observation[key]):
                continue

            image = observation[key]

            # Crop
            if self.crop_params_dict is not None and key in self.crop_params_dict:
                top, left, h, w = self.crop_params_dict[key]
                image = image[top : top + h, left : left + w]

            # Resize
            if self.resize_size is not None:
                target_h, target_w = self.resize_size
                image = cv2.resize(
                    image,
                    (target_w, target_h),  # cv2 uses (w, h)
                    interpolation=self.interpolation,
                )

            new_obs[key] = np.ascontiguousarray(image)

        return new_obs

    # ---- Serialisation ----

    def get_config(self) -> dict[str, Any]:
        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
            "image_keys": self.image_keys,
        }

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Update image feature shapes if resizing is applied."""
        if self.resize_size is None:
            return features
        for key in features.get(PipelineFeatureType.OBSERVATION, {}):
            feat = features[PipelineFeatureType.OBSERVATION][key]
            if len(feat.shape) == 3:  # (C, H, W)
                nb_channel = feat.shape[0]
                features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                    type=feat.type,
                    shape=(nb_channel, *self.resize_size),
                )
        return features

    # ---- Private helpers ----

    def _is_image_key(self, key: str, value: Any) -> bool:
        """Decide whether this key should be processed as an image."""
        if self.image_keys:
            return key in self.image_keys
        # Auto-detect: 3-D numpy array is probably an image
        return isinstance(value, np.ndarray) and value.ndim == 3
