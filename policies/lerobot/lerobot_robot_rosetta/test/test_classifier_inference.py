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

"""ClassifierServer._predict_reward: the actual observation -> reward tensor pipeline.

Uses lerobot's real raw_observation_to_observation/TimedObservation helpers (the
key-mapping, image-extraction, and resize logic rosetta doesn't own) against a
duck-typed fake model exposing only predict_reward(), lerobot.rewards.classifier
.Classifier's real eval method. This is the pipeline test_classifier_factory.py's
routing/caching tests don't cover.
"""

import time
from types import SimpleNamespace

import numpy as np
import torch

from lerobot.async_inference.helpers import TimedObservation
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot_robot_rosetta import classifier_server as cs

IMAGE_KEY = "observation.images.cam0"
STATE_KEY = "observation.state"


class _FakePredictRewardClassifier:
    """Duck-typed like lerobot.rewards.classifier.Classifier: only predict_reward()."""

    def __init__(self, input_features, reward=1.0):
        self.config = SimpleNamespace(input_features=input_features)
        self._reward = reward
        self.calls = []

    def predict_reward(self, batch, threshold=0.5):
        self.calls.append({"batch": batch, "threshold": threshold})
        return torch.tensor([self._reward])


def _server_with_fake_classifier(image_size):
    input_features = {
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    server = cs.ClassifierServer.__new__(cs.ClassifierServer)
    server.classifier = _FakePredictRewardClassifier(input_features)
    server.device = "cpu"
    server.lerobot_features = {
        IMAGE_KEY: {"dtype": "image", "shape": [480, 640, 3], "names": ["height", "width", "channels"]},
        STATE_KEY: {"dtype": "float32", "shape": [2], "names": ["joint1", "joint2"]},
    }
    server._image_size = image_size
    return server


def _timed_observation(timestep=0):
    raw_obs = {
        "cam0": np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "joint1": 0.1,
        "joint2": 0.2,
    }
    return TimedObservation(timestamp=time.time(), timestep=timestep, observation=raw_obs)


def test_predict_reward_resizes_images_and_calls_model_eval_method():
    server = _server_with_fake_classifier(image_size=(128, 128))
    server._predict_reward(_timed_observation())

    assert len(server.classifier.calls) == 1
    call = server.classifier.calls[0]
    assert call["threshold"] == 0.5
    assert call["batch"][IMAGE_KEY].shape[-2:] == (128, 128)


def test_predict_reward_wraps_scalar_as_timed_action_with_advanced_timestep():
    server = _server_with_fake_classifier(image_size=(128, 128))
    obs = _timed_observation(timestep=5)

    result = server._predict_reward(obs)

    assert len(result) == 1
    action = result[0]
    assert action.timestep == 6  # timestep + 1, see _predict_reward docstring
    assert action.action.shape == torch.Size([1])
    assert action.action.item() == 1.0


def test_predict_reward_overrides_to_detected_image_size():
    # raw_observation_to_observation already resizes to the declared
    # input_features shape (128, 128, see _server_with_fake_classifier); a
    # _image_size that disagrees (e.g. from the kernel-heuristic fallback in
    # _detect_image_size) must still win as the final size fed to the model.
    server = _server_with_fake_classifier(image_size=(64, 64))
    server._predict_reward(_timed_observation())

    call = server.classifier.calls[0]
    assert call["batch"][IMAGE_KEY].shape[-2:] == (64, 64)
