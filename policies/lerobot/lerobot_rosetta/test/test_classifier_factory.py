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

"""Classifier-server routing after the lerobot 0.6.0 reward-model split.

Regression guards: `get_policy_class('reward_classifier')` no longer resolves
(reward models moved to lerobot.rewards), and RewardClassifierConfig has no
`image_features` property — visual features must be derived from
input_features.
"""

import pickle
from types import SimpleNamespace

import pytest
from lerobot.async_inference.helpers import RemotePolicyConfig
from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot_rosetta import classifier_server as cs


def test_reward_classifier_resolves_in_installed_lerobot():
    # The real factory must still know 'reward_classifier' (moved, not gone).
    cls = cs.get_reward_model_class("reward_classifier")
    assert cls.__name__ == "Classifier"


# ---------------------------------------------------------------------------
# _detect_image_size: config-first, kernel heuristic as loud fallback.
# ---------------------------------------------------------------------------


def _server_with(input_features, named_parameters=()):
    server = cs.ClassifierServer.__new__(cs.ClassifierServer)
    server.classifier = SimpleNamespace(
        config=SimpleNamespace(input_features=input_features),
        named_parameters=lambda: iter(named_parameters),
    )
    return server


def _visual(shape):
    return PolicyFeature(type=FeatureType.VISUAL, shape=shape)


class _FakeKernel:
    """Duck-typed torch parameter: .dim() and .shape only."""

    def __init__(self, shape):
        self.shape = shape

    def dim(self):
        return len(self.shape)


# A (512, 4, 3, 8) SpatialLearnedEmbeddings kernel implies (4*32, 3*32) = (128, 96).
_KERNEL = [("encoder.spatial.kernel", _FakeKernel((512, 4, 3, 8)))]


@pytest.mark.parametrize(
    ("input_features", "named_parameters", "expected"),
    [
        ({"observation.image.top": _visual((3, 128, 96))}, (), (128, 96)),
        (
            {"observation.image.top": _visual((3, 128, 96)), "observation.image.wrist": _visual((3, 128, 96))},
            (),
            (128, 96),
        ),
        (
            {"observation.image.top": _visual((3, 128, 96)), "observation.image.wrist": _visual((3, 64, 64))},
            _KERNEL,
            (128, 96),
        ),
        ({}, _KERNEL, (128, 96)),
        ({"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))}, (), None),
        # (H, W, C) sneaking into a channel-first field must not become a
        # (640, 3)-ish garbage size; it falls through to the (absent) kernel.
        ({"observation.image.top": _visual((480, 640, 3))}, (), None),
    ],
    ids=["config_shape", "agreeing_cameras", "disagreeing_cameras_use_kernel", "kernel_only", "no_source", "hwc_shape"],
)
def test_detect_image_size(input_features, named_parameters, expected):
    server = _server_with(input_features, named_parameters)
    assert server._detect_image_size() == expected


# ---------------------------------------------------------------------------
# Model cache: load once at preload/first handshake, reuse for identical keys.
# ---------------------------------------------------------------------------


class _FakeClassifier:
    def __init__(self):
        self.config = SimpleNamespace(input_features={})

    def to(self, device):
        return self

    def eval(self):
        return self

    def named_parameters(self):
        return iter(())


def _patch_factories(monkeypatch):
    loads = []

    def _fake_factory(route):
        def factory(policy_type):
            class _Cls:
                @staticmethod
                def from_pretrained(path):
                    loads.append((route, policy_type, path))
                    return _FakeClassifier()

            return _Cls

        return factory

    monkeypatch.setattr(cs, "get_reward_model_class", _fake_factory("reward"))
    monkeypatch.setattr(cs, "get_policy_class", _fake_factory("policy"))
    return loads


def _handshake(server, policy_type="reward_classifier", path="user/repo", device="cpu"):
    spec = RemotePolicyConfig(
        policy_type=policy_type,
        pretrained_name_or_path=path,
        lerobot_features={},
        actions_per_chunk=1,
        device=device,
    )
    request = SimpleNamespace(data=pickle.dumps(spec))
    context = SimpleNamespace(peer=lambda: "test-client")
    server.SendPolicyInstructions(request, context)


def test_identical_handshakes_load_once(monkeypatch):
    loads = _patch_factories(monkeypatch)
    server = cs.ClassifierServer()
    _handshake(server)
    _handshake(server)
    assert loads == [("reward", "reward_classifier", "user/repo")]


def test_preload_makes_handshake_a_cache_hit(monkeypatch):
    loads = _patch_factories(monkeypatch)
    server = cs.ClassifierServer()
    server.preload("reward_classifier", "user/repo", "cpu")
    _handshake(server)
    assert loads == [("reward", "reward_classifier", "user/repo")]
    assert server.device == "cpu"
    assert server.classifier is not None


def test_changed_key_reloads_and_non_reward_routes_to_policy(monkeypatch):
    loads = _patch_factories(monkeypatch)
    server = cs.ClassifierServer()
    _handshake(server)
    _handshake(server, policy_type="act", path="user/other")
    assert loads == [
        ("reward", "reward_classifier", "user/repo"),
        ("policy", "act", "user/other"),
    ]
