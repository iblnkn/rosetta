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

from lerobot.async_inference.helpers import RemotePolicyConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot_robot_rosetta import classifier_server as cs


def test_reward_types_route_to_reward_factory(monkeypatch):
    calls = []
    monkeypatch.setattr(cs, "get_reward_model_class", lambda n: calls.append(("reward", n)))
    monkeypatch.setattr(cs, "get_policy_class", lambda n: calls.append(("policy", n)))

    for name in ("reward_classifier", "act"):
        if name in cs.REWARD_MODEL_TYPES:
            cs.get_reward_model_class(name)
        else:
            cs.get_policy_class(name)

    assert calls == [("reward", "reward_classifier"), ("policy", "act")]


def test_reward_classifier_resolves_in_installed_lerobot():
    # The real factory must still know 'reward_classifier' (moved, not gone).
    cls = cs.get_reward_model_class("reward_classifier")
    assert cls.__name__ == "Classifier"


def test_image_features_derived_from_input_features():
    server = cs.ClassifierServer.__new__(cs.ClassifierServer)
    visual = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))
    state = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    # Reward-model config shape: input_features only, no image_features property.
    server.classifier = SimpleNamespace(
        config=SimpleNamespace(input_features={"observation.image.top": visual, "observation.state": state})
    )
    feats = server._image_features()
    assert feats == {"observation.image.top": visual}


def test_image_features_prefers_policy_property():
    server = cs.ClassifierServer.__new__(cs.ClassifierServer)
    sentinel = {"observation.image.top": "from-property"}
    server.classifier = SimpleNamespace(config=SimpleNamespace(image_features=sentinel, input_features={}))
    assert server._image_features() is sentinel


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


def test_image_size_from_config_shape():
    server = _server_with({"observation.image.top": _visual((3, 128, 96))})
    assert server._detect_image_size() == (128, 96)


def test_image_size_agreeing_cameras():
    server = _server_with(
        {
            "observation.image.top": _visual((3, 128, 96)),
            "observation.image.wrist": _visual((3, 128, 96)),
        }
    )
    assert server._detect_image_size() == (128, 96)


class _FakeKernel:
    """Duck-typed torch parameter: .dim() and .shape only."""

    def __init__(self, shape):
        self.shape = shape

    def dim(self):
        return len(self.shape)


def test_image_size_disagreeing_cameras_falls_to_kernel():
    server = _server_with(
        {
            "observation.image.top": _visual((3, 128, 96)),
            "observation.image.wrist": _visual((3, 64, 64)),
        },
        named_parameters=[("encoder.spatial.kernel", _FakeKernel((512, 4, 3, 8)))],
    )
    assert server._detect_image_size() == (128, 96)


def test_image_size_kernel_only():
    server = _server_with({}, named_parameters=[("encoder.spatial.kernel", _FakeKernel((512, 4, 3, 8)))])
    assert server._detect_image_size() == (128, 96)


def test_image_size_none_when_no_source():
    server = _server_with({"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))})
    assert server._detect_image_size() is None


def test_image_size_suspicious_hwc_shape_ignored():
    # (H, W, C) sneaking into a channel-first field must not become a
    # (640, 3)-ish garbage size; it falls through to the (absent) kernel.
    server = _server_with({"observation.image.top": _visual((480, 640, 3))})
    assert server._detect_image_size() is None


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
