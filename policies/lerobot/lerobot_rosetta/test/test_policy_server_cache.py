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

"""RosettaPolicyServer preload/cache behavior.

The wrapper must load the policy once (at preload or the first handshake) and
reuse it for identical (path, device) requests, while rebuilding the cheap
pre/post processors on every handshake. The policy type comes from the
checkpoint, never from the client. No real weights, no CUDA: the config
loader, policy factory and processor factory are monkeypatched.
"""

import pickle
from types import SimpleNamespace

import pytest
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig

from lerobot_rosetta import policy_server as ps


class _FakePolicy:
    def __init__(self):
        self.config = SimpleNamespace()
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _Recorder:
    """Counts from_pretrained and processor-factory calls via monkeypatch."""

    def __init__(self):
        self.loads: list[tuple[str, str]] = []
        self.processor_builds: list[dict] = []
        self.fail_paths: set[str] = set()
        self.types: dict[str, str] = {}  # checkpoint path -> recorded type; "act" if absent

    def get_policy_class(self, policy_type):
        recorder = self

        class _Cls:
            @staticmethod
            def from_pretrained(path, config):
                assert config.type == policy_type
                if path in recorder.fail_paths:
                    raise OSError(f"no such repo: {path}")
                recorder.loads.append((policy_type, path))
                return _FakePolicy()

        return _Cls

    def make_pre_post_processors(self, *args, **kwargs):
        self.processor_builds.append(kwargs)
        return ("pre", "post")


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(ps, "get_policy_class", rec.get_policy_class)
    monkeypatch.setattr(ps, "make_pre_post_processors", rec.make_pre_post_processors)
    fake_config = SimpleNamespace(from_pretrained=lambda path: SimpleNamespace(type=rec.types.get(path, "act")))
    monkeypatch.setattr(ps, "PreTrainedConfig", fake_config)
    return rec


@pytest.fixture(autouse=True)
def every_device_available(monkeypatch):
    """Cache keys are resolved devices; pin resolution so they don't depend on this host's GPUs."""
    monkeypatch.setattr(ps, "resolve_policy_device", lambda device: device)


@pytest.fixture
def server():
    return ps.RosettaPolicyServer(PolicyServerConfig(host="127.0.0.1", port=8080))


def _handshake(server, **overrides):
    spec_kwargs = dict(
        policy_type="act",
        pretrained_name_or_path="user/repo",
        lerobot_features={},
        actions_per_chunk=10,
        device="cpu",
    )
    spec_kwargs.update(overrides)
    request = SimpleNamespace(data=pickle.dumps(RemotePolicyConfig(**spec_kwargs)))
    context = SimpleNamespace(peer=lambda: "test-client")
    server.Ready(SimpleNamespace(), context)  # what RobotClient.start() does first
    return server.SendPolicyInstructions(request, context)


def test_identical_handshakes_load_once(recorder, server):
    _handshake(server)
    _handshake(server)
    assert recorder.loads == [("act", "user/repo")]
    # Processors are rebuilt per handshake (rename_map comes from the client).
    assert len(recorder.processor_builds) == 2


def test_preload_makes_first_handshake_a_cache_hit(recorder, server):
    server.preload("user/repo", "cpu")
    assert recorder.loads == [("act", "user/repo")]
    _handshake(server)
    assert recorder.loads == [("act", "user/repo")]  # no second load
    assert server.policy is not None
    assert len(recorder.processor_builds) == 1


def test_changed_path_or_device_reloads(recorder, server):
    _handshake(server)
    _handshake(server, pretrained_name_or_path="user/other")
    _handshake(server, pretrained_name_or_path="user/other", device="cuda")
    assert recorder.loads == [
        ("act", "user/repo"),
        ("act", "user/other"),
        ("act", "user/other"),
    ]


def test_server_resolves_the_requested_device_on_its_own_host(recorder, server, monkeypatch):
    # The robot forwards its configured device; a backend this host lacks falls back here.
    monkeypatch.setattr(ps, "resolve_policy_device", lambda device: "cpu")
    _handshake(server, device="cuda")
    assert server.policy.device == "cpu"


def test_failed_load_is_not_cached(recorder, server):
    recorder.fail_paths.add("user/repo")
    with pytest.raises(OSError):
        _handshake(server)

    recorder.fail_paths.clear()
    _handshake(server)  # same key must actually load now, not hit a stale cache
    assert recorder.loads == [("act", "user/repo")]


def test_policy_type_comes_from_the_checkpoint(recorder, server):
    recorder.types["user/repo"] = "diffusion"
    _handshake(server, policy_type="act")
    assert recorder.loads == [("diffusion", "user/repo")]


def test_unsupported_checkpoint_type_rejected(recorder, server):
    recorder.types["user/repo"] = "not_a_policy"
    with pytest.raises(ValueError, match="'not_a_policy'.*does not support"):
        server.preload("user/repo", "cpu")
    assert recorder.loads == []
