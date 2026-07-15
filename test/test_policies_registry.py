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

"""The policy-framework registry: entry-point resolution and fail-fast validation.

Everything here fakes ``importlib.metadata`` (the registry's ``_ilm`` handle),
so no real distribution metadata is involved — same trick as
test_codec_registry.py uses for the codec plugin loader.
"""

import types

import pytest
from rosetta.policies import load_dataset_writer, load_policy_runner, registry
from rosetta.policies.registry import (
    DATASET_WRITER_GROUP,
    POLICY_RUNNER_GROUP,
    available_dataset_writers,
    available_policy_runners,
)


class GoodWriter:
    def open(self, **kwargs):
        pass

    def add_frame(self, frame):
        pass

    def save_episode(self):
        pass

    def discard_episode(self):
        pass

    def finalize(self):
        pass


class GoodRunner:
    def setup(self, node, contract):
        pass

    def run(self, frames, *, task, stop_event):
        pass

    def feedback(self):
        pass

    def request_stop(self):
        pass

    def teardown(self):
        pass


class MissingMethodsWriter:
    """No save_episode/discard_episode/finalize."""

    def open(self, **kwargs):
        pass

    def add_frame(self, frame):
        pass


class NeedsArgs:
    def __init__(self, required):
        self.required = required

    def setup(self, node, contract):
        pass

    def run(self, frames, *, task, stop_event):
        pass

    def feedback(self):
        pass

    def request_stop(self):
        pass

    def teardown(self):
        pass


def _fake_ep(name, obj, dist_name="somedist", group=DATASET_WRITER_GROUP):
    return types.SimpleNamespace(
        name=name,
        group=group,
        dist=types.SimpleNamespace(name=dist_name),
        load=lambda: obj,
    )


def _install(monkeypatch, eps_by_group):
    monkeypatch.setattr(
        registry,
        "_ilm",
        types.SimpleNamespace(entry_points=lambda group=None: eps_by_group.get(group, [])),
    )


def test_load_dataset_writer_happy_path(monkeypatch):
    _install(monkeypatch, {DATASET_WRITER_GROUP: [_fake_ep("good", GoodWriter)]})
    assert isinstance(load_dataset_writer("good"), GoodWriter)


def test_load_policy_runner_happy_path(monkeypatch):
    _install(monkeypatch, {POLICY_RUNNER_GROUP: [_fake_ep("good", GoodRunner)]})
    assert isinstance(load_policy_runner("good"), GoodRunner)


def test_unknown_name_error_lists_available(monkeypatch):
    _install(
        monkeypatch,
        {DATASET_WRITER_GROUP: [_fake_ep("beta", GoodWriter), _fake_ep("alpha", GoodWriter)]},
    )
    with pytest.raises(ValueError, match=r"No framework 'nope'.*Available: alpha, beta"):
        load_dataset_writer("nope")


def test_unknown_name_none_installed(monkeypatch):
    _install(monkeypatch, {})
    with pytest.raises(ValueError, match=r"\(none installed\)"):
        load_policy_runner("anything")


def test_non_conforming_impl_raises_naming_missing_members(monkeypatch):
    _install(monkeypatch, {DATASET_WRITER_GROUP: [_fake_ep("partial", MissingMethodsWriter)]})
    with pytest.raises(TypeError, match=r"does not implement DatasetWriter.*discard_episode.*finalize.*save_episode"):
        load_dataset_writer("partial")


def test_non_zero_arg_constructor_propagates_raw(monkeypatch):
    """Constructor failures propagate unwrapped: the raw traceback names the
    adapter class directly (NeedsArgs.__init__ missing 'required'), which no
    registry wrapper improves on."""
    _install(monkeypatch, {POLICY_RUNNER_GROUP: [_fake_ep("needy", NeedsArgs, dist_name="needy-dist")]})
    with pytest.raises(TypeError, match=r"required"):
        load_policy_runner("needy")


def test_load_failure_propagates_raw(monkeypatch):
    """A broken adapter import chain escapes unwrapped — the traceback already
    names the failing module; callers (node, port CLI) log it in full."""

    def _boom():
        raise ImportError("adapter deps missing")

    ep = types.SimpleNamespace(
        name="broken", group=POLICY_RUNNER_GROUP, dist=types.SimpleNamespace(name="d"), load=_boom
    )
    _install(monkeypatch, {POLICY_RUNNER_GROUP: [ep]})
    with pytest.raises(ImportError, match="adapter deps missing"):
        load_policy_runner("broken")


def test_non_class_entry_point_rejected(monkeypatch):
    """The documented contract is a class; a factory function is rejected even
    if calling it would return a conforming object."""

    def writer_factory():
        return GoodWriter()

    _install(monkeypatch, {DATASET_WRITER_GROUP: [_fake_ep("factory", writer_factory)]})
    with pytest.raises(TypeError, match="must be a class"):
        load_dataset_writer("factory")


def test_none_member_named_in_error(monkeypatch):
    """A member set to None fails isinstance but passes hasattr — the old
    diagnostic printed 'missing: ' with an empty list; the callable sweep
    names it."""

    class NoneFinalizeWriter(GoodWriter):
        finalize = None

    _install(monkeypatch, {DATASET_WRITER_GROUP: [_fake_ep("halfbaked", NoneFinalizeWriter)]})
    with pytest.raises(TypeError, match=r"missing or non-callable: finalize"):
        load_dataset_writer("halfbaked")


def test_duplicate_requested_name_raises_naming_both_dists(monkeypatch):
    _install(
        monkeypatch,
        {
            DATASET_WRITER_GROUP: [
                _fake_ep("dup", GoodWriter, dist_name="dist-a"),
                _fake_ep("dup", GoodWriter, dist_name="dist-b"),
            ]
        },
    )
    with pytest.raises(ValueError, match=r"'dup' is registered more than once.*dist-a, dist-b"):
        load_dataset_writer("dup")


def test_duplicate_of_other_name_does_not_block(monkeypatch):
    _install(
        monkeypatch,
        {
            DATASET_WRITER_GROUP: [
                _fake_ep("good", GoodWriter),
                _fake_ep("dup", GoodWriter, dist_name="dist-a"),
                _fake_ep("dup", GoodWriter, dist_name="dist-b"),
            ]
        },
    )
    assert isinstance(load_dataset_writer("good"), GoodWriter)


def test_available_sorted_and_deduped(monkeypatch):
    _install(
        monkeypatch,
        {
            DATASET_WRITER_GROUP: [_fake_ep("zeta", GoodWriter), _fake_ep("alpha", GoodWriter)],
            POLICY_RUNNER_GROUP: [
                _fake_ep("dup", GoodRunner, dist_name="dist-a", group=POLICY_RUNNER_GROUP),
                _fake_ep("dup", GoodRunner, dist_name="dist-b", group=POLICY_RUNNER_GROUP),
                _fake_ep("alpha", GoodRunner, group=POLICY_RUNNER_GROUP),
            ],
        },
    )
    assert available_dataset_writers() == ["alpha", "zeta"]
    assert available_policy_runners() == ["alpha", "dup"]
