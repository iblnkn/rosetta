"""Tests for the policy registry loader.

The registry may legitimately be empty (a ``policies:`` key holding only
comments parses as None) — e.g. after all entries were retired to unified
contracts — and the client node must still configure in that state.
"""

import textwrap

import pytest

from rosetta.common.policy_registry import PolicyRegistryError, load_registry


def _write(tmp_path, text):
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def test_empty_policies_key_is_empty_registry(tmp_path):
    path = _write(
        tmp_path,
        """\
        policies:
          # no entries yet
        """,
    )
    assert load_registry(str(path)) == {}


def test_missing_policies_key_raises(tmp_path):
    path = _write(tmp_path, "something_else: {}\n")
    with pytest.raises(PolicyRegistryError):
        load_registry(str(path))


def test_unknown_entry_keys_warn_but_load(tmp_path, caplog):
    """Stale keys (e.g. the removed observation_processor_path pin) must not
    be dropped silently — the operator gets a warning naming them."""
    import logging

    path = _write(
        tmp_path,
        """\
        policies:
          pick:
            pretrained_name_or_path: user/model
            policy_type: sns_diffusion
            observation_processor_path: /old/pinned/processor
        """,
    )
    with caplog.at_level(
        logging.WARNING, logger="rosetta.common.policy_registry"
    ):
        reg = load_registry(str(path))
    assert set(reg) == {"pick"}
    assert not hasattr(reg["pick"], "observation_processor_path")
    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "observation_processor_path" in messages
    assert "inline" in messages


def test_entry_parses(tmp_path):
    path = _write(
        tmp_path,
        """\
        policies:
          pick:
            pretrained_name_or_path: user/model
            policy_type: sns_diffusion
            actions_per_chunk: 24
            contract_path: /some/contract.yaml
        """,
    )
    reg = load_registry(str(path))
    assert set(reg) == {"pick"}
    bundle = reg["pick"]
    assert bundle.policy_type == "sns_diffusion"
    assert bundle.actions_per_chunk == 24
    assert bundle.contract_path == "/some/contract.yaml"
