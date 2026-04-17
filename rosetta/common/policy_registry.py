"""Policy registry: declarative per-task model bundles.

Lets a ``RunPolicy`` caller address a model by symbolic name (e.g. ``pick``)
instead of wiring raw pretrained paths in every goal. The YAML file maps
names to ``PolicyBundle`` entries carrying everything that varies per policy:
checkpoint path, policy type, and the RTC-sensitive runtime knobs
(``actions_per_chunk``, ``chunk_size_threshold``, ``aggregate_fn_name``).
Infrastructure knobs like ``fps`` or ``policy_device`` stay on the node.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml


class PolicyRegistryError(ValueError):
    """Raised when the registry YAML is malformed or references bad paths."""


@dataclass
class PolicyBundle:
    """Per-policy configuration resolved from the registry.

    Optional fields are ``None`` when the registry entry did not specify them,
    in which case the caller should fall back to node-level defaults.
    """

    pretrained_name_or_path: str
    policy_type: str
    actions_per_chunk: int | None = None
    chunk_size_threshold: float | None = None
    aggregate_fn_name: str | None = None


def validate_pretrained(path: str) -> str | None:
    """Return a human-readable error if ``path`` is not a plausible model spec.

    Local paths (``/``, ``./``, ``../``) must exist and contain ``config.json``.
    Everything else is treated as an HF repo ID and must match the
    ``namespace/repo_name`` shape (no network check performed).
    """
    if not path:
        return "pretrained_name_or_path is empty"

    looks_local = path.startswith(("/", "./", "../"))
    if looks_local:
        if not os.path.isdir(path):
            return f"Local model path does not exist: {path}"
        if not os.path.isfile(os.path.join(path, "config.json")):
            return f"No config.json under {path}; not a pretrained checkpoint dir"
        return None

    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        return (
            f"'{path}' is neither an absolute local path "
            "nor a valid HF repo ID (expected 'namespace/repo_name')"
        )
    return None


_OPTIONAL_FIELDS: dict[str, type] = {
    "actions_per_chunk": int,
    "chunk_size_threshold": float,
    "aggregate_fn_name": str,
}


def _parse_entry(name: str, raw: dict) -> PolicyBundle:
    if not isinstance(raw, dict):
        raise PolicyRegistryError(
            f"Registry entry '{name}' must be a mapping; got {type(raw).__name__}"
        )

    missing = [k for k in ("pretrained_name_or_path", "policy_type") if k not in raw]
    if missing:
        raise PolicyRegistryError(
            f"Registry entry '{name}' missing required field(s): {missing}"
        )

    pretrained = raw["pretrained_name_or_path"]
    err = validate_pretrained(pretrained)
    if err is not None:
        raise PolicyRegistryError(f"Registry entry '{name}': {err}")

    kwargs: dict = {
        "pretrained_name_or_path": pretrained,
        "policy_type": raw["policy_type"],
    }
    for field, expected in _OPTIONAL_FIELDS.items():
        if field in raw and raw[field] is not None:
            value = raw[field]
            # Accept int where float is expected (YAML often returns ints for "0.5").
            if expected is float and isinstance(value, int):
                value = float(value)
            if not isinstance(value, expected):
                raise PolicyRegistryError(
                    f"Registry entry '{name}' field '{field}' must be "
                    f"{expected.__name__}; got {type(value).__name__}"
                )
            kwargs[field] = value
    return PolicyBundle(**kwargs)


def load_registry(path: str) -> dict[str, PolicyBundle]:
    """Load a policy registry YAML.

    File layout::

        policies:
          <name>:
            pretrained_name_or_path: ...
            policy_type: ...
            actions_per_chunk: 32           # optional
            chunk_size_threshold: 0.7       # optional
            aggregate_fn_name: latest_only  # optional
    """
    if not os.path.isfile(path):
        raise PolicyRegistryError(f"Registry file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict) or "policies" not in data:
        raise PolicyRegistryError(
            f"Registry {path} must have a top-level 'policies' mapping"
        )

    policies = data["policies"]
    if not isinstance(policies, dict):
        raise PolicyRegistryError(
            f"Registry {path}: 'policies' must be a mapping of name -> entry"
        )

    return {name: _parse_entry(name, entry) for name, entry in policies.items()}
