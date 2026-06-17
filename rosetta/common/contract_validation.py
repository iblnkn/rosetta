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
Cross-boundary consistency checks for the robot interface.

The robot's data interface (joint names/order, image keys/shapes, fps,
action/state dims) must agree across recording -> dataset -> training ->
deployment. These checks compare a (unified) contract against the artifacts at
each boundary: a LeRobot ``dataset.meta``, a trained checkpoint ``config.json``,
the inline observation processor, and the deploy registry.

Everything here is ROS-free: it derives the contract interface via
:func:`rosetta.common.contract_utils.contract_interface` (no decoder registry /
rclpy), so it runs in the training conda env and in CI without ROS.

Each ``check_*`` returns a :class:`CheckResult` (errors + warnings). Callers use
``result.raise_or_warn(strict=...)`` to either hard-fail or log warnings; the
default strictness is read from the ``ROSETTA_CONTRACT_STRICT`` env var so the
project can ship in "warn" mode and flip to "enforce" later.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import (
    Contract,
    ROLE_INFERENCE,
    ROLE_RECORD,
    load_contract,
    load_processor_spec,
    load_unified_contract,
)
from .contract_utils import contract_interface

logger = logging.getLogger(__name__)

ENV_STRICT = "ROSETTA_CONTRACT_STRICT"


class ContractConsistencyError(ValueError):
    """Raised when a consistency check fails in strict mode."""

    def __init__(self, errors: list[str], context: str = ""):
        self.errors = list(errors)
        header = "Contract consistency check failed"
        if context:
            header += f" ({context})"
        super().__init__(header + ":\n" + "\n".join(f"  - {e}" for e in self.errors))


def strict_from_env(default: bool = False) -> bool:
    """Resolve strict mode from the ``ROSETTA_CONTRACT_STRICT`` env var."""
    val = os.environ.get(ENV_STRICT)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class CheckResult:
    """Outcome of one (or several merged) consistency checks."""

    context: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: "CheckResult") -> "CheckResult":
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        return self

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_or_warn(self, *, strict: bool | None = None, log: logging.Logger | None = None) -> "CheckResult":
        """Log warnings/errors; raise ``ContractConsistencyError`` if strict and any error.

        Args:
            strict: True -> raise on error; False -> warn only; None -> read env
                (``ROSETTA_CONTRACT_STRICT``, default off).
        """
        if strict is None:
            strict = strict_from_env()
        log = log or logger
        for w in self.warnings:
            log.warning("[contract] %s", w)
        for e in self.errors:
            log.error("[contract] %s", e)
        if self.errors and strict:
            raise ContractConsistencyError(self.errors, self.context)
        return self


# =============================================================================
# Shape / name helpers
# =============================================================================


def _hw_from_image_shape(shape: Any) -> tuple[int, int] | None:
    """Extract (height, width) from an image feature shape.

    Handles both LeRobot dataset video features ``(H, W, C)`` and checkpoint
    policy features ``(C, H, W)`` by dropping the single channel dim (value 1 or
    3) and returning the remaining two in order.
    """
    if shape is None:
        return None
    dims = [int(x) for x in shape]
    if len(dims) == 2:
        return (dims[0], dims[1])
    if len(dims) == 3:
        # Drop one channel dim (1 or 3); keep order of the rest.
        for i, d in enumerate(dims):
            if d in (1, 3):
                rest = dims[:i] + dims[i + 1 :]
                return (rest[0], rest[1])
        return (dims[0], dims[1])  # ambiguous; best effort
    return None


def _dim_from_shape(shape: Any) -> int | None:
    if shape is None:
        return None
    dims = [int(x) for x in shape]
    return dims[0] if dims else None


def _compare_names(expected: list[str], actual: list[str] | None, label: str, result: CheckResult,
                   *, as_error: bool = False) -> None:
    """Compare equal-length ordered name lists for order/identity drift.

    Length (dim) mismatches are reported by the caller's dedicated dim check, so
    this only fires on same-length name/order differences. Because dataset and
    checkpoint naming conventions vary (e.g. ``position.la_..._joint`` vs
    ``la_..._joint_pos``), cross-artifact comparisons treat this as a **warning**
    (``as_error=False``); same-convention comparisons (unified vs legacy) set
    ``as_error=True``.
    """
    if actual is None or len(expected) != len(actual):
        return
    if list(expected) != list(actual):
        diffs = [f"[{i}] {e!r}!={a!r}" for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
        msg = f"{label}: name/order differs: " + ", ".join(diffs[:6]) + ("..." if len(diffs) > 6 else "")
        (result.error if as_error else result.warn)(msg)


# =============================================================================
# Boundary checks
# =============================================================================


def check_contract_vs_dataset_meta(contract: Contract, ds_meta: Any, *, context: str = "") -> CheckResult:
    """Compare a contract's interface against a LeRobot ``dataset.meta``.

    ``ds_meta`` is duck-typed: it must expose ``.features`` (dict of
    ``{key: {dtype, shape, names}}``) and ``.fps``; ``.robot_type`` is optional.
    """
    res = CheckResult(context=context or "contract vs dataset.meta")
    intf = contract_interface(contract)
    feats = getattr(ds_meta, "features", {}) or {}

    # fps
    ds_fps = getattr(ds_meta, "fps", None)
    if ds_fps is not None and int(ds_fps) != int(intf["fps"]):
        res.error(f"fps mismatch: contract {intf['fps']} vs dataset {ds_fps}")

    # robot_type (provenance only)
    ds_rt = getattr(ds_meta, "robot_type", None)
    if ds_rt and ds_rt != intf["robot_type"]:
        res.warn(f"robot_type differs: contract '{intf['robot_type']}' vs dataset '{ds_rt}'")

    # images
    for key, (h, w) in intf["images"].items():
        if key not in feats:
            res.error(f"image '{key}' declared by contract is missing from dataset features")
            continue
        hw = _hw_from_image_shape(feats[key].get("shape"))
        if hw and hw != (h, w):
            res.error(f"image '{key}' shape mismatch: contract {[h, w]} vs dataset {list(hw)}")

    # numeric state + action keys
    for key, info in {**intf["state"], **intf["actions"]}.items():
        if key not in feats:
            res.error(f"feature '{key}' declared by contract is missing from dataset features")
            continue
        ds_dim = _dim_from_shape(feats[key].get("shape"))
        if ds_dim is not None and ds_dim != info["dim"]:
            res.error(f"'{key}' dim mismatch: contract {info['dim']} vs dataset {ds_dim}")
        _compare_names(info["names"], feats[key].get("names"), f"'{key}'", res)

    # extra dataset images not declared by the contract (informational)
    ds_imgs = {k for k, v in feats.items() if str(v.get("dtype")) in ("video", "image")}
    for extra in sorted(ds_imgs - set(intf["images"])):
        res.warn(f"dataset has image '{extra}' not declared by the contract")

    return res


def check_contract_vs_checkpoint(contract: Contract, config_json: dict[str, Any], *, context: str = "") -> CheckResult:
    """Compare a contract's interface against a trained checkpoint ``config.json``.

    Catches deploying a policy against a contract whose cameras/dims differ from
    what the policy was trained on (the silent ``KeyError``/zero-camera class of
    deploy bug).
    """
    res = CheckResult(context=context or "contract vs checkpoint")
    intf = contract_interface(contract)
    in_feats = config_json.get("input_features") or {}
    out_feats = config_json.get("output_features") or {}

    # images: VISUAL input features vs contract image keys
    ckpt_imgs = {k: v for k, v in in_feats.items() if str(v.get("type")).upper() == "VISUAL"}
    contract_img_keys = set(intf["images"])
    missing = contract_img_keys - set(ckpt_imgs)
    extra = set(ckpt_imgs) - contract_img_keys
    for k in sorted(missing):
        res.error(f"contract image '{k}' is not an input of the checkpoint")
    for k in sorted(extra):
        res.error(f"checkpoint expects image '{k}' which the contract does not provide")
    for k in sorted(contract_img_keys & set(ckpt_imgs)):
        hw = _hw_from_image_shape(ckpt_imgs[k].get("shape"))
        want = tuple(intf["images"][k])
        if hw and hw != want:
            res.error(f"image '{k}' shape mismatch: contract {list(want)} vs checkpoint {list(hw)}")

    # state dim
    if "observation.state" in in_feats and "observation.state" in intf["state"]:
        ck = _dim_from_shape(in_feats["observation.state"].get("shape"))
        cn = intf["state"]["observation.state"]["dim"]
        if ck is not None and ck != cn:
            res.error(f"observation.state dim mismatch: contract {cn} vs checkpoint {ck}")

    # action dim + names
    if "action" in out_feats and "action" in intf["actions"]:
        ck = _dim_from_shape(out_feats["action"].get("shape"))
        info = intf["actions"]["action"]
        if ck is not None and ck != info["dim"]:
            res.error(f"action dim mismatch: contract {info['dim']} vs checkpoint {ck}")
        _compare_names(info["names"], config_json.get("action_feature_names"), "action", res)

    return res


def check_processor_vs_contract(processor_spec: dict[str, Any] | None, contract: Contract, *,
                                policy_crop_shape: Any = None, policy_resize_shape: Any = None,
                                context: str = "") -> CheckResult:
    """Validate the (inline) observation processor against the contract images.

    Asserts the processor's ``resize_size`` matches the contract image shape and
    that its ``image_keys`` are known short camera names. Warns when the policy
    ``crop_shape`` differs from the resize while no ``resize_shape`` is set (the
    crop-vs-resize footgun) — that is informational, not an error.
    """
    res = CheckResult(context=context or "processor vs contract")
    if not processor_spec:
        return res
    intf = contract_interface(contract)
    short_keys = {k.removeprefix("observation.images.") for k in intf["images"]}
    contract_shapes = {tuple(v) for v in intf["images"].values()}

    resize_size = None
    for step in processor_spec.get("steps", []):
        cfg = step.get("config", {}) or {}
        if cfg.get("resize_size"):
            resize_size = tuple(int(x) for x in cfg["resize_size"])
        for k in cfg.get("image_keys", []) or []:
            if k not in short_keys:
                res.error(f"processor image_key '{k}' is not a contract camera (known: {sorted(short_keys)})")
        for k in (cfg.get("crop_params_dict") or {}):
            if k not in short_keys:
                res.error(f"processor crop key '{k}' is not a contract camera (known: {sorted(short_keys)})")

    if resize_size is not None and contract_shapes and contract_shapes != {resize_size}:
        res.error(
            f"processor resize_size {list(resize_size)} does not match contract image shape(s) "
            f"{[list(s) for s in sorted(contract_shapes)]}"
        )

    if policy_resize_shape is None and policy_crop_shape and resize_size is not None:
        crop = tuple(int(x) for x in policy_crop_shape)
        if crop != resize_size:
            res.warn(
                f"policy crop_shape {list(crop)} != processor resize_size {list(resize_size)}; "
                "the model crops inside its encoder (RandomCrop in train, CenterCrop at eval) — "
                "confirm this is intended"
            )
    return res


def check_chunk_consistency(n_action_steps: int | None, actions_per_chunk: int | None, *, context: str = "") -> CheckResult:
    """Assert the deploy ``actions_per_chunk`` equals the model's ``n_action_steps``."""
    res = CheckResult(context=context or "chunk consistency")
    if n_action_steps is None or actions_per_chunk is None:
        return res
    if int(n_action_steps) != int(actions_per_chunk):
        res.error(
            f"actions_per_chunk ({actions_per_chunk}) != model n_action_steps ({n_action_steps}); "
            "the chunk will be truncated/padded at deploy"
        )
    return res


def check_unified_contract(path: Path | str, *, context: str = "") -> CheckResult:
    """Validate that a unified contract loads in both roles and has a valid processor."""
    path = Path(path)
    res = CheckResult(context=context or f"unified contract {path.name}")
    roles = {}
    for role in (ROLE_RECORD, ROLE_INFERENCE):
        try:
            roles[role] = load_unified_contract(path, role)
        except Exception as e:  # noqa: BLE001 - report any load/merge error
            res.error(f"role '{role}' failed to load: {e}")
    try:
        # Single-pipeline processors are validated here; task-keyed ones need a task.
        spec = None
        try:
            spec = load_processor_spec(path)
        except Exception:
            spec = None  # task-keyed or absent; validated where a task is known
        if spec and ROLE_RECORD in roles:
            res.merge(check_processor_vs_contract(spec, roles[ROLE_RECORD]))
    except Exception as e:  # noqa: BLE001
        res.error(f"processor validation error: {e}")
    return res


def check_unified_vs_legacy(unified_path: Path | str, record_path: Path | str | None,
                            inference_path: Path | str | None, *, context: str = "") -> CheckResult:
    """Migration aid: confirm a unified contract reproduces the legacy pair.

    The record role must match the legacy record contract exactly (interface).
    For inference, observations/images/fps must match; the **action** layout may
    legitimately differ (e.g. a DMP output), so action differences are warnings.
    """
    res = CheckResult(context=context or f"migrate {Path(unified_path).name}")

    def _cmp(role_intf: dict, legacy_intf: dict, *, strict_actions: bool, tag: str) -> None:
        if role_intf["robot_type"] != legacy_intf["robot_type"]:
            res.warn(f"{tag}: robot_type '{role_intf['robot_type']}' vs legacy '{legacy_intf['robot_type']}'")
        if role_intf["fps"] != legacy_intf["fps"]:
            res.error(f"{tag}: fps {role_intf['fps']} vs legacy {legacy_intf['fps']}")
        if role_intf["images"] != legacy_intf["images"]:
            res.error(f"{tag}: image keys/shapes differ: {role_intf['images']} vs {legacy_intf['images']}")
        if _state_names(role_intf) != _state_names(legacy_intf):
            res.error(f"{tag}: state names differ: {_state_names(role_intf)} vs {_state_names(legacy_intf)}")
        if _action_names(role_intf) != _action_names(legacy_intf):
            msg = f"{tag}: action names differ: {_action_names(role_intf)} vs {_action_names(legacy_intf)}"
            (res.error if strict_actions else res.warn)(msg)

    if record_path:
        try:
            _cmp(contract_interface(load_unified_contract(unified_path, ROLE_RECORD)),
                 contract_interface(load_contract(record_path)), strict_actions=True, tag="record")
        except Exception as e:  # noqa: BLE001
            res.error(f"record comparison failed: {e}")
    if inference_path:
        try:
            _cmp(contract_interface(load_unified_contract(unified_path, ROLE_INFERENCE)),
                 contract_interface(load_contract(inference_path)), strict_actions=False, tag="inference")
        except Exception as e:  # noqa: BLE001
            res.error(f"inference comparison failed: {e}")
    return res


def _state_names(intf: dict) -> dict[str, list[str]]:
    return {k: v["names"] for k, v in intf["state"].items()}


def _action_names(intf: dict) -> dict[str, list[str]]:
    return {k: v["names"] for k, v in intf["actions"].items()}
