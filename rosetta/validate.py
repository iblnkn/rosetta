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
``python -m rosetta.validate`` — offline consistency checker for the robot
interface across recording -> dataset -> training -> deployment.

ROS-free: derives the contract interface without the decoder registry / rclpy,
so it runs in the training conda env and in CI. Subcommands:

  contract   <contract.yaml> [...]            both roles load + processor valid
  migrate    --unified U --record R --inference I   unified reproduces the legacy pair
  dataset    --contract C [--role record] --root R  contract vs dataset meta/info.json
  checkpoint --contract C [--role record] --pretrained-dir D   contract vs checkpoint config.json
  deploy     --registry policy_registry.yaml        per-entry checkpoint/processor/chunk checks

Exit code is non-zero if any check reports an error (warnings do not fail unless
``--warnings-as-errors``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from .common.contract import (
    ROLE_INFERENCE,
    ROLE_RECORD,
    is_unified_contract,
    load_contract,
    load_processor_spec,
    load_unified_contract,
)
from .common import contract_validation as V


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _safe_exists(path: str | Path | None) -> bool:
    """Path.exists() that returns False (not raises) on unreadable paths."""
    if not path:
        return False
    try:
        return Path(path).exists()
    except OSError:
        return False


def _load_for_role(path: str | Path, role: str = ROLE_RECORD):
    """Load a contract (unified -> role view, else legacy) into a Contract."""
    if is_unified_contract(path):
        return load_unified_contract(path, role)
    return load_contract(path)


def _meta_from_info_json(root: str | Path) -> SimpleNamespace:
    """Build a duck-typed dataset meta from ``<root>/meta/info.json``."""
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.exists():
        # Allow passing the meta dir or the info.json directly.
        alt = Path(root) / "info.json"
        info_path = Path(root) if Path(root).name == "info.json" else (alt if alt.exists() else info_path)
    info = json.loads(Path(info_path).read_text())
    return SimpleNamespace(
        features=info.get("features", {}) or {},
        fps=info.get("fps"),
        robot_type=info.get("robot_type"),
    )


def _print_result(res: V.CheckResult) -> None:
    status = "OK" if res.ok else "FAIL"
    print(f"[{status}] {res.context}")
    for w in res.warnings:
        print(f"    warn:  {w}")
    for e in res.errors:
        print(f"    ERROR: {e}")


def _finish(results: list[V.CheckResult], warnings_as_errors: bool) -> int:
    n_err = sum(len(r.errors) for r in results)
    n_warn = sum(len(r.warnings) for r in results)
    print(f"\n{len(results)} check(s): {n_err} error(s), {n_warn} warning(s)")
    if n_err or (warnings_as_errors and n_warn):
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def cmd_contract(args) -> int:
    results = []
    for path in args.paths:
        if is_unified_contract(path):
            r = V.check_unified_contract(path)
        else:
            r = V.CheckResult(context=f"legacy contract {Path(path).name}")
            try:
                load_contract(path)
            except Exception as e:  # noqa: BLE001
                r.error(str(e))
        results.append(r)
        _print_result(r)
    return _finish(results, args.warnings_as_errors)


def cmd_migrate(args) -> int:
    if not args.record and not args.inference:
        # check_unified_vs_legacy compares only the roles it is given, so
        # --unified alone would run zero comparisons yet report [OK].
        print(
            "error: migrate needs at least one of --record/--inference to "
            "compare against; --unified alone checks nothing",
            file=sys.stderr,
        )
        return 2
    r = V.check_unified_vs_legacy(args.unified, args.record, args.inference)
    _print_result(r)
    return _finish([r], args.warnings_as_errors)


def cmd_dataset(args) -> int:
    contract = _load_for_role(args.contract, args.role)
    meta = _meta_from_info_json(args.root)
    r = V.check_contract_vs_dataset_meta(contract, meta,
                                         context=f"{Path(args.contract).name} [{args.role}] vs {args.root}")
    _print_result(r)
    return _finish([r], args.warnings_as_errors)


def cmd_checkpoint(args) -> int:
    # Compare against the record/base role: the checkpoint predicts the trained
    # (base) action, not any inference-time publish transform (e.g. DMP).
    contract = _load_for_role(args.contract, args.role)
    cfg = json.loads((Path(args.pretrained_dir) / "config.json").read_text())
    r = V.check_contract_vs_checkpoint(contract, cfg,
                                       context=f"{Path(args.contract).name} [{args.role}] vs {args.pretrained_dir}")
    # Also surface the crop-vs-resize footgun if a processor is inlined.
    try:
        spec = load_processor_spec(args.contract) if is_unified_contract(args.contract) else None
    except Exception as e:  # noqa: BLE001
        r.error(f"invalid processor block: {e}")
        spec = None
    if spec:
        r.merge(V.check_processor_vs_contract(spec, contract,
                                              policy_crop_shape=cfg.get("crop_shape"),
                                              policy_resize_shape=cfg.get("resize_shape")))
    _print_result(r)
    return _finish([r], args.warnings_as_errors)


def _check_deploy_entry(name: str, entry: dict, r: V.CheckResult) -> None:
    """Run every check for one registry entry, accumulating into ``r``."""
    contract_path = entry.get("contract_path")
    ckpt_dir = entry.get("pretrained_name_or_path")
    actions_per_chunk = entry.get("actions_per_chunk")

    if not _safe_exists(contract_path):
        r.warn(f"contract_path missing/unreachable: {contract_path}")
        return
    contract = _load_for_role(contract_path, ROLE_RECORD)

    cfg = None
    cfg_path = Path(ckpt_dir) / "config.json" if ckpt_dir else None
    if _safe_exists(cfg_path):
        cfg = json.loads(cfg_path.read_text())
        r.merge(V.check_contract_vs_checkpoint(contract, cfg, context=f"{name}: contract vs checkpoint"))
        r.merge(V.check_chunk_consistency(cfg.get("n_action_steps"), actions_per_chunk,
                                          context=f"{name}: chunk"))
    else:
        r.warn(f"checkpoint config.json unreachable: {cfg_path}")

    # processor: the unified contract's inline block (the only deployable
    # form; legacy single-role contracts are reference-only). Mirror the
    # client node's hard gate: anything it would reject is an error here.
    spec = None
    if is_unified_contract(contract_path):
        try:
            spec = load_processor_spec(contract_path)
        except Exception as e:  # noqa: BLE001
            r.error(f"{name}: invalid processor block: {e}")
        else:
            if spec is None:
                r.error(f"{name}: unified contract has no inline "
                        "`processor:` block — the client node rejects "
                        "every goal against this entry")
    else:
        r.error(f"{name}: legacy single-role contract — reference-only, "
                "not deployable; migrate to a unified contract")
    if spec:
        r.merge(V.check_processor_vs_contract(
            spec, contract,
            policy_crop_shape=(cfg or {}).get("crop_shape"),
            policy_resize_shape=(cfg or {}).get("resize_shape"),
            context=f"{name}: processor"))


def cmd_deploy(args) -> int:
    reg = yaml.safe_load(Path(args.registry).read_text()) or {}
    policies = reg.get("policies", {}) or {}
    results = []
    for name, entry in policies.items():
        r = V.CheckResult(context=f"registry entry '{name}'")
        try:
            _check_deploy_entry(name, entry or {}, r)
        except Exception as e:  # noqa: BLE001 - one broken entry (bad
            # contract, malformed config.json, ...) must not stop the rest
            # of the registry from being checked; partial results are kept.
            r.error(str(e))
        results.append(r)
        _print_result(r)
    return _finish(results, args.warnings_as_errors)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m rosetta.validate",
                                description="Robot-interface consistency checker.")
    p.add_argument("--warnings-as-errors", action="store_true",
                   help="Exit non-zero if any warnings are reported.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("contract", help="Validate unified/legacy contract file(s).")
    pc.add_argument("paths", nargs="+")
    pc.set_defaults(func=cmd_contract)

    pm = sub.add_parser("migrate", help="Check a unified contract reproduces a legacy record/inference pair.")
    pm.add_argument("--unified", required=True)
    pm.add_argument("--record")
    pm.add_argument("--inference")
    pm.set_defaults(func=cmd_migrate)

    pd = sub.add_parser("dataset", help="Check a contract against a dataset's meta/info.json.")
    pd.add_argument("--contract", required=True)
    pd.add_argument("--root", required=True)
    pd.add_argument("--role", default=ROLE_RECORD, choices=[ROLE_RECORD, ROLE_INFERENCE])
    pd.set_defaults(func=cmd_dataset)

    pk = sub.add_parser("checkpoint", help="Check a contract against a checkpoint config.json.")
    pk.add_argument("--contract", required=True)
    pk.add_argument("--pretrained-dir", required=True)
    pk.add_argument("--role", default=ROLE_RECORD, choices=[ROLE_RECORD, ROLE_INFERENCE])
    pk.set_defaults(func=cmd_checkpoint)

    pp = sub.add_parser("deploy", help="Check every policy_registry.yaml entry.")
    pp.add_argument("--registry", required=True)
    pp.set_defaults(func=cmd_deploy)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
