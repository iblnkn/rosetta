# server.py
# -*- coding: utf-8 -*-
"""
Inference backends for PolicyBridge.

Provides a stable interface so the node can swap between:
- LocalPolicyServer: in-process LeRobot policy (fastest, no networking)
- LerobotGrpcServer: remote LeRobot async inference server via gRPC

Expected shapes returned by `predict_chunk`:
- torch.Tensor of shape [B, T, D]
- OR dict[str, torch.Tensor] with each value shaped [B, T, D_k]

Notes
-----
* Device selection supports "auto" | "cuda[:N]" | "mps"/"metal" | "cpu".
* When running on Jetson/NGC, prefer `policy_device:=cuda`.
* If `dataset_stats` are available, they are passed into the LeRobot
  pre/post processors for exact training/serving parity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Union

import torch

# LeRobot core
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


TensorOrDict = Union[torch.Tensor, Dict[str, torch.Tensor]]


@dataclass(slots=True)
class ServerConfig:
    policy_path: str
    device: str = "auto"
    actions_per_chunk: int = 25
    dataset_stats: Optional[dict] = None
    rename_map: Optional[Dict[str, str]] = None


class InferenceServer(ABC):
    @abstractmethod
    def load(self, cfg: ServerConfig) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def predict_chunk(self, batch: Dict[str, Any]) -> TensorOrDict: ...


# ---------- helpers ----------

def _device_from_param(requested: Optional[str]) -> torch.device:
    r = (requested or "auto").lower().strip()

    def mps_available() -> bool:
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if r == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if r.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(r)  # "cuda" or "cuda:N"
    if r in {"mps", "metal"}:
        if not mps_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    try:
        return torch.device(r)
    except Exception:
        return torch.device("cpu")


# ---------- Local (in-process) backend ----------

class LocalPolicyServer(InferenceServer):
    """In-process LeRobot policy. Fastest path; zero-copy when on same GPU."""

    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.actions_per_chunk: int = 25

    # Public API --------------------------------------------------------------
    def load(self, cfg: ServerConfig) -> None:
        import json, os

        self.device = _device_from_param(cfg.device)
        self.actions_per_chunk = int(cfg.actions_per_chunk)

        # Try to infer policy type from local config.json, else try common types
        policy_type = ""
        if os.path.isdir(cfg.policy_path):
            cfg_json = os.path.join(cfg.policy_path, "config.json")
            if os.path.exists(cfg_json):
                try:
                    with open(cfg_json, "r", encoding="utf-8") as f:
                        j = json.load(f)
                    policy_type = str(j.get("type", "")).lower()
                except Exception:
                    policy_type = ""

        tried: List[str] = [policy_type] if policy_type else [
            "act", "diffusion", "pi0", "pi05", "smolvla"
        ]
        last_err: Optional[Exception] = None
        for t in tried:
            try:
                PolicyCls = get_policy_class(t)
                self.policy = PolicyCls.from_pretrained(cfg.policy_path)
                self.policy.to(self.device)
                self.policy.eval()
                policy_type = t
                break
            except Exception as e:  # keep trying next type
                last_err = e
                self.policy = None
        if self.policy is None:
            raise RuntimeError(
                f"Could not load policy from {cfg.policy_path!r}. Last error: {last_err!r}"
            )

        # Wire pre/post processors with dataset stats + optional renames
        dev_override = {"device": str(self.device)}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=cfg.policy_path,
            dataset_stats=cfg.dataset_stats,
            preprocessor_overrides={
                "device_processor": dev_override,
                # If available in your LeRobot build; harmless if ignored
                "rename_observations_processor": {"rename_map": cfg.rename_map or {}},
            },
            postprocessor_overrides={"device_processor": dev_override},
        )

        # Best-effort: clamp actions_per_chunk to policy's known maximum if exposed
        try:
            max_steps = None
            if hasattr(self.policy.config, "n_action_steps"):
                max_steps = int(self.policy.config.n_action_steps)
            elif hasattr(self.policy.config, "chunk_size"):
                max_steps = int(self.policy.config.chunk_size)
            if max_steps is not None and self.actions_per_chunk > max_steps:
                self.actions_per_chunk = max_steps
        except Exception:
            pass

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            try:
                self.policy.reset()
            except Exception:
                pass

    def predict_chunk(self, batch: Dict[str, Any]) -> TensorOrDict:
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        with torch.inference_mode():
            # Many LeRobot policies expose predict_action_chunk(); keep API stable.
            out = self.policy.predict_action_chunk(batch)
        # --- enforce configured horizon on the local path ---
        try:
            h = int(self.actions_per_chunk)
            if h > 0:
                if isinstance(out, dict):
                    out = {k: v[:, :h, ...] for k, v in out.items()}
                else:
                    out = out[:, :h, ...]
        except Exception:
            pass
        # Postprocess must preserve leading [B, T, ...]
        return self.postprocessor(out) if self.postprocessor is not None else out


# ---------- Remote (gRPC) backend ----------

class LerobotGrpcServer(InferenceServer):
    """Client for LeRobot's async inference policy server.

    Requires a running server (typically launched with
    `python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080`).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        self.host, self.port = host, port
        self._client = None
        self._cfg: Optional[ServerConfig] = None
        # The node inspects `server.device` for dtype/placement decisions.
        self.device: torch.device = torch.device("cpu")

    def load(self, cfg: ServerConfig) -> None:
        self._cfg = cfg
        try:
            # Import here to avoid hard dependency when only using local backend
            from lerobot.async_inference.policy_client import PolicyClient  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Could not import lerobot.async_inference.policy_client. "
                "Upgrade LeRobot or adjust PYTHONPATH."
            ) from e

        self._client = PolicyClient(host=self.host, port=self.port)
        # Ensure server is ready and send instructions (policy path, device, etc.)
        self._client.ready()
        self._client.send_policy_instructions(
            policy_type=None,  # let the server validate/infer
            pretrained_name_or_path=cfg.policy_path,
            actions_per_chunk=int(cfg.actions_per_chunk),
            device=str(cfg.device or "auto"),
            lerobot_features=None,  # server-side preprocessing
            rename_map=cfg.rename_map or {},
        )

    def reset(self) -> None:
        # Server keeps its own policy state; no local action needed.
        pass

    def predict_chunk(self, batch: Dict[str, Any]) -> TensorOrDict:
        if self._client is None:
            raise RuntimeError("gRPC client not initialized; call load() first")

        # The PolicyClient performs its own preprocessing/encoding
        timed_actions = self._client.get_actions(batch)  # List[TimedAction]
        if not timed_actions:
            # Return an empty chunk shaped [1, 0, D] to signal starvation
            return torch.zeros((1, 0, 1), dtype=torch.float32)

        # TimedAction.action can be a Tensor or a dict[str, Tensor]
        first = timed_actions[0].action
        if isinstance(first, dict):
            keys = list(first.keys())
            out: Dict[str, torch.Tensor] = {}
            for k in keys:
                seq = [ta.action[k] for ta in timed_actions]
                # [T, D_k] -> [1, T, D_k]
                out[k] = torch.stack(seq, dim=0).unsqueeze(0)
            return out
        else:
            seq = [ta.action for ta in timed_actions]  # List[Tensor [D]] or [1, D]
            return torch.stack(seq, dim=0).unsqueeze(0)  # [1, T, D]


__all__ = [
    "ServerConfig",
    "InferenceServer",
    "LocalPolicyServer",
    "LerobotGrpcServer",
]
