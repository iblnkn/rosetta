"""Inference profile schema and loader for backend-agnostic classifier servers.

This module defines a small validated configuration surface for model backends,
input handling, and output emission behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_BACKENDS = frozenset({
    "lerobot_classifier",
    "transformers_vlm",
    "custom_python",
})

SUPPORTED_QUEUE_STRATEGIES = frozenset({"latest_only", "fifo"})
SUPPORTED_ON_ERROR = frozenset({"emit_default", "drop"})
SUPPORTED_OUTPUT_KINDS = frozenset({"tensor", "dict"})
SUPPORTED_TIMESTEP_POLICIES = frozenset({"obs_plus_one", "obs_same"})


class InferenceProfileValidationError(ValueError):
    """Raised when an inference profile is invalid."""


@dataclass(frozen=True, slots=True)
class QueueConfig:
    """Observation queue behavior."""

    strategy: str = "latest_only"
    maxsize: int = 1


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Server runtime behavior."""

    queue: QueueConfig = QueueConfig()
    timeout_s: float = 2.0
    on_error: str = "emit_default"


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Model backend selection and loading hints."""

    kind: str = "lerobot_classifier"
    model_id: str | None = None
    device: str | None = None
    dtype: str | None = None
    quantization: str | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True, slots=True)
class InputConfig:
    """Input feature selection and optional prompt behavior."""

    image_keys: list[str] | None = None
    state_keys: list[str] | None = None
    text_source: str = "task_prompt"
    text_literal: str = ""
    image_resize: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Inference request configuration."""

    mode: str = "classify"
    prompt_template: str = ""
    generation: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """Postprocessing parser configuration."""

    parser: str = "label_map"
    label_map: dict[str, float] | None = None
    default_value: float = 0.0
    binary_threshold: float = 0.5
    callable: str | None = None


@dataclass(frozen=True, slots=True)
class TensorOutputConfig:
    """Tensor output shape/type description."""

    dtype: str = "float32"
    shape: tuple[int, ...] = (1,)


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output packaging configuration."""

    kind: str = "tensor"
    tensor: TensorOutputConfig = TensorOutputConfig()
    timestep_policy: str = "obs_plus_one"


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Runtime validation thresholds."""

    require_nonempty_images: bool = True
    max_latency_ms: int = 2000


@dataclass(frozen=True, slots=True)
class InferenceProfile:
    """Top-level model inference profile."""

    version: int = 1
    server: ServerConfig = ServerConfig()
    backend: BackendConfig = BackendConfig()
    inputs: InputConfig = InputConfig()
    inference: InferenceConfig = InferenceConfig()
    postprocess: PostprocessConfig = PostprocessConfig()
    output: OutputConfig = OutputConfig()
    validation: ValidationConfig = ValidationConfig()


def _ensure_mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise InferenceProfileValidationError(
            f"Expected mapping for '{context}', got {type(value).__name__}"
        )
    return value


def _validate_choice(value: str, valid: frozenset[str], field: str) -> str:
    value = str(value).strip().lower()
    if value not in valid:
        raise InferenceProfileValidationError(
            f"Invalid {field} '{value}'. Must be one of: {sorted(valid)}"
        )
    return value


def _parse_server(data: dict[str, Any]) -> ServerConfig:
    queue_data = _ensure_mapping(data.get("queue"), "server.queue")
    strategy = _validate_choice(
        queue_data.get("strategy", "latest_only"),
        SUPPORTED_QUEUE_STRATEGIES,
        "server.queue.strategy",
    )
    maxsize = int(queue_data.get("maxsize", 1))
    if maxsize <= 0:
        raise InferenceProfileValidationError("server.queue.maxsize must be > 0")

    timeout_s = float(data.get("timeout_s", 2.0))
    if timeout_s <= 0:
        raise InferenceProfileValidationError("server.timeout_s must be > 0")

    on_error = _validate_choice(
        data.get("on_error", "emit_default"),
        SUPPORTED_ON_ERROR,
        "server.on_error",
    )

    return ServerConfig(
        queue=QueueConfig(strategy=strategy, maxsize=maxsize),
        timeout_s=timeout_s,
        on_error=on_error,
    )


def _parse_backend(data: dict[str, Any]) -> BackendConfig:
    kind = _validate_choice(
        data.get("kind", "lerobot_classifier"),
        SUPPORTED_BACKENDS,
        "backend.kind",
    )
    return BackendConfig(
        kind=kind,
        model_id=data.get("model_id"),
        device=data.get("device"),
        dtype=data.get("dtype"),
        quantization=data.get("quantization"),
        trust_remote_code=bool(data.get("trust_remote_code", False)),
    )


def _parse_inputs(data: dict[str, Any]) -> InputConfig:
    image_resize = data.get("image_resize")
    parsed_resize = None
    if image_resize is not None:
        if not isinstance(image_resize, (list, tuple)) or len(image_resize) != 2:
            raise InferenceProfileValidationError(
                "inputs.image_resize must be a two-item list [H, W]"
            )
        parsed_resize = (int(image_resize[0]), int(image_resize[1]))

    image_keys = data.get("image_keys")
    if image_keys is not None:
        if not isinstance(image_keys, list):
            raise InferenceProfileValidationError("inputs.image_keys must be a list")
        image_keys = [str(k) for k in image_keys]

    state_keys = data.get("state_keys")
    if state_keys is not None:
        if not isinstance(state_keys, list):
            raise InferenceProfileValidationError("inputs.state_keys must be a list")
        state_keys = [str(k) for k in state_keys]

    return InputConfig(
        image_keys=image_keys,
        state_keys=state_keys,
        text_source=str(data.get("text_source", "task_prompt")),
        text_literal=str(data.get("text_literal", "")),
        image_resize=parsed_resize,
    )


def _parse_inference(data: dict[str, Any]) -> InferenceConfig:
    return InferenceConfig(
        mode=str(data.get("mode", "classify")),
        prompt_template=str(data.get("prompt_template", "")),
        generation=_ensure_mapping(data.get("generation"), "inference.generation") or None,
    )


def _parse_postprocess(data: dict[str, Any]) -> PostprocessConfig:
    label_map = data.get("label_map")
    parsed_label_map = None
    if label_map is not None:
        if not isinstance(label_map, dict):
            raise InferenceProfileValidationError("postprocess.label_map must be a mapping")
        parsed_label_map = {str(k): float(v) for k, v in label_map.items()}

    threshold = float(data.get("binary_threshold", 0.5))
    if not (0.0 <= threshold <= 1.0):
        raise InferenceProfileValidationError("postprocess.binary_threshold must be in [0, 1]")

    return PostprocessConfig(
        parser=str(data.get("parser", "label_map")),
        label_map=parsed_label_map,
        default_value=float(data.get("default_value", 0.0)),
        binary_threshold=threshold,
        callable=data.get("callable"),
    )


def _parse_output(data: dict[str, Any]) -> OutputConfig:
    kind = _validate_choice(data.get("kind", "tensor"), SUPPORTED_OUTPUT_KINDS, "output.kind")
    tensor_data = _ensure_mapping(data.get("tensor"), "output.tensor")

    shape_raw = tensor_data.get("shape", [1])
    if not isinstance(shape_raw, (list, tuple)) or not shape_raw:
        raise InferenceProfileValidationError("output.tensor.shape must be a non-empty list")
    shape = tuple(int(d) for d in shape_raw)
    if any(d <= 0 for d in shape):
        raise InferenceProfileValidationError("output.tensor.shape dimensions must be > 0")

    timestep_policy = _validate_choice(
        data.get("timestep_policy", "obs_plus_one"),
        SUPPORTED_TIMESTEP_POLICIES,
        "output.timestep_policy",
    )

    return OutputConfig(
        kind=kind,
        tensor=TensorOutputConfig(
            dtype=str(tensor_data.get("dtype", "float32")),
            shape=shape,
        ),
        timestep_policy=timestep_policy,
    )


def _parse_validation(data: dict[str, Any]) -> ValidationConfig:
    max_latency_ms = int(data.get("max_latency_ms", 2000))
    if max_latency_ms <= 0:
        raise InferenceProfileValidationError("validation.max_latency_ms must be > 0")

    return ValidationConfig(
        require_nonempty_images=bool(data.get("require_nonempty_images", True)),
        max_latency_ms=max_latency_ms,
    )


def load_inference_profile(path: Path | str | None = None) -> InferenceProfile:
    """Load and validate an inference profile YAML.

    If path is None, returns a legacy-compatible default profile.
    """
    if path is None or str(path).strip() == "":
        return InferenceProfile()

    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Inference profile file not found: {profile_path}")

    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise InferenceProfileValidationError(
            f"Invalid YAML in inference profile {profile_path}: {e}"
        ) from e

    root = _ensure_mapping(raw, "root")
    version = int(root.get("version", 1))
    if version != 1:
        raise InferenceProfileValidationError(
            f"Unsupported inference profile version {version}. Only version=1 is supported"
        )

    return InferenceProfile(
        version=version,
        server=_parse_server(_ensure_mapping(root.get("server"), "server")),
        backend=_parse_backend(_ensure_mapping(root.get("backend"), "backend")),
        inputs=_parse_inputs(_ensure_mapping(root.get("inputs"), "inputs")),
        inference=_parse_inference(_ensure_mapping(root.get("inference"), "inference")),
        postprocess=_parse_postprocess(_ensure_mapping(root.get("postprocess"), "postprocess")),
        output=_parse_output(_ensure_mapping(root.get("output"), "output")),
        validation=_parse_validation(_ensure_mapping(root.get("validation"), "validation")),
    )
