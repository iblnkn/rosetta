"""Backward-compatible shim for the auxiliary model server.

Deprecated: use `rosetta.common.aux_model_server` instead.
"""

from __future__ import annotations

from lerobot.async_inference.helpers import get_logger

from .aux_model_server import (
    AuxModelServer,
    ClassifierBackend,
    ClassifierServer,
    LerobotClassifierBackend,
    TransformersVLMBackend,
    main as _aux_main,
)

logger = get_logger("classifier_server")


def main() -> None:
    logger.warning(
        "Module 'rosetta.common.classifier_server' is deprecated; "
        "use 'rosetta.common.aux_model_server' instead."
    )
    _aux_main()


__all__ = [
    "AuxModelServer",
    "ClassifierServer",
    "ClassifierBackend",
    "LerobotClassifierBackend",
    "TransformersVLMBackend",
    "main",
]


if __name__ == "__main__":
    main()
