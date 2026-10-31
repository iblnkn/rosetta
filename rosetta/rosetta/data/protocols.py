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

"""The seam a dataset writer implements.

:class:`DatasetWriter` consumes frame dicts offline and writes a dataset in
its framework's format. See :mod:`rosetta.data` for how a package registers
one.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rosetta.contract.schema import Contract

__all__ = ["DatasetWriter"]


@runtime_checkable
class DatasetWriter(Protocol):
    """Consumes frame dicts and writes a dataset.

    Lifecycle: ``open()`` once, then per episode a run of ``add_frame()`` calls
    ending with ``save_episode()`` — or ``discard_episode()`` if the episode
    fails partway — then ``finalize()`` once at the end. If porting fails
    partway, the host does NOT call ``finalize()`` (the porter raises first);
    a writer must not depend on ``finalize()`` for the correctness of
    already-saved episodes.

    A frame dict maps each contract key to a ``np.ndarray`` (or ``str`` for
    task-like values), plus the boundary markers ``is_first``/``is_last``/
    ``is_terminal`` (``(1,)`` bool arrays) and a ``task`` string.
    :func:`rosetta_ros.offline.bag_frames.iter_bag_frames` produces
    exactly this shape.
    """

    def open(
        self,
        *,
        contract: Contract,
        repo_id: str,
        root: Path | None = None,
        contract_path: Path | None = None,
        embed_contract: bool = True,
        **opts: Any,
    ) -> None:
        """Translate the contract into the framework's schema and open output.

        The writer derives whichever spec view it needs from ``contract``:
        :func:`rosetta.contract.specs.iter_specs` is the full recording view —
        exactly the keys the porter feeds :meth:`add_frame` (frameworks that
        record every column, e.g. LeRobot, want this); ``iter_policy_specs``
        is the policy I/O view (frameworks that concatenate state/action
        vectors want this, so extended/teleop columns never leak into them).

        When ``embed_contract`` is true and ``contract_path`` is given, the
        writer must copy that contract YAML into its output so tooling can
        recover it later (see
        :func:`rosetta.contract.sidecar.find_contract_for_pretrained`);
        the location is framework-defined (LeRobot-layout datasets use
        ``meta/rosetta_contract.yaml``). Framework-specific options arrive
        via ``**opts``.
        """
        ...

    def add_frame(self, frame: dict[str, Any]) -> None:
        """Append one resampled frame dict to the current episode."""
        ...

    def save_episode(self) -> None:
        """Flush the buffered episode."""
        ...

    def discard_episode(self) -> None:
        """Drop the partially buffered episode after an episode-level failure.

        Buffered frames must not leak into the next episode. Must be safe to
        call when nothing is buffered.
        """
        ...

    def finalize(self) -> None:
        """Write top-level metadata and close the dataset."""
        ...
