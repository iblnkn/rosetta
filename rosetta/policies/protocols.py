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

"""The two seams a policy framework adapter implements.

:class:`DatasetWriter` (offline) and :class:`PolicyRunner` (online), plus
the value types the runner reports through. See :mod:`rosetta.policies` for
how an adapter registers an implementation of these.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..frames.protocols import FrameIO

if TYPE_CHECKING:
    import threading

    from rosetta.contract.schema import Contract
    from rosetta.contract.specs import StreamSpec


# =============================================================================
# Runner value types
# =============================================================================


@dataclass(frozen=True, slots=True)
class RunnerFeedback:
    """Progress snapshot a :class:`PolicyRunner` emits during execution.

    Keeps the node's ROS feedback message separate from adapter internals.
    """

    queue_depth: int = 0
    published_actions: int = 0
    status: str = "executing"


@dataclass(frozen=True, slots=True)
class RunnerResult:
    """Terminal outcome of a :meth:`PolicyRunner.run` call."""

    success: bool
    message: str = ""


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class DatasetWriter(Protocol):
    """Consumes frame dicts and writes a dataset.

    Lifecycle: ``open()`` once, then per episode a run of ``add_frame()`` calls
    ending with ``save_episode()`` — or ``discard_episode()`` if the episode
    fails partway — then ``finalize()`` once at the end.

    A frame dict is ``{contract_key: np.ndarray | str}`` plus the boundary
    markers ``is_first``/``is_last``/``is_terminal`` ((1,) bool arrays) and a
    ``task`` string. This is what
    :func:`rosetta.robots.ros2.offline.bag_frames.iter_bag_frames` yields.
    """

    def open(
        self,
        *,
        contract: "Contract",
        specs: "list[StreamSpec]",
        repo_id: str,
        root: Path | None = None,
        **opts: Any,
    ) -> None:
        """Translate contract/specs into the framework's schema and open output."""
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


@runtime_checkable
class PolicyRunner(Protocol):
    """Drives a policy against a live robot's :class:`FrameIO`.

    All framework specifics (server/checkpoint params, control-loop strategy)
    live here so the hosting node stays framework-agnostic. The node calls
    :meth:`setup` once (the runner declares and reads its own ROS parameters),
    then :meth:`run` per goal. The node polls :meth:`feedback` on its own
    cadence and may call :meth:`request_stop` to interrupt mid-run (e.g.
    cancel).

    Each ``run()`` owns whatever execution model the framework needs. The
    LeRobot impl delegates to ``RobotClient``'s async control loop; the
    vla_foundry impl runs its own fps loop.
    """

    def setup(self, node: Any, contract: "Contract") -> None:
        """Declare/read runner ROS parameters on ``node`` and prepare to run.

        Called once after the contract is loaded (e.g. in the node's configure
        transition). ``node`` is the hosting ``rclpy`` lifecycle node.
        """
        ...

    def run(
        self,
        frames: FrameIO,
        *,
        task: str,
        stop_event: "threading.Event",
    ) -> RunnerResult:
        """Execute the policy until ``stop_event`` is set or it finishes (blocks)."""
        ...

    def feedback(self) -> RunnerFeedback:
        """Current progress snapshot (the node publishes this on its own cadence)."""
        ...

    def request_stop(self) -> None:
        """Interrupt an in-progress :meth:`run` (e.g. a cancel request)."""
        ...

    def teardown(self) -> None:
        """Release any framework resources (e.g. stop a server subprocess)."""
        ...
