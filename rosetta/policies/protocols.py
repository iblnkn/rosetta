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

if TYPE_CHECKING:
    import threading

    from rosetta.contract.schema import Contract

    from ..frames.protocols import FrameIO


# =============================================================================
# Runner value types
# =============================================================================


@dataclass(frozen=True, slots=True)
class RunnerFeedback:
    """Progress snapshot a :class:`PolicyRunner` emits during execution.

    Keeps the node's ROS feedback message separate from adapter internals.
    The counters map onto ``uint32`` ROS feedback fields and must stay
    non-negative; ``status`` is a short free-form display label
    (conventionally ``"executing"`` or ``"idle"``), not an enum.
    """

    queue_depth: int = 0
    published_actions: int = 0
    status: str = "executing"


@dataclass(frozen=True, slots=True)
class RunnerResult:
    """Terminal outcome of a :meth:`PolicyRunner.run` call.

    ``success`` means the run terminated without error. A run ended by a
    host-requested stop is a *successful* run (conventional ``message``:
    ``"Stopped"``), as is one whose control loop finished naturally
    (``"Completed"``). Reserve ``success=False`` for actual failures
    (connect errors, inference errors, ...).
    """

    success: bool
    message: str = ""


# =============================================================================
# Protocols
# =============================================================================


class NodeLike(Protocol):
    """The slice of the hosting ``rclpy`` node a :class:`PolicyRunner` may use.

    Adapters read their configuration through ROS parameters on the hosting
    node; pinning that surface to these four methods keeps the seam typed
    without importing ``rclpy``. Deliberately not ``runtime_checkable`` —
    it exists for type checking, not isinstance dispatch.
    """

    def declare_parameter(self, name: str, value: Any = ..., descriptor: Any = ...) -> Any: ...

    def has_parameter(self, name: str) -> bool: ...

    def get_parameter(self, name: str) -> Any: ...

    def get_logger(self) -> Any: ...


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
    :func:`rosetta.robots.ros2.offline.bag_frames.iter_bag_frames` produces
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


@runtime_checkable
class PolicyRunner(Protocol):
    """Drives a policy against a live robot's :class:`~rosetta.frames.protocols.FrameIO`.

    All framework specifics (server/checkpoint params, control-loop strategy)
    live here so the hosting node stays framework-agnostic. The node calls
    :meth:`setup` once (the runner declares and reads its own ROS parameters),
    then :meth:`run` per goal. The node polls :meth:`feedback` on its own
    cadence. To stop a run it BOTH sets the run's ``stop_event`` AND calls
    :meth:`request_stop` — see those docstrings for why both exist.

    Threading: :meth:`feedback` and :meth:`request_stop` are called from
    other threads while :meth:`run` blocks; implementations must make those
    two safe against a concurrent ``run()``.

    Each ``run()`` owns whatever execution model the framework needs. The
    LeRobot impl delegates to ``RobotClient``'s async control loop; the
    vla_foundry impl runs its own fps loop.
    """

    def setup(self, node: NodeLike, contract: Contract) -> None:
        """Declare/read runner ROS parameters on ``node`` and prepare to run.

        Called after the contract is loaded (e.g. in the node's configure
        transition). ``node`` is the hosting ``rclpy`` lifecycle node. May run
        again after a cleanup -> configure lifecycle cycle, and the host may
        have pre-declared shared parameters (e.g. ``pretrained_name_or_path``)
        -- guard every declaration with ``has_parameter`` so neither a second
        pass nor a host-declared parameter makes ``declare_parameter`` raise.
        """
        ...

    def run(
        self,
        frames: FrameIO,
        *,
        task: str,
        stop_event: threading.Event,
    ) -> RunnerResult:
        """Execute the policy until ``stop_event`` is set or it finishes (blocks).

        ``stop_event`` is the cooperative stop signal every control loop must
        poll; the host pairs it with :meth:`request_stop` to unblock any
        blocking I/O. A run ended this way returns a *successful*
        :class:`RunnerResult` (see its docstring).
        """
        ...

    def feedback(self) -> RunnerFeedback:
        """Current progress snapshot (the node publishes this on its own cadence)."""
        ...

    def request_stop(self) -> None:
        """Best-effort unblock of an in-progress :meth:`run`.

        Hosts call this alongside setting ``stop_event`` (never instead of
        it) so a run blocked in I/O (a socket ``recv``, a gRPC stream) still
        observes the stop promptly. May be called while no run is active
        (e.g. a late cancel); any state it latches must be cleared by
        :meth:`run` on entry, so a stray idle-time call cannot poison the
        next run.
        """
        ...

    def teardown(self) -> None:
        """Release any framework resources (e.g. stop a server subprocess)."""
        ...
