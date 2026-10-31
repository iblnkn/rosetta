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

"""The seam a policy runner implements.

:class:`PolicyRunner` drives a policy against a live robot, through the
:class:`RunnerHost` its node provides, and reports through the value types
here. See :mod:`rosetta.policies` for how a package registers one.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    Keeps the host's feedback message separate from adapter internals. The
    counters must stay non-negative: hosts may carry them as unsigned fields
    (the ROS 2 host uses ``uint32``).
    """

    queue_depth: int = 0
    published_actions: int = 0


@dataclass(frozen=True, slots=True)
class RunnerResult:
    """Terminal outcome of a :meth:`PolicyRunner.run` call.

    ``success`` means the run terminated without error. A run ended by a
    host-requested stop is a *successful* run (conventional ``message``:
    ``"Stopped"``), as is one whose control loop finished naturally
    (``"Completed"``). Reserve ``success=False`` for actual failures
    (connect errors, inference errors, ...).

    The host maps this onto the action result's ``termination_reason``
    (see ``PolicyRunnerNode._run``): ``success=True`` becomes whichever stop
    reason the node recorded, or ``"completed"`` when nobody asked it to stop;
    ``success=False`` becomes ``"error"``. A runner never needs to know why it
    was stopped -- only that it was.
    """

    success: bool
    message: str = ""


# =============================================================================
# Protocols
# =============================================================================


class RunnerHost(Protocol):
    """What a :class:`PolicyRunner` gets from the process hosting it: configuration and a logger.

    A runner declares each setting it reads, with a default, then reads it by
    name. The host decides where values come from: the ROS 2 host
    (``rosetta_ros``) maps them onto node parameters, so launch files and
    params files set them. Hosts may declare shared settings first (e.g.
    ``contract_path``, ``pretrained_name_or_path``), which is why
    :meth:`declare` must be a no-op for a name that exists. Deliberately not
    ``runtime_checkable``: it exists for type checking, not isinstance dispatch.
    """

    def declare(self, name: str, default: Any, description: str = "", read_only: bool = False) -> None:
        """Declare setting ``name`` with ``default``, unless it is already declared."""
        ...

    def get(self, name: str) -> Any:
        """The current value of declared setting ``name``."""
        ...

    @property
    def logger(self) -> Any:
        """A logger with ``debug``, ``info``, ``warning`` and ``error`` methods."""
        ...


@runtime_checkable
class PolicyRunner(Protocol):
    """Drives a policy against a live robot's :class:`~rosetta.frames.protocols.FrameIO`.

    All framework specifics (server/checkpoint params, control-loop strategy)
    live here so the hosting node stays framework-agnostic. The node calls
    :meth:`setup` once (the runner declares and reads its own settings on the host),
    then :meth:`run` per goal. The node polls :meth:`feedback` on its own
    cadence. To stop a run it BOTH sets the run's ``stop_event`` AND calls
    :meth:`request_stop` — see those docstrings for why both exist.

    Threading: :meth:`feedback` and :meth:`request_stop` are called from
    other threads while :meth:`run` blocks; implementations must make those
    two safe against a concurrent ``run()``.

    Each ``run()`` owns whatever execution model the framework needs. The
    LeRobot impl delegates to ``RobotClient``'s async control loop; other
    implementations run their own fps loops.
    """

    def setup(self, host: RunnerHost, contract: Contract) -> None:
        """Declare and read runner settings on ``host`` and prepare to run.

        Called after the contract is loaded (e.g. in the ROS 2 node's configure
        transition). May run again after a cleanup -> configure lifecycle
        cycle; :meth:`RunnerHost.declare` is a no-op for names that exist, so
        neither a second pass nor a host-declared setting conflicts.
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
