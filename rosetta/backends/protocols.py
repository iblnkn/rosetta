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
Backend protocols and resolution.

Two protocols (:class:`DatasetWriter`, :class:`PolicyRunner`) define what a
backend provides, plus the value types the runner reports through. Backends are
resolved by name from setuptools entry-point groups so ``rosetta`` core never
imports a specific framework.
"""

from __future__ import annotations

import importlib.metadata as _ilm
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    import threading

    from rosetta.core.contract import Contract, StreamSpec
    from rosetta.ros2.topic_bridge import TopicBridge


# =============================================================================
# Entry-point groups
# =============================================================================

DATASET_WRITER_GROUP = 'rosetta.dataset_writers'
"""Entry-point group a backend registers its :class:`DatasetWriter` under."""

POLICY_RUNNER_GROUP = 'rosetta.policy_runners'
"""Entry-point group a backend registers its :class:`PolicyRunner` under."""


# =============================================================================
# Runner value types
# =============================================================================


@dataclass(frozen=True, slots=True)
class RunnerFeedback:
    """Progress snapshot a :class:`PolicyRunner` emits during execution.

    Keeps the node's ROS feedback message separate from backend internals.
    """

    queue_depth: int = 0
    published_actions: int = 0
    status: str = 'executing'


@dataclass(frozen=True, slots=True)
class RunnerResult:
    """Terminal outcome of a :meth:`PolicyRunner.run` call."""

    success: bool
    message: str = ''


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class DatasetWriter(Protocol):
    """Consumes frame dicts and writes a dataset.

    Lifecycle: ``open()`` once, then per episode a run of ``add_frame()`` calls
    ending with ``save_episode()``, then ``finalize()`` once at the end.

    A frame dict is ``{contract_key: np.ndarray | str}`` plus the boundary
    markers ``is_first``/``is_last``/``is_terminal`` ((1,) bool arrays) and a
    ``task`` string. This is what
    :func:`rosetta.ros2.bag_frames.iter_bag_frames` yields.
    """

    def open(
        self,
        *,
        contract: 'Contract',
        specs: 'list[StreamSpec]',
        repo_id: str,
        root: Path | None = None,
        **opts: Any,
    ) -> None:
        """Translate contract/specs into the backend's schema and open output."""
        ...

    def add_frame(self, frame: dict[str, Any]) -> None:
        """Append one resampled frame dict to the current episode."""
        ...

    def save_episode(self) -> None:
        """Flush the buffered episode."""
        ...

    def finalize(self) -> None:
        """Write top-level metadata and close the dataset."""
        ...


@runtime_checkable
class PolicyRunner(Protocol):
    """Drives a policy against a live ``TopicBridge``.

    All backend specifics (server/checkpoint params, control-loop strategy) live
    here so the hosting node stays framework-agnostic. The node calls
    :meth:`setup` once (the runner declares and reads its own ROS parameters),
    then :meth:`run` per goal. The node polls :meth:`feedback` on its own cadence
    and may call :meth:`request_stop` to interrupt mid-run (e.g. cancel).

    Each ``run()`` owns whatever execution model the backend needs. The LeRobot
    impl delegates to ``RobotClient``'s async control loop; the vla_foundry impl
    runs its own fps loop.
    """

    def setup(self, node: Any, contract: 'Contract') -> None:
        """Declare/read backend ROS parameters on ``node`` and prepare to run.

        Called once after the contract is loaded (e.g. in the node's configure
        transition). ``node`` is the hosting ``rclpy`` lifecycle node.
        """
        ...

    def run(
        self,
        bridge: 'TopicBridge',
        *,
        task: str,
        stop_event: 'threading.Event',
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
        """Release any backend resources (e.g. stop a server subprocess)."""
        ...


# =============================================================================
# Resolution by name (entry points)
# =============================================================================


def _load_entry_point(group: str, name: str) -> Any:
    """Load the object a backend registered under ``group`` as ``name``."""
    try:
        eps = _ilm.entry_points(group=group)
    except TypeError:
        # importlib.metadata < 3.10 dict API (ROS 2 Jazzy is 3.12; defensive).
        eps = _ilm.entry_points().get(group, [])

    matches = {ep.name: ep for ep in eps}
    ep = matches.get(name)
    if ep is None:
        known = ', '.join(sorted(matches)) or '(none installed)'
        raise ValueError(
            f"No backend '{name}' registered under entry-point group '{group}'. "
            f'Available: {known}.'
        )
    return ep.load()


def load_dataset_writer(name: str) -> 'DatasetWriter':
    """Instantiate the ``DatasetWriter`` registered as ``name``."""
    cls = _load_entry_point(DATASET_WRITER_GROUP, name)
    return cls()


def load_policy_runner(name: str) -> 'PolicyRunner':
    """Instantiate the ``PolicyRunner`` registered as ``name``."""
    cls = _load_entry_point(POLICY_RUNNER_GROUP, name)
    return cls()


def available_backends(group: str) -> list[str]:
    """List backend names registered under an entry-point ``group``."""
    try:
        eps = _ilm.entry_points(group=group)
    except TypeError:
        eps = _ilm.entry_points().get(group, [])
    return sorted(ep.name for ep in eps)
