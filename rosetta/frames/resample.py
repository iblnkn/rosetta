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

"""Online stream resampling onto the clock defined by the contract's ``fps``.

StreamBuffer is the single resampler behind every place rosetta turns
asynchronous messages into fixed-rate frames: the live topic bridge, the bag
porter (which replays through the same buffers so bag and live frames match
sample-for-sample — see test_bag_live_parity), and warmup gating. One buffer
per stream; ``push()`` on message arrival, ``sample()`` on each frame tick,
per the spec's resample policy (hold / asof / drop).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from ..contract.schema import ResamplePolicy

if TYPE_CHECKING:
    from ..contract.specs import StreamSpec

# A buffered stamp this far in the future of the sampling clock is treated as
# a clock reset (sim restart, bag-replay loop) rather than skew. Cross-host
# NTP skew is normally well under 100 ms; real resets jump backwards by whole
# sessions (seconds+), so 1 s separates the two cleanly.
DEFAULT_RESET_TOL_NS = 1_000_000_000


class StreamBuffer:
    """
    Thread-safe, constant-memory online resampler.

    Policies:
    - "hold": Always return the last value (default)
    - "asof": Return last value only if within tolerance window
    - "drop": Return last value only if it arrived within the step window

    A buffered timestamp slightly ahead of the sampling clock (multi-machine
    header stamps + clock skew) is served as the current value; only a jump
    beyond ``reset_tol_ns`` is treated as a clock reset and cleared.
    """

    def __init__(
        self,
        policy: str,
        step_ns: int,
        tol_ns: int = 0,
        reset_tol_ns: int | None = None,
    ):
        self.policy = policy
        self.step_ns = int(step_ns)
        self.tol_ns = int(tol_ns)
        # Floor of two frame periods keeps the tolerance meaningful at very
        # low contract rates (one period at 2 Hz already exceeds 0.5 s).
        self.reset_tol_ns = max(DEFAULT_RESET_TOL_NS, 2 * self.step_ns) if reset_tol_ns is None else int(reset_tol_ns)
        self.last_ts: int | None = None
        self.last_val: Any | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_spec(cls, spec: StreamSpec) -> "StreamBuffer":
        """Create a StreamBuffer from any resolved stream spec's align block."""
        step_ns = int(1e9 / spec.fps)
        tol_ns = spec.source.align.tolerance_ms * 1_000_000
        return cls(policy=spec.source.align.strategy, step_ns=step_ns, tol_ns=tol_ns)

    def push(self, ts_ns: int, val: Any) -> None:
        """Insert a sample (keeps the newest by timestamp)."""
        with self._lock:
            if self.last_ts is None or ts_ns >= self.last_ts:
                self.last_ts, self.last_val = ts_ns, val

    def sample(self, tick_ns: int) -> Any | None:
        """Sample according to policy at a given tick."""
        with self._lock:
            if self.last_ts is None:
                return None

            # Distinguish clock skew from a clock reset. A stamp slightly in
            # the future (sensor host clock ahead) is the freshest data —
            # serve it as age 0. A jump beyond reset_tol_ns means the sampling
            # clock was reset (sim restart): the buffered data is stale, clear.
            ahead = self.last_ts - tick_ns
            if ahead > self.reset_tol_ns:
                self._clear_unsafe()  # Already holding lock
                return None
            if ahead > 0:
                # Age 0 under every policy (hold trivially; drop: 0 < step;
                # asof: 0 <= tol). last_ts untouched so a persistently-ahead
                # sensor keeps serving each tick.
                return self.last_val

            if self.policy == ResamplePolicy.DROP.value:
                return self.last_val if (self.last_ts > tick_ns - self.step_ns) else None
            if self.policy == ResamplePolicy.ASOF.value:
                return self.last_val if (tick_ns - self.last_ts <= self.tol_ns) else None
            return self.last_val  # hold is default

    def _clear_unsafe(self) -> None:
        """Clear buffered data without acquiring lock (internal use only)."""
        self.last_ts = None
        self.last_val = None

    def reset(self) -> None:
        """Clear buffered data (e.g., between episodes or after sim reset)."""
        with self._lock:
            self._clear_unsafe()
