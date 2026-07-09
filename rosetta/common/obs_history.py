"""Observation-with-history wrapper for async inference.

Upstream ``lerobot.async_inference.TimedObservation`` carries one snapshot of
the robot's state plus its timestep. In async mode the policy server only
receives a new observation at chunk boundaries (~once per second), so its
internal ``_queues`` deque accumulates obs that are ~1 s apart instead of the
training-time 33 ms spacing. For policies with ``n_obs_steps > 1`` (e.g.
diffusion), this feeds stale temporal context to the denoiser.

This subclass lets the client attach a rolling window of the most recent
``n`` raw observations (captured at control rate) so the server can rebuild
the correct ``n_obs_steps`` history instead of falling back on its stale deque.
"""

from dataclasses import dataclass, field

from lerobot.async_inference.helpers import RawObservation, TimedObservation


@dataclass
class TimedObservationWithHistory(TimedObservation):
    """``TimedObservation`` plus a rolling history captured at control rate.

    ``history`` is ordered oldest-first and includes the current observation
    as its last element. Length is at most the client's configured window;
    may be shorter during the first few control-loop ticks while the buffer
    is filling.
    """

    history: list[RawObservation] = field(default_factory=list)
