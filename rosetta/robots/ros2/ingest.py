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

"""StreamIngest: the shared message-ingest policy for live bridge and bag porter.

Everything between "a serialized message arrived" and "a value sits in the
stream buffer" lives here: timeline extraction, decode, and push. The live
bridge and the bag porter both go through this one implementation — like
FrameLayout for frame assembly — so recorded and live frames agree by
construction instead of by parallel copies.

Pipeline-order note: the contract reads channel -> align -> select -> apply,
where align chooses the timeline at sample time. Select/apply are pure
per-message transforms, so they commute with align — running them here at
ingest (once per message, before buffering) is observationally identical to
running them after the tick pick, and cheaper.

Timeline policy: ``spec.source.align.timeline`` names the timeline the contract
chose; contract loading already validated the channel provides it. A message
that still arrives without it (e.g. an uninitialized header stamp) is
dropped with a once-per-stream warning (plus a recovery notice) — never
ingested on a fabricated timeline.

Decode policy (identical live and offline): a malformed message must not
kill the caller — live that would take down the whole inference node's
executor, and offline the port must produce exactly the frames the live
bridge would have served. Drop the message with a once-per-stream warning
(plus a recovery notice); the stream then behaves like a missing stream
(zero-fill + transition logging downstream).

Stream identity is the spec's position in the resolved spec list — specs are
unhashable and (key, topic) is not unique (one topic may feed several specs).
"""

from __future__ import annotations

from typing import Any, Callable

from rosetta.contract.specs import StreamSpec
from rosetta.frames.codecs import decode_value
from rosetta.frames.resample import StreamBuffer
from rosetta.robots.ros2.timelines import get_message_timestamp_ns


class StreamIngest:
    """Per-session ingest state: warn-once bookkeeping over stream indices."""

    def __init__(self, warn: Callable[[str], None], info: Callable[[str], None]):
        self._warn = warn
        self._info = info
        self._timeline_dropped: set[int] = set()
        self._decode_warned: set[int] = set()

    def reset(self) -> None:
        """Forget warn-once state (bridge configure/teardown, new episode)."""
        self._timeline_dropped.clear()
        self._decode_warned.clear()

    def ingest(
        self,
        msg: Any,
        spec: StreamSpec,
        buffer: StreamBuffer,
        index: int,
        receive_ns: int,
    ) -> None:
        """Extract the chosen timeline's timestamp, decode, and push one message.

        ``receive_ns`` is the receive time (node clock live, bag timestamp
        offline) — the 'receive' timeline's value for this message. The two
        are receive-time approximations, not the same instant, so only
        'header'-timeline streams replay bit-exactly offline.
        """
        ts_ns = get_message_timestamp_ns(msg, spec, receive_ns)
        if ts_ns is None:
            if index not in self._timeline_dropped:
                self._warn(
                    f"Message on '{spec.key}' ({spec.source.channel.topic}) is missing its "
                    f"'{spec.source.align.timeline}' timeline; dropping message"
                )
                self._timeline_dropped.add(index)
            return
        if index in self._timeline_dropped:
            self._info(
                f"Timeline '{spec.source.align.timeline}' recovered for '{spec.key}' ({spec.source.channel.topic})"
            )
            self._timeline_dropped.discard(index)

        try:
            value = decode_value(msg, spec)
        except Exception as e:
            if index not in self._decode_warned:
                self._warn(
                    f"Decode failed for '{spec.key}' on {spec.source.channel.topic} "
                    f"({type(e).__name__}: {e}); dropping message"
                )
                self._decode_warned.add(index)
            return
        if index in self._decode_warned:
            self._info(f"Decode recovered for '{spec.key}' ({spec.source.channel.topic})")
            self._decode_warned.discard(index)

        buffer.push(ts_ns, value)
