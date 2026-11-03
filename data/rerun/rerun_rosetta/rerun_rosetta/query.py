# Copyright 2026 Isaac Blankenau
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

"""Compile a rosetta contract into a Rerun dataloader query.

The porter and the live bridge project the contract into eager frame
producers. This projects the same specs into dataloader ``Field``s on a
fixed-rate grid that the catalog server evaluates lazily.

Where each stream's messages sit in a recording comes from the robot
interface's :class:`~.layout.RecordingLayout`; this module knows none.

Limits are enforced at compile time rather than approximated: only ``hold``
alignment (the server's latest-at fill is hold; ``asof``/``drop`` have no
server-side equivalent), and only the recording's receive timeline. The porter
anchors its tick grid at the bag's first receive time whatever timeline a
stream aligns on, while the dataloader anchors the grid on the index timeline
itself, so a query indexed on any other timeline would sample different ticks
than the porter.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rerun.experimental.dataloader import Field, FixedRateSampling

from rosetta.contract.schema import ResamplePolicy
from rosetta.contract.specs import iter_observation_specs, iter_policy_specs

from .decoders import ContractDecoder

if TYPE_CHECKING:
    from rosetta.contract.schema import Contract
    from rosetta.contract.specs import StreamSpec

    from .layout import RecordingLayout


@dataclass(frozen=True)
class CompiledQuery:
    """A contract projected onto the dataloader's query surface.

    ``fields`` are keyed by spec position (``f0``, ``f1``, ...) because contract
    keys are not unique across specs (several topics may feed one key). A spec
    whose topic was not recorded has no field and zero-fills.
    """

    specs: tuple[StreamSpec, ...]
    index: str
    sampling: FixedRateSampling
    fields: dict[str, Field]
    required: tuple[str, ...]  # observation fields: a sample needs all of them (the porter's warmup gate)
    windows: dict[str, tuple[int, int]]  # contract key -> inclusive (start, end) grid offsets


def field_key(position: int) -> str:
    return f"f{position}"


def compile_contract(
    contract: Contract,
    columns: Collection[str],
    layout: RecordingLayout,
    *,
    windows: dict[str, tuple[int, int]] | None = None,
) -> CompiledQuery:
    """Project ``contract`` onto a dataset whose schema has ``columns``. Raises on unsupported contracts.

    ``layout`` is the contract's robot interface's recording layout (see
    :func:`~.layout.load_layout`).

    ``windows`` maps contract keys to inclusive ``(start, end)`` offsets in grid
    steps, e.g. ``{"action": (0, 49)}`` for a 50-step action chunk. A windowed
    key's frame value stacks one value per offset. The dataloader trims each
    episode so that every window lies inside it, so nothing is ever padded.

    Each stream reads the column the layout names. A missing observation topic
    raises, since its frames would be all zeros; any other missing topic warns
    and zero-fills, as in the porter.
    """
    windows = dict(windows or {})
    specs = tuple(iter_policy_specs(contract))
    if not specs:
        raise ValueError("Contract has no policy streams to compile")

    unknown = sorted(set(windows) - {s.key for s in specs})
    if unknown:
        raise ValueError(f"windows refer to keys not in the contract's policy streams: {unknown}")
    for key, (start, end) in windows.items():
        if start > end:
            raise ValueError(f"windows[{key!r}]: start {start} > end {end}")

    unsupported = sorted(
        f"{s.key} ({s.source.align.strategy})" for s in specs if s.source.align.strategy != ResamplePolicy.HOLD
    )
    if unsupported:
        raise ValueError(
            f"Rerun-backed training supports only 'hold' alignment; streams using asof/drop: {unsupported}. "
            f"Change the contract or use the offline porter for this dataset."
        )

    receive = layout.receive_timeline
    not_receive = sorted(f"{s.key} ({s.source.align.timeline})" for s in specs if s.source.align.timeline != receive)
    if not_receive:
        raise ValueError(
            f"Rerun-backed training supports only '{receive}' alignment; streams aligned otherwise: {not_receive}. "
            f"Change the contract or use the offline porter for this dataset."
        )

    columns = set(columns)
    observation_keys = {s.key for s in iter_observation_specs(contract)}
    fields: dict[str, Field] = {}
    required: list[str] = []
    missing: list[str] = []
    for position, spec in enumerate(specs):
        channel = spec.source.channel
        path = layout.column(channel, columns)
        if path is None:
            if spec.key in observation_keys:
                missing.append(f"{spec.key} ({channel.topic} as {channel.type})")
            else:
                logging.warning("Topic %s not in dataset, %s will be zero-filled", channel.topic, spec.key)
            continue

        key = field_key(position)
        window = windows.get(spec.key)
        fields[key] = Field(
            path=path,
            decode=ContractDecoder(spec, layout.message),
            window=tuple(step / contract.fps for step in range(window[0], window[1] + 1)) if window else None,
        )
        if spec.key in observation_keys:
            required.append(key)

    if missing:
        raise ValueError(
            f"Observation topics missing from the recordings: {missing}. Check the contract's topic names and "
            f"types against what was recorded."
        )

    return CompiledQuery(
        specs=specs,
        index=layout.index,
        # Rerun rounds the period to whole ns; see the README for the drift from the porter's grid.
        sampling=FixedRateSampling(rate_hz=float(contract.fps)),
        fields=fields,
        required=tuple(required),
        windows=windows,
    )
