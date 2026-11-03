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

"""Rerun dataset backend for Rosetta.

Train directly from ``.rrd`` recordings registered to a Rerun catalog server.
The contract compiles into a dataloader query (:func:`compile_contract`)
whose decoders reuse rosetta's own ``decode_value``, so frames match the
offline porter and the live bridge by construction. The contract's robot
interface supplies the recording's layout (:class:`RecordingLayout`); this
package knows no robot middleware.

Registration is Rerun's own catalog API::

    from rerun.catalog import CatalogClient
    from rerun_rosetta import contract_dataset

    client = CatalogClient("rerun+http://127.0.0.1:51234")
    client.create_dataset("my_task", exist_ok=True).register_prefix("file:///abs/path/bags").wait()
    frames = contract_dataset("rerun+http://127.0.0.1:51234", "my_task", "contract.yaml")
    loader = torch.utils.data.DataLoader(frames, batch_size=8)
"""

from .dataset import ContractFrames, contract_dataset
from .layout import RecordingLayout, load_layout
from .query import CompiledQuery, compile_contract
from .stats import load_stats, stats_layer_name, write_stats_layer

__all__ = [
    "CompiledQuery",
    "ContractFrames",
    "RecordingLayout",
    "compile_contract",
    "contract_dataset",
    "load_layout",
    "load_stats",
    "stats_layer_name",
    "write_stats_layer",
]
