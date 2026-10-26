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

"""Built-in codecs of the fake robot interface: select names fields of a FakeVector."""

import numpy as np
from fake_interface import VECTOR, FakeVector

from rosetta.frames.codecs import register_decoder, register_encoder


@register_decoder(VECTOR, dtype="float64", requires_select=True)
def decode_vector(msg, spec):
    return np.array([msg.fields[name] for name in spec.source.select], dtype=np.float64)


@register_encoder(VECTOR, requires_select=True)
def encode_vector(action_vec, spec, stamp_ns=None):
    return FakeVector({name: float(v) for name, v in zip(spec.source.select, action_vec, strict=True)})
