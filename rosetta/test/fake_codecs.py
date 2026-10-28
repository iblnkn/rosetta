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

"""Built-in codecs of the fake robot interface.

select names the fields of a FakeVector; fake/Data and fake/Text take no select.
"""

import numpy as np
from fake_interface import DATA, TEXT, VECTOR, FakeData, FakeVector

from rosetta.frames.codecs import register_decoder, register_encoder


@register_decoder(VECTOR, dtype="float64", requires_select=True)
def decode_vector(msg, spec):
    return np.array([msg.fields[name] for name in spec.source.select], dtype=np.float64)


@register_encoder(VECTOR, requires_select=True)
def encode_vector(action_vec, spec, stamp_ns=None):
    return FakeVector({name: float(v) for name, v in zip(spec.source.select, action_vec, strict=True)})


@register_decoder(DATA, dtype="float32")
def decode_data(msg, spec):
    return np.atleast_1d(np.asarray(msg.data, dtype=np.float32))


@register_encoder(DATA)
def encode_data(action_vec, spec, stamp_ns=None):
    return FakeData(float(action_vec[0]))


@register_decoder(TEXT, dtype="string")
def decode_text(msg, spec):
    return msg
