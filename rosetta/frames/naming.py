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
Frame-key naming helpers shared across the framework adapters.

Frame dicts are keyed by contract keys (``observation.images.cam``,
``observation.state``, ``action``, ...). These helpers classify and sanitize
those keys so every framework adapter derives the same names instead of
re-deriving them in parallel.
"""

import re


def classify_key(key: str) -> str:
    """Classify a contract key by prefix.

        observation.images.*  -> 'image'
        action*               -> 'action'
        everything else       -> 'state'

    Shared by the framework adapters so they don't each re-derive it.
    """
    if key.startswith("observation.images."):
        return "image"
    if key.startswith("action"):
        return "action"
    return "state"


def sanitize_field_name(name: str) -> str:
    """Reduce a contract key to a dot-free [A-Za-z0-9_] name."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def camera_short_name(key: str) -> str:
    """Short camera name for an ``observation.images.*`` key.

    Shared by the framework adapters so they agree on the derivation.
    Sanitized because adapters embed it in filenames (e.g. WebDataset tar
    members, where a dot would split the sample key).
    """
    return sanitize_field_name(key.removeprefix("observation.images."))
