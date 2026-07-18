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

"""Frame-key naming helpers shared across the framework adapters.

Frame dicts are keyed by contract keys (``observation.images.cam``,
``observation.state``, ``action``, ...). Two derivations live here.
:func:`camera_name` strips the image prefix to the semantic camera name and
keeps dots (``observation.images.wrist.right`` -> ``wrist.right``). LeRobot and
GR00T consume that name directly. :func:`sanitize_field_name` flattens a name to
a bare ``[A-Za-z0-9_]`` identifier for sinks that need it, such as vla_foundry's
WebDataset tar members where a dot would split the sample key.

A key's role (state vs action) never comes from its spelling. It comes from the
spec's type, via ``KeyLayout.is_action`` in :mod:`rosetta.frames.layout`.
"""

import re

#: Contract-key prefix marking an image stream.
IMAGE_KEY_PREFIX = "observation.images."


def sanitize_field_name(name: str) -> str:
    """Flatten a name to a dot-free ``[A-Za-z0-9_]`` identifier.

    Every character outside that set becomes ``_``. Used by sinks that cannot
    carry dots, such as vla_foundry's WebDataset tar members.

    Args:
        name: A contract key or camera name.

    Returns:
        The name with each non-identifier character replaced by ``_``.

    """
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def camera_name(key: str) -> str:
    """Semantic camera name for an ``observation.images.*`` key.

    Strips the image prefix and returns the remainder verbatim, dots included
    (``observation.images.wrist.right`` -> ``wrist.right``). This is the
    framework-neutral identifier. LeRobot and GR00T use it as-is and keep the
    dotted hierarchy. A backend whose sink needs a flat identifier, such as
    vla_foundry's WebDataset tar members, flattens it via
    :func:`sanitize_field_name`.

    A key without the prefix is returned unchanged, since ``removeprefix`` is a
    no-op when the prefix is absent.

    Args:
        key: A contract frame key.

    Returns:
        The camera name with dots preserved.

    """
    return key.removeprefix(IMAGE_KEY_PREFIX)
