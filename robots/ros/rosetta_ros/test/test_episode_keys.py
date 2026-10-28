# Copyright 2026 Franco Cipollone
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

"""Key decoding and the prompt editor behind episode_keyboard_node."""

import pytest

from rosetta_ros.nodes.episode_keys import KeyDecoder, PromptEditor


def _commands(data: bytes) -> list[str]:
    decoder = KeyDecoder()
    return [c for c in map(decoder.feed, data) if c is not None]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"rR", ["start"] * 2),
        (b"sS", ["stop"] * 2),
        (b"dD\x7f\x08", ["discard"] * 4),
        (b"tT", ["edit"] * 2),
        (b"hH?", ["help"] * 3),
        (b"qQ", ["quit"] * 2),
        (b"x1 ", []),
        (b"\x1b[C", ["start"]),
        (b"\x1b[D", ["stop"]),
        (b"\x1b[A", []),  # up arrow: no command
        (b"\x1bq", ["quit"]),  # ESC then a plain key reads the key
        (b"\x1b[Cr", ["start", "start"]),  # decoder resets after a sequence
    ],
)
def test_key_decoder(data, expected):
    assert _commands(data) == expected


def _edit(start: str, data: bytes) -> tuple[PromptEditor, list]:
    editor = PromptEditor()
    editor.begin(start)
    return editor, [editor.feed(ch) for ch in data]


@pytest.mark.parametrize(
    ("start", "data", "text"),
    [
        ("", b"pick cube", "pick cube"),
        ("cube", b"\x7f\x7f", "cu"),
        ("", b"\x7f", ""),  # backspace on empty text
        ("ab", b"\x08c", "ac"),
        ("", b"a\x01\x1fb", "ab"),  # control bytes are ignored
    ],
)
def test_prompt_editor_text(start, data, text):
    editor, outcomes = _edit(start, data)
    assert editor.text == text
    assert editor.active
    assert set(outcomes) == {None}


@pytest.mark.parametrize(("end", "outcome"), [(b"\r", "applied"), (b"\n", "applied"), (b"\x1b", "cancelled")])
def test_prompt_editor_ends(end, outcome):
    editor, outcomes = _edit("old", b"x" + end)
    assert outcomes[-1] == outcome
    assert not editor.active
