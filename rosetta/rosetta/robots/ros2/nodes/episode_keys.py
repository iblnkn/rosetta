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

"""Raw-terminal input for episode_keyboard_node: key decoding and the prompt editor.

Byte-level and ROS-free: both classes take one input byte at a time.
"""

from __future__ import annotations

_ESC = 27
_BACKSPACE = (127, 8)  # most terminals; Ctrl-H
_ENTER = (13, 10)  # CR, LF

# Arrow keys arrive as ESC [ C (right) and ESC [ D (left).
_ARROWS = {ord("C"): "start", ord("D"): "stop"}
_KEYS = {
    **dict.fromkeys(b"rR", "start"),
    **dict.fromkeys(b"sS", "stop"),
    **dict.fromkeys((*b"dD", *_BACKSPACE), "discard"),
    **dict.fromkeys(b"tT", "edit"),
    **dict.fromkeys(b"hH?", "help"),
    **dict.fromkeys(b"qQ", "quit"),
}


class KeyDecoder:
    """Decode key bytes into commands: start, stop, discard, edit, help, quit."""

    def __init__(self) -> None:
        self._escape = 0  # bytes of an ESC [ prefix seen so far

    def feed(self, ch: int) -> str | None:
        """Consume one byte; return the command it completes, if any."""
        if self._escape == 2:
            self._escape = 0
            return _ARROWS.get(ch)
        if self._escape == 1:
            self._escape = 0
            if ch == ord("["):
                self._escape = 2
                return None
            # Not an escape sequence: read the byte as a plain key.
        elif ch == _ESC:
            self._escape = 1
            return None
        return _KEYS.get(ch)


class PromptEditor:
    """Single-line editor for the task prompt. Enter applies, Esc cancels."""

    def __init__(self) -> None:
        self.active = False
        self.text = ""

    def begin(self, text: str) -> None:
        self.active = True
        self.text = text

    def feed(self, ch: int) -> str | None:
        """Consume one byte; return "applied" or "cancelled" when editing ends."""
        if ch == _ESC:
            self.active = False
            return "cancelled"
        if ch in _ENTER:
            self.active = False
            return "applied"
        if ch in _BACKSPACE:
            self.text = self.text[:-1]
        elif 32 <= ch < 127:  # printable ASCII
            self.text += chr(ch)
        return None
