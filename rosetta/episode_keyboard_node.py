#!/usr/bin/env python3
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

"""
EpisodeKeyboardNode: Keyboard controller for the Rosetta episode recorder.

Inspired by teleop_episode_keyboard.cpp from legalaspro/so101-ros-physical-ai:
  https://github.com/legalaspro/so101-ros-physical-ai/blob/main/episode_recorder/src/teleop_episode_keyboard.cpp

Drives the recorder's service interface without requiring ros2 action send_goal.
Run this in the same terminal after launching the episode recorder.

Services used (on recorder_ns, default /episode_recorder):
  ~/start_recording   (rosetta_interfaces/srv/StartRecording)
  ~/cancel_recording  (std_srvs/srv/Trigger)
  ~/delete_last_bag   (std_srvs/srv/Trigger)

Keys:
  r / →        Start recording
  s / ←        Stop and save
  d / Backspace  Discard episode (stop + delete bag)
  t            Edit task prompt for the next episode
  h / ?        Show help
  q            Quit

Usage:
    ros2 run rosetta episode_keyboard_node
    ros2 run rosetta episode_keyboard_node --ros-args -p recorder_ns:=/my_recorder
    ros2 run rosetta episode_keyboard_node --ros-args -p default_prompt:="pick cube"
"""

from __future__ import annotations

import os
import select
import sys
import termios
import time

import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger
from rosetta_interfaces.srv import StartRecording

# ─────────────────────────────────────────────────────────────────────────────
# Key codes
# ─────────────────────────────────────────────────────────────────────────────
_ESC = 27
_BACKSPACE_1 = 127   # most terminals
_BACKSPACE_2 = 8     # Ctrl-H / some terminals
_ENTER_1 = 13        # CR
_ENTER_2 = 10        # LF
_ARROW_RIGHT = ord('C')  # ESC [ C
_ARROW_LEFT = ord('D')   # ESC [ D


# ─────────────────────────────────────────────────────────────────────────────
# Terminal raw-mode helper
# ─────────────────────────────────────────────────────────────────────────────

class RawTerminal:
    """
    RAII wrapper around termios raw mode.

    Sets VMIN=0, VTIME=0 (non-blocking reads) and disables canonical mode,
    echo, and flow control. ISIG is preserved so Ctrl-C still delivers SIGINT.
    """

    def __init__(self) -> None:
        self._old_settings = None

    def enable(self) -> bool:
        """Enable raw mode. Returns False if stdin is not a TTY."""
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return False
        try:
            self._old_settings = termios.tcgetattr(fd)
            new = termios.tcgetattr(fd)
            # Disable: canonical mode, echo, extended-input processing
            new[3] &= ~(termios.ICANON | termios.ECHO | termios.IEXTEN)
            # Disable: XON/XOFF flow control, CR→NL translation
            new[0] &= ~(termios.IXON | termios.ICRNL)
            # Non-blocking: return immediately with 0 bytes if no data
            new[6][termios.VMIN] = 0
            new[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, new)
            return True
        except termios.error:
            return False

    def restore(self) -> None:
        """Restore original terminal settings."""
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self._old_settings)
            except termios.error:
                pass
            self._old_settings = None

    def read_byte(self) -> int | None:
        """Non-blocking read of one byte. Returns byte value or None."""
        rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not rlist:
            return None
        data = sys.stdin.buffer.read(1)
        return data[0] if data else None


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 node
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeKeyboardNode(Node):
    """Keyboard controller that drives the Rosetta episode recorder via services."""

    def __init__(self, terminal: RawTerminal) -> None:
        super().__init__('episode_keyboard')
        self._term = terminal

        # Parameters
        recorder_ns = self.declare_parameter('recorder_ns', '/episode_recorder').value
        self._current_prompt: str = self.declare_parameter('default_prompt', '').value

        # Service clients
        self._start_client = self.create_client(
            StartRecording, f'{recorder_ns}/start_recording'
        )
        self._cancel_client = self.create_client(
            Trigger, f'{recorder_ns}/cancel_recording'
        )
        self._delete_client = self.create_client(
            Trigger, f'{recorder_ns}/delete_last_bag'
        )

        # State
        self._recording: bool = False
        self._has_last_bag: bool = False  # True after a recording starts; cleared on delete
        self._service_pending: bool = False
        self._rec_start: float | None = None
        self._delete_retries: int = 0
        self._delete_retry_timer = None

        # Task prompt inline editor state
        self._editing_task: bool = False
        self._task_buffer: str = ''

        # ESC sequence state machine (0=none, 1=got ESC, 2=got ESC[)
        self._esc_state: int = 0

        # Timers
        self._key_timer = self.create_timer(0.02, self._poll_keyboard)   # 50 Hz
        self._display_timer = self.create_timer(0.25, self._update_display)  # 4 Hz

        self._print_help()

    # ─────────────────────────────────────────────────────────────────────────
    # Display helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _print_help(self) -> None:
        prompt_line = f'  Current prompt: "{self._current_prompt}"' if self._current_prompt else ''
        lines = [
            '',
            '╔══════════════════════════════════════════╗',
            '║     Episode Recording Controller         ║',
            '╠══════════════════════════════════════════╣',
            '║  → / r  : Start recording                ║',
            '║  ← / s  : Stop recording & save          ║',
            '║  ⌫ / d  : Discard current episode        ║',
            '║  t      : Edit task prompt               ║',
            '║  h / ?  : Help                           ║',
            '║  q      : Quit                           ║',
            '╚══════════════════════════════════════════╝',
        ]
        sys.stdout.write('\n'.join(lines) + '\n')
        if prompt_line:
            sys.stdout.write(prompt_line + '\n')
        sys.stdout.write('\n')
        sys.stdout.flush()

    def _msg(self, text: str) -> None:
        """Clear the current line and print a status message."""
        sys.stdout.write(f'\r\033[K{text}\n')
        sys.stdout.flush()

    def _update_display(self) -> None:
        """Refresh the recording timer on the current line (4 Hz)."""
        if not self._recording or self._editing_task or self._rec_start is None:
            return
        elapsed = time.monotonic() - self._rec_start
        mm = int(elapsed) // 60
        ss = int(elapsed) % 60
        label = self._current_prompt
        if len(label) > 22:
            label = label[:21] + '…'
        sys.stdout.write(f'\r\033[K  \U0001f534 REC  [{label}]  {mm:02d}:{ss:02d}  ')
        sys.stdout.flush()

    # ─────────────────────────────────────────────────────────────────────────
    # Keyboard polling
    # ─────────────────────────────────────────────────────────────────────────

    def _poll_keyboard(self) -> None:
        ch = self._term.read_byte()
        if ch is None:
            return

        # Task edit mode intercepts all keystrokes
        if self._editing_task:
            self._handle_task_input(ch)
            return

        # ESC sequence state machine for arrow keys (ESC [ C / ESC [ D)
        if self._esc_state == 0:
            if ch == _ESC:
                self._esc_state = 1
                return
        elif self._esc_state == 1:
            if ch == ord('['):
                self._esc_state = 2
                return
            self._esc_state = 0
            # Fall through to handle the byte as a regular key
        elif self._esc_state == 2:
            self._esc_state = 0
            if ch == _ARROW_RIGHT:
                self._handle_start()
            elif ch == _ARROW_LEFT:
                self._handle_stop()
            return

        # Single-character keys
        if ch in (ord('r'), ord('R')):
            self._handle_start()
        elif ch in (ord('s'), ord('S')):
            self._handle_stop()
        elif ch in (ord('d'), ord('D'), _BACKSPACE_1, _BACKSPACE_2):
            self._handle_discard()
        elif ch in (ord('t'), ord('T')):
            self._begin_task_edit()
        elif ch in (ord('h'), ord('H'), ord('?')):
            self._print_help()
        elif ch in (ord('q'), ord('Q')):
            self._handle_quit()

    # ─────────────────────────────────────────────────────────────────────────
    # Task prompt inline editor
    # ─────────────────────────────────────────────────────────────────────────

    def _begin_task_edit(self) -> None:
        if self._service_pending:
            return self._msg('⏳ Service call in progress, please wait...')
        self._editing_task = True
        self._task_buffer = self._current_prompt
        self._msg('📝 Edit task prompt (Enter=apply, Esc=cancel):')
        self._redraw_task_prompt()

    def _handle_task_input(self, ch: int) -> None:
        if ch == _ESC:
            self._editing_task = False
            self._task_buffer = ''
            self._msg('✗  Task edit cancelled')
        elif ch in (_ENTER_1, _ENTER_2):
            self._editing_task = False
            self._current_prompt = self._task_buffer
            self._task_buffer = ''
            self._msg(f'✅ Task prompt → "{self._current_prompt}"')
        elif ch in (_BACKSPACE_1, _BACKSPACE_2):
            if self._task_buffer:
                self._task_buffer = self._task_buffer[:-1]
            self._redraw_task_prompt()
        elif 32 <= ch < 127:  # printable ASCII
            self._task_buffer += chr(ch)
            self._redraw_task_prompt()

    def _redraw_task_prompt(self) -> None:
        sys.stdout.write(f'\r\033[KTask> {self._task_buffer}')
        sys.stdout.flush()

    # ─────────────────────────────────────────────────────────────────────────
    # Action handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_start(self) -> None:
        if self._service_pending:
            return self._msg('⏳ Service call in progress, please wait...')
        if self._recording:
            return self._msg('⚠  Already recording — stop or discard first')
        if not self._start_client.service_is_ready():
            return self._msg(
                '✗  start_recording service not available — is the recorder running?'
            )
        prompt = self._current_prompt
        self._msg(f'▶  Starting recording... (prompt: "{prompt}")')
        self._service_pending = True
        req = StartRecording.Request()
        req.prompt = prompt
        future = self._start_client.call_async(req)
        future.add_done_callback(self._on_start_done)

    def _on_start_done(self, future) -> None:
        self._service_pending = False
        try:
            resp = future.result()
            if resp.accepted:
                self._recording = True
                self._has_last_bag = True
                self._rec_start = time.monotonic()
                self._msg('▶  Recording started')
            else:
                self._msg(f'✗  Start failed: {resp.message}')
        except Exception as e:
            self._msg(f'✗  Start exception: {e}')

    def _handle_stop(self) -> None:
        if self._service_pending:
            return self._msg('⏳ Service call in progress, please wait...')
        if not self._recording:
            return self._msg('⚠  Not recording — nothing to stop')
        if not self._cancel_client.service_is_ready():
            return self._msg('✗  cancel_recording service not available')
        self._msg('⏹  Stopping recording...')
        self._service_pending = True
        future = self._cancel_client.call_async(Trigger.Request())
        future.add_done_callback(self._on_stop_done)

    def _on_stop_done(self, future) -> None:
        self._service_pending = False
        self._recording = False
        try:
            resp = future.result()
            if resp.success:
                self._msg('⏹  Saved')
            else:
                self._msg(f'✗  Stop failed: {resp.message}')
        except Exception as e:
            self._msg(f'✗  Stop exception: {e}')

    def _handle_discard(self) -> None:
        if self._service_pending:
            return self._msg('⏳ Service call in progress, please wait...')
        if not self._recording and not self._has_last_bag:
            return self._msg('⚠  No episode to discard')
        self._msg('🗑  Discarding episode...')
        self._service_pending = True
        self._delete_retries = 0
        if self._recording:
            # Still recording: cancel first, then delete
            if not self._cancel_client.service_is_ready():
                self._service_pending = False
                return self._msg('✗  cancel_recording service not available')
            future = self._cancel_client.call_async(Trigger.Request())
            future.add_done_callback(self._on_discard_cancel_done)
        else:
            # Already stopped: go straight to delete
            self._do_delete_last_bag()

    def _on_discard_cancel_done(self, future) -> None:
        try:
            resp = future.result()
            if not resp.success:
                self._service_pending = False
                self._msg(f'✗  Cancel failed: {resp.message}')
                return
            self._recording = False
        except Exception as e:
            self._service_pending = False
            self._msg(f'✗  Cancel exception: {e}')
            return

        # Cancel confirmed — now delete the bag
        self._do_delete_last_bag()

    def _do_delete_last_bag(self) -> None:
        if not self._delete_client.service_is_ready():
            self._service_pending = False
            self._msg('✗  delete_last_bag service not available')
            return
        del_future = self._delete_client.call_async(Trigger.Request())
        del_future.add_done_callback(self._on_discard_delete_done)

    def _on_discard_delete_done(self, future) -> None:
        try:
            resp = future.result()
            if resp.success:
                self._has_last_bag = False
                self._service_pending = False
                self._msg(f'🗑  Discarded: {resp.message}')
                return
            # Recorder stops the bag asynchronously after cancel — retry briefly
            if 'recording in progress' in resp.message.lower():
                self._delete_retries += 1
                if self._delete_retries <= 25:  # up to ~5 s
                    self._delete_retry_timer = self.create_timer(0.2, self._retry_delete_once)
                    return
                self._msg('✗  Discard timed out waiting for recorder to stop')
            else:
                self._msg(f'✗  Delete failed: {resp.message}')
        except Exception as e:
            self._msg(f'✗  Delete exception: {e}')
        self._service_pending = False

    def _retry_delete_once(self) -> None:
        """One-shot timer callback: cancel the recurring timer then retry delete."""
        if self._delete_retry_timer is not None:
            self._delete_retry_timer.cancel()
            self._delete_retry_timer.destroy()
            self._delete_retry_timer = None
        self._do_delete_last_bag()

    def _handle_quit(self) -> None:
        self._msg('\n👋 Shutting down keyboard controller...\n')
        self._key_timer.cancel()
        self._display_timer.cancel()
        self._term.restore()
        rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None) -> int:
    terminal = RawTerminal()
    if not terminal.enable():
        sys.stderr.write(
            'Failed to enable raw terminal mode. '
            'Are you running in an interactive terminal?\n'
        )
        return 1

    try:
        rclpy.init(args=args)
        node = EpisodeKeyboardNode(terminal)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        terminal.restore()
        rclpy.try_shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
