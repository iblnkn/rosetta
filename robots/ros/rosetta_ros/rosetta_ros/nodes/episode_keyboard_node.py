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

Drives the recorder's RecordEpisode action, so each recording's result comes
back to the goal that started it, including a recording that ends on its time
limit. Run this in its own terminal after launching the episode recorder.

Interfaces used:
  recorder_action_name  (rosetta_interfaces/action/RecordEpisode), default /record_episode
  <recorder_ns>/delete_last_bag  (std_srvs/srv/Trigger), default /episode_recorder

Keys:
  r / →        Start recording
  s / ←        Stop and save
  d / Backspace  Discard episode (stop + delete bag)
  t            Edit task prompt for the next episode
  h / ?        Show help
  q            Quit

Usage:
    ros2 run rosetta_ros episode_keyboard_node
    ros2 run rosetta_ros episode_keyboard_node --ros-args -p recorder_ns:=/my_robot/episode_recorder \
        -p recorder_action_name:=/my_robot/record_episode
    ros2 run rosetta_ros episode_keyboard_node --ros-args -p default_prompt:="pick cube"
"""

from __future__ import annotations

import os
import sys
import termios
import time

import rclpy
from action_msgs.srv import CancelGoal
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import Trigger

from rosetta_interfaces.action import RecordEpisode
from rosetta_ros.node_utils import (
    TERMINATION_ERROR,
    TERMINATION_NODE_DEACTIVATED,
    TERMINATION_STOPPED,
    TERMINATION_TIMEOUT,
)
from rosetta_ros.nodes.episode_keys import KeyDecoder, PromptEditor

# A stop the operator asked for needs no reason on the outcome line. These weren't asked for.
_UNASKED_REASONS = {
    TERMINATION_TIMEOUT: " (time limit)",
    TERMINATION_NODE_DEACTIVATED: " (recorder deactivated)",
    TERMINATION_STOPPED: " (stopped by the recorder)",
}


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

    def read_bytes(self) -> bytes:
        """Read all pending input bytes; returns b"" if none.

        Non-blocking because VMIN=0/VTIME=0. Reads the raw fd with os.read —
        never sys.stdin.buffer, whose internal buffering strands bytes and
        makes arrow-key escape sequences lag one keypress behind.
        """
        return os.read(sys.stdin.fileno(), 64)


class EpisodeKeyboardNode(Node):
    """Keyboard controller that drives the Rosetta episode recorder through its action."""

    def __init__(self, terminal: RawTerminal) -> None:
        super().__init__("episode_keyboard")
        self._term = terminal

        # Parameters
        recorder_ns = self.declare_parameter("recorder_ns", "/episode_recorder").value
        action_name = self.declare_parameter("recorder_action_name", "/record_episode").value
        self._current_prompt: str = self.declare_parameter("default_prompt", "").value

        self._action_client = ActionClient(self, RecordEpisode, action_name)
        self._delete_client = self.create_client(Trigger, f"{recorder_ns}/delete_last_bag")

        # State. One goal at a time: from send until its result arrives.
        self._goal_pending: bool = False  # send_goal sent, no response yet
        self._goal_handle = None  # accepted goal whose result hasn't arrived
        self._recording: bool = False  # accepted and not yet stopped by s or d
        self._discard_goal: bool = False  # d asked the recorder to delete this goal's bag
        self._has_last_bag: bool = False  # a finished recording left a bag; cleared on delete
        self._delete_pending: bool = False
        self._rec_start: float | None = None

        self._keys = KeyDecoder()
        self._editor = PromptEditor()
        self._commands = {
            "start": self._handle_start,
            "stop": self._handle_stop,
            "discard": self._handle_discard,
            "edit": self._begin_task_edit,
            "help": self._print_help,
            "quit": self._handle_quit,
        }

        # Timers
        self._key_timer = self.create_timer(0.02, self._poll_keyboard)  # 50 Hz
        self._display_timer = self.create_timer(0.25, self._update_display)  # 4 Hz

        self._print_help()

    def _print_help(self) -> None:
        prompt_line = f'  Current prompt: "{self._current_prompt}"' if self._current_prompt else ""
        lines = [
            "",
            "╔══════════════════════════════════════════╗",
            "║     Episode Recording Controller         ║",
            "╠══════════════════════════════════════════╣",
            "║  → / r  : Start recording                ║",
            "║  ← / s  : Stop recording & save          ║",
            "║  ⌫ / d  : Discard current episode        ║",
            "║  t      : Edit task prompt               ║",
            "║  h / ?  : Help                           ║",
            "║  q      : Quit                           ║",
            "╚══════════════════════════════════════════╝",
        ]
        sys.stdout.write("\n".join(lines) + "\n")
        if prompt_line:
            sys.stdout.write(prompt_line + "\n")
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _msg(self, text: str) -> None:
        """Clear the current line and print a status message."""
        sys.stdout.write(f"\r\033[K{text}\n")
        sys.stdout.flush()

    def _update_display(self) -> None:
        """Refresh the recording timer on the current line (4 Hz)."""
        if not self._recording or self._editor.active or self._rec_start is None:
            return
        elapsed = time.monotonic() - self._rec_start
        mm = int(elapsed) // 60
        ss = int(elapsed) % 60
        label = self._current_prompt
        if len(label) > 22:
            label = label[:21] + "…"
        sys.stdout.write(f"\r\033[K  \U0001f534 REC  [{label}]  {mm:02d}:{ss:02d}  ")
        sys.stdout.flush()

    def _poll_keyboard(self) -> None:
        for ch in self._term.read_bytes():
            self._process_byte(ch)

    def _process_byte(self, ch: int) -> None:
        if self._editor.active:
            self._edit_prompt(ch)
            return
        command = self._keys.feed(ch)
        if command is not None:
            self._commands[command]()

    def _busy(self) -> bool:
        """Report and return True while a start or a delete is in flight."""
        if self._goal_pending or self._delete_pending:
            self._msg("⏳ Request in progress, please wait...")
            return True
        return False

    def _begin_task_edit(self) -> None:
        if self._busy():
            return
        self._editor.begin(self._current_prompt)
        self._msg("📝 Edit task prompt (Enter=apply, Esc=cancel):")
        self._redraw_task_prompt()

    def _edit_prompt(self, ch: int) -> None:
        outcome = self._editor.feed(ch)
        if outcome == "cancelled":
            self._msg("✗  Task edit cancelled")
        elif outcome == "applied":
            self._current_prompt = self._editor.text
            self._msg(f'✅ Task prompt → "{self._current_prompt}"')
        else:
            self._redraw_task_prompt()

    def _redraw_task_prompt(self) -> None:
        sys.stdout.write(f"\r\033[KTask> {self._editor.text}")
        sys.stdout.flush()

    # -------------------- r: start --------------------

    def _handle_start(self) -> None:
        if self._busy():
            return
        if self._goal_handle is not None:
            self._msg("⚠  The last recording hasn't finished — stop or discard first")
            return
        if not self._action_client.server_is_ready():
            self._msg("✗  record_episode action not available — is the recorder running and active?")
            return
        prompt = self._current_prompt
        self._msg(f'▶  Starting recording... (prompt: "{prompt}")')
        self._goal_pending = True
        future = self._action_client.send_goal_async(RecordEpisode.Goal(prompt=prompt))
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future) -> None:
        self._goal_pending = False
        try:
            goal_handle = future.result()
        except Exception as e:
            self._msg(f"✗  Start exception: {e}")
            return
        if not goal_handle.accepted:
            self._msg("✗  Start rejected: the recorder is busy or not active")
            return
        self._goal_handle = goal_handle
        self._recording = True
        self._has_last_bag = False  # the recorder's last bag is this recording's now
        self._rec_start = time.monotonic()
        self._msg("▶  Recording started")
        goal_handle.get_result_async().add_done_callback(self._on_result)

    # -------------------- s: stop --------------------

    def _handle_stop(self) -> None:
        if not self._recording:
            self._msg("⚠  Not recording — nothing to stop")
            return
        self._msg("⏹  Stopping recording...")
        self._cancel()

    def _cancel(self) -> None:
        """Cancel the goal. The outcome arrives with its result (_on_result)."""
        self._recording = False
        self._goal_handle.cancel_goal_async().add_done_callback(self._on_cancel_done)

    def _on_cancel_done(self, future) -> None:
        # A cancel that came too late (the time limit hit first) is fine: the
        # result is on its way. A refused or failed one leaves the recording
        # running, so show it as recording again and let s be pressed again.
        try:
            response = future.result()
        except Exception as e:
            detail = str(e)
        else:
            # rclpy answers a rejected cancel with ERROR_NONE and no goals, so
            # the goal being listed is what says it is stopping.
            if response.goals_canceling or response.return_code == CancelGoal.Response.ERROR_GOAL_TERMINATED:
                return
            detail = f"the recorder refused it, return code {response.return_code}"
        if self._goal_handle is not None:
            self._recording = True
            # A delete the recorder accepted still holds when the recording ends.
            after = ", and its bag is deleted when it ends" if self._discard_goal else ""
            self._msg(f"✗  Stop failed ({detail}). Still recording{after}")

    # -------------------- the goal's result --------------------

    def _on_result(self, future) -> None:
        """Report how the goal's recording ended. A bag d deleted is reported by the delete."""
        self._goal_handle = None
        self._recording = False
        try:
            result = future.result().result
        except Exception as e:
            self._discard_goal = False
            self._msg(f"✗  Result exception: {e}")
            return
        discarded, self._discard_goal = self._discard_goal, False
        self._has_last_bag = bool(result.bag_path) and not discarded
        if result.termination_reason == TERMINATION_ERROR:
            self._msg(f"✗  Recording failed: {result.message}")
            return
        if discarded:
            return
        reason = _UNASKED_REASONS.get(result.termination_reason, "")
        self._msg(f"⏹  Saved {result.messages_written} messages{reason}: {result.bag_path}")
        if result.message:
            self._msg(f"⚠  {result.message}. Press d to discard this episode.")

    # -------------------- d: discard --------------------

    def _handle_discard(self) -> None:
        if self._busy():
            return
        if self._goal_handle is not None:
            # Recording, or stopped with its result still on the way. The
            # recorder's delete_last_bag deletes a recording in progress when
            # it ends, so ask now, before anything else can claim the recorder.
            # Ask once: after the recording ends, a second ask finds no bag.
            if self._discard_goal:
                self._msg("⚠  Already discarding this episode")
                return
            if not self._delete_client.service_is_ready():
                self._msg("✗  delete_last_bag service not available")
                return
            self._msg("🗑  Discarding episode...")
            self._discard_goal = True
            self._delete_last_bag()
            if self._recording:
                self._cancel()
            return
        if not self._has_last_bag:
            self._msg("⚠  No episode to discard")
            return
        self._msg("🗑  Discarding episode...")
        self._delete_last_bag()

    def _delete_last_bag(self) -> None:
        if not self._delete_client.service_is_ready():
            self._msg("✗  delete_last_bag service not available")
            return
        self._delete_pending = True
        self._delete_client.call_async(Trigger.Request()).add_done_callback(self._on_delete_done)

    def _on_delete_done(self, future) -> None:
        self._delete_pending = False
        try:
            resp = future.result()
        except Exception as e:
            self._discard_goal = False
            self._msg(f"✗  Delete exception: {e}")
            return
        if resp.success:
            self._has_last_bag = False
            self._msg(f"🗑  Discarded: {resp.message}")
        else:
            # The bag stays, so its result reports it saved and d can delete it.
            self._discard_goal = False
            self._msg(f"✗  Delete failed: {resp.message}")

    def _handle_quit(self) -> None:
        self._msg("\n👋 Shutting down keyboard controller...\n")
        self._key_timer.cancel()
        self._display_timer.cancel()
        self._term.restore()
        rclpy.shutdown()


def main(args=None) -> int:
    terminal = RawTerminal()
    if not terminal.enable():
        sys.stderr.write("Failed to enable raw terminal mode. Are you running in an interactive terminal?\n")
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


if __name__ == "__main__":
    sys.exit(main())
