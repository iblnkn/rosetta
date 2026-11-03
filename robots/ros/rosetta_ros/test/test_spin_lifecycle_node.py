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

"""spin_lifecycle_node: the shutdown transition is driven on the way out.

destroy_node() never runs on_shutdown -- only the lifecycle state machine
does -- so exiting through spin_lifecycle_node must trigger_shutdown() before
destroying the node, or external resources (an open bag writer, a policy
server subprocess, the safety send) leak on Ctrl-C.

SIGINT and SIGTERM only set a flag, so a signal repeated during teardown (ros2
launch sends SIGINT twice) cannot interrupt it. The in-process tests fake the
executor; the subprocess tests run a real node and real signals.
"""

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest
from rclpy.exceptions import InvalidHandle
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from rosetta_ros import node_utils


class _FakeExecutor:
    """Stands in for the spin loop: the given signal, sent twice, on the first spin."""

    signum = signal.SIGINT
    spins = 0

    def add_node(self, node):
        pass

    def spin_once(self, timeout_sec):
        type(self).spins += 1
        os.kill(os.getpid(), self.signum)
        os.kill(os.getpid(), self.signum)


class _RaisingExecutor(_FakeExecutor):
    """A callback raised: spin_once re-raises it."""

    def spin_once(self, timeout_sec):
        raise RuntimeError("callback failed")


class _ExitingExecutor(_FakeExecutor):
    """A callback called sys.exit(): spin_once raises SystemExit."""

    def spin_once(self, timeout_sec):
        raise SystemExit(0)


class _Node(LifecycleNode):
    def __init__(self, name, *, shutdown_raises=False, shutdown_result=TransitionCallbackReturn.SUCCESS):
        super().__init__(name)
        self.shutdown_saw_live_node = None
        self._shutdown_raises = shutdown_raises
        self._shutdown_result = shutdown_result

    def on_shutdown(self, state):
        # rclpy still runs on_shutdown after destroy_node(), but the node's
        # publishers are gone by then, so a final safety send would fail.
        try:
            self.get_name()
            self.shutdown_saw_live_node = True
        except InvalidHandle:
            self.shutdown_saw_live_node = False
        return self._shutdown_result

    def trigger_shutdown(self):
        if self._shutdown_raises:
            raise RuntimeError("context already invalid")
        return super().trigger_shutdown()


@pytest.fixture
def spin(monkeypatch):
    # The session context belongs to conftest: spin_lifecycle_node must not
    # init or shut it down here.
    monkeypatch.setattr(node_utils.rclpy, "init", lambda **kwargs: None)
    monkeypatch.setattr(node_utils.rclpy, "try_shutdown", lambda **kwargs: None)
    monkeypatch.setattr(node_utils, "EventsExecutor", _FakeExecutor)
    monkeypatch.setattr(_FakeExecutor, "spins", 0)
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    yield node_utils.spin_lifecycle_node
    for signum, handler in saved.items():
        signal.signal(signum, handler)


def test_ctrl_c_runs_on_shutdown_before_destroy(spin):
    node = _Node("spin_shutdown")

    assert spin(lambda: node) == 0

    assert node.shutdown_saw_live_node is True


def test_node_destroyed_even_when_shutdown_transition_fails(spin):
    node = _Node("spin_shutdown_fails", shutdown_raises=True)

    assert spin(lambda: node) == 1  # launch must not report a clean finish

    with pytest.raises(InvalidHandle):
        node.get_name()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signals_stop_the_loop_and_a_caller_handler_is_restored(spin, monkeypatch, signum):
    monkeypatch.setattr(_FakeExecutor, "signum", signum)

    def caller_handler(*_):
        pass

    signal.signal(signum, caller_handler)
    node = _Node(f"spin_signal_{int(signum)}")

    assert spin(lambda: node) == 0

    assert _FakeExecutor.spins == 1
    assert node.shutdown_saw_live_node is True
    assert signal.getsignal(signum) is caller_handler


def test_default_handlers_become_ignore_so_a_late_signal_cannot_crash_the_exit(spin):
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    assert spin(lambda: _Node("spin_late_signal")) == 0

    assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    os.kill(os.getpid(), signal.SIGINT)  # would raise KeyboardInterrupt under the default
    os.kill(os.getpid(), signal.SIGTERM)  # would kill pytest under SIG_DFL


def test_a_failed_shutdown_transition_exits_nonzero(spin):
    # The safety send failing inside on_shutdown must not look like a clean exit.
    node = _Node("spin_shutdown_error", shutdown_result=TransitionCallbackReturn.ERROR)

    assert spin(lambda: node) == 1


def test_a_callback_calling_sys_exit_still_runs_on_shutdown(spin, monkeypatch):
    monkeypatch.setattr(node_utils, "EventsExecutor", _ExitingExecutor)
    node = _Node("spin_sys_exit")

    assert spin(lambda: node) == 1

    assert node.shutdown_saw_live_node is True


class _BusyNode(_Node):
    """Work still finishing on its own thread when the shutdown transition returns."""

    def __init__(self, name):
        super().__init__(name)
        self.busy = True
        self.work_done_before_destroy = None

    def on_shutdown(self, state):
        threading.Thread(target=self._finish, daemon=True).start()
        return super().on_shutdown(state)

    def _finish(self):
        time.sleep(0.3)  # the recorder writing a bag's metadata after its writer closed
        self.busy = False

    def destroy_node(self):
        self.work_done_before_destroy = not self.busy
        super().destroy_node()


def test_work_still_finishing_after_the_shutdown_transition_completes_before_destroy(spin):
    node = _BusyNode("spin_busy")

    assert spin(lambda: node) == 0

    assert node.work_done_before_destroy is True


def test_callback_exception_still_runs_on_shutdown(spin, monkeypatch):
    monkeypatch.setattr(node_utils, "EventsExecutor", _RaisingExecutor)
    node = _Node("spin_raises")

    assert spin(lambda: node) == 1

    assert node.shutdown_saw_live_node is True


# A real process: rclpy.init, the EventsExecutor, real signals. on_shutdown
# sleeps so the later signals land inside it, as ros2 launch's second SIGINT does.
_CHILD = textwrap.dedent(
    """
    import sys, time
    from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
    from rosetta_ros.node_utils import spin_lifecycle_node

    class N(LifecycleNode):
        def __init__(self):
            super().__init__("spin_child")
            self.create_timer(0.01, lambda: None)
            self.trigger_configure()
            self.trigger_activate()
            print("spinning", flush=True)

        def on_shutdown(self, state):
            print("on_shutdown started", flush=True)
            time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 0.3)
            print("on_shutdown ran", flush=True)
            return TransitionCallbackReturn.SUCCESS

    sys.exit(spin_lifecycle_node(N))
    """
)


def _run_child(signals: list[tuple[int, float]], *, shutdown_sec: float = 0.3) -> tuple[int, str]:
    # Read on a thread, with deadlines: the middleware can print to stdout
    # before "spinning" (Fast DDS logs a shared-memory port it cannot lock,
    # say), and a child that never starts must fail the test, not hang it.
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(shutdown_sec)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    lines, spinning = [], threading.Event()

    def read():
        for line in proc.stdout:
            lines.append(line)
            if line.strip() == "spinning":
                spinning.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        assert spinning.wait(30), "".join(lines)
        for signum, gap in signals:
            proc.send_signal(signum)
            time.sleep(gap)
        proc.wait(timeout=20)
    finally:
        proc.kill()
        proc.wait()
    reader.join(5)
    out = "".join(lines)
    return proc.returncode, out[out.index("spinning\n") + len("spinning\n") :]


@pytest.mark.parametrize(
    "signals",
    [
        [(signal.SIGINT, 0.0)],
        [(signal.SIGINT, 0.002), (signal.SIGINT, 0.0)],
        [(signal.SIGINT, 0.0625)] * 5,
        [(signal.SIGINT, 0.0)] + [(signal.SIGINT, 0.1)] * 6,
        [(signal.SIGTERM, 0.0)],
        [(signal.SIGINT, 0.15), (signal.SIGTERM, 0.0)],
    ],
    ids=["sigint", "double-sigint", "five-sigints", "sigints-past-teardown", "sigterm", "sigint-then-sigterm"],
)
def test_process_exits_cleanly_on_signals(signals):
    code, out = _run_child(signals)

    assert code == 0, out
    assert "on_shutdown ran" in out
    assert "Traceback" not in out


def test_a_ctrl_c_after_the_grace_period_quits_a_hung_shutdown():
    # on_shutdown hangs for 60 s. A second Ctrl+C at a terminal must still end
    # the process, while launch's burst of signals (the cases above) does not.
    grace = node_utils.FORCE_QUIT_AFTER_SEC
    started = time.monotonic()
    code, out = _run_child([(signal.SIGINT, grace + 0.5), (signal.SIGINT, 0.0)], shutdown_sec=60.0)

    assert code == 130, out
    assert "on_shutdown started" in out
    assert "on_shutdown ran" not in out
    assert time.monotonic() - started < 30
