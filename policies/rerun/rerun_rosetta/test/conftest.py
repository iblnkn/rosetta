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

"""A local Rerun catalog server, shared by the tests that register recordings."""

import os
import shutil
import signal
import socket
import subprocess
import time

import pytest


@pytest.fixture(scope="session")
def catalog_url():
    if shutil.which("rerun") is None:
        pytest.skip("rerun binary not on PATH")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        ["rerun", "server", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # `rerun` is a Python shim that spawns the server binary
    )
    url = f"rerun+http://127.0.0.1:{port}"
    try:
        _wait_for_port(port)
        yield url
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)


def _wait_for_port(port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"rerun server did not open port {port}")
