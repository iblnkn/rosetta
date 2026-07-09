"""Debug / analysis instrumentation for the action-queue + chunk pipeline.

Every chunk/queue inspection hook lives here, isolated from the control
path. Deleting this module (plus its few one-line call sites) removes the
whole analysis feature; nothing here is required for control.

* **Merge-event dump** (client side). When the node's ``merge_dump_dir``
  param is non-empty, every RunPolicy goal gets its own JSONL file
  (``merge_<stamp>_<task-slug>.jsonl``): a header record with the goal's
  config, then one line per action-queue merge capturing the existing-queue
  tail, the incoming chunk (post-drop and full), and the blended result
  keyed by timestep. The node attaches a ``MergeDumpWriter`` to the goal's
  client as ``client._merge_dump``; one goal = one file = one clean run for
  ``tools/chunk_analysis/build_merge_timeline.py``. Off (zero overhead)
  when the param is empty.

* **Generated-chunk echo** (node side). When the ``publish_debug_chunk``
  node param is true, each model-generated chunk — as received per
  inference, before the client merges it into the execution queue — is
  published as a multi-point ``trajectory_msgs/JointTrajectory`` on
  ``debug_chunk_topic`` (default ``~/debug/generated_chunk``) for
  PlotJuggler / bag recording. ``ChunkDebugPublisher`` owns the publisher;
  ``emit_generated_chunk`` is the receive-path hook that feeds it.
  Inspection only — the robot does not consume this topic.

Everything here is best-effort: diagnostics must never break control.
"""

import json
import os
import re
import time

# ---------------------------------------------------------------------------
# Merge-event dump (per-goal JSONL)
# ---------------------------------------------------------------------------


def _ts_pose(action) -> list[float]:
    """Flatten a TimedAction's pose tensor to a rounded python list."""
    return [
        round(float(x), 6)
        for x in action.get_action().detach().to("cpu").flatten().tolist()
    ]


def new_dump_path(dump_dir: str, task: str) -> str:
    """Claim a fresh per-goal dump path inside ``dump_dir`` (created if needed).

    ``merge_<YYYYmmdd_HHMMSS>_<pid>_<task-slug>.jsonl``, created atomically
    (``O_EXCL``) with a numeric suffix on collision. The PID mostly keeps
    concurrent writers apart (and says who wrote the file), but PIDs can
    coincide across container PID namespaces on a shared dump dir — the
    atomic create is what actually guarantees one goal per file.
    """
    os.makedirs(dump_dir, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (task or "").strip()).strip("-")
    slug = slug[:40] or "task"
    base = f"merge_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{slug}"
    path = os.path.join(dump_dir, f"{base}.jsonl")
    n = 1
    while True:
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            path = os.path.join(dump_dir, f"{base}_{n}.jsonl")
            n += 1
        else:
            return path


class MergeDumpWriter:
    """Per-goal merge-event JSONL writer.

    One instance = one file = one RunPolicy goal: the node creates a writer
    at goal start (when ``merge_dump_dir`` is set) and attaches it to the
    goal's RobotClient as ``client._merge_dump``, so the event counter and
    file lifetime match the goal exactly. The optional ``header`` becomes
    the file's first record as ``{"header": {...}}`` (viewers skip it).
    """

    def __init__(self, path: str, header: dict | None = None):
        self.path = str(path)
        self._event_idx = 0
        if header is not None:
            # Not swallowed: a failure here surfaces at goal start (once),
            # where the node logs it and runs the goal without a dump.
            self._append({"header": header})

    def _append(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def dump(self, latest_action, existing, incoming, merged, incoming_full):
        """Append one merge event (best-effort, debug only).

        ``existing``/``incoming``/``merged`` are ``{timestep: TimedAction}``
        as seen by the merge; ``incoming`` is the post-drop chunk (timesteps
        beyond the cutoff) that enters the merge, while ``incoming_full`` is
        the raw chunk list including the already-passed prefix the client
        drops -- recorded so the viewer can anchor each chunk at
        t_observation and color the dropped prefix.
        """
        try:
            record = {
                "event": self._event_idx,
                "wall_time": time.time(),
                "latest_action": int(latest_action),
                # timestep -> pose, for each of the three queues
                "existing": {int(ts): _ts_pose(a) for ts, a in existing.items()},
                "incoming": {int(ts): _ts_pose(a) for ts, a in incoming.items()},
                "merged": {int(ts): _ts_pose(a) for ts, a in merged.items()},
                # full incoming chunk (pre-drop) keyed by timestep; first point
                # is t_observation (timestep i_0). Points <= latest_action are
                # dropped.
                "incoming_full": {
                    int(a.get_timestep()): _ts_pose(a) for a in incoming_full
                },
            }
            self._append(record)
            self._event_idx += 1
        except Exception:  # pragma: no cover - diagnostics must never break control
            pass


# ---------------------------------------------------------------------------
# Generated-chunk echo (param-gated ROS publisher)
# ---------------------------------------------------------------------------


def emit_generated_chunk(client, timed_actions) -> None:
    """Echo a freshly received (pre-merge) chunk via the node-attached callback.

    The node attaches ``client._debug_chunk_cb`` only when its
    ``publish_debug_chunk`` param is enabled; otherwise this is a getattr +
    None check per chunk. joint_names follow the action-tensor order.
    """
    debug_cb = getattr(client, "_debug_chunk_cb", None)
    if debug_cb is None or not timed_actions:
        return
    try:
        joint_names = list(client.robot.action_features)
        positions = [
            ta.get_action().detach().to("cpu").flatten().tolist()
            for ta in timed_actions
        ]
        debug_cb(joint_names, positions)
    except Exception:
        client.logger.exception("Failed to publish debug action chunk")


class ChunkDebugPublisher:
    """Owns the debug JointTrajectory publisher on behalf of the client node.

    Resolves the message classes at runtime (no import-time dependency on
    trajectory_msgs). ``publish`` turns one model-generated chunk into a
    multi-point trajectory: each chunk step becomes a JointTrajectoryPoint
    at ``i / fps`` seconds.
    """

    def __init__(self, node, topic: str):
        from rosidl_runtime_py.utilities import get_message

        self._node = node
        self._JointTrajectory = get_message("trajectory_msgs/msg/JointTrajectory")
        self._JointTrajectoryPoint = get_message(
            "trajectory_msgs/msg/JointTrajectoryPoint"
        )
        self._pub = node.create_publisher(self._JointTrajectory, topic, 10)
        node.get_logger().info(
            f"Debug: publishing model-generated chunks as JointTrajectory on '{topic}'"
        )

    def publish(
        self, joint_names: list[str], positions: list[list[float]], fps: int
    ) -> None:
        """Publish one chunk; best-effort, debug only."""
        if not positions:
            return
        dt = 1.0 / max(fps, 1)
        msg = self._JointTrajectory()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.joint_names = list(joint_names)
        points = []
        for i, pos in enumerate(positions):
            pt = self._JointTrajectoryPoint()
            pt.positions = [float(x) for x in pos]
            t = i * dt
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
            points.append(pt)
        msg.points = points
        self._pub.publish(msg)
