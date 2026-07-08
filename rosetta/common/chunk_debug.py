"""Debug / analysis instrumentation for the action-queue + chunk pipeline.

Every chunk/queue inspection hook lives here, isolated from the control
path. Deleting this module (plus its few one-line call sites) removes the
whole analysis feature; nothing here is required for control.

* **Merge-event dump** (client side). When the env var ``ROSETTA_MERGE_DUMP``
  names a file, every action-queue merge appends one JSON line capturing the
  existing-queue tail, the incoming chunk (post-drop and full), and the
  blended result keyed by timestep. Consumed offline by the viewers in
  ``tools/chunk_analysis/`` (``build_merge_inspector.py``,
  ``build_merge_timeline.py``). Off (zero overhead) when unset.

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
import time

# ---------------------------------------------------------------------------
# Merge-event dump (env-gated JSONL)
# ---------------------------------------------------------------------------

_MERGE_DUMP_PATH = os.environ.get("ROSETTA_MERGE_DUMP") or None
_merge_event_idx = 0


def _ts_pose(action) -> list[float]:
    """Flatten a TimedAction's pose tensor to a rounded python list."""
    return [
        round(float(x), 6)
        for x in action.get_action().detach().to("cpu").flatten().tolist()
    ]


def maybe_dump_merge_event(latest_action, existing, incoming, merged, incoming_full):
    """Append one merge event to the JSONL dump; no-op unless enabled.

    ``existing``/``incoming``/``merged`` are ``{timestep: TimedAction}`` as
    seen by the merge; ``incoming`` is the post-drop chunk (timesteps beyond
    the cutoff) that enters the merge, while ``incoming_full`` is the raw
    chunk list including the already-passed prefix the client drops --
    recorded so the viewer can anchor each chunk at t_observation and color
    the dropped prefix.
    """
    if _MERGE_DUMP_PATH is None:
        return
    global _merge_event_idx
    try:
        record = {
            "event": _merge_event_idx,
            "wall_time": time.time(),
            "latest_action": int(latest_action),
            # timestep -> pose, for each of the three queues
            "existing": {int(ts): _ts_pose(a) for ts, a in existing.items()},
            "incoming": {int(ts): _ts_pose(a) for ts, a in incoming.items()},
            "merged": {int(ts): _ts_pose(a) for ts, a in merged.items()},
            # full incoming chunk (pre-drop) keyed by timestep; first point is
            # t_observation (timestep i_0). Points <= latest_action are dropped.
            "incoming_full": {
                int(a.get_timestep()): _ts_pose(a) for a in incoming_full
            },
        }
        with open(_MERGE_DUMP_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        _merge_event_idx += 1
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
