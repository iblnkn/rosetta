# Python API

The modules and names other packages may import from `rosetta` (core) and
`rosetta_ros` (the ROS 2 robot interface). Each public module lists its names
in `__all__`; every other module is internal and can change without notice.
Each package's `test/test_public_api.py` pins its list.

## Core modules

Core imports no robot middleware, so these modules work without ROS.

| Module | Provides |
|---|---|
| `rosetta.contract.schema` | `load_contract`, `parse_contract`, `Contract` and its parts, the teleop column prefixes |
| `rosetta.contract.specs` | `StreamSpec` types and the `iter_*_specs` functions |
| `rosetta.contract.errors` | `ContractValidationError` |
| `rosetta.contract.sidecar` | Locating the contract a trained model was built with |
| `rosetta.contract.operators` | `Operator`, `Invertibility`, `register_operator`, for operator plugins |
| `rosetta.frames.codecs` | `decode_value`, `encode_value`, `register_decoder`, `register_encoder`, `discover_codecs` |
| `rosetta.frames.layout` | `FrameLayout`, `build_feature` |
| `rosetta.frames.naming` | Feature and camera naming |
| `rosetta.frames.protocols` | `FrameIO` |
| `rosetta.policies` | `PolicyRunner`, `RunnerHost`, `DatasetWriter`, their result types and loaders |
| `rosetta.robots.protocols` | `RobotInterface`, for robot interface packages |
| `rosetta.robots.registry` | `load_robot_interface`, `available_robot_interfaces` |

## ROS 2 modules

The node toolkit apps such as `rosetta_hil` build on.

| Module | Provides | Needs ROS |
|---|---|---|
| `rosetta_ros.rosetta_lifecycle_node` | `RosettaLifecycleNode`, `BridgeLifecycleNode` | yes |
| `rosetta_ros.node_utils` | Termination reasons, `finish_goal`, `spin_lifecycle_node`, `wait_until`, `positive_rate_descriptor`, `copy_fields`, `STATUS_QOS`/`RESULT_QOS` for `~/status` and `~/result` topics | yes |
| `rosetta_ros.child_goals` | `send_goal`, `cancel_goal`, for running other action servers' goals from inside a node's work | yes |
| `rosetta_ros.topic_bridge` | `TopicBridge` | yes |
| `rosetta_ros.node_host` | `NodeHost` | yes |
| `rosetta_ros.runner_host` | `RosParameterHost`, a `RunnerHost` whose settings are node parameters | yes |
| `rosetta_ros.rclpy_utils` | `qos_profile_from_dict`, `require_transition_success`, `lifecycle_state_label` | yes |
| `rosetta_ros.field_access` | `resolve_indexed`, `dot_get`, `dot_set` | no |
| `rosetta_ros.launch_utils` | `typed_config`, `yaml_params`, `autostart_handlers` | yes |

Plugins register through the entry-point groups listed in
[Packages](packages.md).

## Lifecycle nodes

`RosettaLifecycleNode` is the base class of `episode_recorder_node` and
`policy_runner_node`. It owns the lifecycle transitions and a single work
slot, so a node built on it runs one goal at a time and stops it cleanly on
deactivate.

| Member | Kind | Purpose |
|---|---|---|
| `setup()` | override | Create resources on configure. Raise to fail the transition. |
| `teardown()` | override | Destroy what `setup()` created. Must tolerate partial state. |
| `unblock_stop()` | optional override | Wake work blocked in I/O after a stop. |
| `send_safety_action()` | optional override | Secure hardware after work stops. |
| `on_goal`, `on_cancel` | action callbacks | Claim the work slot; route a cancel to the goal it names. |
| `goal_work(goal_handle)` | context manager | Yields the stop event; releases the slot on every exit. |
| `signal_stop(reason, detail)` | call | Stop the current work and record why. The first reason wins. |
| `stop_reason` | property | The recorded reason, or `None`. |
| `handle_start_service`, `handle_cancel_service` | call | Start or cancel work from a plain service. |

A node never overrides `on_configure`, `on_deactivate` or the other lifecycle
callbacks.

### Example

This node serves the `RunPolicy` action as `wait`. Each goal waits until it is
cancelled or `max_duration_s` passes.

```python
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup

from rosetta_ros.node_utils import TERMINATION_COMPLETED, finish_goal, spin_lifecycle_node
from rosetta_ros.rosetta_lifecycle_node import RosettaLifecycleNode
from rosetta_interfaces.action import RunPolicy


class WaitNode(RosettaLifecycleNode):
    """Runs one goal at a time, until cancelled or for max_duration_s."""

    def __init__(self, **kwargs):
        super().__init__("wait_node", **kwargs)
        self._server = None

    def setup(self):
        self._server = ActionServer(
            self,
            RunPolicy,
            "wait",
            execute_callback=self.execute,
            goal_callback=self.on_goal,
            cancel_callback=self.on_cancel,
            callback_group=ReentrantCallbackGroup(),
        )

    def teardown(self):
        if self._server is not None:
            self._server.destroy()
            self._server = None

    def execute(self, goal_handle):
        result = RunPolicy.Result()
        with self.goal_work(goal_handle) as stop_event:
            stopped = stop_event.wait(goal_handle.request.max_duration_s or None)
            result.termination_reason = self.stop_reason if stopped else TERMINATION_COMPLETED
            finish_goal(goal_handle, result)
        return result


def main():
    return spin_lifecycle_node(WaitNode)


if __name__ == "__main__":
    main()
```

After `ros2 lifecycle set /wait_node configure` and `activate`, a goal with
`max_duration_s: 1.0` ends `SUCCEEDED` with `termination_reason: completed`,
and a cancelled goal ends `CANCELED` with `termination_reason: cancelled`. The
reentrant callback group lets the cancel callback run while `execute` blocks.
