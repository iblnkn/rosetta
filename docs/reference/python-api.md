# Python API

The modules and names other packages may import from `rosetta` (core) and
`rosetta_ros` (the ROS 2 robot interface). Each public module lists its names
in `__all__`. Every other module is internal and can change without notice.
`test/test_public_api.py` in each of the two packages pins the exact names.

## Core modules

Core imports no robot middleware, so every module here imports without ROS.

| Module | Names | Use |
|---|---|---|
| `rosetta.contract.schema` | `load_contract`, `parse_contract`, `Contract`, `FrameEntry`, `Source`, `Channel`, `Align`, `Task`, `Teleop`, `TeleopEventMap`, `TeleopInputSource`, `TeleopFeedbackSource`, `ResamplePolicy`, `SafetyBehavior`, `FieldKind`, `FRAME_SECTIONS`, `TELEOP_INPUT_KEY`, `TELEOP_FEEDBACK_KEY`, `owning_key` | Load and validate a contract. `load_contract(path)` raises `FileNotFoundError` or `ContractValidationError`. `parse_contract(text, source=...)` validates YAML text. The dataclasses mirror the contract YAML. |
| `rosetta.contract.specs` | `StreamSpec`, `ObservationStreamSpec`, `ActionStreamSpec`, `iter_specs`, `iter_policy_specs`, `iter_observation_specs`, `iter_action_specs`, `iter_reward_as_action_specs`, `iter_teleop_input_specs`, `iter_teleop_feedback_specs` | Runtime stream specs resolved from a `Contract`. `iter_specs` yields every recorded stream. `iter_policy_specs` yields observations, then actions. |
| `rosetta.contract.errors` | `ContractValidationError` | Raised for any invalid contract. |
| `rosetta.contract.sidecar` | `CONTRACT_SIDECAR_FILENAME`, `LEROBOT_CONTRACT_SIDECAR_PATH`, `find_contract_for_pretrained`, `scan_inline_codec_paths` | The contract file stored with a dataset (`rosetta_contract.yaml`, at `meta/rosetta_contract.yaml` in a LeRobot dataset). `find_contract_for_pretrained` follows a checkpoint to its training dataset's contract and returns `None` when a link is missing. |
| `rosetta.contract.operators` | `Operator`, `OperatorContext`, `Invertibility`, `register_operator`, `OPERATOR_ENTRY_POINT_GROUP` | Operator plugins for `apply`. |
| `rosetta.frames.codecs` | `register_decoder`, `register_encoder`, `decode_value`, `encode_value`, `discover_codecs`, `DecoderFn`, `EncoderFn`, `NonFiniteActionError`, `SUPPORTED_NUMERIC_DTYPES`, `CODEC_ENTRY_POINT_GROUP` | Codec plugins and the message to array boundary. `encode_value` raises `NonFiniteActionError` for a NaN or Inf action value. |
| `rosetta.frames.layout` | `FrameLayout`, `KeyLayout`, `SpecSlice`, `build_feature` | The positional layout of stream specs within frame keys. `FrameLayout.assemble` builds a frame from per-spec values and zero-fills missing ones. `FrameLayout.split` cuts an action frame into per-spec `float64` vectors. `FrameLayout.lerobot_features` gives LeRobot feature dicts. |
| `rosetta.frames.naming` | `IMAGE_KEY_PREFIX`, `camera_name`, `sanitize_field_name` | `IMAGE_KEY_PREFIX` is `observation.images.`. `camera_name` strips it and keeps dots. `sanitize_field_name` replaces every character outside `[A-Za-z0-9_]` with `_`. |
| `rosetta.frames.protocols` | `FrameIO` | The live robot as a framework adapter sees it. |
| `rosetta.policies` | `DatasetWriter`, `PolicyRunner`, `RunnerHost`, `RunnerFeedback`, `RunnerResult`, `load_dataset_writer`, `load_policy_runner`, `available_dataset_writers`, `available_policy_runners` | Framework adapter protocols and the registry that loads them by name. |
| `rosetta.robots.protocols` | `RobotInterface` | The robot interface protocol. |
| `rosetta.robots.registry` | `load_robot_interface`, `available_robot_interfaces`, `ROBOT_INTERFACE_GROUP` | Robot interfaces by name. |

The registries load plugins from the entry-point groups listed in
[Packages](packages.md#entry-point-groups).

## Protocols

`FrameIO`, from `rosetta.frames.protocols`:

| Member | Kind | Behavior |
|---|---|---|
| `sample_frame()` | method | Returns the current observation frame, `{contract_key: np.ndarray \| str}`. |
| `warmed_up` | property | `True` once every observation stream has delivered at least one value. |
| `publish_frame(action_frame)` | method | Publishes one action frame, `{contract_key: np.ndarray}`, and returns it. |
| `send_safety_action()` | method | Publishes the contract's safety action. |
| `reset_state()` | method | Clears per-episode state such as buffers and warmup. |

`DatasetWriter`, from `rosetta.policies`. A writer is opened once, receives
one run of `add_frame` calls per episode, and is finalized once.

| Member | Behavior |
|---|---|
| `open(*, contract, repo_id, root=None, contract_path=None, embed_contract=True, **opts)` | Translates the contract into the framework's schema and opens the output. With `embed_contract` true and a `contract_path`, copies the contract into the output. |
| `add_frame(frame)` | Appends one frame. Besides the contract keys, a frame holds `is_first`, `is_last` and `is_terminal` as `(1,)` bool arrays and `task` as a string. |
| `save_episode()` | Flushes the buffered episode. |
| `discard_episode()` | Drops a partly buffered episode. Safe to call with nothing buffered. |
| `finalize()` | Writes top-level metadata and closes the dataset. Not called when every episode fails. |

`PolicyRunner`, from `rosetta.policies`:

| Member | Behavior |
|---|---|
| `setup(host, contract)` | Declares and reads settings on `host`. Called at configure, and again after a cleanup and configure. |
| `run(frames, *, task, stop_event)` | Blocks until `stop_event` is set or the policy finishes. Returns a `RunnerResult`. |
| `feedback()` | Returns a `RunnerFeedback`. Called from another thread while `run` blocks. |
| `request_stop()` | Wakes a `run` blocked in I/O. Called from another thread, alongside setting `stop_event`, and also when no run is active. |
| `teardown()` | Releases framework resources. |

`RunnerResult(success, message="")` is a frozen dataclass. `success` is
`True` for a run that ended because it was asked to stop or because it
finished. `policy_runner_node` reports `success=False` as
`termination_reason: error` with `message`.

`RunnerFeedback(queue_depth=0, published_actions=0)` is a frozen dataclass.
Both counters must stay non-negative. `policy_runner_node` carries them as
`uint32`.

`RunnerHost`:

| Member | Behavior |
|---|---|
| `declare(name, default, description="", read_only=False)` | Declares a setting, or does nothing when `name` already exists. |
| `get(name)` | The setting's current value. |
| `logger` | Property. A logger with `debug`, `info`, `warning` and `error`. |

`RobotInterface`, from `rosetta.robots.protocols`:

| Member | Behavior |
|---|---|
| `register_codecs()` | Imports the interface's built-in codecs. Called once per process for every installed interface, before `rosetta.codecs` plugins load. |
| `provided_timelines(msg_type)` | Returns the `frozenset` of timeline names a channel of `msg_type` provides. Raises `ValueError` for an unknown type. |
| `validate_qos(qos)` | Raises `ValueError` when a channel's `qos` mapping does not parse. |

The loaders instantiate each plugin class with no arguments and check it
against its protocol. An unknown name raises `ValueError` listing the
installed names. A class missing a protocol method raises `TypeError` naming
the missing methods.

## ROS 2 modules

The node toolkit that `rosetta_ros_hil` and the LeRobot plugins build on.

| Module | Names | Use | Imports ROS |
|---|---|---|---|
| `rosetta_ros.rosetta_lifecycle_node` | `RosettaLifecycleNode`, `BridgeLifecycleNode` | Base classes for lifecycle nodes. See [Lifecycle nodes](#lifecycle-nodes). | yes |
| `rosetta_ros.node_utils` | `TERMINATION_STOPPED`, `TERMINATION_TIMEOUT`, `TERMINATION_COMPLETED`, `TERMINATION_CANCELLED`, `TERMINATION_NODE_DEACTIVATED`, `TERMINATION_ERROR`, `finish_goal`, `spin_lifecycle_node`, `wait_until`, `positive_rate_descriptor`, `copy_fields`, `STATUS_QOS`, `RESULT_QOS` | The termination reasons, taken from the `.action` files. `finish_goal(goal_handle, result)` ends a goal `CANCELED` when a cancel was requested, `ABORTED` for `error`, `node_deactivated` or `cancelled`, and `SUCCEEDED` otherwise. `spin_lifecycle_node(factory)` runs a node on a 4-thread `MultiThreadedExecutor` and drives the shutdown transition on Ctrl+C. `positive_rate_descriptor` limits a rate parameter to 0.1 to 1000.0. `STATUS_QOS` is reliable, depth 10. `RESULT_QOS` is reliable, transient local, depth 1. | yes |
| `rosetta_ros.child_goals` | `send_goal`, `cancel_goal` | Send and cancel goals on other action servers from inside a node's own work. Both block the calling thread. | yes |
| `rosetta_ros.topic_bridge` | `TopicBridge` | The ROS 2 `FrameIO`: subscriptions, lifecycle publishers and the action watchdog, created on a host lifecycle node by `setup(node)`. | yes |
| `rosetta_ros.node_host` | `NodeHost` | Runs a node on a private rclpy context with its own executor thread, for processes Rosetta does not own. | yes |
| `rosetta_ros.runner_host` | `RosParameterHost` | A `RunnerHost` whose settings are parameters of a node. | yes |
| `rosetta_ros.rclpy_utils` | `qos_profile_from_dict`, `prefixed_topic`, `require_transition_success`, `lifecycle_state_label` | `qos_profile_from_dict` parses a contract `qos` mapping. `prefixed_topic(prefix, topic)` moves a topic under a prefix: `("/hil/policy", "/cmd_vel")` gives `/hil/policy/cmd_vel`. | yes |
| `rosetta_ros.launch_utils` | `typed_config`, `yaml_params`, `autostart_handlers`, `contract_output_remaps` | Helpers for launch files. `yaml_params(path, node_name)` reads one node's parameters from a params file, merging the `/**`, `/**/<node>`, `<node>` and `/<namespace>/<node>` sections in that order. It raises `ValueError` when the file has sections and none applies to the node. `contract_output_remaps(contract_path, prefix)` remaps every action and reward topic under `prefix`, and raises `ValueError` for a prefix with an empty `contract_path`. | yes |
| `rosetta_ros.field_access` | `dot_get`, `dot_set`, `resolve_indexed` | Read and write message fields by dotted path. | no |

## LeRobot packages

These names are each package's `__all__`. No test pins them.

| Module | Names |
|---|---|
| `lerobot_rosetta` | `LeRobotDatasetWriter`, `LeRobotPolicyRunner`, `build_lerobot_features` |
| `lerobot_robot_rosetta` | `Rosetta`, `RosettaConfig` |
| `lerobot_teleoperator_rosetta` | `RosettaTeleop`, `RosettaTeleopConfig` |

## Lifecycle nodes

Base classes for lifecycle nodes, from `rosetta_ros.rosetta_lifecycle_node`.

### RosettaLifecycleNode

`RosettaLifecycleNode` is the base class of `episode_recorder_node`,
`policy_runner_node` and `hil_manager_node`. It owns the lifecycle
transitions and a single work slot, so a node built on it runs one piece of
work at a time and stops it on deactivate.

| Member | Kind | Behavior |
|---|---|---|
| `setup()` | override | Creates resources on configure. Raising fails the transition, and the base logs the traceback and tears down. |
| `teardown()` | override | Destroys what `setup()` created. Runs on cleanup, shutdown and error, so it must tolerate partial state. |
| `unblock_stop()` | optional override | Wakes work blocked in I/O after a stop. Must be safe with no work active. |
| `send_safety_action()` | optional override | Secures hardware after work stops. |
| `on_goal`, `on_cancel` | action callbacks | Pass to the `ActionServer`. `on_goal` claims the work slot or rejects the goal. `on_cancel` stops work only when the cancel names the running goal. |
| `goal_work(goal_handle)` | context manager | Yields the stop event and releases the work slot on every exit. `goal_handle` is `None` for service-started work. |
| `signal_stop(reason, detail="")` | method | Stops the current work and records why. The first reason wins. |
| `stop_reason`, `stop_detail` | properties | The recorded reason, or `None`, and its detail text. |
| `handle_start_service(response, target, args, *, what)` | method | Claims the slot and runs `target` on a thread, for a plain start service. |
| `handle_cancel_service(response, cancel_client, *, what)` | method | Cancels the current work for a `Trigger` service. Reports `success: false` when nothing runs. |
| `cancel_current_work(cancel_client)` | method | Forwards a cancel to the node's own action server, so the goal ends `CANCELED`. Service-started work, which has no goal, is stopped with reason `cancelled`. |
| `try_claim_work()` | method | Claims the work slot. Returns `None`, or the rejection reason. |
| `publish_output(publisher, msg)` | method | Publishes status or result from a work thread, and logs instead of raising when the node is already torn down. |
| `busy`, `is_active`, `is_configured` | properties | Work in progress, and the lifecycle state. |
| `STOP_WORK_TIMEOUT_SEC` | class attribute | Seconds deactivate waits for work to stop before sending the safety action anyway. `5.0`. |

Subclasses never override `on_configure`, `on_deactivate` or the other
lifecycle callbacks. Deactivate closes the work slot, stops the current work
with reason `node_deactivated`, waits up to `STOP_WORK_TIMEOUT_SEC`, then
calls `send_safety_action()`. Cleanup is refused while work is in progress.

### BridgeLifecycleNode

`BridgeLifecycleNode(node_name, observation_specs, action_specs, fps)` adds a
`TopicBridge` as `self.bridge`, set up and torn down with the node, and sends
the bridge's safety action. The LeRobot robot and teleoperator plugins use it.

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
