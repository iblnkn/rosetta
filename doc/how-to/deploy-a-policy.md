# Deploy a Policy

This guide shows you how to run a trained checkpoint on your robot. Full parameter tables: [nodes](../reference/nodes.md#policy_runner_node). Topology background: [LeRobot integration](../explanation/lerobot-integration.md).

## Launch and run

```bash
# Terminal 1: Start the runner
ros2 launch rosetta policy_runner_launch.py \
    contract_path:=/path/to/contract.yaml \
    pretrained_name_or_path:=my-org/my-policy
```

```bash
# Terminal 2: Run the policy
ros2 action send_goal /run_policy \
    rosetta_interfaces/action/RunPolicy "{prompt: 'task description'}"
```

## Deploy without a contract file

Datasets ported with default settings embed the contract. Leave `contract_path` empty and the node resolves the contract from the checkpoint chain: `pretrained_name_or_path` → `train_config.json` → training dataset → `meta/rosetta_contract.yaml`. See [contract resolution](../reference/nodes.md#contract-resolution).

## Run inference on a remote GPU

Set `launch_local_server:=false` and point `server_address` at a machine running the policy server. The server has no ROS2 dependency. Pre-warm so even the first goal skips the model load:

```bash
python -m lerobot_rosetta.policy_server --host=0.0.0.0 --port=8080 \
    --policy-type=act --pretrained-name-or-path=my-org/my-policy --policy-device=cuda
```

Stock `lerobot.async_inference.policy_server` also works, but reloads the checkpoint on every goal.

## Deploy with LeRobot's own tools

The `policy_runner_node` is optional. LeRobot's standard CLI tools run inference directly with the Rosetta robot plugin:

```bash
# Standard LeRobot deployment, no policy_runner_node needed
lerobot-record --robot.type=rosetta --robot.config_path=contract.yaml
```

The `lerobot_robot_rosetta` / `lerobot_teleoperator_rosetta` distributions follow LeRobot's third-party plugin naming convention (`lerobot_robot_*`, `lerobot_teleoperator_*`). LeRobot CLIs and the async robot client auto-discover them when installed. No manual import or registration step. See [Imitation Learning on Real Robots](https://huggingface.co/docs/lerobot/il_robots) for LeRobot's native workflow.

## Tune chunking

`actions_per_chunk`, `chunk_size_threshold`, and `aggregate_fn_name` control how action chunks stream to the robot. If observations get skipped as "too similar", set `obs_similarity_atol:=-1.0` (see the [parameter table](../reference/nodes.md#policy_runner_node)).
