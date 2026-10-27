# lerobot_rosetta

Rosetta's [LeRobot](https://github.com/huggingface/lerobot) framework adapter: dataset writer, policy runner, and gRPC inference servers.

The naming convention is `<framework>_rosetta`, rosetta's adapter to a policy framework. This package is the rosetta-to-LeRobot direction: rosetta loads it by name via entry points. The opposite direction (the Robot and Teleoperator plugins that *LeRobot* auto-discovers) lives in [`lerobot_robot_rosetta`](../lerobot_robot_rosetta) and [`lerobot_teleoperator_rosetta`](../lerobot_teleoperator_rosetta), whose names are forced by LeRobot's `lerobot_robot_*`/`lerobot_teleoperator_*` discovery convention. This package deliberately matches neither prefix, so LeRobot's `register_third_party_plugins()` never imports it. It depends on `lerobot_robot_rosetta` (for `RosettaConfig`, and LeRobot discovers the Robot plugin at policy-run time).

## Entry points

| Group | Name | Target |
|---|---|---|
| `rosetta.dataset_writers` | `lerobot` | `lerobot_rosetta.dataset_writer:LeRobotDatasetWriter` |
| `rosetta.policy_runners` | `lerobot` | `lerobot_rosetta.policy_runner:LeRobotPolicyRunner` |
| `console_scripts` | `rosetta_policy_server` | `lerobot_rosetta.policy_server:main` |
| `console_scripts` | `rosetta_classifier_server` | `lerobot_rosetta.classifier_server:main` |

## Inference servers

Two gRPC servers speak LeRobot's AsyncInference protocol (the `RobotClient` connects to either unchanged):

- **`policy_server.py`** (`rosetta_policy_server`): preload and cache wrapper over LeRobot's stock `lerobot.async_inference.policy_server`. The stock server reloads the checkpoint on every client handshake; this one loads it once and reuses it while the requested `(policy_type, pretrained_name_or_path, device)` is unchanged.
- **`classifier_server.py`** (`rosetta_classifier_server`): reward-classifier variant with the same preload and cache behavior.

Both accept optional preload flags that load the model **before** the port is bound, so a socket-level readiness check implies the model is ready:

```bash
python -m lerobot_rosetta.policy_server --host=127.0.0.1 --port=8080 \
    --policy-type=act --pretrained-name-or-path=my-org/my-policy --policy-device=cuda
```

`LeRobotPolicyRunner` starts the appropriate server with these flags at node configure time (`launch_local_server: true`), so the first `run_policy` goal is as fast as any other.

## Build

```bash
colcon build --packages-select lerobot_rosetta
```

## Documentation

How the LeRobot adapter fits together, and the dataset layout it writes:
[LeRobot integration](https://iblnkn.github.io/rosetta/explanation/design.html)
and [LeRobot data model](https://iblnkn.github.io/rosetta/reference/contract.html).

Full Rosetta documentation: **https://iblnkn.github.io/rosetta/**

## License

Apache-2.0
