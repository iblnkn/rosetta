# lerobot_rosetta

Rosetta's framework adapter for [LeRobot](https://github.com/huggingface/lerobot):
a dataset writer, a policy runner, and two gRPC inference servers. Rosetta
loads it by the name `lerobot`, as `rosetta_port --framework lerobot` and as
`policy_runner_node`'s `framework` parameter, which is the default for both.

This package is the Rosetta-to-LeRobot direction. The other direction, the
`Robot` and `Teleoperator` plugins that LeRobot itself discovers, is in
[`lerobot_robot_rosetta`](../lerobot_robot_rosetta) and
[`lerobot_teleoperator_rosetta`](../lerobot_teleoperator_rosetta). Their names
follow LeRobot's `lerobot_robot_*` and `lerobot_teleoperator_*` discovery
convention. This package matches neither prefix, so LeRobot never imports it.
The policy runner uses `lerobot_robot_rosetta` to drive LeRobot's
`RobotClient`.

## Entry points

| Group | Name | Target |
|---|---|---|
| `rosetta.dataset_writers` | `lerobot` | `lerobot_rosetta.dataset_writer:LeRobotDatasetWriter` |
| `rosetta.policy_runners` | `lerobot` | `lerobot_rosetta.policy_runner:LeRobotPolicyRunner` |
| `console_scripts` | `rosetta_policy_server` | `lerobot_rosetta.policy_server:main` |
| `console_scripts` | `rosetta_classifier_server` | `lerobot_rosetta.classifier_server:main` |

## Inference servers

Both servers speak LeRobot's async-inference gRPC protocol, so LeRobot's
`RobotClient` connects to either unchanged.

- `policy_server` wraps LeRobot's `PolicyServer`. The stock server loads the
  checkpoint on every client handshake. This one keeps the loaded policy and
  reuses it while the requested policy type, checkpoint and device are
  unchanged.
- `classifier_server` serves a LeRobot `reward_classifier` model, returning
  one reward per observation. It caches the model the same way.

Given `--policy-type`, `--pretrained-name-or-path` and `--policy-device`
together, a server loads the model before it opens its port, so a port that
accepts connections means the model is ready:

```bash
python3 -m lerobot_rosetta.policy_server --host=127.0.0.1 --port=8080 \
    --policy-type=act --pretrained-name-or-path=<checkpoint> --policy-device=cuda
```

`<checkpoint>` is a local path or a Hugging Face Hub repo ID. The classifier
server defaults to port 8081. With `launch_local_server: true`,
`policy_runner_node` starts the matching server this way at configure.

## Install

LeRobot has no rosdep key, so `rosdep install` neither installs it nor
reports it missing. Install it with pip first. The policy runner needs
LeRobot's `async` extra, and this package needs a LeRobot older than 0.6.2,
which removes the async-inference module the runner uses. The package is not
released to the ROS buildfarm.

```bash
pip install 'lerobot[dataset,async]<0.6.2'
colcon build --packages-up-to lerobot_rosetta
```

## Documentation

- [Prepare a dataset](https://iblnkn.github.io/rosetta/how-to/prepare-a-dataset.html)
  and [Deploy a policy](https://iblnkn.github.io/rosetta/how-to/deploy-a-policy.html)
- [LeRobot adapter parameters](https://iblnkn.github.io/rosetta/reference/nodes.html#lerobot-adapter-parameters)
  on `policy_runner_node`
- [rosetta_port](https://iblnkn.github.io/rosetta/reference/rosetta-port.html)
- Full Rosetta documentation: [iblnkn.github.io/rosetta](https://iblnkn.github.io/rosetta/)

## License

Apache-2.0
