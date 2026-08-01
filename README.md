<p align="center">
  <img alt="Rosetta" src="media/rosetta_logo.png" width="100%">
</p>

**Rosetta** interfaces ROS 2 robots to robot-learning frameworks like
[LeRobot](https://github.com/huggingface/lerobot).

**Documentation: [iblnkn.github.io/rosetta](https://iblnkn.github.io/rosetta/)**

Between a pub/sub robot and a policy sits a translation: topics
must become training frames, and model output must become messages again.
It has to happen identically at training and at deployment. Rosetta is built around the philosophy that this translation should be defined once and enforced as late as possible.


A YAML contract defines the translation. A contract looks like this:

```yaml
robot_type: my_robot
robot_interface: ros2
fps: 30

observations:
  observation.state:
    channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]

  observation.images.cam:
    channel: {topic: /camera/image_raw/compressed,
              type: sensor_msgs/msg/CompressedImage}
    align: {strategy: hold, timeline: header}
    apply: [resize: [480, 640]]

actions:
  action:
    channel: {topic: /cmd, type: sensor_msgs/msg/JointState}
    align: {strategy: hold, timeline: header}
    select: [position.j1, position.j2]
```


## Getting started

The [rosetta_ws](https://github.com/iblnkn/rosetta_ws) workspace installs
ROS 2 Jazzy, Rosetta, and LeRobot in one command; the
[installation guide](https://iblnkn.github.io/rosetta/installation.html)
covers existing ROS 2 workspaces. The tutorial
[train and deploy your first policy](https://iblnkn.github.io/rosetta/tutorials/first-policy.html)
walks through recording demonstrations with the episode recorder,
converting bags to a dataset with `rosetta_port`, training with LeRobot,
and deploying with the policy runner.


## Packages

| Package | Purpose |
|---------|---------|
| `rosetta` (this repo) | Core library, ROS 2 nodes, bag conversion |
| [`rosetta_interfaces`](https://github.com/iblnkn/rosetta_interfaces) | ROS 2 action and service definitions |
| [`lerobot_rosetta`](https://github.com/iblnkn/lerobot-rosetta) | LeRobot framework adapter: dataset writer, policy runner, inference servers |
| [`lerobot_robot_rosetta`](https://github.com/iblnkn/lerobot-robot-rosetta) | LeRobot Robot plugin |
| [`lerobot_teleoperator_rosetta`](https://github.com/iblnkn/lerobot-teleoperator-rosetta) | LeRobot Teleoperator plugin (experimental) |

## License

[Apache-2.0](LICENSE)
