# Rosetta

**Rosetta** interfaces ROS 2 robots to robot-learning frameworks like
[LeRobot](https://github.com/huggingface/lerobot).

Record demonstrations to rosbag, declare in a YAML
contract how topics become training frames and how model output becomes
messages again. Rosetta executes that contract identically for training preperation and live inference.

> **Getting started?** The [rosetta_ws](https://github.com/iblnkn/rosetta_ws)
> workspace has ROS 2 Jazzy, Rosetta, and LeRobot installed.

```{toctree}
:caption: Getting started
:maxdepth: 1

installation
tutorials/first-policy
```

```{toctree}
:caption: How-to guides
:maxdepth: 1

how-to/write-a-contract
how-to/record-train-deploy
```

```{toctree}
:caption: Reference
:maxdepth: 1

reference/contract
reference/nodes
```

```{toctree}
:caption: Explanation
:maxdepth: 1

explanation/design
```
