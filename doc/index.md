# Rosetta

**Rosetta** brings [LeRobot](https://github.com/huggingface/lerobot) to ROS2 robots.

Define a contract that maps your ROS2 topics to LeRobot features, record demos to
bag files, convert them to a LeRobot dataset, train a policy, and deploy it back
to your robot.

> **Getting started?** The [rosetta_ws](https://github.com/iblnkn/rosetta_ws)
> workspace handles the non-trivial setup of getting ROS2, Rosetta, and LeRobot
> installed together.

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
