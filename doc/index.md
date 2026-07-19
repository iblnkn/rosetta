# Rosetta

Rosetta connects your ROS 2 robot to robot-learning frameworks like [LeRobot](https://github.com/huggingface/lerobot).

- **Tutorials**: learning-oriented. Start with [your first policy](tutorials/first-policy.md).
- **How-to guides**: task-oriented, from [writing a contract](how-to/write-a-contract.md) to [porting existing bags](how-to/port-existing-bags.md).
- **Reference**: information-oriented. The [contract reference](reference/contract.md) and friends.
- **Explanation**: understanding-oriented. Start with [design](explanation/design.md).

```{toctree}
:caption: Tutorials
:maxdepth: 1

tutorials/first-policy
```

```{toctree}
:caption: How-to guides
:maxdepth: 1

how-to/write-a-contract
how-to/record-episodes
how-to/port-existing-bags
how-to/train-a-policy
how-to/deploy-a-policy
how-to/teleop-and-hil
how-to/add-custom-codecs
how-to/add-custom-operators
```

```{toctree}
:caption: Reference
:maxdepth: 1

reference/contract
reference/nodes
reference/porter-cli
reference/lerobot-data-model
```

```{toctree}
:caption: Explanation
:maxdepth: 1

explanation/design
explanation/contract-design
explanation/record-raw-decide-late
explanation/train-serve-skew
explanation/lerobot-integration
```

```{toctree}
:caption: Contributing
:maxdepth: 1

contributing/codebase-guide
```
