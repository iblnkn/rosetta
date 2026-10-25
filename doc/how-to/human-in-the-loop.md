# Run human-in-the-loop episodes

Human-in-the-loop episodes are run by
[`rosetta_hil`](https://github.com/iblnkn/rosetta_hil), a separate app built on
rosetta's recorder, policy runner and public Python API. Its README covers
launching, running an episode, and its parameters and interfaces.

The teleop device is declared in the contract's `teleop` section, described on
the [Contract](../reference/contract.md#teleop) page.
`contracts/so_101_hil.yaml` is a complete example.
