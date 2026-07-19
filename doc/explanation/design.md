# Design

## What is LeRobot?

[LeRobot](https://github.com/huggingface/lerobot) is Hugging Face's open-source framework for [robot learning](https://huggingface.co/spaces/lerobot/robot-learning-tutorial). LeRobot records demonstrations, trains policies (ACT, Diffusion Policy, VLAs like SmolVLA and Pi0), and deploys them on hardware. Datasets use a standard format (v3) built on Parquet files and MP4 videos, with community-contributed datasets and models on the [Hugging Face Hub](https://huggingface.co/datasets?other=LeRobot).

## What is Rosetta?

Every robot-learning pipeline translates robot messages into training frames. Timing, units, field order, image geometry. Write the translation twice, once for your dataset script and again for your deployment node, and nothing checks the two agree. Your policy trains on one translation and runs on another.

Rosetta puts the translation in one place: the contract. One YAML per robot defines the frames and the streams behind them, in both directions. Five properties follow:

- **Parity by construction.** Bag conversion and live inference run the same code from the same contract. There is no second implementation to drift.
- **Record raw, decide late.** Bags keep every message at native rate. The contract assigns meaning at port time, not record time. Change the contract and re-port old bags. No re-collection. See [Record raw, decide late](record-raw-decide-late.md).
- **The contract travels.** Bags and datasets embed the contract text. Deployment resolves the contract from your checkpoint's training dataset. Your policy carries its own translation.
- **Both directions.** One contract records observations and serves actions. Action transforms must invert, and Rosetta verifies the round trip at load.
- **Failure at load time.** Unknown keys, impossible timelines, wrong-dimension kinds, non-invertible action transforms: rejected before a message flows.

The **robot side** (`rosetta.robots`) adapts pub/sub ecosystems (ROS2 today). The **policy side** (`rosetta.policies`) adapts learning frameworks (LeRobot, vla_foundry, starvla). Both sides speak frames, so either swaps without touching the other.

## Package architecture

Inside the `rosetta` package, the tree tells the story:

```
rosetta/
├── contract/    # the declaration: one YAML per robot (schema, specs, operators)
├── frames/      # the interlingua: layout, resampling, codecs, stream protocols
├── robots/      # robot side: pub/sub ecosystems (ros2/ today)
└── policies/    # policy side: DatasetWriter + PolicyRunner seams, entry-point loading
```

The data path uses three words, in pipeline order:

- A **channel** is a declared endpoint in the robot interface's dialect (`schema.Channel`).
- A **stream** is the decoded, timelined sample sequence of one channel *before* alignment. `StreamSpec` describes the stream, `StreamBuffer` lands samples on the clock.
- A **frame** is one synchronized sample of every contract key per clock tick, *after* alignment. `FrameIO` (a `FrameSource` plus a `FrameSink`) is the bidirectional frame surface a `PolicyRunner` drives.

Inside `contract/`, the split is say vs. do. `schema.py` is what the contract **says**: the typed document model of the YAML (`Channel`, `Align`, `Source`, `FrameEntry`, `Contract`), validation, and the loaders. `parse_contract(text)` validates a contract string, so callers embedding the text (the recorder, the porter) hold exactly the bytes they validated. `load_contract(path)` reads a file and delegates to `parse_contract`. `specs.py` is what the runtime **consumes**: the resolved stream specifications (`StreamSpec` family) produced by the `iter_*_specs` pass. A spec is composed, not copied. Each spec carries `source`, the exact declaration `Source` behind the spec (read `spec.source.channel.topic`, `spec.source.align`, `spec.source.kind` through this), plus the computed fields the YAML never states (`names`, `dtype`, `namespace`, `operators`, image geometry). Downstream code takes `list[StreamSpec]` and never re-reads the document. "Spec" always means one of these resolved runtime objects. There is no copy step, so a declaration fact never silently goes missing from a spec.

## Workspace packages

The workspace splits into several packages. Framework adapters register into `rosetta.policies` entry points:

| Package | Purpose |
|---------|---------|
| `rosetta` | Core library, nodes, bag conversion |
| [`rosetta_interfaces`](https://github.com/iblnkn/rosetta_interfaces) | ROS2 action/service definitions |
| [`lerobot_rosetta`](https://github.com/iblnkn/lerobot-rosetta) | LeRobot backend adapter: dataset writer, policy runner, inference servers |
| [`lerobot_robot_rosetta`](https://github.com/iblnkn/lerobot-robot-rosetta) | LeRobot Robot plugin (discovered by LeRobot) |
| [`lerobot_teleoperator_rosetta`](https://github.com/iblnkn/lerobot-teleoperator-rosetta) | LeRobot Teleoperator plugin (experimental) |

Launch files and parameter defaults are cataloged in the [nodes reference](../reference/nodes.md).

## The rosetta_ws workspace

[rosetta_ws](https://github.com/iblnkn/rosetta_ws) is a devcontainer workspace for getting started. Installing ROS2 and LeRobot together is not trivial. The workspace handles this setup.
