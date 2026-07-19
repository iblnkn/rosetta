# Rosetta Onboarding Guide

> Generated from the project knowledge graph (`.understand-anything/knowledge-graph.json`) at commit `1143eee`. Regenerate with `/understand` + `/understand-onboard` after significant structural changes.

## Project Overview

**Rosetta** provides contract-driven translation between pub/sub robots (ROS 2) and policy-learning frameworks like LeRobot: one contract per robot turns messy ROS 2 topics into clean fixed-rate frames for recording, training, and live inference.

- **Languages**: Python (primary), YAML, XML, Markdown, JSON
- **Frameworks**: ROS 2 (Jazzy, `ament_python`), pytest, GitHub Actions (industrial_ci)
- **Package type**: ROS 2 `ament_python` package with five console-script entry points: `episode_recorder_node`, `episode_keyboard_node`, `hil_manager_node`, `policy_runner_node`, `rosetta_port`

The core idea in one sentence: **a single YAML contract per robot declares what topics exist, how to decode them into arrays, and how to align them onto a fixed-rate clock — and every other component (recorder, porter, policy runner) is driven by that contract.**

## Architecture Layers

The codebase splits into nine layers. Dependency direction flows downward: nodes → adapter → frames/contract; tests → everything.

### 1. Contract Layer (`rosetta/contract/`, `contracts/*.yaml`)

The center of the system. Contract schema, spec resolution, operators, plugins, and sidecar publishing.

| File | What it does |
|---|---|
| `contract/schema.py` | YAML parsing, strict validation, loading surface. `load_contract`/`parse_contract` guarantee a returned `Contract` is fully valid in this environment (three load-time layers: shape parsing, timeline attestation, codec/operator resolvability). Highest fan-in file in the codebase (26 incoming edges). |
| `contract/model.py` | Pure-data document model: enums, frozen dataclasses, section descriptor table. No behavior. |
| `contract/specs.py` | Resolves a parsed `Contract` into runtime `StreamSpec` objects: dtype precedence via codec registry, operator pipelines from `apply` clauses, per-consumer projections (policy view, teleop, reward-as-action). Second-highest fan-in (25 edges). |
| `contract/operators.py` | Operator framework: `Operator` base class, invertibility tiers (forward-only / bidirectional / bijective), global registry. |
| `contract/builtin_operators.py` | Built-ins (`rad2deg`, `resize`, `clamp`) registered on import — `schema.py` imports it for the side effect. |
| `contract/plugins.py` | Shared setuptools entry-point loader for both open registries (`rosetta.operators`, `rosetta.codecs`). |
| `contract/sidecar.py` | Canonical on-disk placement of a dataset's sidecar contract; resolves best-effort from local dir or Hugging Face Hub. |
| `contract/errors.py` | `ContractValidationError` — the single user-facing exception; dependency-free leaf. |
| `contracts/stone.yaml` | The "Rosetta stone" reference contract — fictional robot exercising every schema feature; living documentation. |
| `contracts/so_101.yaml`, `so_101_hil.yaml`, `turtlebot3.yaml` | Real contracts for the SO-101 arm (plain + human-in-the-loop) and TurtleBot3. |

### 2. Frames & Codecs (`rosetta/frames/`)

Framework-neutral frame layout, codec registry, naming, resampling. A **frame** is one synchronized dict of every contract key per fps tick — the only value exchanged between the robot side and policy side.

| File | What it does |
|---|---|
| `frames/codecs.py` | The message-to-array boundary: `decode_value` / payload-first `encode_value` plus the global decoder/encoder registry keyed by ROS type string. Fail-fast registration: overriding a nonexistent built-in errors loudly. |
| `frames/layout.py` | `FrameLayout`: single source of truth mapping contract keys to per-spec slices of flat vectors (first-occurrence key order, shared keys concatenate). |
| `frames/resample.py` | `StreamBuffer` — the single online resampler aligning async messages onto the contract fps clock (`hold`/`asof`/`drop` strategies). |
| `frames/protocols.py` | `FrameIO` runtime-checkable Protocol — the duck-typed seam where robot side and policy side meet. |
| `frames/naming.py` | Frame-key naming helpers for framework adapters (`camera_name`, `sanitize_field_name`). |
| `examples/stone_codecs.py` | Worked example custom codec pair referenced by `stone.yaml`; bundled so the contract load-validates anywhere. |

### 3. Policy Framework (`rosetta/policies/`)

The seam that keeps rosetta framework-agnostic.

| File | What it does |
|---|---|
| `policies/protocols.py` | Structural contracts a framework must implement: `DatasetWriter` (offline) and `PolicyRunner` (online inference). |
| `policies/registry.py` | Entry-point plugin registry discovering/instantiating implementations from installed distributions, with protocol-conformance enforcement. |
| `policies/__init__.py` | Barrel re-exporting protocols + registry loaders for adapter packages (e.g. `lerobot_rosetta`). |

### 4. ROS 2 Adapter (`rosetta/robots/ros2/`)

ROS 2 middleware integration, deliberately separated from executable nodes.

| File | What it does |
|---|---|
| `topic_bridge.py` | Backend-neutral observation/action plumbing on a host lifecycle node: subscriptions, per-stream resample buffers, action publishing with safety watchdog. |
| `ingest.py` | Shared message-ingest policy (timeline extraction → decode → buffer push) used by **both** the live bridge and the offline porter — this shared path is why offline and live output match. |
| `decoders.py` / `encoders.py` | Registries of built-in ROS message decoders (Image, CompressedImage, JointState, Joy, IMU, Odometry, …) and encoders (action vectors → outgoing messages). |
| `field_access.py` | Pure-Python resolution of dotted contract selectors (`position.elbow`) against ROS messages, including indexed and parallel-array selectors. |
| `rosetta_lifecycle_node.py` | `RosettaLifecycleNode` — the single lifecycle-node idiom: fail-loud configure, one shared teardown path, work gate. All executable nodes inherit it. |
| `timelines.py` | Timeline registry (receive, header) driving attestation and per-message timestamp extraction. |
| `rclpy_utils.py` | QoS profiles from contract dicts, lifecycle state labels, transition-result checking. |
| `bag_metadata.py` | Read/update the `custom_data` block of rosbag2 `metadata.yaml` (persists the recorded contract text). |
| `node_host.py` | `NodeHost` — owns a private rclpy context, spins a factory-built node on a background executor thread. |
| `launch_utils.py` | Typed launch-configuration helpers shared by all launch files. |

### 5. ROS 2 Nodes (`rosetta/robots/ros2/nodes/`)

Executable lifecycle nodes — thin ROS wrappers over the adapter and core logic.

| File | What it does |
|---|---|
| `episode_recorder_node.py` | Records contract-declared topics straight to rosbag2 (raw messages — no decode at record time), exposing a `RecordEpisode` action plus start/cancel/delete-last-bag services. |
| `hil_manager_node.py` | Human-in-the-loop orchestrator: drives policy inference and bag recording via child action clients, muxes policy output against teleop, edge-detects teleop events. |
| `policy_runner_node.py` | `RunPolicy` action server for live inference: resolves the contract on configure (explicit path or chased from the checkpoint's sidecar), builds a `TopicBridge`, loads a `PolicyRunner` from the registry. |
| `episode_keyboard_node.py` | Raw-terminal keyboard control for episode recording (start/stop/discard/task-edit via async service calls). |
| `node_utils.py` | Node-only helpers: rcl-legal action termination (`finish_goal`), QoS introspection/rosbag2 conversion, bounded polling waits. |

### 6. Offline Porting (`rosetta/robots/ros2/offline/`)

| File | What it does |
|---|---|
| `port.py` | The `rosetta_port` CLI: bags → training dataset via a registry-loaded `DatasetWriter`; warns on contract mismatch, discards failed episodes. |
| `bag_frames.py` | Core of the porter: discovers bag dirs (with sharding), lazily replays messages through the same ingest/decode/resample pipeline as the live path. |

### 7. Launch & Parameters (`launch/`, `params/`)

`hil_launch.py` brings up the four-node HIL stack (robot_policy, optional reward_classifier, episode_recorder, hil_manager); `episode_recorder_launch.py` / `policy_runner_launch.py` / `episode_keyboard_launch.py` cover individual nodes. Each param YAML (`params/*.yaml`) is the single source of truth for its node's defaults.

### 8. Test Suite (`test/`)

55 files following the workspace testing pyramid (see `CLAUDE.md`): ROS-free unit tests for logic, executor-driven component tests for nodes (no sleeps), one `launch_testing` integration test (`test_bridge_launch.py`), and the capstone parity test `test_bag_live_parity.py` proving the offline porter produces frame-for-frame identical output to a live-oracle replay. `conftest.py` provides the session-scoped `rclpy_ctx` fixture (one `rclpy.init()`/`try_shutdown()` per session).

### 9. Project Support

`setup.py` (data files + console scripts), `setup.cfg`, `package.xml`, `.github/workflows/ci.yml` (Docker-native industrial_ci on ROS Jazzy with `rosetta_interfaces` as upstream), `README.md`, `CONTRIBUTING.md`.

## Key Concepts

- **The contract is the single source of truth.** Every component — recorder, porter, policy runner, HIL manager — consumes the same YAML contract. `load_contract` guarantees full validity at load time; nothing downstream re-validates.
- **Record raw, decode late.** Episodes are recorded as raw rosbag2 messages with the contract text embedded in bag metadata. Decoding happens at port/inference time, so recordings survive contract evolution.
- **One ingest path, live and offline.** `ingest.py` + `resample.py` are shared between `TopicBridge` (live) and `bag_frames.py` (offline). `test_bag_live_parity.py` enforces frame-for-frame parity.
- **Plugin seams everywhere.** Operators, codecs, dataset writers, and policy runners are all setuptools entry-point registries with fail-fast registration and protocol conformance checks. Adapter packages (`lerobot_rosetta`, etc.) plug in without rosetta knowing about them.
- **Lifecycle-node discipline.** All executable nodes inherit `RosettaLifecycleNode`: fail-loud configure, safety action published before deactivation, one shared teardown path.
- **ROS-agnostic core.** Contract, frames, and policies layers never import `rclpy` and never take `*_msgs` types in signatures — message ↔ array conversion lives in the adapter layer (`decoders.py`/`encoders.py`/`field_access.py`).

## Guided Tour (recommended reading order)

1. **Project Overview** — `README.md`, `rosetta/__init__.py`
2. **The Contract: One YAML per Robot** — `contracts/stone.yaml` (the annotated reference), then `so_101.yaml`, `turtlebot3.yaml`, `so_101_hil.yaml`
3. **Parsing and Validating Contracts** — `contract/schema.py`, `contract/model.py`, `contract/errors.py`
4. **From Contract to Runtime Specs** — `contract/specs.py`
5. **Operators and the Plugin System** — `contract/operators.py`, `contract/builtin_operators.py`, `contract/plugins.py`
6. **Frames, Codecs, and Naming** — `frames/codecs.py`, `frames/layout.py`, `frames/naming.py`, `examples/stone_codecs.py`
7. **Resampling and the FrameIO Seam** — `frames/resample.py`, `frames/protocols.py`
8. **ROS Messages In, Arrays Out** — `robots/ros2/decoders.py`, `encoders.py`, `field_access.py`
9. **The Live Bridge** — `robots/ros2/topic_bridge.py`, `ingest.py`, `timelines.py`, `rosetta_lifecycle_node.py`, `rclpy_utils.py`
10. **Recording Episodes** — `nodes/episode_recorder_node.py`, `nodes/episode_keyboard_node.py`, `robots/ros2/bag_metadata.py`
11. **Offline Porting: Bags to Datasets** — `offline/port.py`, `offline/bag_frames.py`
12. **The Policy Framework Seam** — `policies/protocols.py`, `policies/registry.py`, `contract/sidecar.py`
13. **Deployment: Policy Runner and HIL** — `nodes/policy_runner_node.py`, `nodes/hil_manager_node.py`, `nodes/node_utils.py`, `robots/ros2/node_host.py`
14. **Launch Files and Parameters** — `launch/hil_launch.py`, `robots/ros2/launch_utils.py`, `params/hil_manager.yaml`, `setup.py`
15. **Testing Pyramid and CI** — `test/conftest.py`, `test/test_bag_live_parity.py`, `test/test_bridge_launch.py`, `.github/workflows/ci.yml`

## Complexity Hotspots

Approach these with care — they carry the most intertwined logic:

| File | Why it's hot |
|---|---|
| `contract/schema.py` (~910 lines) | Three-layer load-time validation; highest fan-in in the codebase. A change here ripples everywhere. |
| `contract/specs.py` (~766 lines) | Spec resolution with dtype precedence, operator building, and multiple per-consumer projections. Second-highest fan-in. |
| `nodes/hil_manager_node.py` (~1085 lines) | Orchestrates policy, recorder, and teleop via child action clients with event edge-detection and muxing. |
| `nodes/episode_recorder_node.py` (~896 lines) | Action/service state machine over rosbag2 with many exit paths (see `test_episode_recorder_record.py`). |
| `frames/codecs.py` / `frames/layout.py` | The decode/encode boundary and the flat-vector slicing rules; correctness here is what the parity test defends. |
| `robots/ros2/topic_bridge.py` | Subscriptions + resampling + safety watchdog + lifecycle interplay. |
| `robots/ros2/rosetta_lifecycle_node.py` | Transition discipline every node depends on (safety publish before deactivate, shared teardown). |
| `offline/bag_frames.py` / `offline/port.py` | Lazy bag replay that must mirror live semantics exactly. |
| `contract/operators.py` | Invertibility tiers and serveability gating — subtle rules about which transforms may run in which direction. |

**Safe starting points** for a first contribution: `frames/naming.py`, `contract/errors.py`, `robots/ros2/field_access.py`, `timelines.py`, or any `simple`-rated test file.

## Working on the Code

- Build/test through the workspace: `pixi run build-with-tests` then `pixi run test`; lint with `pixi run lint`. See `rosetta_ws/CLAUDE.md` for testing standards (no sleeps, executor-driven component tests, session-scoped rclpy fixture).
- CI is per-package: `.github/workflows/ci.yml` runs industrial_ci (Docker, ROS Jazzy) + a separate ruff lint job.
- Explore interactively: `/understand-dashboard` serves the knowledge graph this guide was generated from.
