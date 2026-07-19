# Codebase Guide

For contributors changing Rosetta's code. Users of Rosetta want the [tutorial](../tutorials/first-policy.md) instead. Design rationale lives in [Design](../explanation/design.md). This page covers where things are, what order to read them in, and where to tread carefully.

## Layer map

Dependency direction flows downward: nodes → adapter → frames/contract. Tests reach everything.

| Layer | Where | Role |
|-------|-------|------|
| Contract | `rosetta/contract/` | The center. `schema.py` parses and validates (highest fan-in in the codebase), `model.py` holds the pure-data document model, `specs.py` resolves runtime `StreamSpec`s, `operators.py` + `builtin_operators.py` implement the transform framework, `plugins.py` loads entry-point registries, `sidecar.py` places and resolves embedded contracts, `errors.py` defines `ContractValidationError`. |
| Frames and codecs | `rosetta/frames/` | Framework-neutral interlingua. `codecs.py` is the message-to-array boundary and registry, `layout.py` maps contract keys to flat-vector slices, `resample.py` holds `StreamBuffer` (the one resampler), `protocols.py` defines the `FrameIO` seam, `naming.py` provides key-name helpers. |
| Policy framework | `rosetta/policies/` | The framework-agnostic seam: `DatasetWriter` and `PolicyRunner` protocols plus the entry-point registry adapter packages plug into. |
| ROS2 adapter | `rosetta/robots/ros2/` | Middleware integration without executables. `topic_bridge.py` (live plumbing plus safety watchdog), `ingest.py` (the shared ingest path behind live/offline parity), `decoders.py`/`encoders.py`, `field_access.py` (dotted selectors), `rosetta_lifecycle_node.py` (the lifecycle idiom every node inherits), `timelines.py`, `bag_metadata.py`, `node_host.py`. |
| Nodes | `rosetta/robots/ros2/nodes/` | Thin executable wrappers: recorder, keyboard, policy runner, HIL manager. |
| Offline porting | `rosetta/robots/ros2/offline/` | `port.py` (the `rosetta_port` CLI) and `bag_frames.py` (lazy bag replay through the same ingest pipeline as live). |
| Launch and params | `launch/`, `params/` | Each param YAML is the single source of truth for its node's defaults. |
| Tests | `test/` | Unit tests for ROS-free logic, executor-driven component tests for nodes, one `launch_testing` integration test, and the parity capstone `test_bag_live_parity.py` proving offline output matches a live-oracle replay frame for frame. |
| Contracts | `contracts/` | `stone.yaml` is the annotated reference exercising every schema feature. `so_101.yaml`, `so_101_hil.yaml`, `turtlebot3.yaml` are real robots. |

## Reading order

A dependency-ordered tour. Each step builds on the previous.

1. `README.md`, `rosetta/__init__.py`
2. The contract as a user sees it: `contracts/stone.yaml`, then the real contracts
3. Parsing and validation: `contract/schema.py`, `contract/model.py`, `contract/errors.py`
4. Contract to runtime specs: `contract/specs.py`
5. Operators and plugins: `contract/operators.py`, `contract/builtin_operators.py`, `contract/plugins.py`
6. Frames and codecs: `frames/codecs.py`, `frames/layout.py`, `frames/naming.py`, `examples/stone_codecs.py`
7. Resampling and the seam: `frames/resample.py`, `frames/protocols.py`
8. Messages in, arrays out: `robots/ros2/decoders.py`, `encoders.py`, `field_access.py`
9. The live bridge: `robots/ros2/topic_bridge.py`, `ingest.py`, `timelines.py`, `rosetta_lifecycle_node.py`
10. Recording: `nodes/episode_recorder_node.py`, `nodes/episode_keyboard_node.py`, `robots/ros2/bag_metadata.py`
11. Offline porting: `offline/port.py`, `offline/bag_frames.py`
12. The policy seam: `policies/protocols.py`, `policies/registry.py`, `contract/sidecar.py`
13. Deployment and HIL: `nodes/policy_runner_node.py`, `nodes/hil_manager_node.py`, `robots/ros2/node_host.py`
14. Launch and params: `launch/hil_launch.py`, `params/hil_manager.yaml`, `setup.py`
15. Tests and CI: `test/conftest.py`, `test/test_bag_live_parity.py`, `test/test_bridge_launch.py`, `.github/workflows/ci.yml`

## Complexity hotspots

These files carry the most intertwined logic. Read the matching tests before changing them.

| File | Why |
|------|-----|
| `contract/schema.py` | Three-layer load-time validation (shape, timeline attestation, codec/operator resolvability). Highest fan-in. A change here ripples everywhere. |
| `contract/specs.py` | Spec resolution: dtype precedence, operator building, per-consumer projections. Second-highest fan-in. |
| `nodes/hil_manager_node.py` | Orchestrates policy, recorder, and teleop via child action clients with event edge-detection and muxing. |
| `nodes/episode_recorder_node.py` | Action/service state machine over rosbag2 with many exit paths. |
| `frames/codecs.py`, `frames/layout.py` | The decode/encode boundary and flat-vector slicing. The parity test defends correctness here. |
| `robots/ros2/topic_bridge.py` | Subscriptions, resampling, safety watchdog, and lifecycle interplay in one place. |
| `robots/ros2/rosetta_lifecycle_node.py` | Transition discipline every node depends on: safety publish before deactivate, one shared teardown path. |
| `offline/bag_frames.py`, `offline/port.py` | Lazy bag replay obligated to mirror live semantics exactly. |
| `contract/operators.py` | Invertibility tiers and serveability gating. Subtle rules about which transforms run in which direction. |

Safe starting points for a first contribution: `frames/naming.py`, `contract/errors.py`, `robots/ros2/field_access.py`, `robots/ros2/timelines.py`, or any small test file.

## Working on the code

- Build and test through the workspace: `pixi run build-with-tests`, then `pixi run test`. Lint with `pixi run lint`. The workspace `CLAUDE.md` documents the testing standards (no sleeps, executor-driven component tests, session-scoped rclpy fixture).
- Docs: `pixi run docs-serve` for a live-reload local site. `rosdoc2 build --package-path .` replicates the docs.ros.org build. The docs follow [Diátaxis](https://diataxis.fr/). Place new content with the [compass](https://diataxis.fr/compass/): does the content inform action or cognition, and does the reader acquire skill or apply skill? Action + acquisition → tutorial. Action + application → how-to. Cognition + application → reference. Cognition + acquisition → explanation.
- CI is per-package: `.github/workflows/ci.yml` runs industrial_ci (Docker, ROS Jazzy) plus a separate ruff lint job.
- Invariants to preserve in any change: the contract validates fully at load (nothing downstream re-validates), the core layers never import `rclpy` or take `*_msgs` types, and live and offline share one ingest path (the parity test enforces this).
