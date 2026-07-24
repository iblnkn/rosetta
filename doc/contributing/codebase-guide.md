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
| Nodes | `rosetta/robots/ros2/nodes/` | Thin executable wrappers: recorder, keyboard, policy runner, HIL manager, plus `node_utils.py` for the helpers only they need. |
| Offline porting | `rosetta/robots/ros2/offline/` | `port.py` (the `rosetta_port` CLI) and `bag_frames.py` (lazy bag replay through the same ingest pipeline as live). |
| Launch and params | `launch/`, `params/` | Each param YAML is the single source of truth for its node's defaults; `robots/ros2/launch_utils.py` holds the helpers the launch files share. |
| Tests | `test/` | Unit tests for ROS-free logic, executor-driven component tests for nodes, one `launch_testing` integration test, and the parity capstone `test_bag_live_parity.py` proving offline output matches a live-oracle replay frame for frame. |
| Contracts | `contracts/` | `stone.yaml` is the annotated reference exercising every schema feature. `so_101.yaml`, `so_101_hil.yaml`, `turtlebot3.yaml` are real robots. |

### Helper modules

Three helper modules exist to keep imports honest rather than to collect odds and ends. Which one a helper belongs in is decided by what importing it drags in, so put new helpers where the import rule says, not where the caller happens to live.

| Module | Import rule | Holds |
|--------|-------------|-------|
| `robots/ros2/rclpy_utils.py` | Needs `rclpy` at import, but no node instance | `qos_profile_from_dict` (contract `qos` mapping → `QoSProfile`, raising `ValueError` so a typo dies at contract load instead of silently taking the default policy), `lifecycle_state_label`, `require_transition_success` |
| `robots/ros2/nodes/node_utils.py` | Needs a running node; imported only by the nodes | `finish_goal` (rcl-legal terminal transitions), `qos_to_rosbag2` for bag topic metadata, `wait_until`, `positive_rate_descriptor`, and `spin_lifecycle_node` (the shared `main`) |
| `robots/ros2/launch_utils.py` | Imported only by `launch/`, never by library code | `autostart_handlers` (the configure-on-start / activate-on-inactive chain), `typed_config`, `yaml_params` |

Two rules these encode:

- **Keep `rclpy` out of the ROS-less paths.** `timelines.py` and `field_access.py` are pure Python on purpose — ingest and the codecs use them so offline porting never imports `rclpy`. A helper that needs `rclpy` goes in `rclpy_utils.py`; it does not get inlined into a pure-Python sibling. `launch_utils.py` keeps `launch`/`launch_ros` out of the library import graph the same way.
- **`lifecycle_state_label` is the one sanctioned reach-in.** rclpy has no public synchronous state accessor (the public route is the `GetState` service, which needs an executor round-trip), so that helper reads the private `_state_machine`. Never read `_state_machine` anywhere else — its shape shifts across distros, and one call site bounds the blast radius.

`spin_lifecycle_node` is worth reading before touching any node's shutdown path: it disables rclpy's SIGINT handler and treats `KeyboardInterrupt` as the shutdown trigger, because the default handler tears the context down before `trigger_shutdown()` can run — and `destroy_node()` never fires `on_shutdown`. That transition is what closes an open bag writer and publishes the safety action.

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
9. The live bridge: `robots/ros2/topic_bridge.py`, `ingest.py`, `timelines.py`, `rosetta_lifecycle_node.py`, `rclpy_utils.py`
10. Recording: `nodes/episode_recorder_node.py`, `nodes/episode_keyboard_node.py`, `robots/ros2/bag_metadata.py`
11. Offline porting: `offline/port.py`, `offline/bag_frames.py`
12. The policy seam: `policies/protocols.py`, `policies/registry.py`, `contract/sidecar.py`
13. Deployment and HIL: `nodes/policy_runner_node.py`, `nodes/hil_manager_node.py`, `nodes/node_utils.py`, `robots/ros2/node_host.py`
14. Launch and params: `launch/hil_launch.py`, `robots/ros2/launch_utils.py`, `params/hil_manager.yaml`, `setup.py`
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
- Docs: `pixi run docs-serve` for a live-reload local site. `rosdoc2 build --package-path .` replicates the docs.ros.org build. The docs follow [Diátaxis](https://diataxis.fr/). Place new content with the [compass](https://diataxis.fr/compass/): does the content inform action or cognition, and does the reader acquire skill or apply skill? Action + acquisition → tutorial. Action + application → how-to. Cognition + application → reference. Cognition + acquisition → explanation. Which *repo* a page belongs in is a separate question — see below.
- CI is per-package: `.github/workflows/ci.yml` runs industrial_ci (Docker, ROS Jazzy) plus a separate ruff lint job.
- Invariants to preserve in any change: the contract validates fully at load (nothing downstream re-validates), the core layers never import `rclpy` or take `*_msgs` types, and live and offline share one ingest path (the parity test enforces this).

## Where documentation lives

Rosetta spans several repositories, but readers get **one site**:
<https://iblnkn.github.io/rosetta/>, built by `.github/workflows/docs.yml` from
this `doc/` tree on every push to `main`. Released packages additionally get
per-package `docs.ros.org` pages via `rosdoc2.yaml` — those are API silos with
no shared navigation, so always link the project site as "full documentation".

The current arrangement is a **monolith**: every page lives here, and the
satellite repos (`rosetta_interfaces`, `lerobot_rosetta`,
`lerobot_robot_rosetta`, `lerobot_teleoperator_rosetta`) carry a README only.

A satellite README covers what a reader needs *while looking at that repo* —
what the package is, its entry points, how to build it — and ends with a link
to the project site. It does not restate the contract schema. Two of them did,
both drifted to a schema that no longer parses, and that is the entire argument
for this rule: **the contract is documented in exactly one place**
(`reference/contract.md`). A short illustrative snippet is fine; a schema
reference is not.

Some duplication runs the other way and is accepted deliberately.
`reference/interfaces.md` documents message fields whose source of truth is
`rosetta_interfaces`'s `.action`/`.srv` files, because a reader looking up
`RunPolicy` should not be bounced to another site mid-task.

### When to distribute

Two rules govern any future split:

1. **A satellite earns its own `doc/` when it has two or more pages of genuine
   package-scoped reference or how-to material.** Reference and how-to can live
   next to the code they describe. Tutorials and explanation cannot — the
   tutorial spans four repos, and the design rationale is cross-cutting by
   definition, so the narrative half stays here permanently.
2. **Never split content across repos without the aggregator landing in the
   same change.** A page moving out while nothing joins the sites back together
   is the one state to avoid: readers hit two hosted surfaces with no shared
   navigation. The aggregator belongs in `rosetta_ws`, which already pins every
   satellite in `repos/src.repos` — that manifest is what makes a composed
   build reproducible, and it is why the aggregator goes there rather than
   here.
