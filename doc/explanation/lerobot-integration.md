# LeRobot Integration

## Plugin architecture

The `lerobot_robot_rosetta` and `lerobot_teleoperator_rosetta` packages implement LeRobot's [Robot](https://huggingface.co/docs/lerobot/integrate_hardware) and [Teleoperator](https://huggingface.co/docs/lerobot/integrate_hardware#adding-a-teleoperator) interfaces. They follow LeRobot's [plugin conventions](https://huggingface.co/docs/lerobot/integrate_hardware#using-your-own-lerobot-devices-): each LeRobot CLI scans installed distributions for the `lerobot_robot_*` / `lerobot_teleoperator_*` name prefixes and imports them, which runs the packages' config registrations (`@RobotConfig.register_subclass("rosetta")`). The end user installs the package and passes `--robot.type=rosetta`. No import or registration step beyond installing.

**Typical LeRobot robots** (like `so101_follower`) communicate directly with hardware:

- Motors via serial/CAN (`FeetechMotorsBus`, `DynamixelMotorsBus`)
- Cameras via USB/OpenCV
- The `Robot` class IS the hardware interface

**Rosetta robots** are ROS2 lifecycle nodes:

- Subscribe to ROS2 topics for observations
- Publish to ROS2 topics for actions
- Hardware drivers exist elsewhere in the ROS2 graph
- The contract YAML defines topic-to-feature mapping

**Important:** `lerobot_robot_rosetta` creates a ROS2 lifecycle node internally, so **your system needs ROS2 installed**, even when you invoke the plugin through LeRobot's standard CLI tools. When `policy_runner_node` launches inference, the chain is: `policy_runner_node` (ROS2 node) → LeRobot `RobotClient` → `lerobot_robot_rosetta` (also a ROS2 node) → your robot's ROS2 topics. Both the convenience node and the robot plugin are ROS2 nodes in the same ROS2 graph.

Any ROS2 robot works with LeRobot's tools this way. Define a contract and use `--robot.type=rosetta`.

## Compared with LeRobot's native robots

LeRobot's own `Robot` classes are hardware drivers: open the serial port, read motor registers, grab camera frames, return a dict. This design fits robots LeRobot owns end to end, like an SO-101 on a desk, and integration costs one Python subclass per robot.

Rosetta targets robots where a driver layer already exists as a pub/sub graph: controllers, safety layers, TF, several sensors at different rates. Five differences follow:

- **Declarative vs imperative.** LeRobot integration is code, one class per robot. Rosetta integration is data, one YAML per robot. A generic Robot class executes any contract, so a new robot means a new file, not new code.
- **Where drivers live.** LeRobot puts hardware in-process with the policy loop. Rosetta leaves drivers in the graph and joins as a peer node. Your existing control and safety stack stays untouched.
- **Timing semantics.** LeRobot's record loop samples on demand: call `get_observation()` at fps and take what the bus returns. Reasonable for one arm and one USB camera. A distributed graph has real clocks, jitter, and multi-rate sensors, so Rosetta declares per-channel timelines and alignment instead.
- **Raw preservation.** LeRobot commits to the training format at record time. Rosetta records raw bags and assigns meaning at port time (see [Record raw, decide late](record-raw-decide-late.md)).
- **Complement, not competitor.** Rosetta plugs into LeRobot's own extension points. `lerobot-record`, `lerobot-train`, and the async inference server work unchanged. Rosetta replaces the per-robot Robot subclass and nothing else.

The shortest version: LeRobot connects to hardware, Rosetta connects to middleware. LeRobot asks how to read your motors. Rosetta asks what your topics mean.

## ROS2 lifecycle mapping

LeRobot's `connect()` / `disconnect()` map to ROS2 lifecycle transitions:

| LeRobot Method | Lifecycle Transition | Effect |
|----------------|---------------------|--------|
| `configure()`, or lazily inside `connect()` | `configure` | Create subscriptions (start buffering), create publishers (disabled), create the watchdog timer (inert) |
| `connect()` | `activate` | Enable publishers, watchdog becomes effective |
| `disconnect()` | `deactivate` → `cleanup` | Safety action, disable publishers, destroy resources |

`connect()` configures first when needed, then activates, so a plain connect performs both transitions. Calling `configure()` early splits the expensive setup from the moment publishers go live.

## Policy inference topology

The `policy_runner_node` delegates inference to a gRPC policy server (`lerobot_rosetta.policy_server`, a thin preload/cache wrapper over LeRobot's `lerobot.async_inference.policy_server`). The server has no ROS2 dependency and runs on any machine with LeRobot and a GPU. Benefits:

- Better GPU memory management
- Support for all LeRobot policy types without code changes
- Consistent behavior between training and deployment
- Runs on a remote machine, so a resource-constrained robot offloads inference over the network

When `launch_local_server` is `true`, the node starts the server and fully loads the model at **configure** time, so the first `run_policy` goal costs the same as any other. The configure transition blocks until the model is up (bounded by `server_startup_timeout_sec`), GPU memory is held from startup, and later goals reuse the loaded model instead of re-reading the checkpoint on every handshake.
