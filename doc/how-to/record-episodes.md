# Record Episodes

This guide shows you how to record demonstration bags for training. Full parameter tables live in [nodes](../reference/nodes.md#episode_recorder_node). Why bags at all: [Record raw, decide late](../explanation/record-raw-decide-late.md).

## Start the recorder

```bash
ros2 launch rosetta episode_recorder_launch.py contract_path:=/path/to/contract.yaml
```

## Control recording via ROS2 actions

```bash
# Start an episode. Ctrl-C stops and saves it.
ros2 action send_goal /record_episode \
    rosetta_interfaces/action/RecordEpisode "{prompt: 'task description'}"
```

Or stop via the cancel service:

```bash
ros2 service call /episode_recorder/cancel_recording std_srvs/srv/Trigger
```

Discard a bad episode after saving:

```bash
ros2 service call /episode_recorder/delete_last_bag std_srvs/srv/Trigger
```

## Control recording via keyboard

The `episode_keyboard_node` starts, stops, and discards episodes with single key presses. Run the node in a second terminal while the recorder runs:

```bash
ros2 run rosetta episode_keyboard_node
```

Or via launch:

```bash
ros2 launch rosetta episode_keyboard_launch.py default_prompt:="pick up the cube"
```

Keys: `r` start, `s` stop and save, `d` discard, `t` edit prompt, `q` quit. Full table: [nodes](../reference/nodes.md#episode_keyboard_node).

## Common overrides

```bash
# Override output directory
ros2 launch rosetta episode_recorder_launch.py \
    contract_path:=/path/to/contract.yaml \
    bag_base_dir:=/data/recordings

# Set a max duration and storage format
ros2 launch rosetta episode_recorder_launch.py \
    contract_path:=/path/to/contract.yaml \
    default_max_duration:=600.0 \
    storage_id:=sqlite3
```

## Exclude noisy topics

The recorder captures every topic on the graph by default. Exclude with regex:

```python
# In a launch file
Node(
    package='rosetta',
    executable='episode_recorder_node',
    parameters=[{
        'contract_path': '/path/to/contract.yaml',
        'exclude_topics': ['/camera/.*/debug', '/diagnostics'],
    }],
)
```

Set `record_all: false` to record only contract-declared topics.

## How many episodes?

Plan on 50 to 200+ demonstrations depending on task complexity. Diverse, high-quality demonstrations produce better policies. For data collection tips, see [Collecting Your Dataset](https://abenstirling.com/lerobot/) and [Improving Your Robotics AI Model](https://docs.phospho.ai/learn/improve-robotics-ai-model).

## Record without the recorder node

Any valid rosbag2 file containing the contract's topics works: `ros2 bag record`, custom `rosbag2_py` scripts, third-party tools. See [Port existing bags](port-existing-bags.md) for what the porter expects.
