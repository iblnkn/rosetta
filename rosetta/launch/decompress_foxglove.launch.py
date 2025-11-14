from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch file to republish Foxglove CompressedVideo -> sensor_msgs/Image

    This uses image_transport's `republish` node and the `foxglove` transport
    plugin (foxglove_compressed_video_transport). Make sure that package is
    installed and the transport plugin is available in your ROS2 environment.

    Usage examples:
      ros2 launch rosetta decompress_foxglove.launch.py
      ros2 launch rosetta decompress_foxglove.launch.py input_topic:=/camera/color/foxglove out_topic:=/camera/color/image_raw
    """

    input_topic = LaunchConfiguration("input_topic")
    out_topic = LaunchConfiguration("out_topic")

    ld = LaunchDescription()

    ld.add_action(
        DeclareLaunchArgument(
            "input_topic",
            default_value="/camera/color/foxglove",
            description="Topic that publishes foxglove_msgs/CompressedVideo",
        )
    )

    ld.add_action(
        DeclareLaunchArgument(
            "out_topic",
            default_value="/camera/color/image_raw",
            description="Topic to republish decoded sensor_msgs/Image",
        )
    )

    # image_transport republish node: arguments specify the transport name
    # here we use the foxglove transport. The remapping maps in/foxglove -> input
    # and out -> output topic.
    ld.add_action(
        Node(
            package="image_transport",
            executable="republish",
            name="image_republisher_foxglove",
            output="screen",
            arguments=["foxglove"],
            remappings=[("in/foxglove", input_topic), ("out", out_topic)],
        )
    )

    return ld
