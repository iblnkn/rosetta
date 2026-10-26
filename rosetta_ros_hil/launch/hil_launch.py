# Copyright 2026 Brian Blankenau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Launch the human-in-the-loop system.

Includes rosetta's own launch files for the nodes it builds on, and starts the
HIL manager next to them:

  robot_policy/policy_runner       rosetta's policy_runner_launch.py, its action
                                   output remapped so the manager can mux it
  reward_classifier/policy_runner  the same, as a classifier (optional)
  episode_recorder                 rosetta's episode_recorder_launch.py, recording
                                   the real (muxed) topics
  hil_manager                      this package's node

Each included node gets a params file from this package's params/ layered over
rosetta's defaults; pass your own with robot_policy_params_file:=... and so on.

Usage:
    ros2 launch rosetta_ros_hil hil_launch.py \\
        action_remap_from:=/forward_position_controller/commands \\
        pretrained_name_or_path:=/path/to/policy_model

    ros2 launch rosetta_ros_hil hil_launch.py \\
        enable_reward_classifier:=true \\
        reward_classifier_pretrained_name_or_path:=/path/to/reward_model

    ros2 launch rosetta_ros_hil hil_launch.py configure:=false activate:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EqualsSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import LifecycleNode, PushRosNamespace, SetRemap

from rosetta_ros.launch_utils import autostart_handlers, typed_config, yaml_params


def _validate_remaps(context):
    """Fail at launch when the remap topology doesn't match the contract.

    The mux only works if the remapped topics are ones the contract declares;
    otherwise the policy would publish straight to the real command topic with
    no error anywhere.
    """
    from rosetta.contract.schema import load_contract

    contract = load_contract(LaunchConfiguration("contract_path").perform(context))
    action_topics = sorted({src.channel.topic for entry in contract.actions for src in entry.sources})
    remap_from = LaunchConfiguration("action_remap_from").perform(context)
    if remap_from not in action_topics:
        raise RuntimeError(
            f"action_remap_from '{remap_from}' is not an action topic of the contract "
            f"(action topics: {action_topics}); the HIL mux would never intercept policy output."
        )
    if typed_config(context, "enable_reward_classifier", bool):
        reward_topics = sorted({src.channel.topic for entry in contract.rewards for src in entry.sources})
        reward_from = LaunchConfiguration("reward_remap_from").perform(context)
        if reward_from not in reward_topics:
            raise RuntimeError(
                f"reward_remap_from '{reward_from}' is not a reward topic of the contract "
                f"(reward topics: {reward_topics}); the reward mux would never intercept classifier output."
            )
    return []


def _include(launch_file, *, namespace="", remap=None, condition=None, **arguments):
    """Include one of rosetta's launch files in its own scope, optionally namespaced and remapped."""
    actions = [PushRosNamespace(namespace)] if namespace else []
    if remap is not None:
        actions.append(SetRemap(src=remap[0], dst=remap[1]))
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file),
            launch_arguments={
                **arguments,
                "log_level": LaunchConfiguration("log_level"),
                "configure": LaunchConfiguration("configure"),
                "activate": LaunchConfiguration("activate"),
            }.items(),
        )
    )
    return GroupAction(actions, condition=condition)


def generate_launch_description():
    """Generate the launch description for the human-in-the-loop system."""
    hil_share = get_package_share_directory("rosetta_ros_hil")
    rosetta_launch = os.path.join(get_package_share_directory("rosetta_ros"), "launch")
    hil_params = os.path.join(hil_share, "params")
    hil_cfg = yaml_params(os.path.join(hil_params, "hil_manager.yaml"), "hil_manager")

    def arg(name, default, description):
        return DeclareLaunchArgument(name, default_value=default, description=description)

    launch_args = [
        arg("contract_path", os.path.join(hil_share, "contracts", "so_101_hil.yaml"), "Contract for every node"),
        arg("log_level", "info", "Logging level (debug, info, warn, error)"),
        arg("configure", "true", "Auto-configure every node on startup"),
        arg("activate", "true", "Auto-activate every node after configure"),
        # Robot policy: empty keeps the params-file value.
        arg("robot_policy_params_file", os.path.join(hil_params, "robot_policy.yaml"), "Robot policy params"),
        arg("pretrained_name_or_path", "", "Trained policy (empty = params file)"),
        arg("policy_type", "", "Policy architecture (empty = params file)"),
        arg("policy_device", "", "Inference device for both runners (empty = params file)"),
        arg("server_address", "", "Policy server host:port (empty = params file)"),
        # The policy's output goes to prefix + remap_from; the manager forwards it
        # to remap_from while the policy has control. The prefix is declared first
        # so the derived default resolves.
        arg("policy_remap_prefix", hil_cfg["policy_remap_prefix"], "Prefix for the remapped policy output"),
        arg("action_remap_from", "/leader_arm/joint_states", "Action topic to mux (must be in the contract)"),
        arg(
            "action_remap_to",
            [LaunchConfiguration("policy_remap_prefix"), LaunchConfiguration("action_remap_from")],
            "Remapped policy output topic",
        ),
        # Reward classifier.
        arg("enable_reward_classifier", str(hil_cfg["enable_reward_classifier"]).lower(), "Run a reward classifier"),
        arg(
            "reward_classifier_params_file",
            os.path.join(hil_params, "reward_classifier.yaml"),
            "Reward classifier params",
        ),
        arg("reward_classifier_contract_path", "", "Classifier contract (empty = contract_path)"),
        arg("reward_classifier_pretrained_name_or_path", "", "Trained classifier (empty = params file)"),
        arg("reward_classifier_policy_type", "", "Classifier architecture (empty = params file)"),
        arg("reward_classifier_server_address", "", "Classifier server host:port (empty = params file)"),
        arg("reward_remap_prefix", hil_cfg["reward_remap_prefix"], "Prefix for the remapped classifier output"),
        arg("reward_remap_from", "/reward", "Reward topic to mux (must be in the contract)"),
        arg(
            "reward_remap_to",
            [LaunchConfiguration("reward_remap_prefix"), LaunchConfiguration("reward_remap_from")],
            "Remapped classifier output topic",
        ),
        # Episode recorder.
        arg(
            "episode_recorder_params_file",
            os.path.join(hil_params, "episode_recorder.yaml"),
            "Episode recorder params",
        ),
        arg("bag_base_dir", "", "Bag output directory (empty = params file)"),
        arg("storage_id", "", "Bag storage, mcap or sqlite3 (empty = params file)"),
        # HIL manager.
        arg("enable_recording", str(hil_cfg["enable_recording"]).lower(), "Record each episode"),
        arg(
            "manage_policy_lifecycle",
            str(hil_cfg["manage_policy_lifecycle"]).lower(),
            "Send and cancel a RunPolicy goal per episode",
        ),
        arg("default_prompt", hil_cfg["default_prompt"], "Prompt when a goal, service or button leaves it empty"),
        arg("human_reward_positive", str(hil_cfg["human_reward_positive"]), "Reward for a success label"),
        arg("human_reward_negative", str(hil_cfg["human_reward_negative"]), "Reward for a failure label"),
    ]

    robot_policy = _include(
        os.path.join(rosetta_launch, "policy_runner_launch.py"),
        namespace="robot_policy",
        remap=(LaunchConfiguration("action_remap_from"), LaunchConfiguration("action_remap_to")),
        params_file=LaunchConfiguration("robot_policy_params_file"),
        contract_path=LaunchConfiguration("contract_path"),
        pretrained_name_or_path=LaunchConfiguration("pretrained_name_or_path"),
        policy_type=LaunchConfiguration("policy_type"),
        policy_device=LaunchConfiguration("policy_device"),
        server_address=LaunchConfiguration("server_address"),
    )

    reward_contract = PythonExpression(
        [
            "'",
            LaunchConfiguration("reward_classifier_contract_path"),
            "' or '",
            LaunchConfiguration("contract_path"),
            "'",
        ]
    )
    reward_classifier = _include(
        os.path.join(rosetta_launch, "policy_runner_launch.py"),
        namespace="reward_classifier",
        remap=(LaunchConfiguration("reward_remap_from"), LaunchConfiguration("reward_remap_to")),
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration("enable_reward_classifier"), "true")),
        params_file=LaunchConfiguration("reward_classifier_params_file"),
        contract_path=reward_contract,
        pretrained_name_or_path=LaunchConfiguration("reward_classifier_pretrained_name_or_path"),
        policy_type=LaunchConfiguration("reward_classifier_policy_type"),
        policy_device=LaunchConfiguration("policy_device"),
        server_address=LaunchConfiguration("reward_classifier_server_address"),
    )

    episode_recorder = _include(
        os.path.join(rosetta_launch, "episode_recorder_launch.py"),
        params_file=LaunchConfiguration("episode_recorder_params_file"),
        contract_path=LaunchConfiguration("contract_path"),
        bag_base_dir=LaunchConfiguration("bag_base_dir"),
        storage_id=LaunchConfiguration("storage_id"),
    )

    hil_manager = LifecycleNode(
        package="rosetta_ros_hil",
        executable="hil_manager_node",
        name="hil_manager",
        namespace="",
        output="screen",
        emulate_tty=True,
        parameters=[
            hil_cfg,
            {
                "contract_path": LaunchConfiguration("contract_path"),
                "enable_reward_classifier": LaunchConfiguration("enable_reward_classifier"),
                "enable_recording": LaunchConfiguration("enable_recording"),
                "manage_policy_lifecycle": LaunchConfiguration("manage_policy_lifecycle"),
                "default_prompt": LaunchConfiguration("default_prompt"),
                "policy_remap_prefix": LaunchConfiguration("policy_remap_prefix"),
                "reward_remap_prefix": LaunchConfiguration("reward_remap_prefix"),
                "human_reward_positive": LaunchConfiguration("human_reward_positive"),
                "human_reward_negative": LaunchConfiguration("human_reward_negative"),
            },
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    return LaunchDescription(
        [
            *launch_args,
            OpaqueFunction(function=_validate_remaps),
            robot_policy,
            reward_classifier,
            episode_recorder,
            hil_manager,
            *autostart_handlers(hil_manager),
        ]
    )
