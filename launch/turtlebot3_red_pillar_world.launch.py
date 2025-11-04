#!/usr/bin/env python3
#
# Custom launch file for TurtleBot3 Gazebo with 100Hz clock rate
# Based on the original turtlebot3_world.launch.py
#

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    # Get the models directory path
    # Use source directory directly for custom models (no build required)
    # This allows using custom models with system-installed Gazebo Sim
    launch_file_path = os.path.abspath(__file__)
    # Navigate from: src/action/rosetta/launch/turtlebot3_red_pillar_world.launch.py
    # To: src/simulation/turtlebot3_simulations/turtlebot3_gazebo/models
    # Find workspace root by looking for 'src' directory that contains both 'action' and 'simulation'
    current = launch_file_path
    while current != os.path.dirname(current):  # Stop at root
        src_check = os.path.join(current, 'src')
        if os.path.exists(os.path.join(src_check, 'action')) and \
           os.path.exists(os.path.join(src_check, 'simulation')):
            workspace_root = current
            break
        current = os.path.dirname(current)
    else:
        # Fallback: go up from launch file
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(launch_file_path))))
    
    models_dir = os.path.join(
        workspace_root,
        'src',
        'simulation',
        'turtlebot3_simulations',
        'turtlebot3_gazebo',
        'models'
    )
    
    # Verify the models directory exists
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f'Models directory not found: {models_dir}')
    
    # Set GAZEBO_MODEL_PATH for Gazebo Classic to find custom models
    # Gazebo Classic uses GAZEBO_MODEL_PATH (not GZ_SIM_RESOURCE_PATH which is for Gazebo Sim)
    # Get existing value if it exists, otherwise use empty string
    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    if existing_model_path:
        model_path = f'{models_dir}:{existing_model_path}'
    else:
        model_path = models_dir
    
    print(f'[turtlebot3_red_pillar_world] Using models directory: {models_dir}')
    print(f'[turtlebot3_red_pillar_world] Setting GAZEBO_MODEL_PATH to: {model_path}')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='-2.0')
    y_pose = LaunchConfiguration('y_pose', default='-0.5')
    gui = LaunchConfiguration('gui', default='true')

    # Use the workspace world file if it exists, otherwise fall back to system
    workspace_world = os.path.join(
        workspace_root,
        'src',
        'simulation',
        'turtlebot3_simulations',
        'turtlebot3_gazebo',
        'worlds',
        'turtlebot3_world.world'
    )
    if os.path.exists(workspace_world):
        world = workspace_world
        print(f'[turtlebot3_red_pillar_world] Using workspace world file: {world}')
    else:
        world = os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'worlds',
            'turtlebot3_world.world'
        )
        print(f'[turtlebot3_red_pillar_world] Using system world file: {world}')

    # Path to the parameter file for 100Hz clock rate
    gazebo_params_file = os.path.join(
        get_package_share_directory('rosetta'),
        'params',
        'turtlebot3_red_pillar_world_params.yaml'
    )

    # IMPORTANT: gzserver.launch.py prepends default paths, then appends GAZEBO_MODEL_PATH.
    # To ensure our custom models are found first, we need to set GAZEBO_MODEL_PATH
    # such that our path comes before system turtlebot3_gazebo path.
    # Since gzserver prepends defaults, we'll use SetEnvironmentVariable action
    # to set it for the gzserver process specifically.
    
    # Filter out system turtlebot3_gazebo path from existing paths if present
    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    filtered_paths = [models_dir]  # Always start with our custom models
    
    if existing_model_path:
        for path in existing_model_path.split(':'):
            path = path.strip()
            if path and 'turtlebot3_gazebo' not in path or 'reo_ws' in path:
                if path != models_dir:  # Avoid duplicates
                    filtered_paths.append(path)
    
    # Set for this process too
    final_model_path = ':'.join(filtered_paths)
    os.environ['GAZEBO_MODEL_PATH'] = final_model_path
    
    print(f'[turtlebot3_red_pillar_world] Setting GAZEBO_MODEL_PATH to: {final_model_path}')
    
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={
            'world': world,
            'params_file': gazebo_params_file
        }.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        ),
        condition=IfCondition(gui)
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )

    ld = LaunchDescription()

    # Set environment variable for Gazebo Classic to find custom models
    # gzserver.launch.py reads GAZEBO_MODEL_PATH from os.environ and passes it via additional_env
    ld.add_action(SetEnvironmentVariable('GAZEBO_MODEL_PATH', final_model_path))

    # Declare launch arguments
    ld.add_action(DeclareLaunchArgument('gui', default_value='true', description='Launch Gazebo GUI'))

    # Add the commands to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)

    return ld
