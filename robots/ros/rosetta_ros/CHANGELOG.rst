^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rosetta_ros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* Breaking: ``rosetta_port --framework`` is now ``--format``. It picks a
  dataset writer, and LeRobot's comes from the new ``lerobot_dataset_rosetta``.
* ``policy_runner_node`` with no ``contract_path`` asks its policy runner's
  ``find_contract`` for the contract ``pretrained_name_or_path`` was trained
  with, instead of following LeRobot's layout itself.
* ``episode_recorder_node`` subscribes raw and writes each message's
  serialized bytes as received, with no deserialize and serialize per message.
* Every lifecycle node spins on rclpy's single-threaded ``EventsExecutor``.
  The ``MultiThreadedExecutor`` cost the recorder about 3x the CPU and delayed
  its messages by tens of milliseconds.
* ``RosettaLifecycleNode.create_work_action_server`` serves an action whose
  goals run on their own thread, wired to ``on_goal`` and ``on_cancel``. The
  recorder and the policy runner use it, and no longer set callback groups.
* Breaking: ``child_goals.cancel_goal`` drops ``cancel_timeout``. It sends the
  cancel and, with ``wait_result``, waits only for the result. Without it, it
  returns at once.
* ``params/policy_runner.yaml`` sets ``policy_type`` to empty, so the LeRobot
  runner reads it from the checkpoint.
* Split out of ``rosetta``: the ROS 2 robot interface, its nodes, launch
  files, params, example contracts, ``rosetta_port`` and the node toolkit.
  Earlier history is in the ``rosetta`` changelog.
* Registers the ``ros2`` robot interface (``Ros2Interface``).
* ``RosParameterHost`` hosts a policy runner's settings as node parameters.
* ``policy_runner_launch.py output_prefix:=`` publishes every contract action
  and reward topic under a prefix, with remaps from
  ``launch_utils.contract_output_remaps``.
