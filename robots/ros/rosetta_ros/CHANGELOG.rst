^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rosetta_ros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
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
