^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rosetta_ros_hil
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* Breaking: ``hil_launch.py`` drops ``reward_classifier_policy_type``. The
  classifier is always a ``reward_classifier``.
* Renamed from ``rosetta_hil``: a ROS app built on ``rosetta_ros``, named like
  MoveIt's ``moveit_ros_*`` packages. ``ros2 launch rosetta_ros_hil hil_launch.py``.
* Breaking: ``action_remap_from`` and ``reward_remap_from`` are gone. The policy
  and the classifier publish every contract action and reward topic under
  ``policy_remap_prefix`` / ``reward_remap_prefix``, so takeover covers every
  action topic (before, only one was intercepted).
* Takeover forwarding lives in ``forwarding.py`` and works per topic: a topic
  shared by several sources is forwarded once, each teleop input resolves to
  exactly one action (an ambiguous target fails configure), and teleop
  commands and feedback carry a stamp.
* ``hil_intervention`` is transient-local and published on every change,
  between episodes too, as well as at the feedback rate during an episode.
* ``hil_episode`` imports nothing from ROS.
* Imports the node toolkit from ``rosetta_ros`` and includes its launch files.
* Owns its teleop event names (``HIL_EVENT_NAMES``) and logs a warning at
  configure for contract event names it ignores.
* Ships ``contracts/so_101_hil.yaml``, the launch file's default contract.
* Added ``~/status`` and ``~/result`` topics (``ManageEpisodeStatus``,
  ``ManageEpisodeResult``): the action feedback and result fields for every
  episode, including ones started by service or button. ``~/result`` is
  transient-local.
* ``hil_launch.py`` includes Rosetta's ``policy_runner_launch.py`` and
  ``episode_recorder_launch.py`` instead of declaring those nodes itself. Their
  settings come from ``params/robot_policy.yaml``, ``reward_classifier.yaml``
  and ``episode_recorder.yaml`` (overridable with ``*_params_file:=``), layered
  over Rosetta's defaults. The tuning launch arguments ``actions_per_chunk``,
  ``chunk_size_threshold``, ``aggregate_fn_name``, ``obs_similarity_atol``,
  ``feedback_rate_hz`` and ``default_max_duration_s`` moved into those files.
* ``hil_manager_node``, ``hil_launch.py`` and ``params/hil_manager.yaml`` moved
  here from ``rosetta``. The node uses only Rosetta's public Python API, and its
  interfaces stay in ``rosetta_interfaces``. Launch with
  ``ros2 launch rosetta_ros_hil hil_launch.py``.
