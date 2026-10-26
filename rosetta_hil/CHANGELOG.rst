^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rosetta_hil
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
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
  interfaces moved to ``rosetta_hil_interfaces``. Launch with
  ``ros2 launch rosetta_hil hil_launch.py``.
