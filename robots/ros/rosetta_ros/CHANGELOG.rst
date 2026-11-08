^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rosetta_ros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* ``episode_recorder_node`` names the observation and action topics that
  recorded nothing in the result's ``message`` (``no messages on: <topics>``)
  and logs its per-topic summary as a warning. The bag is kept and the goal
  still ends with its usual state. Before, a recording that lost its joint
  states ended ``SUCCEEDED`` with an empty ``message``.
* ``episode_keyboard_node`` drives the recorder's ``RecordEpisode`` action
  (``recorder_action_name``, default ``/record_episode``) instead of its
  ``start_recording`` and ``cancel_recording`` services, and prints each
  goal's result. A recording that hit its time limit used to print nothing
  and leave the keyboard showing REC. ``d`` pressed before a recording's
  result arrives deletes that recording's bag and doesn't report it saved.
  ``delete_last_bag`` is still a service call. A refused stop leaves the
  keyboard recording, so ``s`` can be pressed again. After ``d``, the bag is
  deleted even if the stop is refused, and a second ``d`` is ignored.
* Breaking: ``rosetta_port`` exits 1 when some bags failed, after writing the rest, and
  names the failed bags. It exited 0, so a script couldn't tell a partial
  port from a complete one.
* ``rosetta_port`` warns when an action topic is in a bag but has no
  messages, since every action in that episode is zero. ``iter_bag_frames``
  takes ``warn_empty_keys`` for this. Its "not in bag" warning names the bag.
* ``rosetta_port`` checks its arguments up front: a ``--contract`` that is not
  a file, a ``--raw-dir`` that is not a directory, an unknown ``--format`` and
  a bad shard are usage errors.
* ``policy_runner_launch.py`` starts the runner with ``output="both"``, so
  its output, and the policy server's, are kept in the launch's log directory
  as well as shown.
* A live connection whose ROS context is shut down outside ``close()`` (by
  ``rclpy.shutdown()`` on it, or by rclpy's signal handler in a process that
  installs one) raises on its next call instead of serving its last frames
  and a stopped clock. Its ``close()`` then logs that the safety action could
  not be sent.
* A live connection that fails to start raises its own error when closing it
  afterwards fails too. The close error is attached as a note.
* The recorder hands the prompt, contract and goal id to rosbag2 as
  ``custom_data`` when it opens a bag, not only after it closes, so a storage
  plugin that keeps the metadata inside the recording (``rosbag2_storage_rrd``)
  records them too.
* Porting reads bags with ``read_next_ext()``. ``read_next()`` is deprecated
  on Lyrical and warned once per message.
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
