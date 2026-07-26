^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rosetta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* **Breaking: action results report how work ended, not whether it "succeeded".**
  ``bool success`` is gone from all three results (see ``rosetta_interfaces``);
  the terminal ``GoalStatus`` is the mechanics and ``termination_reason`` names
  the cause. ``ManageEpisode`` additionally reports ``outcome`` -- whether the
  robot did the task -- which no field carried before.
* **Breaking: ``hil_manager``'s ``~/stop_episode`` split in two.**
  ``~/end_episode`` (``SetBool``: true = success, false = failure) is the
  deliberate, labelled end, mirroring the ``end_success``/``end_failure``
  buttons; ``~/cancel_episode`` (``Trigger``) abandons the take. The single
  service could not say which was meant.
* **Fixed: a failed bag write reported a successful recording.** The write
  exception is caught inside the subscription callback, so it never reached
  ``_record``'s handler: the loop exited normally and the goal SUCCEEDED while
  handing back a truncated bag. Now ``ABORTED`` with
  ``termination_reason: error`` naming the topic that failed.
* **Fixed: a lifecycle deactivate was reported as a human stop.** The HIL
  feedback loop derived its reason from ``is_cancel_requested``, so everything
  that was not an action cancel came back as ``human_stop``.
* **Fixed: the cancel services now really cancel.** ``~/cancel_recording`` used
  to set a flag that ended the goal ``ABORTED``, so a dashboard button and a
  ``ros2 action`` cancel produced different terminal states for the same human
  gesture. The cancel services forward to the action server's own
  ``_action/cancel_goal``, so both paths end ``CANCELED``.
* Every stop signal now names itself. ``_signal_stop(reason)`` records the cause
  under the work gate, first writer wins, and work loops report the recorded
  reason instead of re-deriving it -- there was one stop event but nine stop
  causes, and inferring which had fired is what produced the two bugs above.
* ``PolicyRunnerNode``'s stop hook moved from ``_signal_stop`` to
  ``_unblock_stop``, which never runs while holding the work gate. It calls
  ``runner.request_stop()``, a foreign framework call that could block.
* ``RunnerFeedback.status`` removed along with the ROS feedback field it
  existed to fill (see ``rosetta_interfaces``), so adapters no longer invent an
  ``"executing"``/``"idle"`` label nothing reads.
* The ``TERMINATION_*``/``OUTCOME_*`` values in ``nodes/node_utils.py`` are now
  re-exported from the generated message constants rather than restated, so the
  ``.action`` files are their single source.
* Recordings now carry their provenance: the UUID of the ``RecordEpisode`` goal
  that produced a bag is written to ``rosetta.goal_id`` in the bag's
  ``metadata.yaml``, beside the prompt and contract text. A client that kept its
  goal id can find the bag later. Absent for service-started recordings, which
  have no goal.
* ``max_duration_s`` on a goal overrides the node's ``default_max_duration_s``
  for that one run (see ``rosetta_interfaces``). ``policy_runner`` gained a
  ``default_max_duration_s`` parameter to match the recorder's.
* **Breaking: ``hil_manager``'s ``default_episode_prompt`` renamed to
  ``default_prompt``**, and it now applies to every empty-prompt path -- the
  action goal and the start_episode service as well as the teleop button.
  ``episode_recorder`` and ``policy_runner`` gained the same parameter, so a
  caller with no prompt to give (a dashboard button, a bare ``"{}"`` goal)
  gets the node's configured one. Every goal field now has a zero-value
  default meaning "use the node's", so callers type only what they override.
* Added ``policy_runner`` services ``~/start_policy`` and ``~/cancel_policy``;
  the node previously had no service path at all, so clients that cannot call
  actions could not run or stop a policy.

0.2.0 (2026-07-24)
------------------
* **Breaking: new contract schema.** Sections are mappings keyed by frame key,
  not lists of entries. ``key``/``topic``/``type``/``selector``/``publish``
  are replaced by ``channel`` (``topic``, ``type``, ``qos``, ``safety``),
  ``align`` (``strategy``, ``timeline``), ``select``, and ``apply``. Contracts
  written against 0.1.0 do not load.
* **Breaking: package split into framework-agnostic core plus adapters.**
  ``rosetta.contract``, ``rosetta.frames``, and ``rosetta.policies`` never
  import ``rclpy`` or take ``*_msgs`` types; ROS 2 integration lives under
  ``rosetta.robots.ros2``.
* **Breaking: the LeRobot backend moved out** to the ``lerobot_rosetta``
  package. ``rosetta`` core imports no ML framework; dataset writers and policy
  runners are discovered through the ``rosetta.dataset_writers`` and
  ``rosetta.policy_runners`` entry-point groups.
* **Breaking: nodes and launch files renamed.** ``rosetta_client_node`` is now
  ``policy_runner_node`` (``rosetta_client_launch.py`` →
  ``policy_runner_launch.py``), and ``rosetta_hil_launch.py`` is now
  ``hil_launch.py``. Executables moved to
  ``rosetta.robots.ros2.nodes.*``; bag conversion is the ``rosetta_port`` CLI.
* **Breaking: build type changed** from ``ament_cmake`` to ``ament_python``.
* **Breaking:** ``max_duration_s`` moved from the contract to a ROS parameter
  and now defaults to no limit.
* Record raw, decode late: episodes are recorded as raw rosbag2 messages with
  the exact contract text embedded in bag metadata, so recordings survive
  contract evolution. Decoding happens at port or inference time.
* One ingest path shared by the live bridge and the offline porter, enforced
  frame-for-frame by ``test_bag_live_parity.py``.
* Operators gained invertibility tiers (forward-only, bidirectional, bijective)
  with round-trip gates, so a non-invertible transform on an action is refused
  at contract load.
* Operators and codecs are open registries via the ``rosetta.operators`` and
  ``rosetta.codecs`` entry-point groups.
* The contract travels: bags and datasets embed the contract text, and
  deployment resolves it from the checkpoint's training dataset when
  ``contract_path`` is empty.
* Recording is one stream per camera, pinned by the contract or a whitelist,
  otherwise preferring the compressed topic.
* The rosbag2 storage plugin is selectable as a launch argument.
* Contract validation is fully load-time in three layers (shape, timeline
  attestation, codec/operator resolvability); nothing downstream re-validates.
* Documentation rebuilt as a Diátaxis Sphinx site under ``doc/``, published to
  https://iblnkn.github.io/rosetta/ and via ``rosdoc2.yaml``.
* Declared ``tf2_msgs`` and ``rosgraph_msgs``, which are referenced by type
  string (bundled contracts and the recorder's ``/clock`` topic) rather than
  imported, so rosdep could not resolve them.
* CI moved to ``industrial_ci``; test isolation fixed for ``rclpy``.

0.1.0
-----
* Initial version. Never tagged or released.
