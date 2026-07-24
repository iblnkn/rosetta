^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rosetta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
