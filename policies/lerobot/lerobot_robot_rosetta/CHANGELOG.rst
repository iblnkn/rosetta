^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package lerobot_robot_rosetta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* **Fixed: a failed lifecycle transition was silent.** ``configure()`` and
  ``connect()`` called ``trigger_configure()``/``trigger_activate()`` and
  discarded the result. ``trigger_*`` returns the transition's outcome rather
  than raising, so a failed configure left ``connect()`` reporting success
  against a node with no publishers -- surfacing later as a warmup timeout
  naming the wrong cause. Both now go through ``require_transition_success``,
  matching what the teleoperator adapter already did.

0.2.0 (2026-07-24)
------------------
* **Breaking: requires the 0.2.0 contract schema.** The package consumes
  contracts through ``rosetta``; contracts written against 0.1.0 do not load.
* **Breaking: the rosetta-side LeRobot backend moved out** to the
  ``lerobot_rosetta`` package. This package keeps only the Robot plugin that
  LeRobot itself discovers through the ``lerobot_robot_*`` naming convention —
  the gRPC inference servers, dataset writer, and policy runner now live in
  ``lerobot_rosetta``.
* Rebuilt on ``TopicBridge`` and the restructured contract layer, so
  observations and actions run the same decode, alignment, and operator
  pipeline as the ROS 2-native path.
* Standalone mode builds a ``BridgeLifecycleNode`` rather than a bare
  ``RosettaLifecycleNode``, so the node owns its ``TopicBridge`` instead of the
  adapter wiring one alongside it.
* Fixed standalone-node mode to sample frames through ``node.bridge``.
* Contract paths resolve through ``get_package_share_directory`` rather than
  relative ``parents[]`` walks, so an installed package finds its contracts.
* Fixed pip installation: CPU torch, and a protobuf/grpcio upgrade.
* CI moved to ``industrial_ci``.
* README documented a contract schema that no longer parsed; corrected to the
  0.2.0 syntax, with the full schema left to the contract reference rather
  than restated here.

0.1.0
-----
* Initial version. Never tagged or released.
