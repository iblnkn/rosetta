^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package lerobot_teleoperator_rosetta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.2.0 (2026-07-24)
------------------
* **Breaking: requires the 0.2.0 contract schema.** The ``teleop`` section is
  now role-based: ``input`` and ``feedback`` are lists of independently
  targeted sources naming an existing action (``target``) or observation
  (``origin``) topic, and events use ``channel`` plus ``select`` instead of
  ``topic``/``type``/``mappings``. Contracts written against 0.1.0 do not
  load.
* **Breaking: the event vocabulary is closed** — ``is_intervention``,
  ``start_episode``, ``success``, ``failure``, ``end_success``,
  ``end_failure``. An unknown event name is a load error. ``end_success`` and
  ``end_failure`` replace the former ``terminate_episode`` and
  ``rerecord_episode`` mappings.
* Rebuilt on ``TopicBridge``, with bug fixes and test coverage for the
  lifecycle and shared-topic paths.
* Contract paths resolve through ``get_package_share_directory`` rather than
  relative ``parents[]`` walks.
* Fixed pip installation on Ubuntu 24.04 (``--break-system-packages``).
* CI moved to ``industrial_ci``.
* README documented a ``teleop`` schema that no longer parsed; corrected to
  the 0.2.0 syntax.

0.1.0
-----
* Initial version. Never tagged or released.
