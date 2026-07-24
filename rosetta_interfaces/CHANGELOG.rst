^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rosetta_interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.2.0 (2026-07-24)
------------------
* Version aligned with the rest of the Rosetta package set: a matching minor
  version denotes a compatible set. Supersedes the placeholder ``0.0.0``,
  which ``bloom`` rejects.
* ``StartHILEpisode`` returns once the episode has started rather than
  blocking for its duration, which would hold an executor thread. Results come
  from the ``ManageEpisode`` action result; stop through the
  ``stop_episode`` service.
* Added ``ManageEpisode`` and ``StartHILEpisode`` for human-in-the-loop
  episodes, and ``StartRecording`` for callers that cannot route to the hidden
  ``_action/*`` services (the Foxglove extension, the keyboard controller).
* CI moved to ``industrial_ci``.

0.0.0
-----
* Initial version. Never tagged or released.
