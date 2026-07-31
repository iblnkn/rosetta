^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rosetta_interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* **Breaking: ``bool success`` removed from every action result.** It meant
  three different things at once -- did the goal run to term, did anything
  break, and did the robot do the task -- and disagreed with the goal's
  ``GoalStatus`` on the most common path: cancelling a recording, the only way
  to end an untimed one, reported the take as a failure. ``GoalStatus`` is now
  the sole mechanics signal.
* **Added ``string termination_reason``** to ``RecordEpisode``, ``RunPolicy``,
  and ``ManageEpisode``, with the legal values declared as string constants in
  each ``.action``. ``SUCCEEDED`` means the work reached a defined end,
  ``CANCELED`` that a client took it away, ``ABORTED`` that the server stopped
  it; ``termination_reason`` says which of those, exactly.
* **Added ``string outcome`` to ``ManageEpisode``** (result and feedback):
  whether the robot did the task, as ``success``, ``failure``, or
  ``unlabeled``. Independent of how the episode ended. Previously this was only
  implicit in ``final_reward``, where an unlabeled episode was
  indistinguishable from a classifier honestly reporting ``0.0``.
* ``RecordEpisode`` feedback gained ``elapsed_s`` and dropped
  ``seconds_remaining``. Time remaining needed a ``-1`` sentinel to describe the
  common case (no limit), and now that ``max_duration_s`` is on the goal a
  client that set one can subtract for itself.
* **Breaking: ``status`` removed from all three feedback messages.** It was a
  hardcoded constant at every publish site, advertising states ("starting",
  "stopping") that were never emitted. Feedback arriving at all is the "running"
  signal; ``GoalStatus`` and ``termination_reason`` cover the end.
* ``message`` now carries only the detail of something going wrong, and is empty
  on a clean end rather than restating ``termination_reason`` or
  ``messages_written``. A non-empty ``message`` is a signal.
* **Added ``float64 max_duration_s`` to the ``RecordEpisode`` and ``RunPolicy``
  goals** (``0.0`` = use the node default), matching ``ManageEpisode``. Duration
  was a node parameter only, so a timed recording needed a ``ros2 param set``
  first -- and two clients alternating between goals raced on that write.
  ``RunPolicy`` gained ``TERMINATION_TIMEOUT`` accordingly.
* Added ``StartPolicy`` so callers that cannot route to the hidden ``_action/*``
  services can start policy inference, matching ``StartRecording``. The
  ``Start*`` services deliberately take node defaults and carry no duration: a
  caller that needs per-invocation control sends a goal.
* ``StartHILEpisode``'s note updated: an episode is ended with a verdict through
  ``end_episode`` or abandoned through ``cancel_episode``; the single ambiguous
  ``stop_episode`` service is gone.

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
