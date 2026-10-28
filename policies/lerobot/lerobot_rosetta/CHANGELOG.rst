^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package lerobot_rosetta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* ``policy_type`` defaults to empty, which reads the type from the
  checkpoint's ``config.json`` the way LeRobot does
  (``PreTrainedConfig.from_pretrained``), or uses ``reward_classifier`` with
  ``is_classifier``. Set it only when the runner can't read the checkpoint.
* Breaking: both servers drop ``--policy-type``. The policy server loads the
  type the checkpoint records, so a checkpoint can't be loaded as the wrong
  policy class, and the classifier server always loads a
  ``reward_classifier``. The model cache is keyed on
  ``(pretrained_name_or_path, device)``.

0.2.0 (2026-07-24)
------------------
* First release. The package was split out of ``rosetta`` so the core imports
  no ML framework; it enters the set at 0.2.0 because a matching minor version
  across the Rosetta packages denotes a compatible set.
* ``LeRobotDatasetWriter`` registered on the ``rosetta.dataset_writers``
  entry-point group, and ``LeRobotPolicyRunner`` on
  ``rosetta.policy_runners``.
* ``rosetta_policy_server``: preload-and-cache wrapper over LeRobot's stock
  async-inference policy server. The stock server reloads the checkpoint on
  every client handshake; this one loads it once and reuses it while the
  requested ``(policy_type, pretrained_name_or_path, device)`` is unchanged.
* ``rosetta_classifier_server``: the reward-classifier variant, scoped to
  ``reward_classifier`` and calling its real ``predict_reward()``.
* Both servers accept preload flags that load the model *before* the port is
  bound, so a socket-level readiness check implies the model is ready.
* The dataset writer embeds the contract as a sidecar
  (``meta/rosetta_contract.yaml``) when porting, so a checkpoint can resolve
  its own contract at deployment.
