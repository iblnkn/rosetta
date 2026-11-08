^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package lerobot_dataset_rosetta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* ``find_contract_for_pretrained`` never fetches the dataset's ``root`` from
  the Hub. A relative root such as ``lerobot/gz_reach`` was sent to the Hub as
  a repo id and warned about the connection. When no link finds the contract,
  one warning lists each attempt, and a link that does find it leaves no
  warning from earlier misses.
* ``LeRobotView`` serves any ``rosetta.data.FrameReader`` as a LeRobot
  training dataset: each policy's windows, ``<key>_is_pad``, task and index
  columns, and metadata with the reader's stats. It is read by index, as a
  ported ``LeRobotDataset`` is, so LeRobot's own sampler picks the order.
  ``LeRobotView.select`` gives a view over some episodes, and
  ``train_eval_views`` and ``split_episodes`` apply ``--dataset.episodes`` and
  ``--dataset.eval_split`` per task, as LeRobot does. ``LeRobotStreamView``
  streams a reader in its own order instead.
* ``rosetta-lerobot-train`` runs ``lerobot-train`` on a frame reader
  (``--rosetta.reader``, ``--rosetta.contract``, and the reader's options as
  ``--rosetta.*``), and copies the contract into each checkpoint.
  ``--rosetta.shuffle=block`` streams, and refuses an eval split and
  ``--dataset.episodes``. ``--dataset.streaming=true`` without it is refused,
  since LeRobot would read an indexed view in episode order.
* ``rosetta-lerobot-train`` sets ``--dataset.repo_id`` to
  ``rosetta-lerobot-train/<reader>`` unless one is given.
  ``find_contract_for_pretrained`` never looks a repo in that namespace
  (``READER_DATASET_NAMESPACE``) up on the Hub, and warns to pass
  ``contract_path``. A model pushed to the Hub doesn't carry its contract.
* Needs ``lerobot[training]<0.6.2``, for the accelerate ``lerobot-train``
  uses, so installing it to port datasets also installs LeRobot's training
  dependencies.
* ``find_contract_for_pretrained`` reads a checkpoint's own
  ``rosetta_contract.yaml`` before following ``train_config.json``.
* First release. ``LeRobotDatasetWriter`` moved here from ``lerobot_rosetta``,
  so the data side installs without the policy runner's gRPC and inference
  dependencies. It is still registered on ``rosetta.dataset_writers`` as
  ``lerobot``. Import it from ``lerobot_dataset_rosetta.dataset_writer``.
* ``lerobot_dataset_rosetta.sidecar`` holds the contract sidecar helpers
  (``LEROBOT_CONTRACT_SIDECAR_PATH``, ``find_contract_for_pretrained`` and
  ``resolve_repo_file``), moved from ``rosetta.contract.sidecar``.
  ``find_contract_for_pretrained`` tries the training dataset's ``root``, then
  its ``repo_id``, and warns for every missing link.
