^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package lerobot_dataset_rosetta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* First release. ``LeRobotDatasetWriter`` moved here from ``lerobot_rosetta``,
  so the data side installs without the policy runner's gRPC and inference
  dependencies. It is still registered on ``rosetta.dataset_writers`` as
  ``lerobot``. Import it from ``lerobot_dataset_rosetta.dataset_writer``.
* ``lerobot_dataset_rosetta.sidecar`` holds the contract sidecar helpers
  (``LEROBOT_CONTRACT_SIDECAR_PATH``, ``find_contract_for_pretrained`` and
  ``resolve_repo_file``), moved from ``rosetta.contract.sidecar``.
  ``find_contract_for_pretrained`` tries the training dataset's ``root``, then
  its ``repo_id``, and warns for every missing link.
