^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package lerobot_dataset_rosetta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* First release. ``LeRobotDatasetWriter`` moved here from ``lerobot_rosetta``,
  so the data side installs without the policy runner's gRPC and inference
  dependencies. It is still registered on ``rosetta.dataset_writers`` as
  ``lerobot``. Import it from ``lerobot_dataset_rosetta.dataset_writer``.
