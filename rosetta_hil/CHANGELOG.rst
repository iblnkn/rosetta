^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rosetta_hil
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* ``hil_manager_node``, ``hil_launch.py`` and ``params/hil_manager.yaml`` moved
  here from ``rosetta``. The node uses only Rosetta's public Python API, and its
  interfaces moved to ``rosetta_hil_interfaces``. Launch with
  ``ros2 launch rosetta_hil hil_launch.py``.
