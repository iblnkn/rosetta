"""LeRobot async inference policy server entrypoint for Rosetta.

Importing ``lerobot_policy_sns_diffusion`` registers the ``SNSDiffusionConfig``
subclass with ``PreTrainedConfig`` so that checkpoints trained with it can be
loaded by ``serve()``.

Why the explicit import?
------------------------
LeRobot v0.5.1's third-party plugin discovery
(``lerobot.utils.import_utils.register_third_party_plugins``) auto-imports
distributions whose name starts with ``lerobot_policy_`` — which this plugin
satisfies — but ``lerobot.async_inference.policy_server.serve()`` does not
call that discovery hook (only ``lerobot_train``, ``lerobot_eval``, and
``robot_client`` do). Until upstream adds plugin discovery to the server
entry point, this one-line shim is the minimum glue needed.
"""

import lerobot_policy_sns_diffusion  # noqa: F401 - registers config subclass
from rosetta.common.rtc_policy_server import serve


if __name__ == "__main__":
    serve()
