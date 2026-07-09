"""Policy server entrypoint for Rosetta.

Thin wrapper around :func:`rosetta.common.rtc_policy_server.serve`, which runs an
``RTCPolicyServer`` (a ``PolicyServer`` subclass). The server dispatches by
policy capability: sns_diffusion-family checkpoints get server-built stacked
history + RTC guidance, while ACT / upstream diffusion / smolvla / pi0 take the
plain upstream path.

``serve()`` calls ``register_third_party_plugins()`` at startup, which discovers
installed ``lerobot_policy_*`` distributions (e.g. ``sns_diffusion``) and imports
them so their configs register with ``PreTrainedConfig`` before any checkpoint
loads. That replaces the old explicit ``import lerobot_policy_sns_diffusion`` and
means an ACT-only deployment without that plugin installed no longer fails at
import time.
"""

from rosetta.common.rtc_policy_server import serve

if __name__ == "__main__":
    serve()
