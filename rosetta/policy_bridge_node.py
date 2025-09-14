import os
import importlib.util
from typing import Dict

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rosidl_runtime_py.utilities import get_message

from .contract_utils import load_contract

def _maybe_load_policy():
    path = os.environ.get('ROSETTA_POLICY_PY', '')
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("rosetta_policy", path)
    if not spec or not spec.loader:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, 'infer', None)

class PolicyBridgeNode(Node):
    def __init__(self):
        super().__init__('policy_bridge')
        self.declare_parameter('contract_path', '')
        cp = self.get_parameter('contract_path').get_parameter_value().string_value
        if not cp:
            raise RuntimeError("Parameter 'contract_path' is required (path to YAML).")
        self._contract = load_contract(cp)

        if not self._contract.policy or not self._contract.policy.action_topic:
            raise RuntimeError("Contract missing 'policy' section or 'action_topic'.")

        # Prepare action publisher
        act_spec = self._contract.policy.action_topic
        self._action_type = get_message(act_spec.type)
        self._action_pub = self.create_publisher(self._action_type, act_spec.name, 10)

        # Prepare obs subscriptions
        self._latest: Dict[str, object] = {}
        for t in self._contract.policy.obs_topics:
            msg_cls = get_message(t.type)
            self.create_subscription(msg_cls, t.name, self._mk_obs_cb(t.name), 10)
            self._latest[t.name] = None

        # Policy loader (optional)
        self._infer = _maybe_load_policy()
        if self._infer:
            self.get_logger().info("External policy loaded from ROSETTA_POLICY_PY.")
        else:
            self.get_logger().info("No external policy, publishing default-zero actions.")

        # Timer to publish actions
        hz = float(self._contract.policy.rate_hz or 20.0)
        self._timer = self.create_timer(1.0 / hz, self._on_tick)

    def _mk_obs_cb(self, topic_name: str):
        def cb(msg):
            self._latest[topic_name] = msg
        return cb

    def _on_tick(self):
        # Assemble obs dict (topic -> last msg or None)
        obs = dict(self._latest)

        if self._infer:
            try:
                action_msg = self._infer(obs, self)  # user function
                if action_msg is None:
                    action_msg = self._action_type()  # fallback
            except Exception as e:
                self.get_logger().warn(f"Policy error: {e}; using default action.")
                action_msg = self._action_type()
        else:
            # default-zero action
            action_msg = self._action_type()

        self._action_pub.publish(action_msg)

def main():
    try:
        rclpy.init()
        node = PolicyBridgeNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.shutdown()
