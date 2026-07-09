"""rosetta.common's public surface must match what actually imported.

The ros2_utils import is guarded so contract loading/validation stays
importable without rclpy; __all__ must advertise those names only when
the import succeeded, or star-imports break in exactly the ROS-free
environments the guard exists for.
"""

import rosetta.common as common

ROS2_UTIL_NAMES = (
    "dot_get",
    "dot_set",
    "get_message_timestamp_ns",
    "qos_profile_from_dict",
    "stamp_from_header_ns",
)


def _have_rclpy() -> bool:
    try:
        import rclpy  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def test_star_import_never_raises_and_matches_environment():
    ns: dict = {}
    exec("from rosetta.common import *", ns)  # noqa: S102 - the point of the test
    have = _have_rclpy()
    for name in ROS2_UTIL_NAMES:
        assert (name in ns) == have
    # The always-available core stays exported either way.
    assert "load_contract" in ns
    assert "StreamBuffer" in ns


def test_all_names_actually_exist():
    for name in common.__all__:
        assert hasattr(common, name), f"__all__ advertises missing name {name!r}"
