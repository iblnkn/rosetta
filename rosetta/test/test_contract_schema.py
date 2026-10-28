# Copyright 2026 Isaac Blankenau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The contract loader's strictness matrix, against the fake robot interface.

Validation replaces fallback: unknown keys, missing align, a timeline the
channel cannot provide are load-time errors with a dotted-path context, never
a silent default. Each test pins one rule.
"""

import warnings

import pytest

from rosetta.contract.errors import ContractValidationError
from rosetta.contract.schema import load_contract, parse_contract

pytestmark = pytest.mark.usefixtures("fake_interface")

BASE = """
robot_type: test
robot_interface: fake
fps: 30
observations:
  observation.state:
    channel: {topic: /state, type: fake/Vector}
    align: {strategy: hold, timeline: arrival}
    select: [a]
actions:
  action:
    channel: {topic: /cmd, type: fake/Vector}
    align: {strategy: hold, timeline: arrival}
    select: [a]
"""

ALIGN = "{strategy: hold, timeline: arrival}"
OBS_CHANNEL = "channel: {topic: /state, type: fake/Vector}"
ACT_CHANNEL = "channel: {topic: /cmd, type: fake/Vector}"
OBS_SELECT = "    select: [a]\nactions:"
EVENTS = "teleop:\n  events:\n    channel: {{topic: /joy, type: {type}}}\n    select: {select}\n"
OBS_ENTRY = f"  observation.state:\n    {OBS_CHANNEL}\n    align: {ALIGN}\n    select: [a]\n"


def _load(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return load_contract(p)


def _expect_error(tmp_path, text, match):
    with pytest.raises(ContractValidationError, match=match):
        _load(tmp_path, text)


def _channel(extra, channel=OBS_CHANNEL):
    """BASE with ``extra`` fields added to one channel block."""
    return BASE.replace(channel, channel[:-1] + f", {extra}}}", 1)


def _align(new):
    """BASE with the observation's align block replaced."""
    return BASE.replace(ALIGN, new, 1)


# ---------------------------------------------------------------------------
# Loader I/O and the parsed document
# ---------------------------------------------------------------------------


def test_valid_contract_round_trips(tmp_path):
    contract = _load(tmp_path, BASE)
    assert contract.robot_type == "test"
    assert contract.robot_interface == "fake"
    assert contract.fps == 30
    assert contract.observations[0].key == "observation.state"
    assert contract.actions[0].key == "action"
    src = contract.observations[0].sources[0]
    assert (src.channel.topic, src.align.strategy, src.align.timeline) == ("/state", "hold", "arrival")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"source": "my_robot.yaml"}, "Invalid YAML in my_robot.yaml"), ({}, "Invalid YAML in <contract>")],
)
def test_parse_contract_invalid_yaml_names_source(kwargs, match):
    with pytest.raises(ContractValidationError, match=match):
        parse_contract("robot_type: [unclosed", **kwargs)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_contract(tmp_path / "nope.yaml")


def test_non_mapping_document_rejected(tmp_path):
    _expect_error(tmp_path, "- a\n- b\n", "must be a YAML mapping")


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_rejected(tmp_path):
    _expect_error(tmp_path, BASE + "visualization: {}\n", "visualization")


def test_missing_robot_type_rejected(tmp_path):
    _expect_error(tmp_path, "robot_interface: fake\nfps: 30\n", "robot_type")


def test_missing_robot_interface_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("robot_interface: fake\n", ""), "robot_interface")


def test_missing_fps_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("fps: 30\n", ""), "fps")


def test_non_positive_fps_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("fps: 30", "fps: 0"), "fps")


@pytest.mark.parametrize("bad", ["3.9", "true", "'30'", "[30]"])
def test_fps_rejects_non_integers(bad, tmp_path):
    _expect_error(tmp_path, BASE.replace("fps: 30", f"fps: {bad}"), "integer")


def test_fps_accepts_integral_float(tmp_path):
    assert _load(tmp_path, BASE.replace("fps: 30", "fps: 30.0")).fps == 30


@pytest.mark.parametrize("bad", ["0", "''", "'  '", "false"])
def test_falsy_robot_type_rejected_with_clear_wording(bad, tmp_path):
    _expect_error(tmp_path, BASE.replace("robot_type: test", f"robot_type: {bad}"), "non-empty")


# ---------------------------------------------------------------------------
# Align: mandatory, no defaults, asof <-> tolerance coupling
# ---------------------------------------------------------------------------


def test_missing_align_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace(f"    align: {ALIGN}\n", "", 1), "align")


def test_align_missing_strategy_rejected(tmp_path):
    _expect_error(tmp_path, _align("{timeline: arrival}"), "strategy")


def test_align_missing_timeline_rejected(tmp_path):
    _expect_error(tmp_path, _align("{strategy: hold}"), "timeline")


def test_align_unknown_strategy_rejected(tmp_path):
    _expect_error(tmp_path, _align("{strategy: nearest, timeline: arrival}"), "nearest")


def test_align_unknown_key_rejected(tmp_path):
    _expect_error(tmp_path, _align("{strategy: hold, timeline: arrival, stamp: header}"), "stamp")


def test_asof_requires_tolerance(tmp_path):
    _expect_error(tmp_path, _align("{strategy: asof, timeline: arrival}"), "tolerance_ms")


def test_asof_with_tolerance_accepted(tmp_path):
    c = _load(tmp_path, _align("{strategy: asof, timeline: arrival, tolerance_ms: 10}"))
    src = c.observations[0].sources[0]
    assert (src.align.strategy, src.align.tolerance_ms) == ("asof", 10)


def test_tolerance_without_asof_rejected(tmp_path):
    _expect_error(tmp_path, _align("{strategy: hold, timeline: arrival, tolerance_ms: 10}"), "tolerance_ms")


def test_non_positive_tolerance_rejected(tmp_path):
    _expect_error(tmp_path, _align("{strategy: asof, timeline: arrival, tolerance_ms: 0}"), "tolerance_ms")


@pytest.mark.parametrize("bad", ["12.5", "true", "abc", "[5]"])
def test_tolerance_ms_rejects_non_integers(bad, tmp_path):
    _expect_error(tmp_path, _align(f"{{strategy: asof, timeline: arrival, tolerance_ms: {bad}}}"), "integer")


# ---------------------------------------------------------------------------
# Sources and channels: strict keys, per-section field rules
# ---------------------------------------------------------------------------


def test_unknown_source_key_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace(OBS_SELECT, "    select: [a]\n    serve: {safety: zeros}\nactions:"), "serve")


def test_unknown_channel_key_rejected(tmp_path):
    _expect_error(tmp_path, _channel("stamp: header"), "stamp")


def test_empty_topic_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("topic: /state,", "topic: '',", 1), "topic")


def test_non_string_topic_rejected(tmp_path):
    # No coercion: `topic: 123` is a typo'd contract, not a string in disguise.
    _expect_error(tmp_path, BASE.replace("topic: /cmd,", "topic: 123,", 1), "'topic'")


def test_empty_type_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("type: fake/Vector}", "type: ''}", 1), "'type'")


def test_non_mapping_qos_rejected(tmp_path):
    _expect_error(tmp_path, _channel("qos: 5"), "qos.*mapping")


TELEOP_FEEDBACK = """
teleop:
  feedback:
    - origin: /state
      channel: {topic: /wrist_cmd, type: fake/Vector}
      align: {strategy: hold, timeline: arrival}
      select: [x, y, z, w]
      kind: quaternion
"""


@pytest.mark.parametrize(
    ("text", "key"),
    [
        pytest.param(_channel("safety: zeros"), "safety", id="observations-safety"),
        pytest.param(_channel("encoder: 'json:dumps'"), "encoder", id="observations-encoder"),
        pytest.param(
            BASE + TELEOP_FEEDBACK.replace("topic: /wrist_cmd,", "topic: /wrist_cmd, safety: zeros,"),
            "safety",
            id="teleop-feedback-safety",
        ),
    ],
)
def test_publish_only_channel_key_rejected_where_nothing_publishes(text, key, tmp_path):
    """A channel field the section never acts on is a contract lie: error, not a silent override."""
    _expect_error(tmp_path, text, key)


@pytest.mark.parametrize(("extra", "expected"), [("", "none"), ("safety: zeros, ", "zeros")])
def test_safety_defaults_to_none(extra, expected, tmp_path):
    """An action channel without `safety` must not fabricate safety commands.

    A zero command is only a safe stop under velocity control; under position
    control (the common case) it commands a slam to the zero pose. Safety is an
    explicit, per-channel opt-in.
    """
    contract = _load(tmp_path, BASE.replace("topic: /cmd, ", f"topic: /cmd, {extra}"))
    assert contract.actions[0].sources[0].channel.safety == expected


def test_empty_source_list_rejected(tmp_path):
    text = BASE.replace(OBS_ENTRY, "  observation.state: []\n")
    _expect_error(tmp_path, text, "empty")


def test_shared_key_sources_ordered(tmp_path):
    """Order and per-source align are what concatenation under one key relies on."""
    text = BASE.replace(
        OBS_ENTRY,
        """  observation.state:
    - channel: {topic: /arm/state, type: fake/Vector}
      align: {strategy: hold, timeline: arrival}
      select: [a]
    - channel: {topic: /base/state, type: fake/Vector}
      align: {strategy: drop, timeline: arrival}
      select: [b]
""",
    )
    entry = _load(tmp_path, text).observations[0]
    assert [(s.channel.topic, s.align.strategy) for s in entry.sources] == [
        ("/arm/state", "hold"),
        ("/base/state", "drop"),
    ]


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "bool"])
def test_supported_numeric_dtypes_load(tmp_path, dtype):
    _load(tmp_path, _channel(f"dtype: {dtype}"))


@pytest.mark.parametrize("dtype", ["uint8", "float16", "image", "not_a_dtype", "flot32", "Float32", "'(-1,)f4'", "','"])
def test_unsupported_dtypes_rejected_at_load(tmp_path, dtype):
    """The vocabulary is closed and checked at load: anything FrameLayout
    cannot assemble is refused here, not at bridge configure or writer open."""
    _expect_error(tmp_path, _channel(f"dtype: {dtype}"), "Invalid dtype")


# ---------------------------------------------------------------------------
# kind: a semantic tag whose dimension count select must match
# ---------------------------------------------------------------------------

QUATERNION_FIELDS = ["orientation.x", "orientation.y", "orientation.z", "orientation.w"]


def _kind(kind, select):
    """BASE with the observation's select replaced and, unless None, a kind added."""
    kind_line = f"    kind: {kind}\n" if kind else ""
    return BASE.replace(OBS_SELECT, f"    select: [{', '.join(select)}]\n{kind_line}actions:", 1)


def _fields(n):
    return [f"f{i}" for i in range(n)]


def test_kind_defaults_to_continuous(tmp_path):
    assert _load(tmp_path, BASE).observations[0].sources[0].kind == "continuous"


@pytest.mark.parametrize(
    ("kind", "n"),
    [("quaternion", 4), ("euler_rpy", 3), ("rotation_6d", 6), ("binary", 1), ("continuous", 10)],
)
def test_kind_with_matching_dim_count_loads(kind, n, tmp_path):
    assert _load(tmp_path, _kind(kind, _fields(n))).observations[0].sources[0].kind == kind


@pytest.mark.parametrize(("kind", "n", "expected"), [("quaternion", 3, 4), ("euler_rpy", 4, 3)])
def test_kind_with_wrong_dim_count_rejected(kind, n, expected, tmp_path):
    _expect_error(
        tmp_path, _kind(kind, _fields(n)), rf"kind '{kind}' .* requires {expected} dim\(s\) but select has {n}"
    )


def test_invalid_kind_rejected(tmp_path):
    _expect_error(tmp_path, _kind("rotation_matrix", ["a"]), "Invalid kind 'rotation_matrix'")


@pytest.mark.parametrize(("kind", "select"), [("continuous", QUATERNION_FIELDS), (None, ["pose.quat_w"])])
def test_untagged_rotation_warns(kind, select, tmp_path):
    with pytest.warns(UserWarning, match="looks like a quaternion"):
        _load(tmp_path, _kind(kind, select))


@pytest.mark.parametrize(
    ("kind", "select"),
    [("continuous", ["position.j1", "position.j2", "position.j3"]), ("quaternion", QUATERNION_FIELDS)],
)
def test_no_warning_for_plain_joints_or_tagged_rotations(kind, select, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _load(tmp_path, _kind(kind, select))


# ---------------------------------------------------------------------------
# Spec resolution runs eagerly in load: its errors surface here, not at first frame
# ---------------------------------------------------------------------------


def test_image_without_resize_rejected_at_load(tmp_path):
    image = f"  observation.images.cam:\n    channel: {{topic: /cam, type: fake/Vector}}\n    align: {ALIGN}\nactions:"
    _expect_error(tmp_path, BASE.replace("actions:", image, 1), "output geometry")


def test_unencodable_action_type_rejected_at_load(tmp_path):
    """fake/Text decodes but has no encoder; as an action it must fail at load, not first publish."""
    text = BASE.replace(ACT_CHANNEL, "channel: {topic: /cmd, type: fake/Text}")
    _expect_error(tmp_path, text, "No encoder registered for 'fake/Text'")


# ---------------------------------------------------------------------------
# Extended sections: dtype mandatory, never images
# ---------------------------------------------------------------------------

EXTENDED = (
    BASE
    + """
signals:
  next.done:
    channel: {topic: /done, type: fake/Vector, dtype: bool}
    align: {strategy: hold, timeline: arrival}
    select: [a]
"""
)


def test_extended_section_accepted(tmp_path):
    c = _load(tmp_path, EXTENDED)
    assert c.signals[0].key == "next.done"
    assert c.signals[0].sources[0].channel.dtype == "bool"


def test_extended_without_dtype_rejected(tmp_path):
    _expect_error(tmp_path, EXTENDED.replace(", dtype: bool", ""), "dtype")


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(EXTENDED.replace("next.done", "observation.images.done"), id="signals"),
        pytest.param(
            BASE + "  observation.images.x:\n    channel: {topic: /cmd2, type: fake/Vector}\n"
            "    align: {strategy: hold, timeline: arrival}\n    select: [a]\n",
            id="actions",
        ),
    ],
)
def test_image_key_outside_observations_rejected(text, tmp_path):
    _expect_error(tmp_path, text, "only .*observations")


def test_frame_key_shared_across_sections_rejected(tmp_path):
    """observations.foo + signals.foo would silently merge into one recording
    column while narrower views (iter_policy_specs) disagree about its
    contents; a frame key must belong to exactly one section."""
    text = (
        BASE
        + """
signals:
  observation.state:
    channel: {topic: /battery, type: fake/Vector, dtype: float32}
    align: {strategy: hold, timeline: arrival}
    select: [a]
"""
    )
    _expect_error(tmp_path, text, "exactly one section")


# ---------------------------------------------------------------------------
# Tasks / adjunct shapes
# ---------------------------------------------------------------------------


def test_task_with_align_rejected(tmp_path):
    text = BASE + "tasks:\n  task:\n    channel: {topic: /task, type: fake/Vector}\n    align: " + ALIGN + "\n"
    _expect_error(tmp_path, text, "align")


def test_adjunct_must_be_list(tmp_path):
    _expect_error(tmp_path, BASE + "adjunct:\n  channel: {topic: /tf, type: fake/Vector}\n", "list")


def test_adjunct_extra_channel_fields_rejected(tmp_path):
    _expect_error(tmp_path, BASE + "adjunct:\n  - channel: {topic: /tf, type: fake/Vector, dtype: float64}\n", "dtype")


# ---------------------------------------------------------------------------
# Teleop: roles, refs into the action/observation topics, synthesized keys
# ---------------------------------------------------------------------------


def _teleop(role, ref_field, refs, select="[a]"):
    items = "".join(
        f"    - {ref_field}: {ref}\n"
        f"      channel: {{topic: /leader_{i}, type: fake/Vector}}\n"
        f"      align: {ALIGN}\n"
        f"      select: {select}\n"
        for i, ref in enumerate(refs)
    )
    return BASE + f"teleop:\n  {role}:\n{items}"


def test_teleop_feedback_parses_with_kind(tmp_path):
    fb = _load(tmp_path, BASE + TELEOP_FEEDBACK).teleop.feedback[0].source
    assert fb.channel.safety == "none"
    assert fb.kind == "quaternion"


@pytest.mark.parametrize(
    ("role", "ref_field"),
    [("input", "target"), ("feedback", "origin")],
)
def test_teleop_ref_to_unknown_topic_rejected(role, ref_field, tmp_path):
    """A teleop ref that names no real action (input) or observation (feedback)
    topic is a load-time error, not a message silently going nowhere."""
    _expect_error(tmp_path, _teleop(role, ref_field, ["/no/such/topic"]), f"{ref_field}.*does not match any")


@pytest.mark.parametrize(
    ("role", "ref_field", "topic"),
    [("input", "target", "/cmd"), ("feedback", "origin", "/state")],
)
def test_duplicate_teleop_ref_rejected(role, ref_field, topic, tmp_path):
    """Two inputs driving one action topic (or two feedbacks of one observation)
    would collide on the synthesized teleop.<role>.<key> recording column."""
    _expect_error(tmp_path, _teleop(role, ref_field, [topic, topic]), f"duplicate {ref_field}")


def test_teleop_target_owned_by_multiple_entries_rejected(tmp_path):
    """A topic shared by two action entries makes the owning recording column
    ambiguous, so a teleop target may not reference it."""
    text = _teleop("input", "target", ["/cmd"]).replace(
        "teleop:",
        f"  action.alt:\n    channel: {{topic: /cmd, type: fake/Vector}}\n    align: {ALIGN}\n    select: [b]\nteleop:",
    )
    _expect_error(tmp_path, text, "belongs to multiple entries")


def test_teleop_unknown_role_rejected(tmp_path):
    _expect_error(tmp_path, _teleop("inputs", "target", ["/cmd"]), "inputs")


def test_teleop_events_with_align_rejected(tmp_path):
    text = BASE + EVENTS.format(type="fake/Vector", select="{success: a}") + f"    align: {ALIGN}\n"
    _expect_error(tmp_path, text, "align")


def test_teleop_event_names_are_open(tmp_path):
    """Core doesn't own an event vocabulary: each consumer acts on the names it
    knows, so any name loads."""
    text = BASE + EVENTS.format(type="fake/Vector", select="{rerecord_episode: a}")
    assert _load(tmp_path, text).teleop.events.select == {"rerecord_episode": "a"}


def test_empty_events_select_rejected(tmp_path):
    text = BASE + EVENTS.format(type="fake/Vector", select="{}")
    _expect_error(tmp_path, text, "present but empty")


def test_declared_key_colliding_with_synthesized_teleop_key_rejected(tmp_path):
    """specs.py synthesizes teleop.input.<owning key> recording columns; a
    user-declared key of the same name would silently merge with it."""
    signal = (
        "signals:\n  teleop.input.action:\n    channel: {topic: /sig, type: fake/Vector, dtype: float64}\n"
        f"    align: {ALIGN}\n    select: [a]\n"
    )
    _expect_error(
        tmp_path, _teleop("input", "target", ["/cmd"]).replace("teleop:", signal + "teleop:"), "synthesized teleop"
    )


# ---------------------------------------------------------------------------
# Empty and null collections: present but empty is an error everywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "match"),
    [
        pytest.param(BASE + "signals: {}\n", "present but empty", id="frame-section"),
        pytest.param(BASE + "tasks: {}\n", "present but empty", id="tasks"),
        pytest.param(BASE + "adjunct: []\n", "present but empty", id="adjunct"),
        pytest.param(BASE + "teleop: {}\n", "present but empty", id="teleop-mapping"),
        pytest.param(BASE + "teleop: []\n", "must be a mapping", id="teleop-list"),
        pytest.param(BASE + "teleop:\n  input: []\n", "empty source list", id="teleop-input"),
        pytest.param(BASE + "teleop:\n  feedback: []\n", "empty source list", id="teleop-feedback"),
        pytest.param(BASE + "signals:\n", "present but null", id="null-section"),
        pytest.param(BASE + "teleop:\n  input:\n", "present but null", id="null-teleop-role"),
        pytest.param(
            BASE.replace("  observation.state:\n    channel", "  - key: observation.state\n    channel"),
            "must be a mapping",
            id="list-section",
        ),
    ],
)
def test_present_but_empty_sections_rejected(text, match, tmp_path):
    _expect_error(tmp_path, text, match)


def test_empty_select_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("select: [a]", "select: []", 1), "present but empty")


def test_non_string_select_entry_rejected(tmp_path):
    _expect_error(tmp_path, BASE.replace("select: [a]", "select: [1]", 1), "string field path")


def test_duplicate_select_rejected(tmp_path):
    # Duplicates silently corrupt: the decoder returns the value twice while
    # the encoder's by-name maps last-win, so widths disagree end to end.
    _expect_error(tmp_path, BASE.replace("select: [a]", "select: [a, a]", 1), "duplicates")


# ---------------------------------------------------------------------------
# YAML strictness: duplicate keys are load errors; anchors/aliases still work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(BASE + "fps: 31\n", id="top-level"),
        pytest.param(
            BASE.replace("actions:", f"  observation.state:\n    {OBS_CHANNEL}\n    align: {ALIGN}\nactions:"),
            id="frame-key",
        ),
        pytest.param(BASE.replace("{topic: /state, type", "{topic: /state, topic: /x, type", 1), id="channel-block"),
    ],
)
def test_duplicate_mapping_key_rejected(text, tmp_path):
    """PyYAML silently keeps the last duplicate key, so an entire frame entry
    could vanish from a copy-paste edit. The strict loader rejects it."""
    _expect_error(tmp_path, text, "duplicate key")


def test_qos_anchor_alias_supported_and_not_shared(tmp_path):
    """Anchors/aliases under top-level x-* keys work under the strict loader,
    the x-* key itself is ignored, and aliased qos blocks are copied per
    channel, not shared."""
    text = "x-qos:\n  state: &state_qos {level: 1}\n" + BASE.replace(
        OBS_CHANNEL, OBS_CHANNEL[:-1] + ", qos: *state_qos}"
    ).replace(ACT_CHANNEL, ACT_CHANNEL[:-1] + ", qos: *state_qos}")
    c = _load(tmp_path, text)
    obs_qos = c.observations[0].sources[0].channel.qos
    act_qos = c.actions[0].sources[0].channel.qos
    assert obs_qos == act_qos == {"level": 1}
    assert obs_qos is not act_qos


# ---------------------------------------------------------------------------
# Codec paths and apply: validated (and imported) at load
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "match"),
    [
        pytest.param("nofunc", "Expected format", id="no-colon"),
        pytest.param("'m:'", "Expected format", id="no-function"),
        pytest.param("':f'", "Expected format", id="no-module"),
        pytest.param("no_such_module_xyz:f", "Cannot import", id="unimportable-module"),
        pytest.param("json:nope", "not found", id="missing-attribute"),
    ],
)
def test_invalid_codec_paths_rejected(path, match, tmp_path):
    _expect_error(tmp_path, _channel(f"decoder: {path}"), match)


@pytest.mark.parametrize("bad", ["''", "'   '", "' pkg.mod:fn'"])
def test_blank_or_padded_codec_path_rejected(bad, tmp_path):
    _expect_error(tmp_path, _channel(f"decoder: {bad}"), "codec path")


@pytest.mark.parametrize(
    ("apply_yaml", "match"),
    [
        pytest.param("apply: {rad2deg: null}", "must be a list", id="non-list"),
        pytest.param("apply: [{a: 1, b: 2}]", "single-key", id="multi-key-mapping"),
        pytest.param("apply: [3]", "string or single-key", id="bad-item-type"),
        pytest.param("apply: [warp_drive]", "Unknown operator", id="unknown-operator"),
    ],
)
def test_malformed_apply_rejected(apply_yaml, match, tmp_path):
    _expect_error(tmp_path, BASE.replace(OBS_SELECT, f"    select: [a]\n    {apply_yaml}\nactions:", 1), match)


def test_action_apply_with_serveable_operator_loads(tmp_path):
    text = BASE + "    apply: [{clamp: {min: -1.0, max: 1.0}}]\n"
    assert _load(tmp_path, text).actions[0].sources[0].apply == (("clamp", {"min": -1.0, "max": 1.0}),)


def test_action_apply_with_forward_only_operator_rejected(tmp_path):
    """An action's pipeline runs in the serve direction; resize has none."""
    _expect_error(tmp_path, BASE + "    apply: [{resize: [2, 2]}]\n", "Operator 'resize'.*cannot be used on an action")


# ---------------------------------------------------------------------------
# The interface attests types, including on bare channels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(BASE.replace("type: fake/Vector}", "type: fake/Nope}", 1), id="frame-entry"),
        pytest.param(BASE + "tasks:\n  task:\n    channel: {topic: /p, type: fake/Nope}\n", id="tasks"),
        pytest.param(BASE + "adjunct:\n  - channel: {topic: /d, type: fake/Nope}\n", id="adjunct"),
        pytest.param(
            BASE + EVENTS.format(type="fake/Nope", select="{is_intervention: a}"),
            id="teleop-events",
        ),
    ],
)
def test_unknown_type_rejected_at_load(text, tmp_path):
    """Every channel, aligned or bare, gets its type checked by the interface,
    so a typo'd type fails at load, not at subscription time."""
    _expect_error(tmp_path, text, "Unknown message type 'fake/Nope'")


# ---------------------------------------------------------------------------
# Strict scalars: values are used exactly as written, never case-folded,
# stripped or coerced from another YAML type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("robot_interface: fake", "robot_interface: FAKE", "robot_interface"),
        ("robot_type: test", "robot_type: ' test'", "robot_type"),
        (ALIGN + "\n    select", "{strategy: HOLD, timeline: arrival}\n    select", "strategy"),
        ("timeline: arrival}\n    select", "timeline: 1}\n    select", "timeline"),
        ("timeline: arrival}\n    select", "timeline: ' arrival'}\n    select", "timeline"),
        ("timeline: arrival}\n    select", "timeline: Arrival}\n    select", "Arrival"),
        (OBS_SELECT, "    select: [a]\n    kind: Continuous\nactions:", "kind"),
        ("  observation.state:", "  1:", "frame key"),
        (OBS_SELECT, "    select: [a]\n    apply: [{1: null}]\nactions:", "operator name"),
    ],
)
def test_scalars_are_not_coerced(old, new, match, tmp_path):
    assert old in BASE
    _expect_error(tmp_path, BASE.replace(old, new, 1), match)
