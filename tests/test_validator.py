"""Tests for the fd6.shapes v1 validator.

Every check listed in docs/JSON_SPEC.md § "Validator rules summary"
has at least one positive (passes) and one negative (caught) test
here. When the spec doc adds a new rule, this file gets a new test
in the same commit.

Test naming: `test_<rule>_<positive|caught>` keeps the matrix obvious
at a glance.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from forza_abyss_painter.io.validator import (
    Issue, Severity, validate_document, INJECTOR_SAFE_TYPES,
)


def _good_doc() -> dict:
    """Minimal-but-complete valid v1 document. Tests mutate copies of
    this to isolate specific failures."""
    return {
        "format": "fd6.shapes",
        "version": 1,
        "image_size": [1024, 768],
        "shape_count": 1,
        "generated_at": "2026-05-26T17:42:11Z",
        "profile": "test",
        "sticker_mode": False,
        "shapes": [
            {
                "type": "rotated_ellipse",
                "x": 100.0, "y": 50.0,
                "rx": 30.0, "ry": 15.0,
                "angle": 12.5,
                "color": [200, 12, 80, 180],
            }
        ],
    }


def _errors(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.severity is Severity.ERROR]


def _warnings(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.severity is Severity.WARNING]


def _codes(issues: list[Issue]) -> set[str]:
    return {i.code for i in issues}


# ============================================================ smoke


def test_minimal_good_document_has_zero_errors():
    """A textbook valid doc should validate clean. If this breaks,
    something is over-strict and the spec doc disagrees."""
    issues = validate_document(_good_doc())
    assert _errors(issues) == [], f"unexpected errors: {_errors(issues)}"


def test_not_a_json_object_short_circuits():
    """A non-object root is catastrophic — no point checking further."""
    for bad in (None, [], "string", 42, 3.14):
        issues = validate_document(bad)
        assert len(issues) == 1
        assert issues[0].severity is Severity.ERROR
        assert issues[0].code == "not_an_object"


# ============================================================ format / version


def test_missing_format_caught():
    doc = _good_doc()
    del doc["format"]
    assert "missing_format" in _codes(validate_document(doc))


def test_wrong_format_caught():
    doc = _good_doc()
    doc["format"] = "fd6.geometrize"
    assert "wrong_format" in _codes(validate_document(doc))


def test_missing_version_caught():
    doc = _good_doc()
    del doc["version"]
    assert "missing_version" in _codes(validate_document(doc))


def test_wrong_version_caught():
    doc = _good_doc()
    doc["version"] = 2
    assert "wrong_version" in _codes(validate_document(doc))


# ============================================================ image_size


def test_missing_image_size_caught():
    doc = _good_doc()
    del doc["image_size"]
    assert "missing_image_size" in _codes(validate_document(doc))


def test_image_size_wrong_shape_caught():
    doc = _good_doc()
    doc["image_size"] = [1024]   # length 1, should be 2
    assert "bad_image_size_shape" in _codes(validate_document(doc))


def test_image_size_non_integer_caught():
    doc = _good_doc()
    doc["image_size"] = [1024.5, 768]
    assert "bad_image_size_types" in _codes(validate_document(doc))


def test_image_size_zero_caught():
    """Zero dimensions would make every shape's bbox degenerate."""
    doc = _good_doc()
    doc["image_size"] = [0, 768]
    assert "non_positive_image_size" in _codes(validate_document(doc))


# ============================================================ shapes container


def test_missing_shapes_caught():
    doc = _good_doc()
    del doc["shapes"]
    assert "missing_shapes" in _codes(validate_document(doc))


def test_shapes_not_array_caught():
    doc = _good_doc()
    doc["shapes"] = {"not": "an array"}
    assert "shapes_not_array" in _codes(validate_document(doc))


def test_shape_count_mismatch_warns():
    """The injector reads len(shapes); shape_count is informational —
    warning, not error, so users can still load hand-edited JSONs."""
    doc = _good_doc()
    doc["shape_count"] = 999
    issues = validate_document(doc)
    assert "shape_count_mismatch" in {i.code for i in _warnings(issues)}
    assert _errors(issues) == []


# ============================================================ per-shape geometry


@pytest.mark.parametrize("type_name,required_fields", [
    ("circle",            ("x", "y", "r")),
    ("ellipse",           ("x", "y", "rx", "ry")),
    ("rotated_ellipse",   ("x", "y", "rx", "ry", "angle")),
    ("rectangle",         ("x", "y", "hw", "hh")),
    ("rotated_rectangle", ("x", "y", "hw", "hh", "angle")),
    ("triangle",          ("x1", "y1", "x2", "y2", "x3", "y3")),
])
def test_each_shape_type_validates_with_required_fields(type_name, required_fields):
    """Every registered shape type can be expressed as a valid v1
    document. If a new type is added, parametrize it here."""
    doc = _good_doc()
    shape = {"type": type_name, "color": [255, 0, 0, 255]}
    for fld in required_fields:
        shape[fld] = 10.0
    doc["shapes"] = [shape]
    doc["shape_count"] = 1
    issues = validate_document(doc)
    assert _errors(issues) == [], (
        f"{type_name} with all required fields should validate; "
        f"errors: {_errors(issues)}"
    )


def test_missing_geometry_field_caught():
    doc = _good_doc()
    del doc["shapes"][0]["rx"]
    codes = _codes(validate_document(doc))
    assert "missing_geometry_field" in codes


def test_non_numeric_geometry_field_caught():
    doc = _good_doc()
    doc["shapes"][0]["rx"] = "thirty"
    assert "non_numeric_geometry_field" in _codes(validate_document(doc))


def test_bool_geometry_field_rejected_explicitly():
    """`bool` is a subclass of `int` in Python — without an explicit
    isinstance(_, bool) guard, `True` would sneak through as a
    coordinate. Real-world footgun, hence the explicit test."""
    doc = _good_doc()
    doc["shapes"][0]["rx"] = True
    assert "non_numeric_geometry_field" in _codes(validate_document(doc))


def test_unknown_shape_type_caught():
    doc = _good_doc()
    doc["shapes"][0]["type"] = "hexagon"
    assert "unknown_shape_type" in _codes(validate_document(doc))


def test_shape_missing_type_caught():
    doc = _good_doc()
    del doc["shapes"][0]["type"]
    assert "shape_missing_type" in _codes(validate_document(doc))


def test_shape_not_object_caught():
    doc = _good_doc()
    doc["shapes"][0] = "not a shape"
    assert "shape_not_object" in _codes(validate_document(doc))


# ============================================================ positive extents


@pytest.mark.parametrize("type_name,positive_field,extra_fields", [
    ("circle",          "r",  {"x": 5.0, "y": 5.0}),
    ("ellipse",         "rx", {"x": 5.0, "y": 5.0, "ry": 10.0}),
    ("rotated_ellipse", "ry", {"x": 5.0, "y": 5.0, "rx": 10.0, "angle": 0.0}),
    ("rectangle",       "hw", {"x": 5.0, "y": 5.0, "hh": 10.0}),
    ("rotated_rectangle", "hh", {"x": 5.0, "y": 5.0, "hw": 10.0, "angle": 0.0}),
])
def test_zero_extent_caught(type_name, positive_field, extra_fields):
    """Zero extents render to nothing — would waste an injector slot."""
    doc = _good_doc()
    shape = {"type": type_name, "color": [255, 0, 0, 255], positive_field: 0.0}
    shape.update(extra_fields)
    doc["shapes"] = [shape]
    doc["shape_count"] = 1
    assert "non_positive_extent" in _codes(validate_document(doc))


def test_negative_radius_caught():
    doc = _good_doc()
    doc["shapes"][0]["rx"] = -5.0
    assert "non_positive_extent" in _codes(validate_document(doc))


def test_angle_zero_is_valid():
    """angle is signed — 0° is a legitimate value, not zero-extent."""
    doc = _good_doc()
    doc["shapes"][0]["angle"] = 0.0
    issues = validate_document(doc)
    assert _errors(issues) == []


# ============================================================ color


def test_missing_color_caught():
    doc = _good_doc()
    del doc["shapes"][0]["color"]
    assert "missing_color" in _codes(validate_document(doc))


def test_color_not_array_caught():
    doc = _good_doc()
    doc["shapes"][0]["color"] = "red"
    assert "color_not_array" in _codes(validate_document(doc))


def test_color_wrong_length_caught():
    doc = _good_doc()
    doc["shapes"][0]["color"] = [255, 0, 0]   # missing alpha
    assert "color_wrong_length" in _codes(validate_document(doc))


def test_color_channel_out_of_range_caught():
    doc = _good_doc()
    doc["shapes"][0]["color"] = [256, 0, 0, 255]
    assert "color_channel_out_of_range" in _codes(validate_document(doc))


def test_color_channel_non_int_caught():
    doc = _good_doc()
    doc["shapes"][0]["color"] = [255.5, 0, 0, 255]
    assert "color_channel_not_int" in _codes(validate_document(doc))


# ============================================================ alpha=0 / is_mask


def test_alpha_zero_without_is_mask_warns():
    """Materialize_shapes filters these silently — surface as warning
    so users don't lose shapes to the void."""
    doc = _good_doc()
    doc["shapes"][0]["color"] = [255, 0, 0, 0]
    issues = validate_document(doc)
    assert "invisible_shape" in {i.code for i in _warnings(issues)}
    assert _errors(issues) == []


def test_alpha_zero_with_is_mask_clean():
    """Boundary mask shapes are explicitly preserved — no warning."""
    doc = _good_doc()
    doc["shapes"][0]["color"] = [255, 0, 0, 0]
    doc["shapes"][0]["is_mask"] = True
    issues = validate_document(doc)
    assert "invisible_shape" not in _codes(issues)
    assert _errors(issues) == []


# ============================================================ injector-safety rollup


def test_only_triangles_warns_no_injector_safe():
    """A doc with zero injector-safe shapes would inject nothing —
    that's a footgun worth surfacing prominently."""
    doc = _good_doc()
    doc["shapes"] = [
        {
            "type": "triangle",
            "x1": 0, "y1": 0, "x2": 10, "y2": 0, "x3": 5, "y3": 10,
            "color": [255, 0, 0, 255],
        }
    ]
    doc["shape_count"] = 1
    warnings = _warnings(validate_document(doc))
    assert "no_injector_safe_shapes" in {w.code for w in warnings}
    assert "non_injector_safe_type_present" in {w.code for w in warnings}


def test_mixed_shapes_warns_per_unsupported_type_once():
    """Multiple triangles in the same doc should generate ONE warning
    about triangles, not one per shape — keeps the report digestible."""
    doc = _good_doc()
    doc["shapes"] = [
        # Two triangles + one safe ellipse.
        {"type": "triangle", "x1": 0, "y1": 0, "x2": 10, "y2": 0,
         "x3": 5, "y3": 10, "color": [255, 0, 0, 255]},
        {"type": "triangle", "x1": 1, "y1": 1, "x2": 11, "y2": 1,
         "x3": 6, "y3": 11, "color": [0, 255, 0, 255]},
        {"type": "rotated_ellipse", "x": 50, "y": 50, "rx": 10, "ry": 5,
         "angle": 0, "color": [0, 0, 255, 255]},
    ]
    doc["shape_count"] = 3
    warnings = _warnings(validate_document(doc))
    triangle_warns = [w for w in warnings
                      if w.code == "non_injector_safe_type_present"]
    assert len(triangle_warns) == 1, (
        f"expected one rolled-up triangle warning, got {triangle_warns}"
    )
    # Has at least one injector-safe shape → the no_injector_safe
    # warning must NOT fire.
    assert "no_injector_safe_shapes" not in {w.code for w in warnings}


def test_injector_safe_types_set_matches_spec():
    """Schema-doc check: the validator's INJECTOR_SAFE_TYPES set
    must match docs/JSON_SPEC.md §Injector-safe subset. If the doc
    adds/removes a type, this test catches the drift."""
    assert INJECTOR_SAFE_TYPES == frozenset({
        "rotated_ellipse", "ellipse", "circle", "rectangle",
    })


# ============================================================ misc


def test_unknown_top_level_field_info_only():
    """External tooling commonly attaches metadata — accept silently
    at INFO, never warning."""
    doc = _good_doc()
    doc["my_extra"] = "tag"
    issues = validate_document(doc)
    assert "unknown_top_level_field" in {
        i.code for i in issues if i.severity is Severity.INFO
    }
    assert _errors(issues) == []


def test_bad_generated_at_is_info_not_error():
    """Provenance only — don't reject loadable JSON over a malformed
    timestamp."""
    doc = _good_doc()
    doc["generated_at"] = "yesterday"
    issues = validate_document(doc)
    info_codes = {i.code for i in issues if i.severity is Severity.INFO}
    assert "bad_generated_at" in info_codes
    assert _errors(issues) == []


def test_issue_path_pinpoints_offending_shape():
    """The GUI hook will use `path` to highlight the bad row;
    confirm we emit it for shape-level issues."""
    doc = _good_doc()
    doc["shapes"].append({
        "type": "rotated_ellipse",
        "x": 0, "y": 0, "rx": 5, "ry": -1,    # negative ry on shape #1
        "angle": 0, "color": [0, 0, 0, 255],
    })
    doc["shape_count"] = 2
    errors = _errors(validate_document(doc))
    paths = {e.path for e in errors if e.code == "non_positive_extent"}
    assert "shapes[1].ry" in paths


def test_issues_are_hashable_for_dedupe():
    """frozen=True on Issue means consumers can stash them in a set
    to dedupe across multiple validations of the same file."""
    a = Issue(Severity.ERROR, "x", "msg", "shapes[0].rx")
    b = Issue(Severity.ERROR, "x", "msg", "shapes[0].rx")
    assert {a, b} == {a}


# ============================================================ regression bait


def test_validator_does_not_mutate_input():
    """Pure function contract — caller can re-use the dict after
    validation. Catches accidental `.pop()` or in-place edits."""
    doc = _good_doc()
    snapshot = deepcopy(doc)
    validate_document(doc)
    assert doc == snapshot, "validator mutated the input document"
