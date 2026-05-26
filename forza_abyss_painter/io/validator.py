"""Validator for the `fd6.shapes` v1 document format.

Implements every check listed in `docs/JSON_SPEC.md` § "Validator
rules summary". The CLI (`fap-validate`) and the GUI auto-validate
hook both call into `validate_document` here. This is the single
source of validation truth — if the spec doc and this module
disagree, the module wins and the doc is wrong.

## Usage

    from forza_abyss_painter.io.validator import validate_document, Severity

    issues = validate_document(json.loads(text))
    errors = [i for i in issues if i.severity is Severity.ERROR]
    if errors:
        raise BadJSON(errors)

## Design choices

- **Issues, not exceptions.** A single document can have many
  problems; collecting them lets the CLI / GUI show them all at
  once instead of fixing-one-at-a-time. Only catastrophic parse
  errors (not-a-JSON-object) raise.
- **Severity tiers** (ERROR/WARNING/INFO) match the spec doc. The
  validator does NOT decide what to do with them — callers do.
- **Path-aware.** Each `Issue.path` is a JSONPath-ish string like
  `"shapes[42].rx"` so the GUI can highlight the offending shape.
- **No side-effects.** Pure function of the input dict. Safe to
  run on untrusted JSON in any context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from forza_abyss_painter.io.json_schema import FD6_FORMAT, FD6_VERSION


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Issue:
    """One validation finding. Immutable so callers can de-dupe via
    set membership without surprises."""

    severity: Severity
    code: str          # short machine-readable code, e.g. "missing_format"
    message: str       # human-readable, suitable for a dialog or log line
    path: str = ""     # JSONPath-ish location of the problem, e.g. "shapes[3].rx"


# Geometry fields required by each shape type. The keys here are the
# `type` strings from the spec; the values are the names of fields
# that MUST be present and numeric. `color` is handled separately
# because it's required on every shape type.
_SHAPE_GEOMETRY_FIELDS: dict[str, tuple[str, ...]] = {
    "circle":            ("x", "y", "r"),
    "ellipse":           ("x", "y", "rx", "ry"),
    "rotated_ellipse":   ("x", "y", "rx", "ry", "angle"),
    "rectangle":         ("x", "y", "hw", "hh"),
    "rotated_rectangle": ("x", "y", "hw", "hh", "angle"),
    "triangle":          ("x1", "y1", "x2", "y2", "x3", "y3"),
}

# Fields on each shape that must be > 0 (extents/radii — `0` would
# mean a degenerate zero-area shape that the renderer would drop
# silently). `angle` is signed and can legitimately be `0`, so it's
# NOT included here.
_SHAPE_POSITIVE_FIELDS: dict[str, tuple[str, ...]] = {
    "circle":            ("r",),
    "ellipse":           ("rx", "ry"),
    "rotated_ellipse":   ("rx", "ry"),
    "rectangle":         ("hw", "hh"),
    "rotated_rectangle": ("hw", "hh"),
    "triangle":          (),
}

# Shape types the FH6 injector ships today (see docs/JSON_SPEC.md
# §Injector-safe subset). Anything outside this set is schema-valid
# but emits a WARNING because injection will silently drop it.
INJECTOR_SAFE_TYPES: frozenset[str] = frozenset({
    "rotated_ellipse", "ellipse", "circle", "rectangle",
})

# Top-level fields the spec defines. Unknown top-level fields are
# accepted with an INFO log line — common for non-injector tooling
# to attach metadata, and explicitly allowed by §Stability promise.
_KNOWN_TOP_LEVEL_FIELDS: frozenset[str] = frozenset({
    "format", "version", "source_image", "image_size",
    "shape_count", "generated_at", "profile", "sticker_mode",
    "shapes",
})


def validate_document(data: Any) -> list[Issue]:
    """Validate `data` as an `fd6.shapes` v1 document.

    Returns a flat list of `Issue`s. Order is stable: top-level
    checks first, then shapes in array order. Callers split by
    `Issue.severity`.

    Catastrophic inputs (`None`, `[]`, a string) produce a single
    ERROR issue and short-circuit — there's nothing further to check.
    """
    issues: list[Issue] = []

    if not isinstance(data, dict):
        return [Issue(Severity.ERROR, "not_an_object",
                      f"document root must be a JSON object, "
                      f"got {type(data).__name__}")]

    _check_top_level(data, issues)
    _check_shapes(data, issues)
    return issues


# ----------------------------------------------------------------- top-level


def _check_top_level(data: dict, issues: list[Issue]) -> None:
    """Validate the document envelope (format, version, image_size, etc.).

    These checks gate the whole document — `format != "fd6.shapes"`
    is an ERROR because everything downstream assumes our schema.
    """
    fmt = data.get("format")
    if fmt is None:
        issues.append(Issue(
            Severity.ERROR, "missing_format",
            f"document is missing required 'format' field "
            f"(expected '{FD6_FORMAT}'). This may be a legacy "
            f"geometrize JSON — convert via FD6Document.from_dict "
            f"first.",
            path="format",
        ))
    elif fmt != FD6_FORMAT:
        issues.append(Issue(
            Severity.ERROR, "wrong_format",
            f"document 'format' is {fmt!r}; expected {FD6_FORMAT!r}",
            path="format",
        ))

    ver = data.get("version")
    if ver is None:
        issues.append(Issue(
            Severity.ERROR, "missing_version",
            f"document is missing required 'version' field "
            f"(expected {FD6_VERSION})",
            path="version",
        ))
    elif ver != FD6_VERSION:
        issues.append(Issue(
            Severity.ERROR, "wrong_version",
            f"document 'version' is {ver!r}; this validator only "
            f"understands version {FD6_VERSION}",
            path="version",
        ))

    img = data.get("image_size")
    if img is None:
        issues.append(Issue(
            Severity.ERROR, "missing_image_size",
            "document is missing required 'image_size' [width, height]",
            path="image_size",
        ))
    elif not isinstance(img, (list, tuple)) or len(img) != 2:
        issues.append(Issue(
            Severity.ERROR, "bad_image_size_shape",
            f"'image_size' must be [width, height]; got {img!r}",
            path="image_size",
        ))
    else:
        w, h = img[0], img[1]
        if not (isinstance(w, int) and isinstance(h, int)):
            issues.append(Issue(
                Severity.ERROR, "bad_image_size_types",
                f"'image_size' entries must be integers; got "
                f"({type(w).__name__}, {type(h).__name__})",
                path="image_size",
            ))
        elif w <= 0 or h <= 0:
            issues.append(Issue(
                Severity.ERROR, "non_positive_image_size",
                f"'image_size' entries must be > 0; got [{w}, {h}]",
                path="image_size",
            ))

    shapes = data.get("shapes")
    if shapes is None:
        issues.append(Issue(
            Severity.ERROR, "missing_shapes",
            "document is missing required 'shapes' array",
            path="shapes",
        ))
    elif not isinstance(shapes, list):
        issues.append(Issue(
            Severity.ERROR, "shapes_not_array",
            f"'shapes' must be an array; got {type(shapes).__name__}",
            path="shapes",
        ))

    # shape_count is informational; mismatch is a warning so users can
    # still load JSONs that were hand-edited (a common workflow when
    # debugging the injector).
    shape_count = data.get("shape_count")
    if isinstance(shape_count, int) and isinstance(shapes, list):
        if shape_count != len(shapes):
            issues.append(Issue(
                Severity.WARNING, "shape_count_mismatch",
                f"'shape_count' is {shape_count} but 'shapes' has "
                f"{len(shapes)} entries — the array is what gets injected",
                path="shape_count",
            ))

    generated = data.get("generated_at")
    if isinstance(generated, str) and generated:
        try:
            # Spec format is "YYYY-MM-DDTHH:MM:SSZ"; fromisoformat
            # accepts the 'Z' suffix on 3.11+ but we go via replace
            # for older versions too. Pure provenance check.
            datetime.fromisoformat(generated.replace("Z", "+00:00"))
        except ValueError:
            issues.append(Issue(
                Severity.INFO, "bad_generated_at",
                f"'generated_at' is not parseable as ISO-8601 "
                f"({generated!r}) — provenance only, not blocking",
                path="generated_at",
            ))

    # Unknown top-level fields are explicitly allowed per the spec's
    # §Stability promise — flag them at INFO so tools can spot drift.
    for key in data.keys():
        if key not in _KNOWN_TOP_LEVEL_FIELDS:
            issues.append(Issue(
                Severity.INFO, "unknown_top_level_field",
                f"unknown top-level field {key!r} — accepted but not "
                f"defined by the v1 spec",
                path=key,
            ))


# --------------------------------------------------------------------- shapes


def _check_shapes(data: dict, issues: list[Issue]) -> None:
    """Validate each entry in `shapes[]`. Already gated on `shapes`
    being a list (errored at top level if not), but we double-check
    here so the function is safe to call in isolation."""
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return

    any_injector_safe = False
    saw_non_injector_safe_types: set[str] = set()

    for idx, shape in enumerate(shapes):
        path_prefix = f"shapes[{idx}]"
        if not isinstance(shape, dict):
            issues.append(Issue(
                Severity.ERROR, "shape_not_object",
                f"shape #{idx} is not a JSON object; got "
                f"{type(shape).__name__}",
                path=path_prefix,
            ))
            continue

        type_name = shape.get("type")
        if not isinstance(type_name, str):
            issues.append(Issue(
                Severity.ERROR, "shape_missing_type",
                f"shape #{idx} is missing 'type' (or 'type' is not a "
                f"string). Legacy integer-coded shapes must be "
                f"converted first.",
                path=f"{path_prefix}.type",
            ))
            continue

        required = _SHAPE_GEOMETRY_FIELDS.get(type_name)
        if required is None:
            issues.append(Issue(
                Severity.ERROR, "unknown_shape_type",
                f"shape #{idx} has unknown type {type_name!r}; valid "
                f"types are {sorted(_SHAPE_GEOMETRY_FIELDS)}",
                path=f"{path_prefix}.type",
            ))
            continue

        # Track injector-safety for the document-level rollup. We do
        # this before per-field validation so even malformed
        # injector-safe shapes count toward the "at least one safe
        # shape" warning suppression — a malformed shape gets its own
        # error, no need to compound it.
        if type_name in INJECTOR_SAFE_TYPES:
            any_injector_safe = True
        else:
            saw_non_injector_safe_types.add(type_name)

        _check_geometry_fields(shape, type_name, required, path_prefix, issues)
        _check_positive_fields(shape, type_name, path_prefix, issues)
        _check_color(shape, path_prefix, issues)
        _check_is_mask_alpha(shape, path_prefix, issues)

    if shapes and not any_injector_safe:
        issues.append(Issue(
            Severity.WARNING, "no_injector_safe_shapes",
            "document contains no injector-safe shape types — the FH6 "
            "injector will silently skip every shape. Convert "
            "triangles/rotated_rectangles to ellipses before injecting.",
            path="shapes",
        ))

    for type_name in sorted(saw_non_injector_safe_types):
        issues.append(Issue(
            Severity.WARNING, "non_injector_safe_type_present",
            f"document contains {type_name!r} shapes which the FH6 "
            f"injector cannot consume yet (see "
            f"docs/MULTISHAPE_FH6_RECON.md). These will be skipped at "
            f"inject time.",
            path="shapes",
        ))


def _check_geometry_fields(
    shape: dict, type_name: str, required: tuple[str, ...],
    path_prefix: str, issues: list[Issue],
) -> None:
    """Each required field must be present AND numeric. Missing fields
    and wrong types are both ERRORs."""
    for fld in required:
        if fld not in shape:
            issues.append(Issue(
                Severity.ERROR, "missing_geometry_field",
                f"shape of type {type_name!r} is missing required "
                f"field {fld!r}",
                path=f"{path_prefix}.{fld}",
            ))
            continue
        val = shape[fld]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            # bool is a subclass of int in Python — reject explicitly,
            # injecting a coordinate of `True` would be a real footgun.
            issues.append(Issue(
                Severity.ERROR, "non_numeric_geometry_field",
                f"shape of type {type_name!r} field {fld!r} must be "
                f"numeric; got {type(val).__name__} = {val!r}",
                path=f"{path_prefix}.{fld}",
            ))


def _check_positive_fields(
    shape: dict, type_name: str, path_prefix: str, issues: list[Issue],
) -> None:
    """Half-extents and radii must be `> 0`. A zero value would
    silently render to nothing; we'd rather error out than waste
    injector slots."""
    for fld in _SHAPE_POSITIVE_FIELDS.get(type_name, ()):
        val = shape.get(fld)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if val <= 0:
                issues.append(Issue(
                    Severity.ERROR, "non_positive_extent",
                    f"shape of type {type_name!r} field {fld!r} must "
                    f"be > 0; got {val!r}. A zero/negative extent "
                    f"renders to nothing and wastes an injector slot.",
                    path=f"{path_prefix}.{fld}",
                ))


def _check_color(shape: dict, path_prefix: str, issues: list[Issue]) -> None:
    """Every shape needs an RGBA uint8[4] color."""
    color = shape.get("color")
    if color is None:
        issues.append(Issue(
            Severity.ERROR, "missing_color",
            "shape is missing required 'color' [r, g, b, a]",
            path=f"{path_prefix}.color",
        ))
        return
    if not isinstance(color, (list, tuple)):
        issues.append(Issue(
            Severity.ERROR, "color_not_array",
            f"'color' must be an array of 4 uint8 RGBA values; got "
            f"{type(color).__name__}",
            path=f"{path_prefix}.color",
        ))
        return
    if len(color) != 4:
        issues.append(Issue(
            Severity.ERROR, "color_wrong_length",
            f"'color' must have exactly 4 RGBA channels; got {len(color)}",
            path=f"{path_prefix}.color",
        ))
        return
    for ch_idx, ch in enumerate(color):
        if not isinstance(ch, int) or isinstance(ch, bool):
            issues.append(Issue(
                Severity.ERROR, "color_channel_not_int",
                f"'color[{ch_idx}]' must be an integer 0-255; got "
                f"{type(ch).__name__} = {ch!r}",
                path=f"{path_prefix}.color[{ch_idx}]",
            ))
        elif not (0 <= ch <= 255):
            issues.append(Issue(
                Severity.ERROR, "color_channel_out_of_range",
                f"'color[{ch_idx}]' must be in [0, 255]; got {ch}",
                path=f"{path_prefix}.color[{ch_idx}]",
            ))


def _check_is_mask_alpha(
    shape: dict, path_prefix: str, issues: list[Issue],
) -> None:
    """Alpha-0 shapes without `is_mask: true` get silently filtered by
    materialize_shapes — surface that as a warning so users don't
    wonder where their shapes went."""
    color = shape.get("color")
    if not isinstance(color, (list, tuple)) or len(color) != 4:
        return  # color check above will have errored
    if not isinstance(color[3], int):
        return
    if color[3] != 0:
        return
    is_mask = bool(shape.get("is_mask", False))
    if not is_mask:
        issues.append(Issue(
            Severity.WARNING, "invisible_shape",
            "shape has alpha=0 but no 'is_mask: true' flag — the "
            "injector and materialize_shapes() will silently filter "
            "it out. Set is_mask:true to keep it as a boundary mask.",
            path=f"{path_prefix}.color",
        ))
