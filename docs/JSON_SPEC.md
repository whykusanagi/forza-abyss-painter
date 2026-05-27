# `fd6.shapes` v1 — canonical JSON specification

This document is the authoritative schema for **`fd6.shapes` version 1**,
the JSON format produced by Forza Abyss Painter's shape-gen pipeline
and consumed by the FH6 vinyl injector.

The source of truth in code is `forza_abyss_painter/io/json_schema.py`
(top-level document) and `forza_abyss_painter/shapegen/shapes/*.py`
(per-shape geometry). This doc and those modules must stay in sync —
when they disagree, the code wins and this doc is wrong; please file a
fix.

## Why this spec exists

- **Validator (`#100`) ground truth.** `fap-validate` reads this doc's
  rules verbatim. New shape types or fields go here BEFORE they ship.
- **Injector contract.** The FH6 injector parses a specific subset of
  shape types and rejects anything else. The schema documents
  injector-safe vs. shape-gen-only types explicitly.
- **External tooling.** Tutorial creators, third-party importers, and
  reverse-engineering work (see `docs/MULTISHAPE_FH6_RECON.md`) all
  rely on a stable, written contract.

## Document structure

A complete v1 document is a JSON object with these top-level fields:

```json
{
  "format": "fd6.shapes",
  "version": 1,
  "source_image": "input.png",
  "image_size": [1024, 768],
  "shape_count": 1000,
  "generated_at": "2026-05-26T17:42:11Z",
  "profile": "medium_1000",
  "sticker_mode": false,
  "shapes": [ ... ]
}
```

### Top-level fields

| Field            | Type                  | Required | Description                                                                                                                                                                                                                                                                  |
| ---------------- | --------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`         | string                | **yes**  | Must be literally `"fd6.shapes"`. Wrong value → reject.                                                                                                                                                                                                                       |
| `version`        | integer               | **yes**  | Must be `1` for this spec. Future spec revisions bump this.                                                                                                                                                                                                                  |
| `source_image`   | string                | no       | Original source image filename (no path). For provenance only; not validated.                                                                                                                                                                                                |
| `image_size`     | `[int, int]`          | **yes**  | `[width, height]` of the canvas the shapes are addressed against, in pixels. Both > 0. Coordinates of every shape are in this pixel space.                                                                                                                                   |
| `shape_count`    | integer               | no       | `len(shapes)`. Informational — validators may warn on mismatch but must not reject.                                                                                                                                                                                          |
| `generated_at`   | ISO-8601 UTC string   | no       | When the JSON was emitted. Format: `YYYY-MM-DDTHH:MM:SSZ`. Provenance only.                                                                                                                                                                                                  |
| `profile`        | string                | no       | Preset name (e.g. `medium_1000`, `highres_3000`, `legacy-geometrize-import`). Provenance only.                                                                                                                                                                               |
| `sticker_mode`   | boolean               | no       | Default `false`. `true` if the JSON was generated with transparent backdrop (the "add white background" checkbox was UNCHECKED). Used by GUI re-render. Older JSONs without this field render with a white canvas for backward compat.                                       |
| `shapes`         | array of shape dicts  | **yes**  | The painted shapes, in z-order from back to front. Each entry is a shape object (next section).                                                                                                                                                                              |

### Coordinate convention

- `(x, y) = (0, 0)` is the **top-left pixel**.
- `x` increases rightward; `y` increases downward.
- Shape centers (`x`, `y`) and triangle vertices (`x1..x3, y1..y3`) may
  be **outside `[0, width-1] × [0, height-1]`** — the renderer/injector
  clips. (The optimizer occasionally places shape centers slightly
  outside the canvas because their painted extent still overlaps it.)
- Half-extents (`hw`, `hh`, `rx`, `ry`, `r`) must be `> 0`.
- Angles are in **degrees**, no fixed range. `0°` = horizontal axis;
  positive = counter-clockwise. Both `+370°` and `+10°` are accepted.

### Color

Every shape has a `"color"` field: a 4-element array of `uint8` (0-255)
RGBA values: `[r, g, b, a]`.

- All four channels are required.
- `a == 0` shapes are valid in the schema but get filtered by the
  injector and by `materialize_shapes()`. To keep them, set `is_mask:
  true` (see "Optional per-shape flags").

## Shape types

### `rotated_ellipse`  *(injector-safe — primary)*

The only shape type the EXE production pipeline emits. Maps directly to
FH6's `ROTATED_ELLIPSE = 16` legacy code.

```json
{
  "type": "rotated_ellipse",
  "x": 512.0, "y": 384.0,
  "rx": 42.5, "ry": 17.0,
  "angle": 12.5,
  "color": [200, 12, 80, 180]
}
```

| Field   | Type            | Constraint        |
| ------- | --------------- | ----------------- |
| `type`  | string          | `"rotated_ellipse"` |
| `x, y`  | number (float)  | shape center, in canvas pixels |
| `rx, ry`| number          | radii (NOT diameters), `> 0` |
| `angle` | number          | degrees, signed   |
| `color` | uint8[4]        | RGBA              |

### `ellipse`  *(injector-safe via `angle=0` rotated_ellipse)*

Axis-aligned ellipse. Schema-valid. The injector promotes it to a
zero-angle `rotated_ellipse`; see `forza_abyss_painter/io/exporter.py`.

```json
{
  "type": "ellipse",
  "x": 512.0, "y": 384.0,
  "rx": 42.5, "ry": 17.0,
  "color": [200, 12, 80, 180]
}
```

### `circle`  *(injector-safe via `rx=ry` rotated_ellipse)*

Special case of `rotated_ellipse` with one radius and no angle. Schema-
valid. Injector promotes.

```json
{
  "type": "circle",
  "x": 512.0, "y": 384.0,
  "r": 25.0,
  "color": [200, 12, 80, 180]
}
```

| Field   | Type     | Constraint              |
| ------- | -------- | ----------------------- |
| `r`     | number   | radius, `> 0`           |

### `rectangle`  *(injector-safe — primary)*

Axis-aligned rectangle. Maps to FH6's `RECTANGLE = 1` legacy code.

```json
{
  "type": "rectangle",
  "x": 512.0, "y": 384.0,
  "hw": 80.0, "hh": 40.0,
  "color": [200, 12, 80, 180]
}
```

| Field    | Type     | Constraint                                                  |
| -------- | -------- | ----------------------------------------------------------- |
| `x, y`   | number   | center                                                      |
| `hw, hh` | number   | **half-width / half-height** (so total extents are `2·hw × 2·hh`). `> 0`. |

> **Coordinate gotcha:** the schema's `hw, hh` are HALF extents (radii-
> style). The legacy `geometrize` format wrote `[x, y, w, h]` as FULL
> extents — that's why `io/json_schema.py::_convert_legacy_shape`
> halves `w/2, h/2` when importing legacy. New JSONs emit native
> `hw, hh`. See [Legacy compatibility](#legacy-compatibility).

### `rotated_rectangle`  *(shape-gen only — NOT injector-safe)*

Rotated rectangle. Currently emitted by the multi-shape eval pipeline
but **the FH6 injector cannot consume this yet** — see
`docs/MULTISHAPE_FH6_RECON.md`. Validator must flag it as "injector-
incompatible" with a warning, not an error.

```json
{
  "type": "rotated_rectangle",
  "x": 512.0, "y": 384.0,
  "hw": 80.0, "hh": 40.0,
  "angle": 12.5,
  "color": [200, 12, 80, 180]
}
```

### `triangle`  *(shape-gen only — NOT injector-safe)*

Three-vertex triangle. Same status as `rotated_rectangle` — emitted by
the eval pipeline but not consumed by the FH6 injector. Validator must
warn, not error.

```json
{
  "type": "triangle",
  "x1": 100.0, "y1": 50.0,
  "x2": 200.0, "y2": 50.0,
  "x3": 150.0, "y3": 150.0,
  "color": [200, 12, 80, 180]
}
```

| Field                 | Type   | Constraint                                       |
| --------------------- | ------ | ------------------------------------------------ |
| `x1, y1, x2, y2, x3, y3` | number | three vertices in canvas pixels               |

## Injector-safe subset

Only these types are emitted into FH6 vinyl groups today:

| Type                | Injector path                                        |
| ------------------- | ---------------------------------------------------- |
| `rotated_ellipse`   | direct → `ROTATED_ELLIPSE = 16`                     |
| `ellipse`           | promoted to `rotated_ellipse` with `angle = 0`      |
| `circle`            | promoted to `rotated_ellipse` with `rx = ry = r`    |
| `rectangle`         | direct → `RECTANGLE = 1`                            |

Anything else in `shapes[]` is a schema-valid shape that the validator
emits at WARNING severity. The injector skips them silently.

## Optional per-shape flags

These flags are accepted on any shape type:

| Field      | Type    | Default | Effect                                                                                                                                                  |
| ---------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `is_mask`  | boolean | `false` | When `true`, the renderer/injector keeps this shape even if `color[3] == 0` (alpha 0). Used by boundary-mask preprocessing for transparent backdrops. |

## Optional metadata fields (`_*`)

Top-level keys prefixed with `_` are reserved for non-spec metadata
written by tools that produce or consume `fd6.shapes` documents.
Validators MUST NOT warn or error on these keys.

Currently reserved:

- `_run_config` — runner state, written by the GPU/CPU shape-gen
  runner into snapshot JSONs (and final output JSONs) so partial runs
  can be resumed with the original parameters. Shape varies by runner
  version; consumers should treat unknown sub-keys as opaque.

Future tools may add other `_*` keys (e.g. `_diag`, `_telemetry`).
Producers SHOULD prefix any non-spec metadata with `_` to stay out
of the validator's strict path.

## Legacy compatibility

Documents WITHOUT a `format` key are heuristically detected as
**geometrize / forza-painter** JSON and converted by
`io/json_schema.py::_from_legacy_geometrize`. They are NOT valid v1
documents — validators must convert them BEFORE validating against this
spec.

Legacy detection rule: no `format` key AND `shapes[]` contains entries
where `type` is an integer and `data` is an array.

Legacy integer type codes:
| Legacy code | Native type        | Coordinate quirk                                |
| ----------- | ------------------ | ----------------------------------------------- |
| `1`         | `rectangle`        | `data = [x, y, w, h]` (FULL extents — halved on import) |
| `16`        | `rotated_ellipse`  | `data = [x, y, w, h, rot]` where `w, h` are ALREADY RADII (passed through) |

The first shape in a legacy document is conventionally a full-canvas
background rectangle whose `data[2:4]` carries the image dimensions —
that's where the importer reads `image_size` from.

## Validator rules summary

The `fap-validate` CLI (`#100`) implements these checks. Severity:

- **ERROR** — document is rejected; injector will not run.
- **WARNING** — document loads but a downstream tool may behave unexpectedly.
- **INFO** — informational only.

| Check | Severity |
| ----- | -------- |
| `format == "fd6.shapes"` | ERROR |
| `version == 1` | ERROR |
| `image_size` is `[int, int]`, both `> 0` | ERROR |
| `shapes` is an array | ERROR |
| each shape has a valid `type` from the registry | ERROR |
| each shape has all required geometry fields for its type | ERROR |
| each shape's `color` is `uint8[4]` | ERROR |
| half-extents and radii are `> 0` | ERROR |
| `shape_count` matches `len(shapes)` | WARNING |
| at least one shape is injector-safe | WARNING |
| any `rotated_rectangle` or `triangle` present | WARNING (not injector-safe) |
| no shapes have `color[3] == 0` without `is_mask: true` | WARNING |
| `generated_at` parses as ISO-8601 UTC | INFO |

## Stability promise

Within **v1** of `fd6.shapes`:

- Existing shape types and their required fields **will not change**
  meaning. A v1 JSON written today must validate against future v1
  validators.
- New OPTIONAL fields may be added on existing shape types. Validators
  must accept unknown optional fields with a single INFO log line
  (not WARNING — common for non-injector tooling to add metadata).
- New shape types may be registered. They join `shapes/` and become
  schema-valid v1. Injector compatibility is a SEPARATE flag.
- The `format` value stays `"fd6.shapes"` and `version` stays `1`.

Breaking changes (renamed fields, type-meaning changes, removed
required fields) bump to `version: 2`.
