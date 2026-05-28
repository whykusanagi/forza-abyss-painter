# Tier B — Re-shape-gen (#85) + Polish loaded JSON (#86) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two buttons to the EXE's upload panel — "Re-shape-gen at higher budget…" (#85) and "Polish loaded JSON…" (#86) — both visible only when a JSON is loaded and their feature flags are flipped. Re-shape-gen reuses the existing GenerateLocallyDialog pre-filled with the source image from the loaded JSON. Polish opens a small new dialog (steps slider + lock-alpha checkbox) and runs the existing `joint_polish()` via a new `mode: "polish_only"` branch in `torch_runner`.

**Architecture:**
- `RunConfig` gains three optional fields: `mode`, `input_shapes_path`, `polish_steps_override`. `run()` adds a single top-level branch — `if cfg.mode == "polish_only": return _run_polish_only(cfg)` — leaving the existing fresh-gen path untouched.
- GUI surfaces are gated by two new flags in `feature_flags.py`. Both default `False` during dev; flipped to `True` in the same commit that lands smoke-green plumbing (CLAUDE.md §1c).
- Polish operates on `rotated_ellipse` shapes only (matches engine.py:697); non-ellipse loaded JSONs fail with a clear error event before any GPU work starts.
- Source-image discovery: same-folder heuristic (`<json_path.parent>/<doc.source_image>`) with file-picker fallback when the sibling is missing.

**Tech Stack:** Python 3.10+, PySide6 (Qt), PyTorch (via embedded runtime), pytest, pytest-qt offscreen mode.

**Spec:** `docs/superpowers/specs/2026-05-26-tier-b-reshapegen-polish-design.md`

---

## File Structure

### New files

| Path | Purpose |
|---|---|
| `forza_abyss_painter/gui/polish_dialog.py` | Small `QDialog` with polish-iterations spinbox + lock-alpha checkbox; exposes `.values()` after accept. |
| `tests/test_reshape_polish_flags_gating.py` | Pins `RESHAPE_GEN_AVAILABLE` and `POLISH_LOADED_AVAILABLE` defaults; documents the flip checklist. |
| `tests/test_generate_dialog_initial_source.py` | Verifies the new `initial_source_path` kwarg pre-fills the source field + output suggestion. |
| `tests/test_upload_panel_reshape_polish_buttons.py` | Offscreen-Qt: buttons hidden by default; visible after `set_json_loaded` AND flags on; emitted signals match. |
| `tests/test_resolve_source_image.py` | Unit test for `_resolve_source_image()` helper: sibling exists → returns it; missing → returns `None`. |
| `tests/test_polish_dialog_values.py` | `PolishDialog` defaults; `.values()` shape; output-override toggling. |
| `tests/test_torch_runner_polish_mode.py` | `RunConfig.from_dict({"mode": "polish_only", ...})` parsing + validation; default `"fresh"`; unknown mode raises. |
| `tests/test_polish_runner_integration.py` | Subprocess spawn with 3-ellipse JSON + 32×32 image; output exists, shape count preserved, validator-clean, geometry unchanged. |

### Modified files

| Path | Change |
|---|---|
| `forza_abyss_painter/gui/feature_flags.py` | Add `RESHAPE_GEN_AVAILABLE = False`, `POLISH_LOADED_AVAILABLE = False`. |
| `forza_abyss_painter/gui/generate_dialog.py` | Add optional `initial_source_path: Path \| None` kwarg to `__init__`. |
| `forza_abyss_painter/gui/upload_panel.py` | Two new buttons, two new signals (`reshape_requested`, `polish_requested`), `set_json_loaded(path)` slot. |
| `forza_abyss_painter/gui/main_window.py` | Slot `_on_reshape_requested`, slot `_on_polish_requested`, helper `_resolve_source_image`, call `upload.set_json_loaded(...)` after successful load. |
| `forza_abyss_painter/runtime/torch_runner.py` | Add `mode`, `input_shapes_path`, `polish_steps_override` to `RunConfig`; conditional-required validation; `_run_polish_only(cfg)` helper; one-line dispatch at the top of `run()`. |
| `forza_abyss_painter/gui/gpu_gen_worker.py` | Add `build_polish_config(...)` helper; pass-through `mode` if present in `build_run_config()` (unchanged default). |

### Deliberately untouched

- `forza_abyss_painter/io/json_schema.py` — no schema change.
- `forza_abyss_painter/io/validator.py` — existing path already covers polish output.
- `forza_abyss_painter/shapegen/gpu/joint_polish.py` — used as-is.
- `forza_abyss_painter/shapegen/presets.py` — no quality-knob changes.

---

## Task 1: Feature flags

**Files:**
- Modify: `forza_abyss_painter/gui/feature_flags.py:23-23` (append two flags)
- Create: `tests/test_reshape_polish_flags_gating.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_reshape_polish_flags_gating.py`:

```python
"""Pin Tier-B feature flags at False during development.

Mirrors tests/test_gpu_phase3_flag.py — the flag value cannot drift
independently of the plumbing landing. When the smoke is green and the
buttons are wired end-to-end, flip the flag in the SAME commit and
update this test (or delete it once stable, like GPU_PHASE_3).

Plumbing checklist for RESHAPE_GEN_AVAILABLE → True:
  - upload_panel emits `reshape_requested` on click
  - main_window slot `_on_reshape_requested` constructs GenerateLocallyDialog
    with `initial_source_path` from the loaded JSON
  - same-folder heuristic + picker fallback for source resolution
  - local smoke: real MainWindow loads a real JSON, clicks the button,
    fresh-gen completes end-to-end, output JSON lands + validates clean

Plumbing checklist for POLISH_LOADED_AVAILABLE → True:
  - upload_panel emits `polish_requested` on click
  - PolishDialog exposes (steps, lock_alpha, output_path) via .values()
  - torch_runner.RunConfig.mode == "polish_only" branch implemented
  - gpu_gen_worker.build_polish_config() writes valid config
  - local smoke: real MainWindow loads a real JSON, clicks the button,
    polish completes end-to-end, output _polished.json lands + validates
"""
from forza_abyss_painter.gui import feature_flags


def test_reshape_gen_flag_default_is_false():
    assert feature_flags.RESHAPE_GEN_AVAILABLE is False, (
        "Don't flip RESHAPE_GEN_AVAILABLE until the plumbing checklist "
        "in this test's docstring is fully landed and smoke-tested."
    )


def test_polish_loaded_flag_default_is_false():
    assert feature_flags.POLISH_LOADED_AVAILABLE is False, (
        "Don't flip POLISH_LOADED_AVAILABLE until the plumbing checklist "
        "in this test's docstring is fully landed and smoke-tested."
    )
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `pytest tests/test_reshape_polish_flags_gating.py -v`

Expected: FAIL with `AttributeError: module 'forza_abyss_painter.gui.feature_flags' has no attribute 'RESHAPE_GEN_AVAILABLE'`.

- [ ] **Step 1.3: Add the flags**

Edit `forza_abyss_painter/gui/feature_flags.py` — append after the existing `GPU_PHASE_3_AVAILABLE` line:

```python
# Re-shape-gen from a loaded JSON (#85). Visible when the upload_panel
# detects a JSON is loaded. Plumbing: upload_panel button + signal,
# main_window slot, GenerateLocallyDialog accepts initial_source_path.
# Flip True in the same commit that lands smoke-tested plumbing.
RESHAPE_GEN_AVAILABLE: bool = False

# Polish a loaded JSON via joint_polish (#86). Adds a new mode
# ("polish_only") to torch_runner.RunConfig. PolishDialog exposes
# polish iterations + lock_alpha. Output is <input>_polished.json.
# Flip True in the same commit that lands smoke-tested plumbing.
POLISH_LOADED_AVAILABLE: bool = False
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `pytest tests/test_reshape_polish_flags_gating.py -v`

Expected: PASS — both tests green.

- [ ] **Step 1.5: Commit**

```bash
git add forza_abyss_painter/gui/feature_flags.py tests/test_reshape_polish_flags_gating.py
git commit -m "$(cat <<'EOF'
feat(flags): add RESHAPE_GEN_AVAILABLE + POLISH_LOADED_AVAILABLE (#85 #86)

Both default False per CLAUDE.md §1c — flipped in the same commit that
lands smoke-tested plumbing. Pinning test enumerates the checklist that
must land before each flip.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `initial_source_path` kwarg on GenerateLocallyDialog (#85 prerequisite)

**Files:**
- Modify: `forza_abyss_painter/gui/generate_dialog.py:75-202` (constructor + `_on_browse_source` helper invocation)
- Create: `tests/test_generate_dialog_initial_source.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_generate_dialog_initial_source.py`:

```python
"""GenerateLocallyDialog accepts an `initial_source_path` kwarg so the
#85 Re-shape-gen flow can pre-fill the source image from a loaded JSON."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui.generate_dialog import GenerateLocallyDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_no_initial_source_path_keeps_source_field_empty(qapp, tmp_path):
    dlg = GenerateLocallyDialog(parent=None)
    assert dlg.source_path is None
    assert dlg.source_field.text() == ""
    assert dlg.generate_btn.isEnabled() is False
    dlg.deleteLater()


def test_initial_source_path_prefills_source_field(qapp, tmp_path):
    img = tmp_path / "nikke.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    dlg = GenerateLocallyDialog(parent=None, initial_source_path=img)
    assert dlg.source_path == img
    assert dlg.source_field.text() == str(img)
    assert dlg.generate_btn.isEnabled() is True
    # Output placeholder should reflect the pre-filled source.
    placeholder = dlg.output_field.placeholderText()
    assert "nikke" in placeholder
    assert placeholder.endswith(".json")
    dlg.deleteLater()


def test_initial_source_path_missing_file_keeps_source_unset(qapp, tmp_path):
    img = tmp_path / "missing.png"   # not created
    dlg = GenerateLocallyDialog(parent=None, initial_source_path=img)
    assert dlg.source_path is None, (
        "constructor must reject a missing initial_source_path so "
        "the user re-picks rather than running on a stale path"
    )
    assert dlg.generate_btn.isEnabled() is False
    dlg.deleteLater()
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `pytest tests/test_generate_dialog_initial_source.py -v`

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'initial_source_path'`.

- [ ] **Step 2.3: Add the kwarg**

Edit `forza_abyss_painter/gui/generate_dialog.py`. Change the constructor signature at line 75:

```python
    def __init__(self, parent=None, initial_source_path: Path | None = None) -> None:
```

At the END of `__init__` (after the existing `btn_row.addLayout(btn_row); root.addLayout(btn_row)`), append:

```python
        # Pre-fill source from #85 re-shape-gen flow if caller provided it
        # AND the file exists. Missing-file case falls through silently
        # so the user re-picks (avoids running on a stale path).
        if initial_source_path is not None and Path(initial_source_path).is_file():
            self.source_path = Path(initial_source_path)
            self.source_field.setText(str(self.source_path))
            self.generate_btn.setEnabled(True)
            preset = self.preset_combo.currentData()
            if preset:
                stem = self.source_path.stem
                suggested = (self.source_path.parent /
                             f"{stem}_{preset['num_shapes']}.json")
                self.output_field.setPlaceholderText(str(suggested))
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `pytest tests/test_generate_dialog_initial_source.py -v`

Expected: PASS — three tests green.

- [ ] **Step 2.5: Commit**

```bash
git add forza_abyss_painter/gui/generate_dialog.py tests/test_generate_dialog_initial_source.py
git commit -m "$(cat <<'EOF'
feat(gui): GenerateLocallyDialog accepts initial_source_path kwarg (#85)

Prereq for the Re-shape-gen flow: when invoked from a loaded JSON, the
dialog opens with the source image pre-filled so the user only has to
pick a preset. Missing-file values fall through silently so the user
re-picks rather than running on a stale path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Source-image resolver helper

**Files:**
- Modify: `forza_abyss_painter/gui/main_window.py` (add helper near other private methods around line 800)
- Create: `tests/test_resolve_source_image.py`

The helper does NOT use Qt — it's a pure function so we can unit-test without an event loop. Pull it into a module-level function inside `main_window.py` (or split to a sibling module if it bothers later cleanup; not required for this task).

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_resolve_source_image.py`:

```python
"""Source-image resolution for #85/#86: try <json_dir>/<doc.source_image>
first; return None if missing so the caller can open a file picker."""
from __future__ import annotations

from pathlib import Path

from forza_abyss_painter.gui.main_window import _resolve_source_image_path


def test_sibling_exists_returns_sibling_path(tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")
    image = tmp_path / "nikke.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    resolved = _resolve_source_image_path(json_path, "nikke.png")
    assert resolved == image


def test_sibling_missing_returns_none(tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")

    resolved = _resolve_source_image_path(json_path, "nikke.png")
    assert resolved is None


def test_empty_source_image_returns_none(tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")

    resolved = _resolve_source_image_path(json_path, "")
    assert resolved is None


def test_source_image_with_pathlike_chars_uses_basename(tmp_path):
    # Defensive: source_image is supposed to be a filename only, but if
    # a malformed JSON has "subdir/nikke.png", we still resolve to the
    # sibling rather than executing the subpath.
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")
    image = tmp_path / "nikke.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    resolved = _resolve_source_image_path(json_path, "subdir/nikke.png")
    assert resolved == image
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `pytest tests/test_resolve_source_image.py -v`

Expected: FAIL with `ImportError: cannot import name '_resolve_source_image_path' from 'forza_abyss_painter.gui.main_window'`.

- [ ] **Step 3.3: Add the helper**

Edit `forza_abyss_painter/gui/main_window.py`. Add as a **module-level** function (NOT a method on MainWindow) near the top of the file, right after the imports block:

```python
def _resolve_source_image_path(json_path: Path, source_image_name: str) -> Path | None:
    """Resolve the source image for a loaded JSON via the same-folder
    heuristic. `source_image_name` is the JSON's `source_image` field
    (canonically a bare filename); if it accidentally contains path
    separators we use only the basename for safety. Returns the path
    if the sibling exists, otherwise None — caller falls back to a
    file picker.
    """
    if not source_image_name:
        return None
    bare = Path(source_image_name).name   # strip any embedded path
    candidate = json_path.parent / bare
    return candidate if candidate.is_file() else None
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `pytest tests/test_resolve_source_image.py -v`

Expected: PASS — four tests green.

- [ ] **Step 3.5: Commit**

```bash
git add forza_abyss_painter/gui/main_window.py tests/test_resolve_source_image.py
git commit -m "$(cat <<'EOF'
feat(gui): _resolve_source_image_path helper for #85 #86

Same-folder heuristic + None on miss; caller handles file-picker fallback.
Treats source_image basename only so a malformed JSON with embedded path
separators can't escape the JSON's parent directory.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Upload panel — buttons + signals + visibility

**Files:**
- Modify: `forza_abyss_painter/gui/upload_panel.py:19-152`
- Create: `tests/test_upload_panel_reshape_polish_buttons.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/test_upload_panel_reshape_polish_buttons.py`:

```python
"""upload_panel exposes two new actions for loaded JSONs (#85 #86).
Visibility is gated on BOTH the feature flag AND the loaded-JSON state."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui import feature_flags  # noqa: E402
from forza_abyss_painter.gui.upload_panel import UploadPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def both_flags_on(monkeypatch):
    monkeypatch.setattr(feature_flags, "RESHAPE_GEN_AVAILABLE", True)
    monkeypatch.setattr(feature_flags, "POLISH_LOADED_AVAILABLE", True)


def test_buttons_hidden_when_flags_off(qapp):
    # Defaults: both flags False
    panel = UploadPanel()
    assert panel.reshape_btn.isHidden()
    assert panel.polish_btn.isHidden()
    panel.deleteLater()


def test_buttons_hidden_when_flags_on_but_no_json(qapp, both_flags_on):
    panel = UploadPanel()
    assert panel.reshape_btn.isHidden()
    assert panel.polish_btn.isHidden()
    panel.deleteLater()


def test_buttons_visible_when_flags_on_and_json_loaded(qapp, both_flags_on, tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")
    panel = UploadPanel()
    panel.set_json_loaded(json_path)
    assert panel.reshape_btn.isVisible() is True
    assert panel.polish_btn.isVisible() is True
    panel.deleteLater()


def test_buttons_hidden_again_when_json_cleared(qapp, both_flags_on, tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")
    panel = UploadPanel()
    panel.set_json_loaded(json_path)
    panel.set_json_loaded(None)
    assert panel.reshape_btn.isHidden()
    assert panel.polish_btn.isHidden()
    panel.deleteLater()


def test_reshape_button_emits_signal(qapp, both_flags_on, tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")
    panel = UploadPanel()
    panel.set_json_loaded(json_path)
    received: list[Path] = []
    panel.reshape_requested.connect(lambda p: received.append(p))
    panel.reshape_btn.click()
    assert received == [json_path]
    panel.deleteLater()


def test_polish_button_emits_signal(qapp, both_flags_on, tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")
    panel = UploadPanel()
    panel.set_json_loaded(json_path)
    received: list[Path] = []
    panel.polish_requested.connect(lambda p: received.append(p))
    panel.polish_btn.click()
    assert received == [json_path]
    panel.deleteLater()
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `pytest tests/test_upload_panel_reshape_polish_buttons.py -v`

Expected: FAIL with `AttributeError: 'UploadPanel' object has no attribute 'reshape_btn'`.

- [ ] **Step 4.3: Implement the buttons + signals + visibility**

Edit `forza_abyss_painter/gui/upload_panel.py`. Add to imports at the top:

```python
from forza_abyss_painter.gui import feature_flags
```

In the class, add to the signals block at the top of `UploadPanel` (around line 20):

```python
    reshape_requested = Signal(Path)     # User wants to re-shape-gen using the loaded JSON's source image
    polish_requested = Signal(Path)      # User wants to polish the loaded JSON in place
```

At the END of `__init__` (after `layout.addWidget(self.stack, stretch=1)` on line 74), append:

```python
        # Re-shape-gen + Polish (#85 #86). Both hidden until a JSON is loaded
        # AND the corresponding feature flag is True. Construction-time flag
        # reads are correct because flags are build-time constants.
        self._loaded_json_path: Path | None = None
        reshape_polish_row = QHBoxLayout()
        self.reshape_btn = QPushButton("Re-shape-gen at higher budget…", self)
        self.reshape_btn.setToolTip(
            "Re-run shape-gen on the same source image at a different shape "
            "budget. Opens the Generate dialog pre-filled with the source "
            "image from the loaded JSON."
        )
        self.reshape_btn.clicked.connect(self._on_reshape_clicked)
        self.reshape_btn.setVisible(False)
        self.polish_btn = QPushButton("Polish loaded JSON…", self)
        self.polish_btn.setToolTip(
            "Refine the colors of the shapes in the loaded JSON without "
            "generating new geometry. Output is saved as "
            "<input>_polished.json next to the loaded file."
        )
        self.polish_btn.clicked.connect(self._on_polish_clicked)
        self.polish_btn.setVisible(False)
        reshape_polish_row.addWidget(self.reshape_btn)
        reshape_polish_row.addWidget(self.polish_btn)
        layout.addLayout(reshape_polish_row)
```

Then add three new methods to `UploadPanel`:

```python
    def set_json_loaded(self, json_path: Path | None) -> None:
        """Called by MainWindow after Upload JSON succeeds (path) or fails
        (None). Toggles the Re-shape-gen + Polish buttons accordingly,
        respecting the feature flags. Construction-time flag values gate
        the *maximum* visibility; the loaded-JSON state gates the *actual*
        visibility within that maximum."""
        self._loaded_json_path = json_path if json_path is not None else None
        has_json = self._loaded_json_path is not None
        self.reshape_btn.setVisible(has_json and feature_flags.RESHAPE_GEN_AVAILABLE)
        self.polish_btn.setVisible(has_json and feature_flags.POLISH_LOADED_AVAILABLE)

    def _on_reshape_clicked(self) -> None:
        if self._loaded_json_path is not None:
            self.reshape_requested.emit(self._loaded_json_path)

    def _on_polish_clicked(self) -> None:
        if self._loaded_json_path is not None:
            self.polish_requested.emit(self._loaded_json_path)
```

- [ ] **Step 4.4: Run test to verify it passes**

Run: `pytest tests/test_upload_panel_reshape_polish_buttons.py -v`

Expected: PASS — six tests green.

- [ ] **Step 4.5: Commit**

```bash
git add forza_abyss_painter/gui/upload_panel.py tests/test_upload_panel_reshape_polish_buttons.py
git commit -m "$(cat <<'EOF'
feat(gui): upload_panel buttons + signals for #85 #86

Two new buttons (Re-shape-gen, Polish loaded JSON) wired to two new
signals (reshape_requested, polish_requested). Visibility gated on both
feature flags AND the loaded-JSON state via set_json_loaded(). Stays
hidden by default — flags both default False.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Polish dialog

**Files:**
- Create: `forza_abyss_painter/gui/polish_dialog.py`
- Create: `tests/test_polish_dialog_values.py`

- [ ] **Step 5.1: Write the failing test**

Create `tests/test_polish_dialog_values.py`:

```python
"""PolishDialog exposes (steps, lock_alpha, output_path) via .values()."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui.polish_dialog import PolishDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_defaults(qapp, tmp_path):
    loaded = tmp_path / "shapes.json"
    loaded.write_text("{}")
    src = tmp_path / "nikke.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = PolishDialog(parent=None, loaded_json_path=loaded, source_image_path=src)

    values = dlg.values()
    assert values["steps"] == 150
    assert values["lock_alpha"] is True
    # Default output: <loaded_stem>_polished.json next to the loaded JSON.
    assert values["output_path"] == loaded.parent / "shapes_polished.json"
    dlg.deleteLater()


def test_steps_range_clamped(qapp, tmp_path):
    loaded = tmp_path / "shapes.json"
    loaded.write_text("{}")
    src = tmp_path / "nikke.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = PolishDialog(parent=None, loaded_json_path=loaded, source_image_path=src)

    # Spin box bounds: 50..500.
    assert dlg.steps_spinbox.minimum() == 50
    assert dlg.steps_spinbox.maximum() == 500
    dlg.steps_spinbox.setValue(75)
    assert dlg.values()["steps"] == 75
    dlg.deleteLater()


def test_lock_alpha_toggle(qapp, tmp_path):
    loaded = tmp_path / "shapes.json"
    loaded.write_text("{}")
    src = tmp_path / "nikke.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = PolishDialog(parent=None, loaded_json_path=loaded, source_image_path=src)
    dlg.lock_alpha_cb.setChecked(False)
    assert dlg.values()["lock_alpha"] is False
    dlg.deleteLater()


def test_output_path_override(qapp, tmp_path):
    loaded = tmp_path / "shapes.json"
    loaded.write_text("{}")
    src = tmp_path / "nikke.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    override = tmp_path / "custom_out.json"

    dlg = PolishDialog(parent=None, loaded_json_path=loaded, source_image_path=src)
    dlg.set_output_path(override)
    assert dlg.values()["output_path"] == override
    dlg.deleteLater()
```

- [ ] **Step 5.2: Run test to verify it fails**

Run: `pytest tests/test_polish_dialog_values.py -v`

Expected: FAIL with `ImportError: No module named 'forza_abyss_painter.gui.polish_dialog'`.

- [ ] **Step 5.3: Implement the dialog**

Create `forza_abyss_painter/gui/polish_dialog.py`:

```python
"""Modal dialog for #86 Polish loaded JSON.

Two user-facing controls: polish iterations (50–500, default 150) and
lock alpha (default True). Output path defaults to
<input_json_stem>_polished.json next to the loaded JSON; user can
override via Choose output…
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QVBoxLayout,
)


_DEFAULT_STEPS = 150
_STEPS_MIN = 50
_STEPS_MAX = 500
_STEPS_STEP = 10


class PolishDialog(QDialog):
    """Modal dialog that gathers polish parameters."""

    def __init__(
        self,
        parent=None,
        *,
        loaded_json_path: Path,
        source_image_path: Path,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Polish loaded JSON")
        self.setModal(True)
        self.setMinimumWidth(520)

        self._loaded_json_path = Path(loaded_json_path)
        self._source_image_path = Path(source_image_path)
        self._output_path = self._default_output_path()

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        intro = QLabel(
            "Refine the colors and opacity of the shapes already in the "
            "loaded JSON. Geometry (positions, sizes, angles) is NOT "
            "changed. Output is saved as a new file alongside the input."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #999;")
        root.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.addRow("Loaded JSON:", QLabel(self._loaded_json_path.name))
        form.addRow("Source image:", QLabel(self._source_image_path.name))

        self.steps_spinbox = QSpinBox(self)
        self.steps_spinbox.setRange(_STEPS_MIN, _STEPS_MAX)
        self.steps_spinbox.setSingleStep(_STEPS_STEP)
        self.steps_spinbox.setValue(_DEFAULT_STEPS)
        self.steps_spinbox.setToolTip(
            "Number of Adam optimization steps. Higher = slower but "
            "potentially better color match. Default 150 matches the "
            "engine's joint_polish budget for the Medium 1000 preset."
        )
        form.addRow("Polish iterations:", self.steps_spinbox)

        self.lock_alpha_cb = QCheckBox("Lock alpha to 255", self)
        self.lock_alpha_cb.setChecked(True)
        self.lock_alpha_cb.setToolTip(
            "Keep every shape fully opaque (required for FH6 injection). "
            "Uncheck only for diagnostic experiments."
        )
        form.addRow(self.lock_alpha_cb)

        out_row = QHBoxLayout()
        self.output_label = QLabel(self._output_path.name)
        self.output_label.setStyleSheet("color: #ccc;")
        out_row.addWidget(self.output_label, stretch=1)
        self.output_choose_btn = QPushButton("Choose output…", self)
        self.output_choose_btn.clicked.connect(self._on_choose_output)
        out_row.addWidget(self.output_choose_btn)
        form.addRow("Output:", out_row)

        root.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)
        self.polish_btn = QPushButton("Polish", self)
        self.polish_btn.setDefault(True)
        self.polish_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.polish_btn)
        root.addLayout(btn_row)

    def _default_output_path(self) -> Path:
        return self._loaded_json_path.parent / f"{self._loaded_json_path.stem}_polished.json"

    def set_output_path(self, path: Path) -> None:
        self._output_path = Path(path)
        self.output_label.setText(self._output_path.name)

    def _on_choose_output(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self, "Polish output JSON",
            str(self._output_path),
            "Forza Abyss Painter shapes (*.json);;All files (*)",
        )
        if path:
            self.set_output_path(Path(path))

    def values(self) -> dict:
        """Return the user-selected polish parameters as a plain dict.
        Caller hands these to gpu_gen_worker.build_polish_config()."""
        return {
            "steps": int(self.steps_spinbox.value()),
            "lock_alpha": bool(self.lock_alpha_cb.isChecked()),
            "output_path": self._output_path,
        }
```

- [ ] **Step 5.4: Run test to verify it passes**

Run: `pytest tests/test_polish_dialog_values.py -v`

Expected: PASS — four tests green.

- [ ] **Step 5.5: Commit**

```bash
git add forza_abyss_painter/gui/polish_dialog.py tests/test_polish_dialog_values.py
git commit -m "$(cat <<'EOF'
feat(gui): PolishDialog for #86 — polish iterations + lock_alpha

Two-control dialog (50-500 step spinbox, lock-alpha checkbox) that
collects polish parameters for the new mode='polish_only' runner branch.
Default output is <loaded_stem>_polished.json next to the loaded file;
user can override via Choose output…

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: torch_runner — `mode` field + RunConfig validation

**Files:**
- Modify: `forza_abyss_painter/runtime/torch_runner.py:80-169` (`RunConfig` + `from_dict`)
- Create: `tests/test_torch_runner_polish_mode.py`

- [ ] **Step 6.1: Write the failing test**

Create `tests/test_torch_runner_polish_mode.py`:

```python
"""RunConfig accepts mode='polish_only' with conditional-required fields."""
from __future__ import annotations

from pathlib import Path

import pytest

from forza_abyss_painter.runtime.torch_runner import RunConfig


def _polish_dict(tmp_path) -> dict:
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    shapes = tmp_path / "shapes.json"
    shapes.write_text("{}")
    return {
        "image_path": str(image),
        "output_json_path": str(tmp_path / "out.json"),
        "mode": "polish_only",
        "input_shapes_path": str(shapes),
        "polish_steps_override": 100,
        # Fresh-mode required fields stay absent in polish_only.
    }


def _fresh_dict(tmp_path) -> dict:
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return {
        "image_path": str(image),
        "output_json_path": str(tmp_path / "out.json"),
        "num_shapes": 100,
        "max_resolution": 480,
        "random_samples": 1024,
    }


def test_default_mode_is_fresh(tmp_path):
    cfg = RunConfig.from_dict(_fresh_dict(tmp_path))
    assert cfg.mode == "fresh"
    assert cfg.input_shapes_path is None
    assert cfg.polish_steps_override is None


def test_polish_only_mode_parses(tmp_path):
    cfg = RunConfig.from_dict(_polish_dict(tmp_path))
    assert cfg.mode == "polish_only"
    assert cfg.input_shapes_path == Path(_polish_dict(tmp_path)["input_shapes_path"])
    assert cfg.polish_steps_override == 100


def test_polish_only_does_not_require_num_shapes(tmp_path):
    """num_shapes / max_resolution / random_samples are required for fresh
    but not for polish_only. Their absence in polish_only is OK."""
    d = _polish_dict(tmp_path)
    cfg = RunConfig.from_dict(d)
    assert cfg.num_shapes == 0   # placeholder default in polish_only
    assert cfg.max_resolution == 0
    assert cfg.random_samples == 0


def test_polish_only_requires_input_shapes_path(tmp_path):
    d = _polish_dict(tmp_path)
    del d["input_shapes_path"]
    with pytest.raises(ValueError, match="input_shapes_path"):
        RunConfig.from_dict(d)


def test_polish_only_requires_existing_input_shapes_path(tmp_path):
    d = _polish_dict(tmp_path)
    d["input_shapes_path"] = str(tmp_path / "does_not_exist.json")
    with pytest.raises(ValueError, match="not found"):
        RunConfig.from_dict(d)


def test_unknown_mode_raises(tmp_path):
    d = _polish_dict(tmp_path)
    d["mode"] = "frobnicate"
    with pytest.raises(ValueError, match="mode"):
        RunConfig.from_dict(d)


def test_fresh_mode_still_requires_num_shapes(tmp_path):
    d = _fresh_dict(tmp_path)
    del d["num_shapes"]
    with pytest.raises(ValueError, match="num_shapes"):
        RunConfig.from_dict(d)


def test_polish_steps_override_optional(tmp_path):
    d = _polish_dict(tmp_path)
    del d["polish_steps_override"]
    cfg = RunConfig.from_dict(d)
    assert cfg.polish_steps_override is None
```

- [ ] **Step 6.2: Run test to verify it fails**

Run: `pytest tests/test_torch_runner_polish_mode.py -v`

Expected: FAIL — multiple errors (no `mode` field, `from_dict` raises on missing `num_shapes` etc.).

- [ ] **Step 6.3: Extend RunConfig**

Edit `forza_abyss_painter/runtime/torch_runner.py`. Add to the `RunConfig` dataclass (after `preset_label: str = ""` on line 114):

```python
    # --- Mode dispatch (#85 #86) ---------------------------------------
    # mode == "fresh"        → generate shapes from image (the default,
    #                          unchanged historic behavior)
    # mode == "polish_only"  → load shapes from input_shapes_path + run
    #                          joint_polish() on them with the supplied
    #                          image as target. num_shapes, max_resolution,
    #                          random_samples are not required in this
    #                          mode (canvas size comes from the loaded
    #                          doc.image_size).
    mode: str = "fresh"
    input_shapes_path: Path | None = None
    polish_steps_override: int | None = None
```

Replace the body of `from_dict` (lines 117-169) entirely with:

```python
    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        """Parse + validate. Raises ValueError on missing required fields,
        wrong types, or out-of-range values. Conditional requirements
        depend on `mode`:

          mode='fresh' (default)  — image_path, output_json_path,
              num_shapes, max_resolution, random_samples are all required.
          mode='polish_only'       — image_path, output_json_path,
              input_shapes_path are required. The shape-budget fields are
              ignored (canvas dims come from the loaded JSON's image_size).
        """
        mode = str(d.get("mode", "fresh"))
        if mode not in ("fresh", "polish_only"):
            raise ValueError(
                f"unknown mode {mode!r} — must be 'fresh' or 'polish_only'"
            )

        # Always-required fields, regardless of mode.
        try:
            image_path = Path(d["image_path"])
            output_json_path = Path(d["output_json_path"])
        except KeyError as exc:
            raise ValueError(f"missing required config field: {exc}") from exc

        if mode == "polish_only":
            # polish_only fields.
            isp = d.get("input_shapes_path")
            if not isp:
                raise ValueError(
                    "input_shapes_path is required when mode='polish_only'"
                )
            input_shapes_path = Path(isp)
            if not input_shapes_path.is_file():
                raise ValueError(
                    f"input_shapes_path not found: {input_shapes_path}"
                )
            polish_steps_override = d.get("polish_steps_override")
            if polish_steps_override is not None:
                polish_steps_override = int(polish_steps_override)
                if polish_steps_override < 1:
                    raise ValueError(
                        f"polish_steps_override must be >= 1, "
                        f"got {polish_steps_override}"
                    )
            num_shapes = 0
            max_resolution = 0
            random_samples = 0
        else:
            # Fresh-mode required fields.
            try:
                num_shapes = int(d["num_shapes"])
                max_resolution = int(d["max_resolution"])
                random_samples = int(d["random_samples"])
            except KeyError as exc:
                raise ValueError(
                    f"missing required config field: {exc}"
                ) from exc
            if num_shapes < 1:
                raise ValueError(
                    f"num_shapes must be >= 1, got {num_shapes}"
                )
            if max_resolution < 64:
                raise ValueError(
                    f"max_resolution must be >= 64, got {max_resolution}"
                )
            if random_samples < 1:
                raise ValueError(
                    f"random_samples must be >= 1, got {random_samples}"
                )
            input_shapes_path = None
            polish_steps_override = None

        # Optional with type coercion (same across modes).
        lock_alpha = bool(d.get("lock_alpha", True))
        if not lock_alpha:
            raise ValueError(
                "lock_alpha=False is not a supported value — the Forza "
                "injector writes alpha=255 at inject time and any non-opaque "
                "JSON breaks preview/in-game parity."
            )
        device = str(d.get("device", "cuda"))
        if device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {device!r}")
        return cls(
            image_path=image_path,
            output_json_path=output_json_path,
            num_shapes=num_shapes,
            max_resolution=max_resolution,
            random_samples=random_samples,
            sticker_mode=bool(d.get("sticker_mode", False)),
            seed=int(d.get("seed", 0)),
            edge_strength=float(d.get("edge_strength", 0.0)),
            posterize_levels=int(d.get("posterize_levels", 0)),
            bbox_local=bool(d.get("bbox_local", True)),
            joint_polish_steps=int(d.get("joint_polish_steps", 0)),
            vram_budget_gib=float(d.get("vram_budget_gib", 0.0)),
            lock_alpha=lock_alpha,
            progress_every=int(d.get("progress_every", 0)),
            checkpoint_every=int(d.get("checkpoint_every", 0)),
            device=device,
            preset_label=str(d.get("preset_label", "")),
            mode=mode,
            input_shapes_path=input_shapes_path,
            polish_steps_override=polish_steps_override,
        )
```

- [ ] **Step 6.4: Run test to verify it passes**

Run: `pytest tests/test_torch_runner_polish_mode.py -v`

Expected: PASS — eight tests green.

- [ ] **Step 6.5: Run existing torch_runner tests to verify no regression**

Run: `pytest tests/test_torch_runner.py -v`

Expected: PASS — all existing tests still green (they don't pass `mode`, so default `"fresh"` keeps the historic path).

- [ ] **Step 6.6: Commit**

```bash
git add forza_abyss_painter/runtime/torch_runner.py tests/test_torch_runner_polish_mode.py
git commit -m "$(cat <<'EOF'
feat(runtime): RunConfig.mode + input_shapes_path + polish_steps_override

Adds mode dispatch to torch_runner's IPC schema:
- mode='fresh' (default) preserves all historic behavior — num_shapes,
  max_resolution, random_samples remain required.
- mode='polish_only' relaxes those three to optional and adds
  input_shapes_path (required) + polish_steps_override (optional).
- unknown modes raise ValueError before any subprocess work happens.

The run() branch + _run_polish_only() helper land in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: torch_runner — `_run_polish_only` implementation

**Files:**
- Modify: `forza_abyss_painter/runtime/torch_runner.py:292-446` (add `_run_polish_only` + dispatch in `run`)
- Create: `tests/test_polish_runner_integration.py`

This task adds the actual polish branch + a subprocess-driven integration test.

- [ ] **Step 7.1: Write the failing test**

Create `tests/test_polish_runner_integration.py`:

```python
"""End-to-end: spawn the runner subprocess in mode='polish_only' with a
real (small) image + a real (3-ellipse) shapes JSON, assert the output
file lands, has the same shape count, and the geometry is bit-identical
to the input (freeze_geometry=True).

Skipped when torch is not importable in the host env — this is an
integration test that exercises the real joint_polish call path.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")


def _write_test_image(path: Path, h: int = 32, w: int = 32) -> None:
    from PIL import Image
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :w // 2] = (200, 80, 80)   # left half: dusty red
    arr[:, w // 2:] = (80, 80, 200)   # right half: dusty blue
    Image.fromarray(arr, "RGB").save(path)


def _write_test_shapes_json(path: Path, w: int, h: int) -> dict:
    """Three rotated_ellipses in a v1 fd6.shapes document."""
    shapes = [
        {"type": "rotated_ellipse", "x": 8.0,  "y": 16.0, "rx": 6.0, "ry": 6.0,
         "angle": 0.0,  "color": [128, 128, 128, 255]},
        {"type": "rotated_ellipse", "x": 16.0, "y": 16.0, "rx": 6.0, "ry": 6.0,
         "angle": 30.0, "color": [128, 128, 128, 255]},
        {"type": "rotated_ellipse", "x": 24.0, "y": 16.0, "rx": 6.0, "ry": 6.0,
         "angle": 60.0, "color": [128, 128, 128, 255]},
    ]
    doc = {
        "format": "fd6.shapes",
        "version": 1,
        "source_image": "img.png",
        "image_size": [w, h],
        "shape_count": len(shapes),
        "generated_at": "",
        "profile": "test",
        "sticker_mode": False,
        "shapes": shapes,
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return doc


@pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("FAP_POLISH_TEST_FORCE_CPU") != "1",
    reason="Polish runs much faster on CUDA; skipping on CPU-only env. "
           "Set FAP_POLISH_TEST_FORCE_CPU=1 to force.",
)
def test_polish_only_subprocess_end_to_end(tmp_path):
    image = tmp_path / "img.png"
    _write_test_image(image, h=32, w=32)
    shapes_path = tmp_path / "shapes.json"
    in_doc = _write_test_shapes_json(shapes_path, w=32, h=32)

    cfg_path = tmp_path / "cfg.json"
    out_path = tmp_path / "out.json"
    cfg = {
        "image_path": str(image),
        "output_json_path": str(out_path),
        "mode": "polish_only",
        "input_shapes_path": str(shapes_path),
        "polish_steps_override": 10,   # tiny for test speed
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lock_alpha": True,
    }
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "forza_abyss_painter.runtime.torch_runner",
         "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, (
        f"runner exited {result.returncode}\nstderr:\n{result.stderr}"
    )
    assert out_path.is_file(), "polish runner did not write the output JSON"

    out_doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert out_doc["format"] == "fd6.shapes"
    assert out_doc["version"] == 1
    assert out_doc["shape_count"] == 3

    # Geometry frozen — x, y, rx, ry, angle preserved (within float
    # round-trip tolerance, which is large because rounding to 3 decimals
    # in joint_polish output is exact for these inputs).
    for i, (a, b) in enumerate(zip(in_doc["shapes"], out_doc["shapes"])):
        for key in ("x", "y", "rx", "ry", "angle"):
            assert abs(a[key] - b[key]) < 1e-3, (
                f"shape {i} key {key}: input {a[key]} != output {b[key]} "
                f"— geometry was not frozen"
            )
        # Alpha must remain 255 (lock_alpha=True).
        assert b["color"][3] == 255

    # Each shape's RGB should differ from the input gray (128,128,128)
    # along at least one channel, because polish color-snapped toward
    # the local target (red on left, blue on right). If RGB is exactly
    # the input gray for every shape, polish didn't actually run.
    rgb_changed_count = 0
    for a, b in zip(in_doc["shapes"], out_doc["shapes"]):
        if (a["color"][0], a["color"][1], a["color"][2]) != (
                b["color"][0], b["color"][1], b["color"][2]):
            rgb_changed_count += 1
    assert rgb_changed_count >= 1, (
        "polish did not change any shape's RGB — joint_polish may not "
        "have run, or the optimizer didn't move from the gray init"
    )


def test_polish_only_subprocess_rejects_non_ellipse(tmp_path):
    image = tmp_path / "img.png"
    _write_test_image(image, h=32, w=32)
    # JSON with a non-rotated_ellipse shape — engine.py:697 only polishes
    # all-ellipse documents. Runner should emit a clean error.
    shapes_path = tmp_path / "shapes.json"
    shapes_path.write_text(json.dumps({
        "format": "fd6.shapes",
        "version": 1,
        "source_image": "img.png",
        "image_size": [32, 32],
        "shape_count": 1,
        "generated_at": "",
        "profile": "test",
        "sticker_mode": False,
        "shapes": [
            {"type": "rectangle", "x": 16.0, "y": 16.0, "hw": 10.0, "hh": 10.0,
             "color": [100, 100, 100, 255]},
        ],
    }), encoding="utf-8")

    cfg_path = tmp_path / "cfg.json"
    out_path = tmp_path / "out.json"
    cfg_path.write_text(json.dumps({
        "image_path": str(image),
        "output_json_path": str(out_path),
        "mode": "polish_only",
        "input_shapes_path": str(shapes_path),
        "polish_steps_override": 10,
        "device": "cpu",
        "lock_alpha": True,
    }), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "forza_abyss_painter.runtime.torch_runner",
         "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=60,
    )

    assert result.returncode != 0, "runner should reject non-ellipse polish"
    # Stderr should contain a 'polish_only_unsupported_shape' or similar
    # stage in an error event. Verify the error event was emitted.
    assert "error" in result.stderr.lower()
    assert "ellipse" in result.stderr.lower() or "unsupported" in result.stderr.lower()
    assert not out_path.exists(), (
        "runner wrote output JSON despite reporting an error — IPC violation"
    )
```

- [ ] **Step 7.2: Run test to verify it fails**

Run: `pytest tests/test_polish_runner_integration.py -v`

Expected: FAIL — the runner doesn't have a polish branch yet, so the subprocess will either run the fresh-gen path (and crash because `num_shapes=0`) or exit non-zero with an unrelated error.

- [ ] **Step 7.3: Implement `_run_polish_only`**

Edit `forza_abyss_painter/runtime/torch_runner.py`. Add a new helper function right BEFORE the existing `run()` function (insert at line 292, pushing the existing `run` down):

```python
def _run_polish_only(cfg: RunConfig, stream, logger) -> int:
    """Polish branch for mode='polish_only'. Loads the source image +
    the existing shapes JSON, builds the joint_polish target (matching
    engine.py's construction at lines 460-570 of engine.py), runs the
    polish optimizer with freeze_geometry=True, saves the refined
    shapes via the canonical save_json path.

    All non-ellipse shapes are rejected here so the user gets a clean
    error before any GPU work starts (joint_polish only handles
    rotated_ellipse — engine.py:697 has the matching gate).
    """
    # Import inside the function so the import-error exit code (3) still
    # applies if torch isn't installed.
    try:
        with logger.start_phase("import_engine"):
            import torch  # noqa: F401
            import numpy as np
            from forza_abyss_painter.shapegen.gpu.joint_polish import joint_polish
            from forza_abyss_painter.shapegen.gpu.engine import (
                _posterize, _edge_weight_map, DTYPE, get_device,
            )
    except ImportError as exc:
        emit(stream, {
            "kind": "error", "stage": "import_engine",
            "message": (
                f"{type(exc).__name__}: {exc}. The GPU runtime isn't "
                f"fully installed in this Python — run the runtime "
                f"installer from Tools → Generate shapes locally first."
            ),
        })
        return 3

    # Load shapes JSON first — canvas dims come from doc.image_size, NOT
    # from cfg.max_resolution. This is the key polish_only invariant:
    # polish operates on the canvas the shapes were generated for.
    try:
        with logger.start_phase("load_shapes_json",
                                  shapes_path=str(cfg.input_shapes_path)):
            from forza_abyss_painter.io.exporter import load_json, save_json
            from forza_abyss_painter.io.json_schema import FD6Document
            doc = load_json(str(cfg.input_shapes_path))
            shapes_json = list(doc.shapes)
            logger.log("shapes_loaded",
                       shape_count=len(shapes_json),
                       image_size=list(doc.image_size))
    except (OSError, ValueError, KeyError) as exc:
        emit(stream, {
            "kind": "error", "stage": "load_shapes_json",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    # Polish only supports rotated_ellipse (joint_polish builds a
    # (N, 5) geometry tensor expecting x/y/rx/ry/angle). Reject other
    # shape types early.
    if not shapes_json:
        emit(stream, {
            "kind": "error", "stage": "polish_only_empty_input",
            "message": (
                f"input_shapes_path {cfg.input_shapes_path} has zero "
                f"shapes — nothing to polish."
            ),
        })
        return 1
    non_ellipse = [s for s in shapes_json if s.get("type") != "rotated_ellipse"]
    if non_ellipse:
        kinds = sorted({s.get("type", "?") for s in non_ellipse})
        emit(stream, {
            "kind": "error", "stage": "polish_only_unsupported_shape",
            "message": (
                f"polish_only supports rotated_ellipse only; found "
                f"{len(non_ellipse)} non-ellipse shape(s) of type(s) "
                f"{kinds} in {cfg.input_shapes_path.name}. Polish "
                f"skipped."
            ),
        })
        return 1

    if doc.image_size[0] <= 0 or doc.image_size[1] <= 0:
        emit(stream, {
            "kind": "error", "stage": "polish_only_invalid_canvas",
            "message": (
                f"loaded JSON has image_size={doc.image_size}; polish "
                f"requires a positive canvas size."
            ),
        })
        return 1
    canvas_w, canvas_h = int(doc.image_size[0]), int(doc.image_size[1])

    # Load + resize source image to the loaded JSON's canvas.
    try:
        with logger.start_phase("load_image", image_path=str(cfg.image_path)):
            from PIL import Image
            img = Image.open(cfg.image_path)
            sticker = cfg.sticker_mode or bool(getattr(doc, "sticker_mode", False))
            if sticker:
                rgba = img.convert("RGBA").resize(
                    (canvas_w, canvas_h), Image.LANCZOS)
                arr = np.asarray(rgba, dtype=np.uint8)
                rgb = arr[:, :, :3].copy()
                alpha_mask = arr[:, :, 3].copy()
            else:
                rgb = np.asarray(
                    img.convert("RGB").resize((canvas_w, canvas_h), Image.LANCZOS),
                    dtype=np.uint8,
                )
                alpha_mask = None
            logger.log("image_loaded_for_polish",
                       canvas_size=[canvas_w, canvas_h],
                       has_alpha=alpha_mask is not None)
    except (OSError, ValueError) as exc:
        emit(stream, {
            "kind": "error", "stage": "load_image",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    # Build polish inputs mirroring engine.py's construction. Posterize +
    # alpha substrate fill happen the same way; the only difference is we
    # start from the FINAL canvas (rendered from the loaded shapes) so
    # joint_polish sees the same baseline the engine would have at this
    # point in a fresh run.
    try:
        with logger.start_phase("build_polish_inputs"):
            device = get_device() if cfg.device == "cuda" else "cpu"
            if cfg.posterize_levels:
                rgb = _posterize(rgb, cfg.posterize_levels)
            if alpha_mask is not None:
                opaque_mask3 = (alpha_mask > 0)[:, :, None].astype(np.uint8)
                substrate = np.full_like(rgb, 40)
                target_np = np.where(opaque_mask3 > 0, rgb, substrate)
            else:
                target_np = rgb
            target = torch.from_numpy(target_np).to(device)
            alpha_t = None
            alpha_mask_f = None
            if alpha_mask is not None:
                alpha_t = torch.from_numpy(alpha_mask).to(device)
                alpha_mask_f = alpha_t.to(DTYPE) / 255.0
            edge_weight = (_edge_weight_map(target, cfg.edge_strength)
                            if cfg.edge_strength > 0 else None)
            # canvas_init is the mean of the target (or substrate-40 for
            # sticker) — same construction as engine.py:566-570.
            if alpha_mask is not None:
                canvas_init = torch.full(
                    (canvas_h, canvas_w, 3), 40,
                    dtype=torch.uint8, device=device,
                )
            else:
                mean = target.to(DTYPE).reshape(-1, 3).mean(dim=0).round() \
                    .clamp(0, 255).to(torch.uint8)
                canvas_init = mean.view(1, 1, 3) \
                    .expand(canvas_h, canvas_w, 3).contiguous().clone()
    except Exception as exc:  # pragma: no cover — defensive
        emit(stream, {
            "kind": "error", "stage": "build_polish_inputs",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    steps = cfg.polish_steps_override if cfg.polish_steps_override is not None else 150
    emit(stream, {"kind": "progress", "shape_count": 0, "total": steps})

    try:
        with logger.start_phase("joint_polish",
                                  steps=steps,
                                  shape_count=len(shapes_json)):
            refined, _canvas_np = joint_polish(
                shapes_json, target, alpha_t, alpha_mask_f, edge_weight,
                canvas_init, canvas_h, canvas_w, steps,
                lock_alpha=cfg.lock_alpha,
                purity_penalty=0.0,
                freeze_geometry=True,
            )
            logger.log("joint_polish_done", refined_count=len(refined))
    except RuntimeError as exc:
        # joint_polish + OOM are converted to RuntimeError upstream;
        # surface verbatim like the fresh path.
        emit(stream, {
            "kind": "error", "stage": "joint_polish",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1
    except Exception as exc:  # pragma: no cover — defensive
        emit(stream, {
            "kind": "error", "stage": "joint_polish",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    # Save via the canonical exporter so the validator hook (#100) runs.
    try:
        with logger.start_phase("save_json",
                                  output_path=str(cfg.output_json_path)):
            polished_doc = FD6Document(
                source_image=cfg.image_path.name,
                image_size=(canvas_w, canvas_h),
                shape_count=len(refined),
                generated_at=doc.generated_at,
                profile=f"{doc.profile} (polished)" if doc.profile else "polished",
                sticker_mode=sticker,
                shapes=refined,
            )
            save_json(polished_doc, cfg.output_json_path)
    except (OSError, ValueError, KeyError) as exc:
        emit(stream, {
            "kind": "error", "stage": "save_json",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    logger.log("polish_only_done",
               output_path=str(cfg.output_json_path),
               shape_count=len(refined))
    emit(stream, {
        "kind": "done",
        "output_path": str(cfg.output_json_path),
        "shape_count": len(refined),
    })
    return 0
```

Now add the dispatch in `run()`. At the top of the existing `run()` (right after the `_install_graceful_shutdown_handler(stream, logger)` line, around the new line equivalent of 315), insert:

```python
    emit(stream, {"kind": "started", "cfg_summary": cfg.summary()})

    # Mode dispatch (#85 #86). polish_only branches off here; fresh
    # continues with the historic path below.
    if cfg.mode == "polish_only":
        return _run_polish_only(cfg, stream, logger)
```

Make sure the existing `emit(stream, {"kind": "started", ...})` line that was previously the first thing emitted in `run()` is removed (it's now in the block above — don't double-emit).

- [ ] **Step 7.4: Run test to verify it passes**

Run: `pytest tests/test_polish_runner_integration.py -v`

Expected: PASS — both tests green. If the CUDA test is skipped because the host has no CUDA, that's OK; the non-ellipse rejection test still runs on CPU.

- [ ] **Step 7.5: Run existing torch_runner tests to verify no regression**

Run: `pytest tests/test_torch_runner.py tests/test_torch_runner_polish_mode.py -v`

Expected: PASS — all green. The fresh-mode path is unchanged.

- [ ] **Step 7.6: Commit**

```bash
git add forza_abyss_painter/runtime/torch_runner.py tests/test_polish_runner_integration.py
git commit -m "$(cat <<'EOF'
feat(runtime): _run_polish_only branch for mode='polish_only' (#86)

Loads source image + shapes JSON, resizes image to doc.image_size (NOT
max_resolution — shapes live in the canvas they were generated for),
mirrors engine.py's polish-target construction (posterize + alpha
substrate + edge weight), runs joint_polish with freeze_geometry=True,
saves via canonical save_json (validator hook from #100 runs).

Non-ellipse shapes rejected with a clean error event before any GPU
work happens — joint_polish only supports rotated_ellipse.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: gpu_gen_worker — `build_polish_config` helper

**Files:**
- Modify: `forza_abyss_painter/gui/gpu_gen_worker.py:376-421` (append helper)
- Test: extend `tests/test_torch_runner_polish_mode.py` with a builder check, or write a new tiny test. We'll add to a new test file for clarity.

- [ ] **Step 8.1: Write the failing test**

Append to `tests/test_torch_runner_polish_mode.py` (or create `tests/test_build_polish_config.py` — using the latter for cleanliness):

Create `tests/test_build_polish_config.py`:

```python
"""build_polish_config converts PolishDialog values + paths into a
RunConfig-compatible dict that mode='polish_only' accepts."""
from __future__ import annotations

from pathlib import Path

from forza_abyss_painter.gui.gpu_gen_worker import build_polish_config
from forza_abyss_painter.runtime.torch_runner import RunConfig


def test_build_polish_config_round_trips_through_RunConfig(tmp_path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    shapes = tmp_path / "shapes.json"
    shapes.write_text("{}")
    out = tmp_path / "out.json"

    cfg_dict = build_polish_config(
        source_image_path=image,
        input_shapes_path=shapes,
        output_path=out,
        steps=200,
        lock_alpha=True,
        sticker_mode=False,
    )
    assert cfg_dict["mode"] == "polish_only"
    assert cfg_dict["image_path"] == str(image)
    assert cfg_dict["input_shapes_path"] == str(shapes)
    assert cfg_dict["output_json_path"] == str(out)
    assert cfg_dict["polish_steps_override"] == 200
    assert cfg_dict["lock_alpha"] is True

    # Round-trip through the parser proves the schema is valid.
    parsed = RunConfig.from_dict(cfg_dict)
    assert parsed.mode == "polish_only"
    assert parsed.input_shapes_path == shapes
    assert parsed.polish_steps_override == 200


def test_build_polish_config_passes_sticker_mode(tmp_path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    shapes = tmp_path / "shapes.json"
    shapes.write_text("{}")
    out = tmp_path / "out.json"

    cfg_dict = build_polish_config(
        source_image_path=image,
        input_shapes_path=shapes,
        output_path=out,
        steps=100,
        lock_alpha=True,
        sticker_mode=True,
    )
    assert cfg_dict["sticker_mode"] is True
```

- [ ] **Step 8.2: Run test to verify it fails**

Run: `pytest tests/test_build_polish_config.py -v`

Expected: FAIL with `ImportError: cannot import name 'build_polish_config'`.

- [ ] **Step 8.3: Implement the helper**

Edit `forza_abyss_painter/gui/gpu_gen_worker.py`. At the very END of the file (after `build_run_config`), append:

```python
def build_polish_config(
    source_image_path: Path,
    input_shapes_path: Path,
    output_path: Path,
    steps: int,
    lock_alpha: bool = True,
    sticker_mode: bool = False,
) -> dict:
    """Map PolishDialog values + paths → torch_runner RunConfig dict
    with mode='polish_only'. Sibling to build_run_config for #86.

    num_shapes / max_resolution / random_samples are intentionally
    omitted — RunConfig.from_dict treats them as optional under
    polish_only and ignores them at the runner level (canvas dims come
    from the loaded JSON's image_size).
    """
    return {
        "image_path": str(source_image_path),
        "output_json_path": str(output_path),
        "mode": "polish_only",
        "input_shapes_path": str(input_shapes_path),
        "polish_steps_override": int(steps),
        "lock_alpha": bool(lock_alpha),
        "sticker_mode": bool(sticker_mode),
        # Hold to the same defensive defaults the fresh builder applies.
        "bbox_local": True,
        "vram_budget_gib": 0.0,
        "preset_label": "polish_only",
    }
```

- [ ] **Step 8.4: Run test to verify it passes**

Run: `pytest tests/test_build_polish_config.py -v`

Expected: PASS — two tests green.

- [ ] **Step 8.5: Commit**

```bash
git add forza_abyss_painter/gui/gpu_gen_worker.py tests/test_build_polish_config.py
git commit -m "$(cat <<'EOF'
feat(gui): build_polish_config helper for #86

Sibling to build_run_config — assembles a mode='polish_only' RunConfig
dict from PolishDialog values + paths. Round-trip through
RunConfig.from_dict is pinned by the test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: MainWindow — wire upload_panel signals to slots

**Files:**
- Modify: `forza_abyss_painter/gui/main_window.py:83` (connect new signals) and add two slots + update `_on_json_loaded_for_preview` to call `upload.set_json_loaded(...)`

No new test file — this wiring is exercised by the local smoke (Task 10) which constructs a real `MainWindow`.

- [ ] **Step 9.1: Connect signals + clear-on-load-error**

Edit `forza_abyss_painter/gui/main_window.py` around line 83 where `self.upload.json_loaded.connect(...)` already lives. Add the two new connections immediately after:

```python
        self.upload.reshape_requested.connect(self._on_reshape_requested)
        self.upload.polish_requested.connect(self._on_polish_requested)
```

- [ ] **Step 9.2: Update `_on_json_loaded_for_preview` to notify upload_panel**

Locate `_on_json_loaded_for_preview` (around line 1264). At the very end of the SUCCESS path (right after `self._loaded_json_path = json_path` at line 1329), add:

```python
        self.upload.set_json_loaded(json_path)
```

In the **error** paths inside this function (the `return` after `QMessageBox.critical(self, "Load failed", …)` around line 1283, and the `return` after the validation-errors block around line 1302), add right before each `return`:

```python
        self.upload.set_json_loaded(None)
```

- [ ] **Step 9.3: Add the two slots**

Add at module level (after `_resolve_source_image_path` from Task 3) OR as methods on `MainWindow` (preferred — they need access to `self`). Put them just before `_on_json_loaded_for_preview`:

```python
    def _prompt_source_image_picker(self, hint_filename: str) -> Path | None:
        """Fallback file picker when the sibling source image is missing.
        Returns the user-picked path or None on cancel."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        QMessageBox.information(
            self, "Source image not found",
            f"The loaded JSON refers to source image '{hint_filename}', "
            f"but no file with that name exists next to the JSON. "
            f"Pick the source image manually.",
        )
        path, _ = QFileDialog.getOpenFileName(
            self, f"Pick source image (looking for '{hint_filename}')",
            "", "Images (*.png *.jpg *.jpeg *.webp);;All files (*)",
        )
        return Path(path) if path else None

    def _resolve_source_for_loaded_json(self, json_path: Path) -> Path | None:
        """Resolve the source image for a loaded JSON. Returns None when
        the user cancels the picker fallback — caller aborts silently."""
        from forza_abyss_painter.io.exporter import load_json
        try:
            doc = load_json(str(json_path))
        except Exception:
            # Validator already gated us on load; getting here means the
            # file changed between load and this call. Treat as fatal.
            return None
        sibling = _resolve_source_image_path(json_path, doc.source_image)
        if sibling is not None:
            return sibling
        return self._prompt_source_image_picker(doc.source_image or "<unknown>")

    def _on_reshape_requested(self, json_path: Path) -> None:
        """#85 — user clicked Re-shape-gen at higher budget. Resolve the
        source image, then open the existing GenerateLocallyDialog
        pre-populated with that source."""
        source = self._resolve_source_for_loaded_json(json_path)
        if source is None:
            return
        from forza_abyss_painter.gui.generate_dialog import GenerateLocallyDialog
        from forza_abyss_painter.gui.runtime_install_dialog import (
            prompt_install_or_use_existing,
        )
        if not prompt_install_or_use_existing(self):
            return
        dlg = GenerateLocallyDialog(self, initial_source_path=source)
        if dlg.exec() == dlg.Accepted and dlg.output_path:
            self._on_json_loaded_for_preview(dlg.output_path)

    def _on_polish_requested(self, json_path: Path) -> None:
        """#86 — user clicked Polish loaded JSON. Resolve the source
        image, open PolishDialog, on accept spawn GpuGenWorker with the
        polish config."""
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QMessageBox
        source = self._resolve_source_for_loaded_json(json_path)
        if source is None:
            return
        from forza_abyss_painter.gui.polish_dialog import PolishDialog
        from forza_abyss_painter.gui.gpu_gen_worker import (
            GpuGenWorker, build_polish_config,
        )
        from forza_abyss_painter.gui.runtime_install_dialog import (
            prompt_install_or_use_existing,
        )
        from forza_abyss_painter.runtime.torch_installer import embedded_python_exe
        if not prompt_install_or_use_existing(self):
            return
        dlg = PolishDialog(self,
                            loaded_json_path=json_path,
                            source_image_path=source)
        if dlg.exec() != dlg.Accepted:
            return
        values = dlg.values()

        # Detect sticker mode from the source image's transparency, same
        # as the renderer does. If the user set the sticker checkbox, that
        # also flips this on (matches fresh-gen ergonomics).
        sticker = bool(getattr(self.settings_panel, "sticker_mode_cb", None) and
                        not self.settings_panel.sticker_mode_cb.isChecked())

        config = build_polish_config(
            source_image_path=source,
            input_shapes_path=json_path,
            output_path=values["output_path"],
            steps=values["steps"],
            lock_alpha=values["lock_alpha"],
            sticker_mode=sticker,
        )
        config_path = json_path.parent / f".{json_path.stem}_polish_config.json"
        try:
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Couldn't start polish",
                                  f"Failed to write polish config: {exc}")
            return
        py = embedded_python_exe()
        if not py.exists():
            QMessageBox.critical(
                self, "GPU runtime missing",
                f"Embedded Python not found at {py}. "
                f"Open Tools → Generate shapes locally to install the runtime.",
            )
            return

        # Spawn the worker on a QThread — same pattern GenerateLocallyDialog uses.
        self._polish_thread = QThread(self)
        self._polish_worker = GpuGenWorker(
            embedded_python_exe=py,
            config_path=config_path,
        )
        self._polish_worker.moveToThread(self._polish_thread)
        self._polish_thread.started.connect(self._polish_worker.run)
        self._polish_worker.started.connect(self._on_polish_started)
        self._polish_worker.progress.connect(self._on_polish_progress)
        self._polish_worker.checkpoint.connect(self._on_polish_progress)
        self._polish_worker.done.connect(self._on_polish_done)
        self._polish_worker.error.connect(self._on_polish_error)
        self._polish_worker.finished.connect(self._polish_thread.quit)
        self._polish_worker.finished.connect(self._polish_worker.deleteLater)
        self._polish_thread.finished.connect(self._polish_thread.deleteLater)
        self.statusBar().showMessage("Polishing — running optimizer on GPU…")
        self._polish_thread.start()

    def _on_polish_started(self, summary: dict) -> None:
        self.statusBar().showMessage("Polish started — running joint_polish on GPU…")

    def _on_polish_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.statusBar().showMessage(f"Polish — step {current}/{total}")

    def _on_polish_done(self, output_path: str, shape_count: int) -> None:
        from pathlib import Path as _Path
        path = _Path(output_path)
        self.statusBar().showMessage(
            f"Polish done — {shape_count} shapes saved to {path.name}", 10000,
        )
        # Auto-load the polished JSON into the preview so the user can
        # compare visually + click Inject when ready.
        self._on_json_loaded_for_preview(path)

    def _on_polish_error(self, stage: str, message: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, f"Polish failed — {stage}",
                              f"Stage: {stage}\n\n{message}")
        self.statusBar().showMessage(f"Polish failed at {stage}.", 10000)
```

Confirm `json` is imported at the top of `main_window.py` — it almost certainly already is. If not, add `import json` near the top.

- [ ] **Step 9.4: Run existing main_window tests to verify no regression**

Run: `pytest tests/test_main_window.py tests/test_gpu_bundle_gui.py -v` (or whichever main_window-touching tests exist — search via `grep -l "MainWindow" tests/`).

Expected: PASS — slots are pure additions; existing behavior is unchanged.

- [ ] **Step 9.5: Commit**

```bash
git add forza_abyss_painter/gui/main_window.py
git commit -m "$(cat <<'EOF'
feat(gui): wire reshape_requested + polish_requested slots in MainWindow

#85 slot opens GenerateLocallyDialog with the loaded JSON's source
image pre-filled. #86 slot opens PolishDialog, on accept spawns a
GpuGenWorker with mode='polish_only' and auto-loads the polished JSON
into the preview when done.

Source image resolution: same-folder heuristic with file-picker fallback
when the sibling is missing.

upload.set_json_loaded(...) is called on every JSON-load success
(enables the buttons) and on every load failure (hides them again).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Local smoke + flag flip

This is the gate that ships the feature, per CLAUDE.md §1c.

**Files:**
- Create: `tests/test_tier_b_local_smoke.py`
- Modify: `forza_abyss_painter/gui/feature_flags.py` (flip both flags to `True`)
- Modify: `tests/test_reshape_polish_flags_gating.py` (flip the assertions to `True`)

- [ ] **Step 10.1: Write the smoke test**

Create `tests/test_tier_b_local_smoke.py`. Per CLAUDE.md §8g this MUST
construct the real `MainWindow` and walk through to verify the buttons
render. Per §8h, only ONE `MainWindow` per process — both checks share
one construction.

```python
"""Local smoke (CLAUDE.md §1b + §8g) — construct real MainWindow under
offscreen Qt, load a real JSON, verify Re-shape-gen + Polish buttons
become visible. Then call _run_polish_only directly with real inputs
and assert the output JSON validates clean.

Per CLAUDE.md §8h: ONE MainWindow per process. Both assertions share
the single construction.

Skipped when torch is not importable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui import feature_flags  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def both_flags_on(monkeypatch):
    monkeypatch.setattr(feature_flags, "RESHAPE_GEN_AVAILABLE", True)
    monkeypatch.setattr(feature_flags, "POLISH_LOADED_AVAILABLE", True)


def _write_image(path: Path, h=64, w=64):
    from PIL import Image
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :w // 2] = (200, 80, 80)
    arr[:, w // 2:] = (80, 80, 200)
    Image.fromarray(arr, "RGB").save(path)


def _write_shapes_json(path: Path, w=64, h=64):
    doc = {
        "format": "fd6.shapes",
        "version": 1,
        "source_image": "img.png",
        "image_size": [w, h],
        "shape_count": 3,
        "generated_at": "",
        "profile": "smoke",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 16.0, "y": 32.0, "rx": 8.0,
             "ry": 8.0, "angle": 0.0, "color": [128, 128, 128, 255]},
            {"type": "rotated_ellipse", "x": 32.0, "y": 32.0, "rx": 8.0,
             "ry": 8.0, "angle": 30.0, "color": [128, 128, 128, 255]},
            {"type": "rotated_ellipse", "x": 48.0, "y": 32.0, "rx": 8.0,
             "ry": 8.0, "angle": 60.0, "color": [128, 128, 128, 255]},
        ],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_tier_b_smoke(qapp, tmp_path):
    """Single test, single MainWindow:
      Part A — construct MainWindow, simulate JSON load via the same slot
               the upload panel signals, assert both buttons become visible.
      Part B — directly invoke _run_polish_only with the same inputs,
               assert the output JSON validates clean.

    Part B doesn't go through GpuGenWorker (which would need the embedded
    python that isn't installed in test envs). The subprocess hop is
    covered by tests/test_polish_runner_integration.py."""
    img = tmp_path / "img.png"
    _write_image(img)
    shapes = tmp_path / "shapes.json"
    _write_shapes_json(shapes)

    # ---- Part A: real MainWindow + button visibility ----
    from forza_abyss_painter.gui.main_window import MainWindow
    win = MainWindow()
    try:
        win._on_json_loaded_for_preview(shapes)
        assert win._loaded_json_path == shapes
        assert win.upload.reshape_btn.isVisible(), (
            "Re-shape-gen button did not become visible after JSON load — "
            "set_json_loaded wiring is missing"
        )
        assert win.upload.polish_btn.isVisible(), (
            "Polish button did not become visible after JSON load — "
            "set_json_loaded wiring is missing"
        )
    finally:
        # Explicit cleanup so any second test construction wouldn't trip
        # the deleteLater hazard (CLAUDE.md §8h).
        win.close()
        win.deleteLater()

    # ---- Part B: polish runner end-to-end ----
    from forza_abyss_painter.runtime.torch_runner import RunConfig, _run_polish_only
    from forza_abyss_painter.io.exporter import load_json
    from forza_abyss_painter.io.validator import Severity, validate_document

    out_path = tmp_path / "shapes_polished.json"
    cfg = RunConfig.from_dict({
        "image_path": str(img),
        "output_json_path": str(out_path),
        "mode": "polish_only",
        "input_shapes_path": str(shapes),
        "polish_steps_override": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lock_alpha": True,
    })

    class _StubLogger:
        def log(self, *a, **kw): pass
        def log_exception(self, *a, **kw): pass
        def start_phase(self, *a, **kw):
            from contextlib import nullcontext
            return nullcontext()

    rc = _run_polish_only(cfg, sys.stderr, _StubLogger())
    assert rc == 0, f"polish_only exited {rc}"
    assert out_path.is_file()

    polished = load_json(str(out_path))
    issues = validate_document(polished.to_dict())
    errors = [i for i in issues if i.severity is Severity.ERROR]
    assert not errors, f"polished JSON failed validation: {errors}"
    assert polished.shape_count == 3
```

- [ ] **Step 10.2: Run the smoke**

Run: `pytest tests/test_tier_b_local_smoke.py -v`

Expected: PASS (or SKIPPED if torch missing — confirm the skip reason is `torch`, not a real error).

- [ ] **Step 10.3: Flip the flags**

Edit `forza_abyss_painter/gui/feature_flags.py`:

```python
RESHAPE_GEN_AVAILABLE: bool = True
POLISH_LOADED_AVAILABLE: bool = True
```

- [ ] **Step 10.4: Update the pinning test to match the flip**

Edit `tests/test_reshape_polish_flags_gating.py`. Change both assertions:

```python
def test_reshape_gen_flag_default_is_true():
    assert feature_flags.RESHAPE_GEN_AVAILABLE is True


def test_polish_loaded_flag_default_is_true():
    assert feature_flags.POLISH_LOADED_AVAILABLE is True
```

Also rename the function names + assertion-text references in this file to reflect the new state (delete the long checklist docstring — it has shipped).

- [ ] **Step 10.5: Re-run all changed tests**

Run: `pytest tests/test_reshape_polish_flags_gating.py tests/test_tier_b_local_smoke.py tests/test_upload_panel_reshape_polish_buttons.py -v`

Expected: PASS — all green. The upload_panel button-visibility tests already monkey-patch the flags, so flipping the defaults doesn't break them.

- [ ] **Step 10.6: Run the FULL test suite**

Run: `pytest -x -q`

Expected: PASS (or pre-existing failures unrelated to this work). If any regressions surface, fix them BEFORE the commit — don't ship broken tests under a flag flip.

- [ ] **Step 10.7: Commit the flip + smoke**

```bash
git add forza_abyss_painter/gui/feature_flags.py tests/test_reshape_polish_flags_gating.py tests/test_tier_b_local_smoke.py
git commit -m "$(cat <<'EOF'
feat(gui): flip RESHAPE_GEN_AVAILABLE + POLISH_LOADED_AVAILABLE to True (#85 #86)

Plumbing is complete and smoke-tested per CLAUDE.md §1b:
- Buttons render in upload_panel
- Source image resolves via same-folder + picker fallback
- #85 opens GenerateLocallyDialog with the source pre-filled
- #86 opens PolishDialog, runs mode='polish_only' through joint_polish,
  saves <input>_polished.json validator-clean
- Local smoke loads real JSON + image, runs polish end-to-end, asserts
  the output validates

Per CLAUDE.md §1c, flag flip lands in the SAME commit that lands the
smoke-green plumbing — never independently.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: SMB sync + tester handoff

Per CLAUDE.md §6e: sync to SMB AFTER local smoke green, BEFORE asking QUASAR/Cursor to validate.

- [ ] **Step 11.1: Verify SMB share is mounted**

Run: `ls /Volumes/ContentCreation/ForzaAbyssPainter_build/source/`

Expected: directory listing. If the share isn't mounted, ask the user to mount it before continuing.

- [ ] **Step 11.2: Sync the source tree**

The project's existing SMB workflow uses robocopy from QUASAR's side, but the macOS-side sync command is `rsync` or whatever the user has scripted. Check existing tooling first:

Run: `ls scripts/ Makefile 2>/dev/null | grep -i smb`

If no script exists, sync with rsync (confirm with user before running — this is a write to a shared share):

```bash
rsync -av --delete \
  --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' \
  --exclude='dist' --exclude='build' --exclude='.pytest_cache' \
  /Users/kusanagi/Development/forza-abyss-painter/ \
  /Volumes/ContentCreation/ForzaAbyssPainter_build/source/
```

- [ ] **Step 11.3: Update CURSOR_NEXT_RUN.md on SMB**

Add a new Run-6 section pointing at the two new buttons and what to validate:

- Load any existing JSON + click Re-shape-gen → confirm GenerateLocallyDialog opens with source pre-filled → run at a higher preset → confirm output JSON renders + injects.
- Load any existing JSON + click Polish loaded JSON → confirm PolishDialog appears → run with 150 steps → confirm `_polished.json` lands + renders + injects.
- Edge case: rename the source image after generating, reload the JSON, click Polish → confirm the file-picker fallback fires.
- Edge case: load a rectangle-containing JSON (if one exists) → click Polish → confirm clean error rather than crash.

- [ ] **Step 11.4: Notify the user**

Output a one-line message:

> "Tier B plumbing complete and smoke-green. SMB sync done; QUASAR can rebuild + validate Run 6 (re-shape-gen + polish). PR is not yet opened per CLAUDE.md §6d — awaiting tester confirmation."

---

## Self-Review Summary

After completing all tasks, verify against the spec:

| Spec section | Implementing task(s) |
|---|---|
| §3.1 Re-shape-gen UX flow | Task 2 (initial_source_path), Task 4 (button), Task 9 (slot) |
| §3.2 Polish flow | Task 4 (button), Task 5 (dialog), Task 7 (runner), Task 8 (config), Task 9 (slot) |
| §4.1 Source-image resolution | Task 3 (helper), Task 9 (picker fallback in `_resolve_source_for_loaded_json`) |
| §4.2 Output naming | Task 5 (PolishDialog default), Task 7 (FD6Document construction) |
| §4.3 Validator integration | Task 7 (`save_json` reuse) |
| §4.4 Feature flags | Task 1 (add), Task 10 (flip) |
| §5.1 RunConfig extensions | Task 6 |
| §5.2 run() dispatch | Task 7 |
| §5.3 IPC schema additions | Task 8 (`build_polish_config`) |
| §6 GUI changes | Tasks 2, 4, 5, 9 |
| §7 Testing strategy | Tasks 1, 2, 3, 4, 5, 6, 7, 8, 10 |
| §9 Acceptance criteria | Task 10 (smoke + flag flip), Task 11 (SMB + tester) |

If any spec section above lacks a task, add a task before invoking execution.
