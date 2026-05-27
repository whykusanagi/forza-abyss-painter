# GPU snapshots, live preview, resume, and auto-polish — design

**Date:** 2026-05-27
**Status:** Approved for implementation planning
**Scope:** EXE (consumer + enterprise GPU paths). CPU side is mostly already
correct; small additive changes only.
**Closes:** Task #40 (live preview, deferred). New feature: resume from
failed run. Calibration change: auto-polish on fresh GPU runs.

---

## 1. Purpose

Three intertwined gaps in the current GPU pipeline:

1. **No live preview during GPU runs.** The middle PreviewPanel shows the
   source image (after the recent fix) + progress bar, but never renders
   the partial output. CPU side does this every 50 shapes via an
   in-process numpy emit; GPU runs across a subprocess boundary and the
   shapes list is discarded at the IPC layer.
2. **All-or-nothing failure mode.** If a 3000-shape GPU run crashes at
   shape 2900 (OOM, kill-ladder fired, FH6 stole VRAM mid-run, etc.), the
   user loses everything. The runner doesn't persist intermediate state.
3. **No resume mechanism in the GUI.** `io/resume.py::load_resume()` and
   `Engine.seed_shapes()` exist as a programmatic API but nothing in the
   GUI wires them up. Even the CPU path can't recover from a crash
   without dropping into Python.

Plus one calibration alignment: fresh GPU runs currently skip
`joint_polish` entirely (`LOCAL_PRESETS` doesn't set
`joint_polish_steps`; `build_run_config` doesn't forward it). CPU
presets DO polish. This spec brings GPU presets to parity.

## 2. Non-goals

- **No resume for `polish_only` mode.** Polish is atomic + fast (~15-60s);
  a crash mid-polish costs the user no meaningful time. Not worth the
  complexity of mid-polish snapshots.
- **No mid-run refill.** `clean_and_refill` runs once at end-of-greedy
  (current behavior). Adding it at every snapshot would re-rasterize the
  full shape set + add 5-10s per checkpoint. Not justified by current
  failure data.
- **No resume across preset changes.** Snapshots embed `_run_config`;
  resume uses those exact params. Bumping K or `max_resolution` on resume
  produces inconsistent output (the seeded shapes were generated against
  different scoring parameters). If user wants different params, they
  start fresh.
- **No snapshot pruning.** Snapshots survive successful runs (matches CPU
  behavior). At ~80KB × 30 checkpoints = ~2.4MB per 3000-shape run,
  acceptable.
- **No embedded preview PNG in the snapshot JSON.** Render-from-JSON in
  GUI is fast enough (~150-250ms per render) and keeps snapshots small.
- **No multi-shape resume.** Multi-shape (#65) gated on #129 anyway.

## 3. User-facing behavior

### 3.1 Live preview during GPU run

When a GPU run is generating shapes, the middle PreviewPanel renders the
partial canvas every 100 shapes (every checkpoint). The user sees:

- The source image on the left (already shipped in the recent fix).
- A live preview on the right, refreshed each checkpoint, showing the
  current N committed shapes rendered onto the canvas.
- Progress bar + "Shape N/T" status text (already shipped).

Cadence is the user-configurable `checkpoint_every` (default 100, min
100 for `device="cuda"`, min 1 for `device="cpu"`). Each render takes
~150-250ms in a background QThreadPool worker; if a render is still
in-flight when the next snapshot fires, the older render is dropped.

### 3.2 Periodic snapshots to disk

At each checkpoint during a fresh run (CPU or GPU), the partial output
is written to `<output_stem>_<shape_count>.json` next to the final
output:

```
Downloads/ziz_dnace/
  ziz_dnace_100.json
  ziz_dnace_200.json
  ...
  ziz_dnace_2900.json     ← if run died here, this is the resume point
  ziz_dnace.json          ← only exists if run completed successfully
```

CPU side already does this (`worker.py:246-261`). GPU side gains the
same behavior via the new IPC `snapshot` event (see §5.1).

**Snapshots survive successful runs** (matches CPU). User gets a
milestone history for A/B comparison and a safety net if the final
output write somehow gets corrupted.

### 3.3 Resume from a partial run

A new **Resume from snapshot…** button appears on `upload_panel`
alongside Upload Image / Upload JSON. Always visible (no feature
flag — recovering failed runs is a day-one concern).

Click flow:

1. `QFileDialog.getOpenFileName` with filter
   `Forza Abyss Painter snapshots (*_*.json)`.
2. `load_json(path)` parses the snapshot.
3. Validate: `shape_count > 0`. If `_run_config.target_shape_count`
   exists AND `> shape_count` → one-click resume. Otherwise → fall back
   to "pick a target preset" dialog.
4. Resolve source image via the same-folder heuristic + picker
   fallback (`_resolve_source_image_path` from Tier B).
5. `ResumeDialog` shows:
   > "Continue **2900 → 3000** shapes from `ziz_dnace_2900.json` using
   > original settings (K=1000, max_res=1200, preset `_default`).
   > Source: `ziz_dnace.gif`."

   Buttons: **Resume** / **Cancel**.
6. On Resume: spawn `GpuGenWorker` with a `seed_shapes_path` config
   field. Runner replays the snapshot shapes onto the canvas, then
   continues the greedy loop to target. Polish + refill run at end as
   usual.

### 3.4 Auto-polish on fresh GPU runs

`LOCAL_PRESETS` (`generate_dialog.py:42-67`) gains `joint_polish_steps`
matching the CPU calibration:

| GPU preset | num_shapes | joint_polish_steps | Rationale |
|---|---|---|---|
| Lineart — 400 shapes | 400 | **100** | Lineart's flat colors → less to polish; small N makes polish fast |
| Headshot — 700 shapes | 700 | **150** | Faces benefit from color refinement; matches CPU medium_1000 polish value |
| Medium — 1000 shapes | 1000 | **150** | Match CPU `medium_1000` baseline exactly |
| Hi-Res — 3000 shapes | 3000 | **250** | Match CPU `highres_3000` baseline exactly |

`build_run_config` (gpu_gen_worker.py) forwards `joint_polish_steps`
from the preset into the IPC config. `RunConfig.from_dict` already
accepts it (line 250).

**This is a CLAUDE.md §8a quality-knob baseline change**, called out
explicitly: GPU output quality will visibly differ from pre-spec runs
(better color match, possibly slightly different geometry edges where
the optimizer nudged them). The user approved this change during
brainstorming on 2026-05-27.

Per-run wall-time impact: polish adds ~30-60s at 1000 shapes /
~60-120s at 3000 shapes on consumer hardware (the joint_polish optimizer
runs once over all committed shapes). User can disable by setting
`joint_polish_steps=0` in a custom preset.

## 4. Snapshot file format

Snapshots are standard `fd6.shapes` v1 documents PLUS one optional
top-level field:

```json
{
  "format": "fd6.shapes",
  "version": 1,
  "source_image": "ziz_dnace.gif",
  "image_size": [1200, 981],
  "shape_count": 2900,
  "generated_at": "2026-05-27T15:04:12Z",
  "profile": "medium_1000",
  "sticker_mode": false,
  "shapes": [ /* 2900 ellipses */ ],
  "_run_config": {
    "target_shape_count": 3000,
    "random_samples": 1000,
    "max_resolution": 1200,
    "edge_strength": 0.0,
    "posterize_levels": 0,
    "sticker_mode": false,
    "lock_alpha": true,
    "bbox_local": true,
    "joint_polish_steps": 150,
    "vram_budget_gib": 0.0,
    "preset_label": "Medium — 1000 shapes"
  }
}
```

`_run_config` is **optional**. Snapshots without it load fine as
plain shape documents (legacy interop). Resume falls back to a
preset-picker when missing.

**Validator behavior** (`io/validator.py`): `_run_config` is added to
the known-keys allowlist. No per-field rules — the dict is metadata,
not contract. Future field additions don't break old snapshots.

**Schema doc update** (`docs/JSON_SPEC.md`): add an "Optional metadata
fields" section. `_run_config` is reserved for runner state. Other
non-standard top-level keys starting with `_` are also tolerated by
the validator (warn but don't error).

**Final output** (when the run completes successfully) ALSO embeds
`_run_config`. So a user inspecting a finished `.json` can see what
preset produced it. This is a small but useful provenance win.

## 5. Architecture

### 5.1 GPU snapshot IPC

`torch_runner.py` runs `engine.run_gpu(...)` which already accepts a
`checkpoint_cb(shape_idx, shapes_so_far)` callback. Currently the
callback emits only `{"kind": "checkpoint", "shape_count": N, "total": T}`
and discards `shapes_so_far`.

**Change:**

```python
def _checkpoint_cb(shape_idx: int, shapes_so_far: list) -> None:
    snapshot_path = _snapshot_path_for(cfg.output_json_path, shape_idx)
    try:
        _write_snapshot(cfg, shapes_so_far, shape_idx, snapshot_path)
    except OSError as exc:
        logger.log_exception("snapshot_write_failed", exc)
        # Don't crash the run — emit the bare checkpoint event so the
        # progress bar still moves. The snapshot is a safety net, not
        # a hard requirement.
        emit(stream, {"kind": "checkpoint",
                       "shape_count": shape_idx,
                       "total": cfg.num_shapes})
        return
    # Both events so existing GUI listeners (progress bar) keep working
    # AND the new snapshot path is available for preview/resume.
    emit(stream, {"kind": "checkpoint",
                   "shape_count": shape_idx,
                   "total": cfg.num_shapes})
    emit(stream, {"kind": "snapshot",
                   "shape_count": shape_idx,
                   "total": cfg.num_shapes,
                   "path": str(snapshot_path)})
```

**`_write_snapshot`** uses `save_json` (so #100 validator hook fires)
and includes `_run_config` derived from `cfg`. Located in
`torch_runner.py` alongside `_run_polish_only` since it's runner state.

**`_snapshot_path_for(output_json_path, count) -> Path`** is shared
helper in `io/exporter.py` (or new `io/snapshots.py`) so CPU + GPU paths
use the same naming. `Path(output).parent / f"{output.stem}_{count}.json"`.

### 5.2 GUI snapshot subscription

`GpuGenWorker` gains one new signal:

```python
snapshot = Signal(int, int, str)  # shape_count, total, path
```

Emitted from `_dispatch()` when `event["kind"] == "snapshot"`. The
existing `checkpoint` signal stays (carries count + total only, used
by the progress bar).

**MainWindow wires:**

```python
self._worker.snapshot.connect(self._on_gpu_snapshot)
```

**Slot:**

```python
def _on_gpu_snapshot(self, count: int, total: int, snapshot_path: str) -> None:
    self.statusBar().showMessage(
        f"GPU: snapshot saved at {count}/{total} "
        f"→ {Path(snapshot_path).name}", 2000,
    )
    self._render_snapshot_async(snapshot_path)
```

### 5.3 GUI render worker

New file `forza_abyss_painter/gui/snapshot_render.py`:

```python
class _RenderSnapshotJob(QRunnable):
    """Off-thread render: snapshot JSON → numpy canvas → PreviewPanel."""

    def __init__(self, snapshot_path: Path, preview: PreviewPanel) -> None:
        super().__init__()
        self._path = snapshot_path
        self._preview = preview

    def run(self) -> None:
        from forza_abyss_painter.io.exporter import load_json
        from forza_abyss_painter.shapegen.render import render_shapes
        try:
            doc = load_json(str(self._path))
            shapes = doc.materialize_shapes()
            w, h = doc.image_size
            canvas = render_shapes(
                shapes, w, h,
                background=(255, 255, 255),
                transparent_bg=bool(doc.sticker_mode),
            )
        except Exception:
            # Snapshot may be mid-write or transiently invalid.
            # Best-effort: skip this render, next one will fire soon.
            return
        # Marshal back to GUI thread.
        QMetaObject.invokeMethod(
            self._preview, "on_preview",
            Qt.QueuedConnection,
            Q_ARG(object, canvas),   # numpy array
        )
```

**Throttling**: MainWindow holds a single-slot `_pending_snapshot_path:
str | None`. When a new snapshot event arrives:
- If a render is already in-flight, just overwrite `_pending_snapshot_path`.
- When the in-flight render finishes (via a `renderFinished` signal),
  if `_pending_snapshot_path is not None`, dispatch a new render and
  clear the slot.

Avoids backlog under fast-fire snapshots (rare but possible at low
checkpoint cadence on fast GPUs).

### 5.4 Resume flow — GUI

**New button on `upload_panel.py`**. After the recent layout fix that
moved the Tier B row above the Recent stack, the order is:

```
[Upload Image…]
[ Drop zone ]
[Upload JSON…] [Download JSON]                ← existing JSON row
[Re-shape-gen…] [Polish loaded JSON…]         ← Tier B row (visible after Upload JSON)
[Resume from snapshot…]                       ← NEW row
[Recent: ...stack...]
```

Resume row goes BELOW Tier B (resume is rarer than the Tier B actions,
and it's a standalone action that doesn't depend on a JSON being
loaded — so it stays visible regardless of `_loaded_json_path`).
Always visible. No feature flag.

`UploadPanel` gains:
- New signal: `resume_requested = Signal(Path)` emitted with the
  selected snapshot path.
- Click handler opens a `QFileDialog.getOpenFileName` filtered to
  `*_*.json` files in any directory the user picks. (No auto-scan of
  output dirs — explicit user pick keeps this scoped.)
- On signal: MainWindow's `_on_resume_requested(snapshot_path)`
  loads, validates, resolves source, and opens `ResumeDialog`.

**`ResumeDialog`** (new `gui/resume_dialog.py`):

- Modal. Constructed with `snapshot_path` and the resolved source
  image path.
- Reads `doc = load_json(snapshot_path)`.
- Reads `_run_config` if present; otherwise displays a preset picker
  to fill in target_shape_count + random_samples + max_resolution.
- Body shows the summary text (snapshot filename, current → target
  shape count, K, max_res, source filename, preset label).
- Buttons: **Resume** (default), **Cancel**.
- `.values()` returns the full `RunConfig` dict ready for
  `GpuGenWorker`.

On Resume: dialog `accept()`s. MainWindow spawns `GpuGenWorker` with
the values dict. Same code path as fresh generation, but with
`seed_shapes_path` set.

### 5.5 Resume flow — runner

**`RunConfig` gains one field:**

```python
seed_shapes_path: Path | None = None
```

`from_dict` validation:
- If set: must be an existing file. Same `is_file()` check as
  `input_shapes_path`.
- Allowed only when `mode == "fresh"`. `mode == "polish_only"`
  with `seed_shapes_path` → `ValueError("seed_shapes_path is not
  supported in polish_only mode")`.

**`torch_runner.run()` fresh branch**: when `cfg.seed_shapes_path is
not None`, load the snapshot doc, extract `shapes`, pass them into
`run_gpu(...)` as a new optional `seed_shapes` arg.

**`shapegen/gpu/engine.run_gpu(...)` gains:**

```python
def run_gpu(
    target_rgb, cfg, alpha_mask=None,
    progress_every=0, checkpoint_cb=None, checkpoint_every=0,
    seed_shapes: list[dict] | None = None,    # NEW
):
    ...
    shapes: list[dict] = []
    if seed_shapes:
        # Replay each seeded shape onto the canvas + commit to shapes list.
        for s in seed_shapes:
            params = torch.tensor([[s["x"], s["y"], s["rx"], s["ry"], s["angle"]]],
                                    dtype=DTYPE, device=device)
            color = torch.tensor(s["color"][:3], dtype=DTYPE, device=device)
            alpha = int(s["color"][3])
            canvas = _composite_one(
                kinds[0], canvas, params.squeeze(0), color,
                h, w, alpha_mask_f, alpha,
            )
            shapes.append(s)
        shape_idx = len(shapes)
        logger.log("seeded", count=shape_idx)
    ...
    # Existing while loop uses shape_idx as the starting point.
```

**Critical:** seed replay assumes rotated_ellipse only (matches the
production `bbox_local=True` shape_types). The runner validates the
seed shapes upstream: any non-`rotated_ellipse` type → emit
`{"kind": "error", "stage": "resume_unsupported_shape", ...}` and
return 1 before invoking `run_gpu`. Avoids the engine crashing
inside `_composite_one` with an unfamiliar kind.

**End-of-loop polish + refill** behave normally on the FULL shape list
(seeded + newly generated). Same final state as a fresh successful run
of the same length.

### 5.6 Auto-polish wiring

`generate_dialog.py:LOCAL_PRESETS`: add `joint_polish_steps` field per
the table in §3.4.

`gpu_gen_worker.build_run_config`: forward `preset["joint_polish_steps"]`
(default 0 when missing, for back-compat with code paths that build
custom presets) into the config dict.

`RunConfig.from_dict`: already accepts `joint_polish_steps`. No change.

`engine.run_gpu`: already runs polish when `cfg.joint_polish_steps > 0`.
No change.

## 6. Min cadence enforcement

The "Generate locally (GPU)" dialog is GPU-only (CPU shape-gen runs
through a different path with its own profile-baked cadence). So the
GUI knob is GPU-only.

`generate_dialog.py` adds a single `QSpinBox`:

| Field | Min | Max | Step | Default |
|---|---|---|---|---|
| "Snapshot every N shapes" | 100 | 1000 | 50 | 100 |

The spinbox writes to `checkpoint_every` in the IPC config (replaces
the current `checkpoint_every: max(1, num_shapes // 20)` heuristic in
`build_run_config`, which can produce values below 100 for small
runs — e.g. num_shapes=400 → 20).

**Defense in depth — runner enforcement:** `RunConfig.from_dict`
raises `ValueError` if `device == "cuda" AND 0 < checkpoint_every < 100`.
The `0` (disabled) case is preserved for callers that don't want
snapshots (e.g. unit tests).

**CPU side is unaffected.** CPU shape-gen uses
`presets.py:Profile.save_every` (default 100) + `save_at` (milestone
list). No user-facing GUI knob exists or is added for CPU cadence —
the CPU `worker.py:_apply_upload_cap` path remains unchanged.

## 7. Testing strategy

### 7.1 Pure / unit

| File | Covers |
|---|---|
| `tests/test_snapshot_path_for.py` | `_snapshot_path_for(output, count)` naming + edge cases (no extension, dots in filename) |
| `tests/test_run_config_seed_shapes_path.py` | `seed_shapes_path` parses; missing file raises; polish_only rejects |
| `tests/test_run_config_cuda_min_cadence.py` | `cuda + checkpoint_every=50` raises; `=100` OK; `=0` OK; `cpu + =10` OK |
| `tests/test_run_config_emits_run_config_block.py` | Snapshots written via the runner contain `_run_config` with all expected keys |
| `tests/test_validator_tolerates_underscore_keys.py` | Validator passes a snapshot with `_run_config` (and rejects malformed ones) |

### 7.2 GUI / offscreen Qt

| File | Covers |
|---|---|
| `tests/test_upload_panel_resume_button.py` | Resume button visible always; emits `resume_requested` with the picked path |
| `tests/test_resume_dialog_values.py` | Reads `_run_config`; computes "X → Y"; returns the full RunConfig dict; falls back to preset-picker when `_run_config` missing |
| `tests/test_snapshot_render_job.py` | `_RenderSnapshotJob` reads a fixture snapshot, renders, calls `on_preview` with a numpy array of correct shape |
| `tests/test_main_window_snapshot_throttle.py` | Two fast-fire snapshot events → only the most recent renders |

### 7.3 Engine / runner integration (gated on torch)

| File | Covers |
|---|---|
| `tests/test_engine_seed_shapes.py` | `run_gpu(seed_shapes=[3 ellipses], num_shapes=6)` → output has 6 shapes, first 3 bit-identical to seed |
| `tests/test_polish_rejects_seed_shapes.py` | `mode=polish_only + seed_shapes_path` → ValueError before subprocess launch |
| `tests/test_snapshot_runner_integration.py` | Subprocess run with `checkpoint_every=100, num_shapes=300` → 3 snapshots land at the expected paths, each with `shape_count` matching the filename |
| `tests/test_resume_runner_integration.py` | Subprocess fresh run with `seed_shapes_path=<3-shape snapshot>, num_shapes=6` → output has 6 shapes, first 3 from seed |

### 7.4 Local smoke (CLAUDE.md §1b)

Construct real `MainWindow()` under offscreen Qt:

1. Load a 3-ellipse snapshot via `_on_resume_requested(snapshot_path)`.
2. Mock `prompt_install_or_use_existing` to True.
3. Auto-accept the ResumeDialog.
4. Spawn the runner directly via `_run_polish_only` equivalent — but
   for fresh+seed (no public helper yet; mirror the Tier B smoke
   pattern: call `engine.run_gpu(seed_shapes=..., num_shapes=6)` with a
   stub logger).
5. Assert output snapshot lands + has 6 shapes + validates clean.

Single `MainWindow` per process (§8h).

## 8. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Snapshot write fails mid-run (disk full, permissions) | Caught in `_checkpoint_cb` try/except; emit bare `checkpoint` event (progress bar still moves) + log exception. Existing snapshots remain on disk; resume from the previous good one. |
| Render-job thread leak | `QThreadPool.globalInstance()` reuses threads with bounded concurrency. Single-slot queue prevents pileup. |
| Snapshot read race (preview fires mid-write) | Render-job swallows parse errors silently; next snapshot fires within seconds at GPU cadence. Acceptable best-effort behavior. |
| `_run_config` field collision with future fd6.shapes spec | Underscore-prefixed keys are reserved for non-spec metadata per the schema-doc update. v1 schema doesn't define `_*` keys; future spec versions can codify the convention. |
| Resume with stale source image (user moved the image) | Same as Tier B: `_resolve_source_image_path` + picker fallback. |
| Resume with a corrupted snapshot | `load_json` raises; ResumeDialog shows the error + lets user pick another snapshot. |
| Auto-polish slows runs more than expected | User can pick the smaller preset; or future "Disable polish" checkbox in the dialog (out of scope for this spec). |
| Seeded shapes produce different scoring than fresh-from-scratch | Acceptable: replayed shapes are bit-identical to what was committed in the original run. The greedy loop's RANDOM search is the only nondeterministic part, and it depends only on the current canvas state which the replay reproduces. |
| Cancelled run mid-snapshot-write | Snapshot may be incomplete. Validator on next load catches it. Resume falls back to the previous snapshot. |

## 9. Acceptance criteria

1. All tests in §7 are green.
2. Local smoke (§7.4) is green.
3. SMB sync per CLAUDE.md §6e — after local smoke green, before Cursor
   validation.
4. Cursor confirms on QUASAR (Run 8 candidate):
   - Snapshots land on disk at every 100 shapes during a GPU run.
   - Middle PreviewPanel shows a live partial render at each checkpoint.
   - Killing the runner mid-run leaves the most recent snapshot
     intact + valid.
   - Resume from snapshot button on upload_panel + dialog flow
     completes successfully, producing output with `len(snapshot) +
     remaining` shapes.
   - Fresh GPU runs now produce polished output (compare ziz_dnace
     before vs after — color match should be visibly better).

## 10. File-touch summary

**New:**
- `forza_abyss_painter/gui/snapshot_render.py` — `_RenderSnapshotJob`
- `forza_abyss_painter/gui/resume_dialog.py` — `ResumeDialog`
- `forza_abyss_painter/io/snapshots.py` (or extend `io/exporter.py`) — `snapshot_path_for(output_path, count)` helper
- Test files per §7.

**Modified:**
- `forza_abyss_painter/runtime/torch_runner.py` — add `seed_shapes_path` to `RunConfig`; rewrite `_checkpoint_cb` to write snapshot + emit `snapshot` event; min-cadence validation.
- `forza_abyss_painter/shapegen/gpu/engine.py:run_gpu` — accept optional `seed_shapes` arg + replay loop.
- `forza_abyss_painter/gui/gpu_gen_worker.py` — new `snapshot` signal; forward `joint_polish_steps` in `build_run_config`.
- `forza_abyss_painter/gui/generate_dialog.py:LOCAL_PRESETS` — add `joint_polish_steps` per §3.4 table. Add device-aware checkpoint-cadence spinbox.
- `forza_abyss_painter/gui/upload_panel.py` — add Resume from snapshot button + signal.
- `forza_abyss_painter/gui/main_window.py` — `_on_gpu_snapshot` slot; `_on_resume_requested` slot; render-job dispatch + single-slot throttle.
- `forza_abyss_painter/io/validator.py` — allow `_*` top-level keys without warning; document `_run_config` as the reserved metadata field.
- `docs/JSON_SPEC.md` — Optional metadata fields section.

**Deliberately untouched:**
- `forza_abyss_painter/shapegen/worker.py` — CPU snapshot machinery already works. The new shared helper `snapshot_path_for` replaces the inline name construction; behavior unchanged.
- `engine.refill_dead_shapes` default + ordering. End-of-run refill is the current behavior; no change.
