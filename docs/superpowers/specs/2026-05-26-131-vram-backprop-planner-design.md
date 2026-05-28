# #131 — VRAM Back-Prop Resolution Planner Design

**Date:** 2026-05-26
**Status:** Approved for implementation planning
**Scope:** EXE (consumer GPU path). Colab notebooks may reuse the helper later.
**Replaces:** static `DEFAULT_UPLOAD_CAP_PX = 720` hard cap in `shapegen/worker.py`.

---

## 1. Purpose

The static 720-pixel cap in `shapegen/worker.py:19` throws away most of
the available VRAM on big-VRAM cards. Planning doc §4 evidence: a
Nikke 2048×2048 source gets downscaled 2.8× to 720×720 on a 32 GiB GPU,
losing ~80% of usable resolution headroom. On enterprise cards
(L4/A100/RTX PRO 6000 Blackwell, 40–100 GiB), the loss is closer to
99%.

The planner math in `vram_planner.estimate_peak_vram_gib` is already
calibrated against PyTorch's actual peak allocations (15× safety
multiplier validated by Cursor's QUASAR Run 4). Invert it: given the
free VRAM budget, solve for the largest `max_resolution` that fits.

## 2. Non-goals

- **Not tuning K or `bbox_crop_max`** in this spike. They stay knobs the
  user controls. K-tuning lives in a future session if needed.
- **Not removing the 720 cap.** It becomes a SAFETY FLOOR — back-prop
  only raises the recommended value, never lowers it below 720. Protects
  the consumer-card / unprobed-GPU paths from regression.
- **Not changing the Colab path.** Colab notebooks have their own VRAM
  ceiling logic; this is EXE-only.
- **Not auto-applying without user visibility.** The recommendation is
  surfaced in the Generate dialog with the GiB-free context.

## 3. User-facing behavior

There are two distinct surfaces the back-prop affects — the GPU
preset's working canvas (`max_resolution`) and the CPU upload-time
downscale cap (`DEFAULT_UPLOAD_CAP_PX`). They share the same math but
fire at different points in the pipeline.

### 3.1 GPU path — `GenerateLocallyDialog`

When the user opens the **Generate locally (GPU)…** dialog or changes
the preset:

1. `probe_free_vram(force=True)` runs (existing #125 plumbing).
2. For the currently-selected preset, `recommend_max_resolution(...)` is
   called with the preset's `K`, the probe's free GiB, and the existing
   `bbox_crop_max`.
3. The dialog's preset-description block (currently at
   `generate_dialog.py:206-214`) gets a new line:

   > *"Recommended max_resolution: **1200 px** (auto-tuned to fit
   > 27.4 GiB free on RTX 5090). Floor: 720."*

4. The recommendation is computed as
   `max(preset.max_resolution, recommend_max_resolution(...))` — back-prop
   never LOWERS the preset's baked value. (The Item E hard-block catches
   "preset doesn't fit at all" separately.) The effective value is
   written into the preset dict the dialog hands to `build_run_config`.
5. If `probe.available is False` (non-NVIDIA, WSL no passthrough,
   nvidia-smi missing), the label reads: *"Recommended: 720 px (VRAM
   probe unavailable; using safety floor)."* and no auto-bump happens;
   the preset's baked `max_resolution` is used unchanged.

### 3.2 CPU path — `shapegen/worker.py`

When CPU shape-gen runs (`fap-paint` / `generate_cpu`):

- `_apply_upload_cap()` accepts an OPTIONAL kwarg `upload_cap_override`.
- When set, the function uses `max(DEFAULT_UPLOAD_CAP_PX, override)`
  instead of the static `DEFAULT_UPLOAD_CAP_PX`.
- When omitted, behavior is unchanged (back-compat for any existing
  scripts that call into `worker._apply_upload_cap` directly).
- The CPU caller (typically `gui/main_window.py::_start_cpu`) is
  responsible for computing the recommendation and passing it through;
  the `_apply_upload_cap` helper stays pure.

### 3.3 No user-editable override field added

The Generate dialog currently exposes a preset combo + output path, NOT
a max_resolution input. We do not add one in this spec — back-prop +
preset selection together determine the effective canvas. Users who
want a custom value still have `fap-generate --max-resolution N` on the
CLI.

## 4. The math

`vram_planner._peak_bytes_per_candidate(bbox_local, max_resolution, bbox_crop_max)`
is the canonical footprint formula. For bbox-local mode:

```
crop_e = min(bbox_crop_max, max(1, max_resolution // 8))
footprint = min((2 * crop_e + 1)² , max_resolution²)
bytes_per_cand = footprint * 12 * BBOX_LOCAL_SAFETY   # 15.0
peak_bytes = K * bytes_per_cand
```

Back-prop solves: *given* `target_bytes = free_gib * 0.85 * 1e9` *and*
fixed K + `bbox_crop_max`, find the largest `max_resolution` such that
`peak_bytes ≤ target_bytes`.

Two regimes depending on whether `crop_e` is capped by `bbox_crop_max`
or by `max_resolution // 8`:

**Regime A — `max_resolution // 8 < bbox_crop_max`** (small canvas):

```
crop_e = max_resolution // 8
footprint = (2 * crop_e + 1)²
```

As `max_resolution` grows linearly, `crop_e` grows linearly, footprint
grows quadratically. Solving:

```
target_bytes / (K * 12 * 15) ≥ (2 * (max_res // 8) + 1)²
sqrt(target_bytes / (K * 180)) ≥ 2 * (max_res // 8) + 1
max_res ≤ 8 * ((sqrt(target_bytes / (K * 180)) - 1) / 2)
```

**Regime B — `max_resolution // 8 ≥ bbox_crop_max`** (canvas big enough
that `crop_e` is pinned at the cap):

```
crop_e = bbox_crop_max         # constant
footprint = (2 * bbox_crop_max + 1)²
```

`max_resolution` no longer affects `bytes_per_cand` in this regime —
once we're past the `bbox_crop_max * 8` threshold, the K-batch peak is
constant. So if **Regime B fits at all**, `max_resolution` can be
arbitrarily large (clamped only by source image resolution and
practical canvas-tensor allocation costs not modeled here).

**Combined:**

1. Check Regime B: if `K * (2*bbox_crop_max+1)² * 180 ≤ target_bytes`,
   the K-batch budget is fine at any canvas. Return a generous value
   (e.g., `min(source_long_side, 4096)`).
2. Else solve Regime A: `max_res ≤ floor(8 * (sqrt(target_bytes / (K*180)) - 1) / 2)`.
3. Floor with the safety constant (720). Ceiling with the source image's
   long side (we never recommend upscaling).

## 5. API

### 5.1 New function — `vram_planner.recommend_max_resolution`

```python
def recommend_max_resolution(
    free_gib: float,
    K: int,
    *,
    bbox_local: bool = True,
    bbox_crop_max: int = 256,
    safety_floor_px: int = 720,
    safety_margin: float = 0.85,
    absolute_ceiling_px: int = 4096,
) -> int:
    """Return the largest max_resolution that fits in `safety_margin *
    free_gib` of VRAM for the given K, in the bbox_local (or
    full_canvas) scoring regime.

    Always returns at least `safety_floor_px`. Caller separately clamps
    by source image long side (we never recommend upscaling). When
    `free_gib <= 0` (probe unavailable), returns `safety_floor_px`."""
```

`full_canvas=False` path uses `FULL_CANVAS_SAFETY` (8.0) and the
quadratic-in-canvas formula — same shape, different multiplier. Left as
a parameter for future use; the EXE always passes `bbox_local=True`.

### 5.2 Touched callers

- **`gui/generate_dialog.py`** — calls `recommend_max_resolution`
  whenever the preset changes (`_on_preset_changed`) or the dialog
  opens. Updates the description label + the `output_field`
  placeholder. Writes the recommended value into the preset dict the
  dialog hands to `build_run_config`.

- **`shapegen/worker.py::_apply_upload_cap`** — accepts an
  `upload_cap_override: int | None = None` kwarg. When set, it
  replaces `DEFAULT_UPLOAD_CAP_PX` for that call. Default behavior
  unchanged.

- **`gui/main_window.py::_start_gpu`** — when computing the preset
  dict to hand to `build_run_config`, calls `recommend_max_resolution`
  and passes the result as the preset's `max_resolution`. Status bar
  shows: *"Auto-tuned max_res to 1200 px (RTX 5090, 27.4 GiB free)."*

### 5.3 Tests

| File | Covers |
|---|---|
| `tests/test_recommend_max_resolution.py` | Regime A solving; Regime B short-circuit; floor + ceiling clamps; probe-unavailable fallback. Real `vram_planner` math; no mocks. |
| `tests/test_generate_dialog_recommendation_label.py` | Offscreen Qt: dialog opens, monkey-patched `probe_free_vram` returns canned values, label text contains "Recommended:" + the right number. |
| `tests/test_apply_upload_cap_override.py` | `worker._apply_upload_cap(upload_cap_override=N)` uses N instead of `DEFAULT_UPLOAD_CAP_PX`. Default kwarg preserves old behavior. |
| `tests/test_main_window_autotune_status.py` | Offscreen Qt: `_start_gpu` slot logs the auto-tune line in the status bar with the right number. |

## 6. Behavior matrix

| GPU / scenario | Free GiB | K | Old cap | New recommendation |
|---|---|---|---|---|
| RTX 5090 idle | 27 | 8192 | 720 | ~720 (Regime A, crop_e at 90 already maxes at 22 GiB peak; back-prop math returns ~480, floor wins) |
| RTX 5090 idle | 27 | 4096 | 720 | ~1024 (more budget per candidate; crop_e=128 fits) |
| RTX PRO 6000 (98 GiB) | 95 | 8192 | 720 | ~1400 (Regime A solves comfortably) |
| L4 / A100 (~24/40 GiB) | 38 | 8192 | 720 | ~700 (still close to floor on K=8192) |
| nvidia-smi unavailable | None | * | 720 | 720 (floor) |
| Source image 480×480 | 95 | 4096 | 720 → capped to 480 by clamp | 480 (source clamp) |
| FH6 open | 22 | 8192 | 720 | 720 (floor — Run 4 scenario; preflight HARD BLOCK from Item E still fires if peak > free) |

The hard block from Item E is a different gate — it's the "this preset
won't fit at all" check. Back-prop runs BEFORE the hard block: pick the
best canvas, then verify the resulting peak fits. If the math returns
720 (floor) and even 720 doesn't fit on the user's free VRAM, the
hard-block modal still fires. No regression.

## 7. Risk and mitigations

| Risk | Mitigation |
|---|---|
| Math drift if `BBOX_LOCAL_SAFETY` re-calibrates | The new function uses `_peak_bytes_per_candidate` directly — no constant duplication. Recalibration auto-flows through. |
| Big recommendation on small source | `min(recommendation, source_long_side)` clamp at call site. We never upscale. |
| User confusion about "Recommended" vs preset's `max_resolution` | Dialog text explains the override + shows the free-VRAM context. Settings panel reflects the value. |
| Math regression masks a real OOM | Item E's hard block stays in place; if back-prop's recommendation still exceeds free VRAM (e.g. shape allocator overhead grows), the modal fires. |
| Recommendation oscillates as free VRAM fluctuates | Probe is cached for `_CACHE_TTL_SECONDS` (5s). Recommendation is computed at dialog-open + on preset-change, not on every keystroke. |

## 8. Out of scope (explicit)

- K-tuning (raise K when canvas is the bottleneck).
- `bbox_crop_max` auto-tune (parking-lot item from planning doc §8).
- Per-card profile presets (encode card → known good settings).
- Colab notebook port.
- Multi-shape (#65) interaction — gated on #129 first.

## 9. Acceptance criteria

1. New `recommend_max_resolution` ships with unit tests pinning Regime
   A, Regime B, floor, and probe-unavailable paths.
2. `GenerateLocallyDialog` shows the recommendation label and auto-bumps
   the preset's `max_resolution` when the user has not overridden it.
3. `worker._apply_upload_cap` accepts `upload_cap_override`; existing
   call sites pass `None` and behave identically.
4. `_start_gpu` writes the auto-tune line to the status bar with the
   correct GiB-free context.
5. Existing 720-related tests still pass (no constant change to
   `DEFAULT_UPLOAD_CAP_PX`).
6. Local smoke under offscreen Qt confirms the dialog label renders
   correctly under both `probe.available=True` and `False`.

## 10. File-touch summary

**Modified:**
- `forza_abyss_painter/shapegen/gpu/vram_planner.py` — add `recommend_max_resolution`.
- `forza_abyss_painter/shapegen/worker.py` — add `upload_cap_override` kwarg to `_apply_upload_cap`.
- `forza_abyss_painter/gui/generate_dialog.py` — probe + compute + label + auto-bump.
- `forza_abyss_painter/gui/main_window.py::_start_gpu` — apply recommendation to preset dict.

**New:**
- `tests/test_recommend_max_resolution.py`
- `tests/test_generate_dialog_recommendation_label.py`
- `tests/test_apply_upload_cap_override.py`
- `tests/test_main_window_autotune_status.py`

**Untouched:**
- `DEFAULT_UPLOAD_CAP_PX = 720` stays as the floor constant; only its
  usage gains an optional override.
- `vram_planner._peak_bytes_per_candidate`, `BBOX_LOCAL_SAFETY` — no
  recalibration in this spec.
