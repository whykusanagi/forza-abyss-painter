"""Source-mean padding regression for the EXE's CPU shape-gen worker.

Ports the v0.1.2 'opaque padding fill = source mean' fix from the colab
GPU pipeline (sister repo fd6/shapegen/gpu/engine.py:407-408) into the
EXE side. Without this, non-square images and the universal 8% edge
buffer were filled with pure white, and the shape-generator spent
hundreds of shapes trying to cover that phantom white border before it
ever got to the real content. After the fix, padding fills with the
source image's own mean RGB so it blends and doesn't burn shape budget.

The helper `_pad_with_source_mean` is what the worker calls; testing it
directly keeps the asserts byte-precise without needing to spin up a
real Engine + GPU/CPU shape-gen pass per assertion.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from forza_abyss_painter.shapegen.worker import _pad_with_source_mean


# ------------------------------------------------------------- non-sticker mode


def test_non_square_padding_fills_with_source_mean_not_white():
    """Solid-color non-square input → square pad pixels must equal the
    source mean, NOT (255, 255, 255). This is the headline regression:
    if this asserts pad == (255, 255, 255) we've reverted v0.1.2."""
    src_rgb = (77, 144, 33)
    img = Image.new("RGB", (100, 50), src_rgb)   # rectangular → triggers square-up
    padded, _alpha = _pad_with_source_mean(
        img, sticker_mode=False, alpha_mask=None
    )
    padded_arr = np.asarray(padded)
    # Top-left pixel is pure pad (the edge buffer alone is ≥8 px wide).
    pad_pixel = tuple(int(c) for c in padded_arr[0, 0])
    assert pad_pixel == src_rgb, (
        f"expected pad pixel == source mean {src_rgb}, got {pad_pixel} — "
        f"the source-mean padding fix has regressed back to hardcoded white"
    )
    assert pad_pixel != (255, 255, 255), (
        "pad pixel is hardcoded white — phantom-border regression"
    )


def test_edge_buffer_padding_applies_to_square_inputs_too():
    """Square input → no square-up step, but the 8% edge buffer still
    fires and must also use source-mean fill. This is the 'every
    generation' branch — if it slips back to white, square inputs get
    the same phantom-border tax that the v0.1.2 fix removed."""
    src_rgb = (12, 220, 90)
    img = Image.new("RGB", (64, 64), src_rgb)   # already square
    padded, _alpha = _pad_with_source_mean(
        img, sticker_mode=False, alpha_mask=None
    )
    padded_arr = np.asarray(padded)
    assert padded_arr.shape[0] > 64 and padded_arr.shape[1] > 64, (
        "edge buffer didn't fire on a square input — buffer_frac path broke"
    )
    pad_pixel = tuple(int(c) for c in padded_arr[0, 0])
    assert pad_pixel == src_rgb


def test_padding_preserves_source_pixel_interior():
    """Pad fill must NOT overwrite the source image's interior pixels —
    the paste(offset) call has to land the original content unchanged
    so the engine sees real content, not pad-bled-into-content."""
    src_rgb = (200, 50, 100)
    img = Image.new("RGB", (40, 40), src_rgb)
    padded, _alpha = _pad_with_source_mean(
        img, sticker_mode=False, alpha_mask=None
    )
    padded_arr = np.asarray(padded)
    # The center of the padded canvas must still be the original source color.
    h, w = padded_arr.shape[:2]
    center = tuple(int(c) for c in padded_arr[h // 2, w // 2])
    assert center == src_rgb, (
        f"source interior was overwritten by padding — expected {src_rgb}, "
        f"got {center} at canvas center ({h // 2}, {w // 2})"
    )


def test_source_mean_computed_before_non_square_pad_not_after():
    """When the non-square pad expands the canvas, the edge-buffer step
    that runs AFTER it must still fill with the original source mean —
    not the post-non-square-pad mean (which would already be diluted
    toward the previous pad fill). This pins the 'compute fill ONCE up
    front' contract documented in the helper's docstring."""
    # Image where the non-square pad will roughly double the canvas:
    # 200×50 → 200×200 square, then +edge buffer.
    src_rgb = (10, 200, 30)
    img = Image.new("RGB", (200, 50), src_rgb)
    padded, _alpha = _pad_with_source_mean(
        img, sticker_mode=False, alpha_mask=None
    )
    padded_arr = np.asarray(padded)
    # Top-left is pure edge-buffer pad — must still match the ORIGINAL
    # source mean, not the mean after square-up (which would skew because
    # the square-up step added 150 rows of fill pixels above + below).
    pad_pixel = tuple(int(c) for c in padded_arr[0, 0])
    assert pad_pixel == src_rgb, (
        f"edge-buffer pad pixel {pad_pixel} != original source mean {src_rgb} — "
        f"src_mean was recomputed AFTER the non-square pad, drifting toward "
        f"the previous fill color"
    )


def test_mixed_color_source_uses_arithmetic_mean():
    """Half-and-half image: red top half, blue bottom half → mean is
    purplish. Pad must be that mean (within rounding), not white."""
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:20, :, 0] = 240   # top half: pure red
    arr[20:, :, 2] = 160   # bottom half: pure blue
    img = Image.fromarray(arr, "RGB")
    expected_mean = tuple(int(c) for c in arr.reshape(-1, 3).mean(axis=0).round().clip(0, 255))
    padded, _alpha = _pad_with_source_mean(
        img, sticker_mode=False, alpha_mask=None
    )
    pad_pixel = tuple(int(c) for c in np.asarray(padded)[0, 0])
    assert pad_pixel == expected_mean, (
        f"expected arithmetic-mean pad {expected_mean}, got {pad_pixel}"
    )


# ----------------------------------------------------------------- sticker mode


def test_sticker_mode_keeps_black_padding():
    """Sticker mode uses alpha_mask as the real content gate, so the
    pad color is upstream-compatible black (0, 0, 0). DO NOT switch
    sticker mode to source-mean — would silently change scoring behavior
    around transparent boundaries vs the upstream baseline."""
    src_rgb = (200, 50, 100)
    img = Image.new("RGB", (40, 40), src_rgb)
    alpha = np.full((40, 40), 255, dtype=np.uint8)
    padded, padded_alpha = _pad_with_source_mean(
        img, sticker_mode=True, alpha_mask=alpha
    )
    pad_pixel = tuple(int(c) for c in np.asarray(padded)[0, 0])
    assert pad_pixel == (0, 0, 0), (
        f"sticker mode pad must stay (0, 0, 0), got {pad_pixel} — "
        f"source-mean leaked into the sticker branch"
    )
    assert padded_alpha is not None
    # Alpha pad pixels around the edge must be zero (transparent gate).
    assert padded_alpha[0, 0] == 0


def test_sticker_mode_alpha_mask_padded_to_new_canvas_size():
    """Sticker mode's alpha mask must grow with the padded canvas so
    the engine still gets a mask the same H×W as the padded target."""
    img = Image.new("RGB", (40, 40), (100, 100, 100))
    alpha = np.full((40, 40), 255, dtype=np.uint8)
    padded, padded_alpha = _pad_with_source_mean(
        img, sticker_mode=True, alpha_mask=alpha
    )
    assert padded_alpha.shape == (padded.size[1], padded.size[0]), (
        f"alpha mask shape {padded_alpha.shape} mismatched padded canvas "
        f"({padded.size[1]}, {padded.size[0]})"
    )


def test_sticker_mode_skips_non_square_pad():
    """Sticker mode preserves the original behavior of NOT squaring up
    non-square inputs (alpha-driven content stays the natural canvas
    shape; only the edge buffer fires). Regression guard against
    accidentally enabling the square-up branch under sticker mode."""
    img = Image.new("RGB", (200, 50), (100, 100, 100))
    alpha = np.full((50, 200), 255, dtype=np.uint8)
    padded, _alpha = _pad_with_source_mean(
        img, sticker_mode=True, alpha_mask=alpha
    )
    # Only the edge buffer should have expanded the canvas — width is
    # still LARGER than height by ~150 px (the original 150 px aspect
    # difference is preserved, just both grown by 2*pad_px).
    assert padded.size[0] > padded.size[1], (
        "sticker mode unexpectedly squared up the canvas — non-square "
        "skip-branch broke"
    )
    assert padded.size[0] - padded.size[1] == 200 - 50, (
        f"sticker mode's edge buffer didn't preserve aspect: padded "
        f"diff={padded.size[0] - padded.size[1]} vs expected 150"
    )


# ------------------------------------------------------------------ basic contracts


def test_padding_grows_canvas_in_both_dimensions():
    """Edge buffer must be at least 8 px per side (max(8, 0.08*max(size))),
    so the padded canvas is always strictly larger than the source."""
    img = Image.new("RGB", (40, 40), (123, 45, 67))
    padded, _alpha = _pad_with_source_mean(
        img, sticker_mode=False, alpha_mask=None
    )
    assert padded.size[0] >= 40 + 2 * 8
    assert padded.size[1] >= 40 + 2 * 8


def test_padding_returns_rgb_mode_image():
    """Engine consumes a uint8 RGB ndarray; padded image must round-trip
    cleanly as RGB. Slipping into RGBA here would propagate down and
    break the (H, W, 3) shape contract upstream of engine init."""
    img = Image.new("RGB", (40, 30), (50, 50, 50))
    padded, _alpha = _pad_with_source_mean(
        img, sticker_mode=False, alpha_mask=None
    )
    assert padded.mode == "RGB"
    arr = np.asarray(padded, dtype=np.uint8)
    assert arr.ndim == 3 and arr.shape[2] == 3
