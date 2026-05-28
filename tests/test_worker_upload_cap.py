"""Upload-cap auto-resize regression for the EXE's CPU shape-gen worker.

Ports v0.1.5 from the colab pipeline: when a user feeds a huge source
image (4K phone photo, scanned poster), shrink it to a safe size BEFORE
the padding step so we don't burn memory + CPU on an oversized canvas
the engine would immediately downscale anyway. The cap reserves the
8%-per-side padding budget up front so the post-pad long side lands at
or below the configured `upload_cap_px`.

Defaults to 720 px — matches the colab pipeline's
UPLOAD_MAX_LONG_SIDE. Callers can pass 0 to disable (e.g., overnight
batch users who already control input size).
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from forza_abyss_painter.shapegen.worker import (
    DEFAULT_UPLOAD_CAP_PX,
    _apply_upload_cap,
    _pad_with_source_mean,
)


# ----------------------------------------------------- core cap behavior


def test_default_cap_is_720():
    """The 720 default mirrors the colab pipeline's UPLOAD_MAX_LONG_SIDE.
    Changing it drifts the EXE away from the notebook baseline and
    breaks user expectations on cross-tool consistency."""
    assert DEFAULT_UPLOAD_CAP_PX == 720


def test_oversized_input_gets_resized_below_effective_input_max():
    """4000 px long-side input with cap 720 and a normal profile must
    resize down to ≤ 720/1.16 ≈ 620 px so the post-pad canvas lands
    ≤ 720 px. This is the headline regression — without it the worker
    pads a 4K canvas before noticing the size."""
    img = Image.new("RGB", (4000, 2000), (50, 50, 50))
    capped_img, _alpha, was_capped = _apply_upload_cap(
        img, alpha_mask=None,
        profile_max_resolution=1600, upload_cap_px=720,
    )
    assert was_capped is True
    # effective_max = min(1600, 720) = 720; / 1.16 = 620.69 → 620
    assert max(capped_img.size) == pytest.approx(620, abs=1)
    # Aspect ratio preserved (2:1 input → 2:1 output)
    assert capped_img.size[0] / capped_img.size[1] == pytest.approx(2.0, abs=0.02)


def test_final_padded_canvas_stays_under_cap():
    """End-to-end contract: feed the cap helper a giant image, then the
    pad helper, and the final padded long side must land at or below
    the cap. If this fails the cap math is wrong by ≥1 unit somewhere."""
    img = Image.new("RGB", (4000, 4000), (128, 128, 128))
    capped, alpha, _ = _apply_upload_cap(
        img, alpha_mask=None,
        profile_max_resolution=1600, upload_cap_px=720,
    )
    padded, _ = _pad_with_source_mean(
        capped, sticker_mode=False, alpha_mask=alpha
    )
    assert max(padded.size) <= 720, (
        f"final padded canvas {max(padded.size)} > cap 720 — "
        f"the effective_input_max math under-reserves padding budget"
    )


def test_already_small_input_is_not_resized():
    """Input already under the effective_input_max ceiling stays
    untouched. The cap must NOT upsample small inputs (which would
    destroy detail) and must NOT no-op-resize (which would still cost
    a LANCZOS pass for nothing)."""
    img = Image.new("RGB", (400, 300), (10, 20, 30))
    capped, _alpha, was_capped = _apply_upload_cap(
        img, alpha_mask=None,
        profile_max_resolution=1600, upload_cap_px=720,
    )
    assert was_capped is False
    assert capped.size == (400, 300)
    # Bitwise identity (PIL doesn't copy on no-op).
    assert capped is img


def test_cap_zero_disables_safety_net():
    """upload_cap_px <= 0 disables the cap entirely — used by overnight
    batch flows where the user wants the original input untouched."""
    img = Image.new("RGB", (4000, 2000), (255, 0, 0))
    capped, _alpha, was_capped = _apply_upload_cap(
        img, alpha_mask=None,
        profile_max_resolution=1600, upload_cap_px=0,
    )
    assert was_capped is False
    assert capped.size == (4000, 2000)


def test_cap_negative_also_disables():
    """Guard against accidental negative values producing nonsense
    downscale factors — treat any non-positive cap as disabled."""
    img = Image.new("RGB", (4000, 2000), (255, 0, 0))
    capped, _alpha, was_capped = _apply_upload_cap(
        img, alpha_mask=None,
        profile_max_resolution=1600, upload_cap_px=-100,
    )
    assert was_capped is False
    assert capped.size == (4000, 2000)


def test_cap_floors_at_profile_max_resolution_not_above():
    """The effective cap is `min(profile_max_resolution, upload_cap_px)`.
    A preset whose max_resolution is 480 with upload_cap_px=720 should
    cap at 480, not 720 — shrinking further would just lose detail
    without speeding anything up (the engine's own downscale fires
    at 480 anyway)."""
    img = Image.new("RGB", (2000, 1000), (200, 200, 200))
    capped, _alpha, was_capped = _apply_upload_cap(
        img, alpha_mask=None,
        profile_max_resolution=480, upload_cap_px=720,
    )
    # effective_max = min(480, 720) = 480; / 1.16 ≈ 413
    assert was_capped is True
    assert max(capped.size) == pytest.approx(413, abs=2)


# ----------------------------------------------------- alpha-mask handling


def test_alpha_mask_resized_to_match_capped_image():
    """When alpha mask is present, it must downscale alongside RGB to
    the same new size — engine's (H, W, 3) target + (H, W) alpha shape
    contract breaks instantly if these drift."""
    img = Image.new("RGB", (2000, 1000), (100, 100, 100))
    alpha = np.full((1000, 2000), 200, dtype=np.uint8)
    capped, capped_alpha, was_capped = _apply_upload_cap(
        img, alpha_mask=alpha,
        profile_max_resolution=1600, upload_cap_px=720,
    )
    assert was_capped is True
    assert capped_alpha is not None
    assert capped_alpha.shape == (capped.size[1], capped.size[0]), (
        f"alpha {capped_alpha.shape} != image {(capped.size[1], capped.size[0])}"
    )
    assert capped_alpha.dtype == np.uint8


def test_alpha_mask_passthrough_when_cap_disabled():
    """Disabled cap returns the same alpha mask object — no copy."""
    img = Image.new("RGB", (4000, 2000), (255, 0, 0))
    alpha = np.full((2000, 4000), 128, dtype=np.uint8)
    _capped, capped_alpha, _was = _apply_upload_cap(
        img, alpha_mask=alpha,
        profile_max_resolution=1600, upload_cap_px=0,
    )
    assert capped_alpha is alpha


def test_no_alpha_mask_returns_none():
    """alpha_mask=None must pass through as None on both code paths
    (capped and not-capped). Returning np.array([]) or a zeros mask
    would change the engine's "no transparency" branch into the
    sticker-mode branch silently."""
    img = Image.new("RGB", (4000, 2000), (0, 0, 0))
    _, alpha_after, _ = _apply_upload_cap(
        img, alpha_mask=None,
        profile_max_resolution=1600, upload_cap_px=720,
    )
    assert alpha_after is None


# ----------------------------------------------------- aspect ratio preservation


def test_extreme_aspect_ratio_preserved():
    """A 3000×500 panorama (6:1) capped at 720 must stay 6:1, never get
    cropped. The shape-gen depends on the aspect to land shapes where
    the source had content."""
    img = Image.new("RGB", (3000, 500), (100, 100, 100))
    capped, _alpha, _was = _apply_upload_cap(
        img, alpha_mask=None,
        profile_max_resolution=1600, upload_cap_px=720,
    )
    ratio = capped.size[0] / capped.size[1]
    assert ratio == pytest.approx(6.0, abs=0.05), (
        f"aspect ratio drift: {ratio} vs expected 6.0 — cap is cropping"
    )


def test_portrait_orientation_preserved():
    """Tall image (1000×3000) — long side is the height. Cap must shrink
    based on max(size), not assume landscape."""
    img = Image.new("RGB", (1000, 3000), (100, 100, 100))
    capped, _alpha, was_capped = _apply_upload_cap(
        img, alpha_mask=None,
        profile_max_resolution=1600, upload_cap_px=720,
    )
    assert was_capped is True
    # Tall side capped
    assert max(capped.size) == pytest.approx(620, abs=1)
    # Still portrait
    assert capped.size[1] > capped.size[0]


# ----------------------------------------------------- worker constructor wiring


def test_worker_constructor_accepts_upload_cap_kwarg(tmp_path):
    """The worker exposes upload_cap_px as a constructor kwarg so the
    GUI layer can override the default (overnight batch flows) without
    monkeypatching module state."""
    from forza_abyss_painter.shapegen.worker import GenerationWorker
    from forza_abyss_painter.shapegen.profile import Profile
    # Minimal fake image path; we don't run() the worker, just verify
    # constructor wiring.
    img_path = tmp_path / "fake.png"
    Image.new("RGB", (10, 10), (0, 0, 0)).save(img_path)
    profile = Profile(name="test", max_resolution=480, stop_at=100)
    w = GenerationWorker(img_path, profile, upload_cap_px=1024)
    assert w.upload_cap_px == 1024


def test_worker_constructor_default_matches_module_default(tmp_path):
    """Constructor default must read from DEFAULT_UPLOAD_CAP_PX so
    bumping the module-level default propagates without missed
    callsites."""
    from forza_abyss_painter.shapegen.worker import GenerationWorker
    from forza_abyss_painter.shapegen.profile import Profile
    img_path = tmp_path / "fake.png"
    Image.new("RGB", (10, 10), (0, 0, 0)).save(img_path)
    profile = Profile(name="test", max_resolution=480, stop_at=100)
    w = GenerationWorker(img_path, profile)
    assert w.upload_cap_px == DEFAULT_UPLOAD_CAP_PX
