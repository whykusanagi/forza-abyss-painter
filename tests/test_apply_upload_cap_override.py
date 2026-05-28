"""upload_cap_override kwarg on _apply_upload_cap lets callers pass a
back-prop-derived value instead of the static DEFAULT_UPLOAD_CAP_PX.

Backward-compat: default kwarg is None → behavior unchanged.
Override never goes BELOW the static floor (720) — back-prop only
raises the cap, never lowers it (the safety floor is the consumer-card
production stance).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from forza_abyss_painter.shapegen.worker import (
    _apply_upload_cap,
    DEFAULT_UPLOAD_CAP_PX,
)


def _src(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 128, 128))


def test_default_kwarg_preserves_old_behavior():
    """Override=None → uses DEFAULT_UPLOAD_CAP_PX (720). A 2000-px
    source should get downscaled to a value derived from 720 / pad_factor."""
    img, _, was_resized = _apply_upload_cap(
        _src(2000, 2000), None,
        profile_max_resolution=720,
        upload_cap_px=DEFAULT_UPLOAD_CAP_PX,
    )
    assert was_resized is True
    assert max(img.size) <= 720


def test_override_raises_cap():
    """Override=1200 + profile_max_resolution=1200 → 2000-px source
    survives at higher resolution than 720 would allow."""
    img_default, _, _ = _apply_upload_cap(
        _src(2000, 2000), None,
        profile_max_resolution=1200,
        upload_cap_px=DEFAULT_UPLOAD_CAP_PX,
        upload_cap_override=None,
    )
    img_override, _, _ = _apply_upload_cap(
        _src(2000, 2000), None,
        profile_max_resolution=1200,
        upload_cap_px=DEFAULT_UPLOAD_CAP_PX,
        upload_cap_override=1200,
    )
    assert max(img_default.size) <= 720
    assert max(img_override.size) > max(img_default.size)
    assert max(img_override.size) <= 1200


def test_override_never_below_floor():
    """upload_cap_override < DEFAULT_UPLOAD_CAP_PX → use the floor.
    Back-prop must never lower the cap below the safety floor."""
    img, _, _ = _apply_upload_cap(
        _src(2000, 2000), None,
        profile_max_resolution=720,
        upload_cap_px=DEFAULT_UPLOAD_CAP_PX,
        upload_cap_override=480,  # below floor
    )
    # Behavior should match upload_cap_override=None (uses 720 floor).
    img_ref, _, _ = _apply_upload_cap(
        _src(2000, 2000), None,
        profile_max_resolution=720,
        upload_cap_px=DEFAULT_UPLOAD_CAP_PX,
        upload_cap_override=None,
    )
    assert img.size == img_ref.size


def test_override_zero_disables_cap_like_legacy():
    """upload_cap_override=0 → disable cap (legacy upload_cap_px=0 behavior).
    The override is OPT-OUT compatible: passing 0 means 'no cap'."""
    img, _, was_resized = _apply_upload_cap(
        _src(2000, 2000), None,
        profile_max_resolution=2000,
        upload_cap_px=0,
        upload_cap_override=0,
    )
    assert was_resized is False
    assert img.size == (2000, 2000)


def test_alpha_mask_resizes_with_override(tmp_path):
    """When override raises the cap, an alpha mask should resize to
    the same dimensions as the RGB image."""
    img, alpha_out, was_resized = _apply_upload_cap(
        _src(2000, 2000),
        np.zeros((2000, 2000), dtype=np.uint8),
        profile_max_resolution=1200,
        upload_cap_px=DEFAULT_UPLOAD_CAP_PX,
        upload_cap_override=1200,
    )
    assert was_resized is True
    assert alpha_out is not None
    assert alpha_out.shape == (img.size[1], img.size[0])
