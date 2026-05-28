"""Pure math tests for recommend_max_resolution.

Inverts _peak_bytes_per_candidate's footprint formula: given a free
VRAM budget + K, returns the largest max_resolution that fits.
Returns safety_floor_px when probe is unavailable or math says <floor.
"""
from __future__ import annotations

import math

import pytest

from forza_abyss_painter.shapegen.gpu.vram_planner import (
    recommend_max_resolution,
    estimate_peak_vram_gib,
    BBOX_LOCAL_SAFETY,
)


def test_probe_unavailable_returns_floor():
    assert recommend_max_resolution(free_gib=0.0, K=8192) == 720
    assert recommend_max_resolution(free_gib=-1.0, K=8192) == 720


def test_tiny_budget_clamps_to_floor():
    """Tight free VRAM (e.g. FH6 squatting on a 32 GiB card) →
    return the safety floor; don't suggest something even smaller."""
    rec = recommend_max_resolution(free_gib=1.0, K=8192)
    assert rec == 720


def test_big_card_idle_solves_regime_a():
    """RTX PRO 6000 Blackwell at K=8192, ~95 GiB free →
    Regime A math solves for a max_res well above 720."""
    rec = recommend_max_resolution(free_gib=95.0, K=8192)
    assert rec > 720
    # Sanity: the returned value must actually fit in the budget at
    # 0.85 safety margin.
    peak = estimate_peak_vram_gib(K=8192, bbox_local=True, max_resolution=rec)
    assert peak <= 95.0 * 0.85, (
        f"recommendation {rec}px → peak {peak:.2f} GiB > budget"
    )


def test_regime_b_short_circuit():
    """When crop_e is pinned at bbox_crop_max=256, increasing
    max_resolution beyond 8*256=2048 doesn't change peak_bytes
    (footprint becomes constant (513)² · K · 12 · 15). So if Regime B
    fits at all, we can return the absolute ceiling."""
    # K small enough that Regime B fits comfortably
    rec = recommend_max_resolution(free_gib=200.0, K=1024,
                                     absolute_ceiling_px=4096)
    assert rec == 4096


def test_absolute_ceiling_clamps_high_recommendation():
    """When math says > absolute_ceiling, clamp."""
    rec = recommend_max_resolution(free_gib=500.0, K=512,
                                     absolute_ceiling_px=2048)
    assert rec <= 2048


def test_monotonic_in_free_gib():
    """More free VRAM → at-least-as-much recommended resolution."""
    rec_low = recommend_max_resolution(free_gib=10.0, K=8192)
    rec_mid = recommend_max_resolution(free_gib=40.0, K=8192)
    rec_high = recommend_max_resolution(free_gib=95.0, K=8192)
    assert rec_low <= rec_mid <= rec_high


def test_monotonic_inverse_in_k():
    """At fixed budget, lower K → at-least-as-much recommended res
    (each candidate is cheaper)."""
    rec_low_k = recommend_max_resolution(free_gib=40.0, K=2048)
    rec_high_k = recommend_max_resolution(free_gib=40.0, K=16384)
    assert rec_low_k >= rec_high_k


def test_floor_can_be_overridden():
    """Caller can pass a lower safety floor (e.g. for testing)."""
    rec = recommend_max_resolution(free_gib=1.0, K=8192,
                                     safety_floor_px=128)
    assert rec >= 128


def test_returned_value_actually_fits():
    """For a range of (free_gib, K), the returned max_res must keep
    estimate_peak_vram_gib at or below free_gib * 0.85."""
    for free_gib in (5.0, 12.0, 27.0, 48.0, 95.0):
        for K in (1024, 4096, 8192, 16384):
            rec = recommend_max_resolution(free_gib=free_gib, K=K)
            # Only check when recommendation is in Regime A territory;
            # Regime B returns the absolute_ceiling which by construction
            # fits.
            peak = estimate_peak_vram_gib(K=K, bbox_local=True,
                                            max_resolution=rec)
            # Floor case is allowed to over-shoot the budget — Item E's
            # hard-block catches that scenario separately.
            if rec > 720:
                assert peak <= free_gib * 0.85 + 1e-6, (
                    f"free={free_gib} K={K} rec={rec} peak={peak:.2f} "
                    f"exceeds 0.85*{free_gib}={free_gib*0.85:.2f}"
                )


def test_full_canvas_path():
    """full_canvas=False uses FULL_CANVAS_SAFETY (8.0). For the same
    budget, full_canvas yields LOWER max_res than bbox_local (per-cand
    cost is much higher)."""
    bbox = recommend_max_resolution(free_gib=95.0, K=4096, bbox_local=True)
    full = recommend_max_resolution(free_gib=95.0, K=4096, bbox_local=False)
    assert full < bbox or full == 720, (
        f"full_canvas={full} should be <= bbox={bbox} (or both at floor)"
    )
