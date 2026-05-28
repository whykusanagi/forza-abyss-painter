"""estimate_full_pipeline_gib returns K-scorer estimate + calibrated
overhead. Calibration from Run 4 (47.5 GiB measured) + QUASAR
2026-05-27 (53.7 GiB measured at Hi-Res 3000 on 32 GiB card)."""
from __future__ import annotations

from forza_abyss_painter.shapegen.gpu.vram_planner import (
    estimate_peak_vram_gib,
    estimate_full_pipeline_gib,
    PIPELINE_OVERHEAD_GIB,
)


def test_constant_is_35_gib():
    assert PIPELINE_OVERHEAD_GIB == 35.0


def test_full_pipeline_adds_overhead_to_k_estimate():
    """The full estimate is the K-scorer batch peak PLUS a fixed
    overhead for canvas + refill + joint_polish + allocator
    fragmentation. Calibrated from Run 4/6 evidence."""
    k_only = estimate_peak_vram_gib(
        K=8192, bbox_local=True, max_resolution=720,
    )
    full = estimate_full_pipeline_gib(
        K=8192, bbox_local=True, max_resolution=720,
    )
    assert full == k_only + PIPELINE_OVERHEAD_GIB


def test_run_4_scenario_estimates_at_least_40_gib():
    """Run 4: K=8192, max_resolution=720, bbox_local. Measured 47.5
    GiB allocated on a 32 GiB card. New estimate must be at least
    40 GiB — well into 'won't fit' territory."""
    est = estimate_full_pipeline_gib(
        K=8192, bbox_local=True, max_resolution=720,
    )
    assert est >= 40.0, (
        f"Run-4 scenario estimate {est:.1f} GiB is below the 40 GiB "
        f"floor; calibration insufficient to block this OOM scenario."
    )


def test_hi_res_3000_scenario_estimates_at_least_50_gib():
    """QUASAR 2026-05-27 manual test: Hi-Res preset (K=12288,
    max_resolution=1000, bbox_local). Measured 53.7 GiB at OOM on
    32 GiB card. New estimate must be at least 50 GiB."""
    est = estimate_full_pipeline_gib(
        K=12288, bbox_local=True, max_resolution=1000,
    )
    assert est >= 50.0, (
        f"Hi-Res 3000 scenario estimate {est:.1f} GiB is below the "
        f"50 GiB floor; would not block the QUASAR OOM."
    )


def test_small_run_includes_overhead_too():
    """Even a small run (K=1024, max_res=480) has the pipeline
    overhead — polish + canvas + refill don't scale to zero."""
    est = estimate_full_pipeline_gib(
        K=1024, bbox_local=True, max_resolution=480,
    )
    # Small K, small canvas → K-only is ~1-3 GiB; + 35 = 36-38 GiB.
    assert est >= 35.0
    assert est <= 40.0   # not absurdly larger


def test_zero_k_returns_overhead_only():
    """Edge case: K=0 → no K-scorer batch, just the fixed overhead."""
    est = estimate_full_pipeline_gib(
        K=0, bbox_local=True, max_resolution=480,
    )
    assert est == PIPELINE_OVERHEAD_GIB
