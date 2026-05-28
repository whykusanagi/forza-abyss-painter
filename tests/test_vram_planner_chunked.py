"""estimate_effective_peak_gib computes the chunk-aware peak that
matches what engine.run_gpu actually allocates at runtime.

Anchors:
  - K=12288, max_res=1000, budget=17 (Hi-Res 3000 on 32G card with FH6)
    -> chunks > 1, peak < budget
  - K=1000, max_res=1200, budget=17 (ziz_dance config)
    -> chunks == 1, peak ~16 GiB
  - K=12288, max_res=1000, budget=0 (workstation / unlimited)
    -> chunks == 1, peak ~139 GiB (matches full K-only)
"""
from forza_abyss_painter.shapegen.gpu.vram_planner import (
    estimate_effective_peak_gib,
    estimate_peak_vram_gib,
    resolve_k_chunk_size,
)


def test_unchunked_when_budget_unlimited():
    """budget=0 means no chunking; matches full K-only formula."""
    peak, chunks = estimate_effective_peak_gib(
        K=12288, max_resolution=1000, budget_gib=0.0,
    )
    full = estimate_peak_vram_gib(K=12288, bbox_local=True, max_resolution=1000)
    assert chunks == 1
    assert abs(peak - full) < 0.1


def test_chunked_when_budget_tight():
    """Hi-Res 3000 on 17 GiB budget engages chunking."""
    peak, chunks = estimate_effective_peak_gib(
        K=12288, max_resolution=1000, budget_gib=17.0,
    )
    assert chunks > 1
    # resolve_k_chunk_size sizes chunks to fit within the full budget
    # (no safety_margin applied inside; that's a caller concern).
    assert peak <= 17.0


def test_ziz_dance_no_chunking_needed():
    """K=1000, max_res=1200 fits in 17 GiB without chunking."""
    peak, chunks = estimate_effective_peak_gib(
        K=1000, max_resolution=1200, budget_gib=17.0,
    )
    assert chunks == 1
    assert 14.0 < peak < 19.0


def test_lineart_no_chunking_needed():
    """Lineart 400 fits in 4 GiB without chunking."""
    peak, chunks = estimate_effective_peak_gib(
        K=1024, max_resolution=360, budget_gib=4.0,
    )
    assert chunks == 1
    assert peak < 4.0


def test_chunks_per_shape_math_correct():
    """chunks_per_shape = ceil(K / chunk_size)."""
    peak, chunks = estimate_effective_peak_gib(
        K=8192, max_resolution=720, budget_gib=17.0,
    )
    chunk = resolve_k_chunk_size(
        K=8192, max_resolution=720, bbox_local=True, vram_budget_gib=17.0,
    )
    if chunk > 0:
        expected = (8192 + chunk - 1) // chunk
        assert chunks == expected
    else:
        assert chunks == 1
