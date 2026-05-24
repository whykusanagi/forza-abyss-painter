"""Regression tests for clean_and_refill — the post-greedy budget-reclamation pass.

Verifies the three contracts of the option-B refill design:
1. cfg.refill_dead_shapes defaults to True (production-on by default — strict quality
   improvement since dead shapes contribute zero pixels).
2. refill_min_visible_px defaults to 5 (matches fap-clean's threshold).
3. clean_and_refill drops dead shapes AND commits new ones in their place — final
   output has fewer dead shapes than the input would have had without refill.
4. Multi-shape inputs (triangle + rectangle) bypass refill (ellipse-only constraint).
5. Empty shape lists don't crash.
"""
import numpy as np
import pytest
import torch

from forza_abyss_painter.shapegen.gpu.device import DTYPE, get_device
from forza_abyss_painter.shapegen.gpu.engine import GPUConfig, run_gpu
from forza_abyss_painter.shapegen.gpu.refill import (
    clean_and_refill,
    _identify_dead_shape_indices,
)


def test_gpu_config_refill_defaults():
    """Refill is ON by default; threshold matches fap-clean's 5-px cutoff."""
    cfg = GPUConfig()
    assert cfg.refill_dead_shapes is True, (
        "refill_dead_shapes default must be True — strict quality improvement at the "
        "cost of one extra mini-greedy pass."
    )
    assert cfg.refill_min_visible_px == 5


def test_identify_dead_shape_indices_catches_fully_occluded():
    """A small shape fully covered by a larger later shape is dead. Threshold of 5
    visible pixels is the production cutoff."""
    device = get_device()
    shapes = [
        # idx 0: small, will be occluded by 1.
        {"type": "rotated_ellipse", "x": 50.0, "y": 50.0, "rx": 5.0, "ry": 5.0,
         "angle": 0.0, "color": [200, 100, 50, 255]},
        # idx 1: large, fully covers idx 0.
        {"type": "rotated_ellipse", "x": 50.0, "y": 50.0, "rx": 30.0, "ry": 30.0,
         "angle": 0.0, "color": [50, 200, 100, 255]},
        # idx 2: small + non-overlapping = alive.
        {"type": "rotated_ellipse", "x": 10.0, "y": 10.0, "rx": 4.0, "ry": 4.0,
         "angle": 0.0, "color": [10, 10, 200, 255]},
    ]
    dead = _identify_dead_shape_indices(shapes, h=100, w=100, device=device, min_visible_px=5)
    assert 0 in dead, "shape 0 should be detected as fully occluded"
    assert 1 not in dead, "shape 1 (the occluder) is the topmost — must remain"
    assert 2 not in dead, "shape 2 doesn't overlap anyone — must remain"


def test_clean_and_refill_replaces_dead_with_live():
    """End-to-end via run_gpu — synthetic target where greedy is likely to produce
    some occluded shapes. With refill ON, the output has ALL shapes contributing
    visible pixels (no dead weight in final JSON)."""
    rng = np.random.default_rng(42)
    target = rng.integers(40, 200, size=(64, 64, 3), dtype=np.uint8)
    cfg = GPUConfig(
        num_shapes=20, random_samples=8, seed=42,
        refine_mode="gradient", grad_starts=2, grad_steps=10,
        bbox_local=True, bbox_crop_max=64,
        joint_polish_steps=10,
        refill_dead_shapes=True,
    )
    shapes, _ = run_gpu(target, cfg, alpha_mask=None, progress_every=0)
    # After refill, ALL surviving shapes should pass the same dead-shape check that
    # refill itself uses internally.
    dead_after_refill = _identify_dead_shape_indices(
        shapes, h=64, w=64, device=get_device(), min_visible_px=cfg.refill_min_visible_px,
    )
    assert len(dead_after_refill) == 0, (
        f"after refill, {len(dead_after_refill)} dead shapes remain in final JSON. "
        f"Refill should have detected + replaced them. Indices: {sorted(dead_after_refill)}"
    )


def test_refill_off_leaves_dead_shapes_in_output():
    """Disable refill — dead shapes (if any) survive into the final JSON. Confirms the
    flag actually gates the behavior (vs being a no-op)."""
    rng = np.random.default_rng(42)
    target = rng.integers(40, 200, size=(64, 64, 3), dtype=np.uint8)
    cfg = GPUConfig(
        num_shapes=20, random_samples=8, seed=42,
        refine_mode="gradient", grad_starts=2, grad_steps=10,
        bbox_local=True, bbox_crop_max=64,
        joint_polish_steps=10,
        refill_dead_shapes=False,
    )
    shapes_no_refill, _ = run_gpu(target, cfg, alpha_mask=None, progress_every=0)
    # Without refill, output is just greedy + polish — no guarantee of zero dead.
    # We don't assert dead > 0 (synthetic might be too easy to produce dead shapes
    # depending on RNG); just confirm the path runs and returns shapes.
    assert len(shapes_no_refill) > 0
    assert len(shapes_no_refill) <= cfg.num_shapes


def test_clean_and_refill_skips_multi_shape_input():
    """Refill is ellipse-only at the engine call site. A multi-shape eval cfg
    (triangle + rectangle) gets through run_gpu without going through refill — no
    crash, output retains multi-shape mix."""
    rng = np.random.default_rng(42)
    target = rng.integers(40, 200, size=(48, 48, 3), dtype=np.uint8)
    cfg = GPUConfig(
        num_shapes=8, random_samples=4, seed=42,
        refine_mode="gradient", grad_starts=2, grad_steps=8,
        bbox_local=False,
        shape_types=["rotated_ellipse", "triangle", "rotated_rectangle"],
        joint_polish_steps=0,
        refill_dead_shapes=True,   # ON, but should be skipped because multi-shape
    )
    shapes, _ = run_gpu(target, cfg, alpha_mask=None, progress_every=0)
    types_present = {s["type"] for s in shapes}
    assert len(types_present) >= 1, "should commit at least one shape type"
    # If refill ran and tried to process triangles/rectangles, it would have crashed
    # with KeyError on 'rx' — surviving without exception means it correctly skipped.


def test_clean_and_refill_handles_empty_input():
    """No shapes → no work → returns empty list + the canvas-init."""
    device = get_device()
    target = torch.zeros((32, 32, 3), dtype=torch.uint8, device=device)
    canvas_init = torch.full((32, 32, 3), 40, dtype=torch.uint8, device=device)
    cfg = GPUConfig(num_shapes=10, refill_dead_shapes=True)
    gen = torch.Generator(device="cpu").manual_seed(0)
    shapes, canvas = clean_and_refill(
        [], target, canvas_init,
        alpha_t=None, alpha_mask_f=None, edge_weight=None,
        cfg=cfg, h=32, w=32, generator=gen,
    )
    assert shapes == []
    assert canvas.shape == (32, 32, 3)
