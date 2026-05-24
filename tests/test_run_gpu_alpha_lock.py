import numpy as np
import pytest

from forza_abyss_painter.shapegen.gpu.engine import GPUConfig, run_gpu


def _white_target(h=32, w=32):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_lock_alpha_default_is_true_system_constraint():
    """lock_alpha is a HARD SYSTEM CONSTRAINT of the Forza injector — alpha=255 is the
    only valid value the game accepts. Default flipped from False to True in v0.1.7;
    False is no longer a supported config."""
    cfg = GPUConfig()
    assert cfg.lock_alpha is True, (
        "GPUConfig.lock_alpha default must be True. False is not a supported value — "
        "the Forza injector writes alpha=255 unconditionally; soft-alpha JSONs render "
        "one way in the engine preview and another way in-game."
    )
    assert cfg.prune_threshold == 0.5


def test_run_gpu_rejects_lock_alpha_false_with_clear_error():
    """run_gpu raises ValueError at the API boundary if lock_alpha=False, preventing the
    invalid state from silently producing a broken JSON. Error message references the
    underlying game constraint so users understand why this isn't optional."""
    cfg = GPUConfig(num_shapes=3, random_samples=4, lock_alpha=False)
    target = _white_target()
    with pytest.raises(ValueError, match="lock_alpha"):
        run_gpu(target, cfg, alpha_mask=None, progress_every=0)


def test_run_gpu_with_lock_alpha_produces_alpha_255():
    """Net-new generation with lock_alpha=True yields a JSON whose every shape has alpha=255."""
    cfg = GPUConfig(num_shapes=10, random_samples=8, joint_polish_steps=3,
                    lock_alpha=True, edge_strength=0.0)
    target = _white_target()
    shapes_json, _ = run_gpu(target, cfg, alpha_mask=None, progress_every=0)
    assert all(s["color"][3] == 255 for s in shapes_json)


def test_run_gpu_drawable_count_matches_num_shapes():
    """num_shapes now means exactly N drawables — no slot reservation for boundary masks."""
    cfg = GPUConfig(num_shapes=10, random_samples=8, joint_polish_steps=0,
                    lock_alpha=True)
    target = _white_target()
    shapes_json, _ = run_gpu(target, cfg, alpha_mask=None, progress_every=0)
    assert len(shapes_json) == 10
    drawables = shapes_json
    assert len(drawables) == 10
