"""GPU runs enforce a minimum checkpoint cadence of 100 to keep snapshot
write frequency reasonable on fast cards. CPU runs can checkpoint every
shape."""
from __future__ import annotations

import pytest

from forza_abyss_painter.runtime.torch_runner import RunConfig


def _fresh_dict(tmp_path) -> dict:
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return {
        "image_path": str(image),
        "output_json_path": str(tmp_path / "out.json"),
        "num_shapes": 100,
        "max_resolution": 360,
        "random_samples": 1024,
    }


def test_cuda_checkpoint_every_50_rejected(tmp_path):
    d = _fresh_dict(tmp_path)
    d["device"] = "cuda"
    d["checkpoint_every"] = 50
    with pytest.raises(ValueError, match="checkpoint_every"):
        RunConfig.from_dict(d)


def test_cuda_checkpoint_every_100_accepted(tmp_path):
    d = _fresh_dict(tmp_path)
    d["device"] = "cuda"
    d["checkpoint_every"] = 100
    cfg = RunConfig.from_dict(d)
    assert cfg.checkpoint_every == 100


def test_cuda_checkpoint_every_zero_accepted(tmp_path):
    """0 means 'disabled' — explicit opt-out from snapshots. Preserved
    for unit-test callers + power users."""
    d = _fresh_dict(tmp_path)
    d["device"] = "cuda"
    d["checkpoint_every"] = 0
    cfg = RunConfig.from_dict(d)
    assert cfg.checkpoint_every == 0


def test_cpu_checkpoint_every_10_accepted(tmp_path):
    d = _fresh_dict(tmp_path)
    d["device"] = "cpu"
    d["checkpoint_every"] = 10
    cfg = RunConfig.from_dict(d)
    assert cfg.checkpoint_every == 10


def test_cpu_checkpoint_every_1_accepted(tmp_path):
    d = _fresh_dict(tmp_path)
    d["device"] = "cpu"
    d["checkpoint_every"] = 1
    cfg = RunConfig.from_dict(d)
    assert cfg.checkpoint_every == 1
