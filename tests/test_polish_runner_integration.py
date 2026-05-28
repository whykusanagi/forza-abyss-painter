"""End-to-end: spawn the runner subprocess in mode='polish_only' with a
real (small) image + a real (3-ellipse) shapes JSON, assert the output
file lands, has the same shape count, and the geometry is bit-identical
to the input (freeze_geometry=True).

Skipped when torch is not importable in the host env — this is an
integration test that exercises the real joint_polish call path.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")


def _write_test_image(path: Path, h: int = 32, w: int = 32) -> None:
    from PIL import Image
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :w // 2] = (200, 80, 80)   # left half: dusty red
    arr[:, w // 2:] = (80, 80, 200)   # right half: dusty blue
    Image.fromarray(arr, "RGB").save(path)


def _write_test_shapes_json(path: Path, w: int, h: int) -> dict:
    """Three rotated_ellipses in a v1 fd6.shapes document."""
    shapes = [
        {"type": "rotated_ellipse", "x": 8.0,  "y": 16.0, "rx": 6.0, "ry": 6.0,
         "angle": 0.0,  "color": [128, 128, 128, 255]},
        {"type": "rotated_ellipse", "x": 16.0, "y": 16.0, "rx": 6.0, "ry": 6.0,
         "angle": 30.0, "color": [128, 128, 128, 255]},
        {"type": "rotated_ellipse", "x": 24.0, "y": 16.0, "rx": 6.0, "ry": 6.0,
         "angle": 60.0, "color": [128, 128, 128, 255]},
    ]
    doc = {
        "format": "fd6.shapes",
        "version": 1,
        "source_image": "img.png",
        "image_size": [w, h],
        "shape_count": len(shapes),
        "generated_at": "",
        "profile": "test",
        "sticker_mode": False,
        "shapes": shapes,
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return doc


@pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("FAP_POLISH_TEST_FORCE_CPU") != "1",
    reason="Polish runs much faster on CUDA; skipping on CPU-only env. "
           "Set FAP_POLISH_TEST_FORCE_CPU=1 to force.",
)
def test_polish_only_subprocess_end_to_end(tmp_path):
    image = tmp_path / "img.png"
    _write_test_image(image, h=32, w=32)
    shapes_path = tmp_path / "shapes.json"
    in_doc = _write_test_shapes_json(shapes_path, w=32, h=32)

    cfg_path = tmp_path / "cfg.json"
    out_path = tmp_path / "out.json"
    cfg = {
        "image_path": str(image),
        "output_json_path": str(out_path),
        "mode": "polish_only",
        "input_shapes_path": str(shapes_path),
        "polish_steps_override": 10,   # tiny for test speed
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lock_alpha": True,
    }
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "forza_abyss_painter.runtime.torch_runner",
         "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, (
        f"runner exited {result.returncode}\nstderr:\n{result.stderr}"
    )
    assert out_path.is_file(), "polish runner did not write the output JSON"

    out_doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert out_doc["format"] == "fd6.shapes"
    assert out_doc["version"] == 1
    assert out_doc["shape_count"] == 3

    # Geometry frozen — x, y, rx, ry, angle preserved (within float
    # round-trip tolerance, which is large because rounding to 3 decimals
    # in joint_polish output is exact for these inputs).
    for i, (a, b) in enumerate(zip(in_doc["shapes"], out_doc["shapes"])):
        for key in ("x", "y", "rx", "ry", "angle"):
            assert abs(a[key] - b[key]) < 1e-3, (
                f"shape {i} key {key}: input {a[key]} != output {b[key]} "
                f"— geometry was not frozen"
            )
        # Alpha must remain 255 (lock_alpha=True).
        assert b["color"][3] == 255

    # Each shape's RGB should differ from the input gray (128,128,128)
    # along at least one channel, because polish color-snapped toward
    # the local target (red on left, blue on right). If RGB is exactly
    # the input gray for every shape, polish didn't actually run.
    rgb_changed_count = 0
    for a, b in zip(in_doc["shapes"], out_doc["shapes"]):
        if (a["color"][0], a["color"][1], a["color"][2]) != (
                b["color"][0], b["color"][1], b["color"][2]):
            rgb_changed_count += 1
    assert rgb_changed_count >= 1, (
        "polish did not change any shape's RGB — joint_polish may not "
        "have run, or the optimizer didn't move from the gray init"
    )


def test_polish_only_subprocess_rejects_non_ellipse(tmp_path):
    image = tmp_path / "img.png"
    _write_test_image(image, h=32, w=32)
    # JSON with a non-rotated_ellipse shape — engine.py:697 only polishes
    # all-ellipse documents. Runner should emit a clean error.
    shapes_path = tmp_path / "shapes.json"
    shapes_path.write_text(json.dumps({
        "format": "fd6.shapes",
        "version": 1,
        "source_image": "img.png",
        "image_size": [32, 32],
        "shape_count": 1,
        "generated_at": "",
        "profile": "test",
        "sticker_mode": False,
        "shapes": [
            {"type": "rectangle", "x": 16.0, "y": 16.0, "hw": 10.0, "hh": 10.0,
             "color": [100, 100, 100, 255]},
        ],
    }), encoding="utf-8")

    cfg_path = tmp_path / "cfg.json"
    out_path = tmp_path / "out.json"
    cfg_path.write_text(json.dumps({
        "image_path": str(image),
        "output_json_path": str(out_path),
        "mode": "polish_only",
        "input_shapes_path": str(shapes_path),
        "polish_steps_override": 10,
        "device": "cpu",
        "lock_alpha": True,
    }), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "forza_abyss_painter.runtime.torch_runner",
         "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=60,
    )

    assert result.returncode != 0, "runner should reject non-ellipse polish"
    # Stderr should contain a 'polish_only_unsupported_shape' or similar
    # stage in an error event. Verify the error event was emitted.
    assert "error" in result.stderr.lower()
    assert "ellipse" in result.stderr.lower() or "unsupported" in result.stderr.lower()
    assert not out_path.exists(), (
        "runner wrote output JSON despite reporting an error — IPC violation"
    )
