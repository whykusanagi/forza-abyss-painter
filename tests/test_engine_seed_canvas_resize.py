"""_scale_seed_shapes is a pure-Python helper that scales seeded shape
coordinates by target/snap canvas ratio. Pinned for resume + future
shape-upscaling flows.
"""
import pytest

# Engine module imports torch at top-level for the run_gpu pipeline; the
# helper under test is pure Python, but we still have to load the module
# to reach it. Skip the file on dev boxes without torch (Mac); CI/Windows
# tester boxes have torch and will exercise the assertions.
pytest.importorskip("torch")

from forza_abyss_painter.shapegen.gpu.engine import _scale_seed_shapes  # noqa: E402


def test_2x_downscale_halves_geometry():
    shapes = [
        {"type": "rotated_ellipse", "x": 600.0, "y": 400.0,
         "rx": 100.0, "ry": 50.0, "angle": 0.5,
         "rgba": [1.0, 0.5, 0.0, 0.8]},
    ]
    out = _scale_seed_shapes(shapes, snap=(1200, 800), target=(600, 400))
    s = out[0]
    assert s["x"] == 300.0
    assert s["y"] == 200.0
    assert s["rx"] == 50.0
    assert s["ry"] == 25.0
    assert s["angle"] == 0.5
    assert s["rgba"] == [1.0, 0.5, 0.0, 0.8]


def test_2x_upscale_doubles_geometry():
    shapes = [{"type": "rotated_ellipse", "x": 100.0, "y": 100.0,
               "rx": 20.0, "ry": 10.0, "angle": 0.0,
               "rgba": [0.5, 0.5, 0.5, 1.0]}]
    out = _scale_seed_shapes(shapes, snap=(500, 500), target=(1000, 1000))
    assert out[0]["x"] == 200.0
    assert out[0]["rx"] == 40.0


def test_identity_when_sizes_match():
    shapes = [{"type": "rotated_ellipse", "x": 100.0, "y": 50.0,
               "rx": 10.0, "ry": 5.0, "angle": 0.1,
               "rgba": [0.0, 0.0, 0.0, 1.0]}]
    out = _scale_seed_shapes(shapes, snap=(800, 400), target=(800, 400))
    assert out == shapes


def test_extreme_scale_raises():
    """Spec guard: scale factor < 0.3 or > 3.0 refuses."""
    shapes = []
    with pytest.raises(ValueError, match="scale factor"):
        _scale_seed_shapes(shapes, snap=(3000, 3000), target=(400, 400))


def test_alpha_and_color_invariant():
    """rgba and angle are dimensionless; never scaled."""
    shapes = [{"type": "rotated_ellipse", "x": 50.0, "y": 50.0,
               "rx": 5.0, "ry": 5.0, "angle": 1.57,
               "rgba": [0.5, 0.25, 0.75, 0.9]}]
    out = _scale_seed_shapes(shapes, snap=(100, 100), target=(150, 150))
    assert out[0]["rgba"] == [0.5, 0.25, 0.75, 0.9]
    assert out[0]["angle"] == 1.57
