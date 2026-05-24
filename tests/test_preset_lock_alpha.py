"""PRESETS lives in notebooks/build_colab_notebook.py on this branch base.
The build_colab_notebook module has top-level side effects (it builds notebooks
when imported), so we extract PRESETS via runpy with a __main__ guard check
or by parsing — but the simpler path is to import via a controlled exec.

Approach: re-exec the builder module under runpy with run_name set so the
__name__ == '__main__' block (if any) doesn't fire, and read PRESETS from the
returned module-globals dict.

If the builder is rewritten or PRESETS moves (e.g. gpu-fitsize lands and extracts
to fd6.shapegen.presets), update this test to import from the new location.
"""
import runpy
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[1] / "notebooks" / "build_colab_notebook.py"


def _presets():
    # run_name='__not_main__' ensures any `if __name__ == "__main__":` build loop
    # doesn't actually run during the test.
    g = runpy.run_path(str(BUILDER), run_name="__not_main__")
    return g["PRESETS"]


def test_highres_3000_locks_alpha():
    assert _presets()["highres_3000"]["lock_alpha"] is True


def test_medium_1000_locks_alpha():
    assert _presets()["medium_1000"]["lock_alpha"] is True


def test_multi_shape_eval_presets_also_lock_alpha():
    """Multi-shape eval presets MUST ALSO lock alpha. Earlier this test asserted the
    opposite (on the grounds that they're 'not injectable yet') — that was a wrong
    policy that allowed the renderer-vs-injector divergence to ship in their JSONs.
    The engine-vs-injector render contract applies whenever you compare the engine
    PNG to what would happen in-game, which IS the whole point of evaluating richer
    shapes. v0.1.6 flipped the policy + added a comprehensive 'every preset locks
    alpha' sweep in tests/test_lock_alpha_default.py to catch any future regression."""
    p = _presets()
    assert p["shapes_highres_3000"]["lock_alpha"] is True
    assert p["shapes_medium_1000"]["lock_alpha"] is True
