"""Regression test pinning LOCK_ALPHA=True everywhere — across every production preset
AND as the builder's defensive setdefault.

WHY THIS MATTERS: the in-game injector forces binary alpha (255) at write time. A
JSON generated with soft alpha (e.g. 96/160/255 levels) renders one way in the
notebook's engine preview and ANOTHER way in-game — the contract breaks. Every
preset must lock alpha. Every NEW preset added in the future must also lock alpha,
even if the author forgets to set the field explicitly. The builder's `setdefault`
is the safety net for that.

This test was added in v0.1.6 after the user reported the multi-shape eval presets
(shapes_highres_3000, shapes_medium_1000) were shipping with LOCK_ALPHA=False because
they were missing the field in presets.py and falling through to setdefault(False).
Two simultaneous fixes: explicit True in those presets + flipped setdefault default.
"""
import json
import re
from pathlib import Path

import pytest

from forza_abyss_painter.shapegen.presets import PRESETS


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_GLOB = "notebooks/fap_gpu_colab_*.ipynb"


def test_every_preset_locks_alpha_in_source_of_truth():
    """Every entry in PRESETS — including multi-shape eval presets — must explicitly
    set lock_alpha=True. If anyone adds a new preset and forgets, this fails."""
    for name, preset in PRESETS.items():
        assert preset.get("lock_alpha") is True, (
            f"preset {name!r}: lock_alpha is {preset.get('lock_alpha')!r}, must be True. "
            f"The injector forces binary alpha at write time; soft-alpha JSONs render "
            f"differently in-game than in the engine preview. ALWAYS True."
        )


def test_every_production_notebook_ships_with_lock_alpha_true():
    """Generated notebooks (the artifacts users actually upload to Colab) must have
    LOCK_ALPHA = True in their Configure cell."""
    for nb_path in sorted(REPO_ROOT.glob(NOTEBOOK_GLOB)):
        nb = json.loads(nb_path.read_text())
        src_all = "\n".join("".join(c.get("source", []))
                            for c in nb["cells"] if c["cell_type"] == "code")
        m = re.search(r"LOCK_ALPHA\s*=\s*(True|False)", src_all)
        assert m, f"{nb_path.name}: LOCK_ALPHA not defined in Configure cell"
        assert m.group(1) == "True", (
            f"{nb_path.name}: LOCK_ALPHA = {m.group(1)} in generated notebook. Must be True. "
            f"Either the preset is missing lock_alpha:True or the builder's setdefault "
            f"regressed to False — check forza_abyss_painter/shapegen/presets.py and "
            f"notebooks/build_colab_notebook.py's cell_knobs() setdefault."
        )


def test_builder_setdefault_for_lock_alpha_is_true():
    """The builder's defensive default in cell_knobs() must be True. This is the safety
    net that catches new presets added by future contributors who forget the field —
    a False default would silently ship broken JSONs."""
    builder_src = (REPO_ROOT / "notebooks" / "build_colab_notebook.py").read_text()
    # Match the setdefault line for lock_alpha. Allow whitespace + comment variants.
    m = re.search(r'setdefault\(\s*"lock_alpha"\s*,\s*(True|False)\s*\)', builder_src)
    assert m, "lock_alpha setdefault not found in build_colab_notebook.py"
    assert m.group(1) == "True", (
        f"builder's setdefault for lock_alpha is {m.group(1)}. Must be True. "
        f"A False default would mean any new preset that forgets to specify lock_alpha "
        f"ships a broken JSON whose engine PNG diverges from the in-game render."
    )
