"""Snapshot file naming + helpers.

Shared by:
- CPU `shapegen/worker.py` (was inline construction; switch to this).
- GPU `runtime/torch_runner.py` (new code).
- GUI `gui/main_window.py` (resume flow needs to look for siblings).

Snapshots live next to the final output JSON. Naming pattern matches
the CPU side's historical convention so existing artifacts on disk
keep round-tripping.
"""
from __future__ import annotations

from pathlib import Path


def snapshot_path_for(output_path: str | Path, count: int) -> Path:
    """Return the snapshot path for `count` shapes alongside the final
    output JSON.

    E.g. `snapshot_path_for("Downloads/x/ziz.json", 2900)` →
    `PosixPath('Downloads/x/ziz_2900.json')`.

    Uses `Path.stem` so filenames with dots (`ziz.dance.v2.json`) get
    the count appended before the LAST suffix only. Always returns
    `.json` extension regardless of the input's extension (or absence
    thereof), so a caller that passes a stem-only path still gets a
    canonical snapshot name.
    """
    output = Path(output_path)
    parent = output.parent
    stem = output.stem
    return parent / f"{stem}_{count}.json"
