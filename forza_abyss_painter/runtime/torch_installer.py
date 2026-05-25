"""On-demand PyTorch runtime installer for the EXE's local GPU shape-gen.

The main EXE ships at ~320 MB (no torch). When the user clicks "Generate
shapes locally" for the first time, this module:

  1. Downloads python.org's embeddable Python 3.11 zip (~25 MB) to
     %LOCALAPPDATA%/ForzaAbyssPainter/runtime/python311/
  2. Bootstraps pip into that embedded Python
  3. Uses that pip to install torch + numpy from the PyTorch CUDA wheel
     index (~2 GB) into the embedded Python's site-packages
  4. Writes a marker file with the installed version + verification
     (torch.cuda.is_available() must return True for the install to count
     as successful)

Subsequent runs reuse the cached install. Inject-only users never pay the
2 GB cost — they don't trigger this code path.

Why a subprocess-isolated embedded Python instead of in-process pip-install?
PyInstaller-bundled EXEs have a frozen import system that doesn't accept
new packages at runtime, and even if it did, torch + CUDA DLLs need careful
LD_LIBRARY_PATH / PATH manipulation that's hostile to the main app's
process state. Spawning the embedded Python as a worker subprocess (via
`torch_runner.py` — separate module) isolates the torch dependency
completely from the main EXE.

This module is the INSTALL side. `torch_runner.py` is the RUN side.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


# Versions pinned for reproducibility. Bump deliberately when picking up new
# torch/CUDA releases — bundle size + CUDA driver compatibility both change.
EMBED_PYTHON_VERSION = "3.11.9"
EMBED_PYTHON_URL = (
    f"https://www.python.org/ftp/python/{EMBED_PYTHON_VERSION}/"
    f"python-{EMBED_PYTHON_VERSION}-embed-amd64.zip"
)
TORCH_VERSION = "2.4.1"
# PyTorch CUDA index URL for the matching wheel set. The bare version
# (without +cu121) installs from CPU-only PyPI; this index serves the
# CUDA-enabled wheels.
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def runtime_root() -> Path:
    """Return the per-user runtime directory. Created if missing.

    Windows: %LOCALAPPDATA%/ForzaAbyssPainter/runtime/
    macOS:   ~/Library/Application Support/ForzaAbyssPainter/runtime/
    Linux:   $XDG_DATA_HOME/ForzaAbyssPainter/runtime/ (or ~/.local/share/...)

    On non-Windows platforms this exists for testing only — the EXE is
    Windows-only and the runtime download only fires on Windows in prod.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or
                    Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or
                    Path.home() / ".local" / "share")
    root = base / "ForzaAbyssPainter" / "runtime"
    root.mkdir(parents=True, exist_ok=True)
    return root


def embedded_python_dir() -> Path:
    """Path to the embedded Python directory (created by install_runtime).
    Doesn't necessarily exist yet — call is_runtime_installed() first."""
    return runtime_root() / "python311"


def embedded_python_exe() -> Path:
    """Path to the embedded Python interpreter executable. On Windows this
    is python.exe in the embed dir; on other platforms it's a placeholder
    that won't actually run (those platforms don't use this runtime)."""
    suffix = ".exe" if sys.platform == "win32" else ""
    return embedded_python_dir() / f"python{suffix}"


def runtime_marker() -> Path:
    """Path to the JSON marker file written when install completes.
    Presence + valid contents = runtime is ready to use."""
    return runtime_root() / "installed.json"


@dataclass(frozen=True)
class RuntimeInfo:
    """Captured state of an installed runtime. Written to runtime_marker()
    when install_runtime() completes successfully so subsequent launches
    can verify the install without re-running pip."""
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_device_name: str   # "" if cuda_available is False
    installed_at_utc: str   # ISO 8601

    def to_dict(self) -> dict:
        return {
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "installed_at_utc": self.installed_at_utc,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RuntimeInfo":
        return cls(
            python_version=str(d.get("python_version", "")),
            torch_version=str(d.get("torch_version", "")),
            cuda_available=bool(d.get("cuda_available", False)),
            cuda_device_name=str(d.get("cuda_device_name", "")),
            installed_at_utc=str(d.get("installed_at_utc", "")),
        )


def installed_runtime_info() -> RuntimeInfo | None:
    """Read the install marker, return parsed RuntimeInfo, or None if no
    valid install is present. Used by the GUI to decide whether to show
    "Install GPU runtime (~2 GB download)" vs "Generate locally" buttons."""
    marker = runtime_marker()
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        info = RuntimeInfo.from_dict(data)
        # Also sanity-check the python executable still exists — user might
        # have deleted the runtime dir between sessions.
        if not embedded_python_exe().exists():
            return None
        return info
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return None


def is_runtime_installed() -> bool:
    """Cheap check: is a usable runtime present? False if marker is missing
    OR the embedded Python binary is gone OR the marker says CUDA isn't
    available (which means a partial / broken install)."""
    info = installed_runtime_info()
    if info is None:
        return False
    return info.cuda_available


def estimated_download_bytes() -> int:
    """Rough size of what install_runtime() will download. Surfaced in the
    GUI's "first-run download" confirmation prompt. Numbers are approximate:
        - Embedded Python 3.11.9 amd64: ~10 MiB
        - get-pip.py: ~2 MiB
        - torch 2.4.1+cu121: ~2.4 GiB
        - numpy + nvidia-* dep wheels: ~1.5 GiB cumulative
    Total fence-post: ~4 GiB. Confirmation prompt should round up.
    """
    return 4 * (1 << 30)   # 4 GiB
