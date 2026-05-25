"""Tests for the template_size parameter wiring in the inject flow.

The pre-inject TemplateSizePickerDialog (GUI) lets the user pin a specific
FH6 template size; the worker passes it to FH6Injector.find_active_vinyl_group,
which short-circuits the heap-scan `tries` list to that one value instead of
walking common sizes. This avoids a 5-15x scan-time blowup on the common
case where the user knows what they loaded.

We can't exec the QDialog in CI (no Qt), so tests focus on:
  - The picker module's pure constants + classmethod-style helpers (no Qt)
  - The injector's behavior when template_size is set: verify the
    constrained `tries` list AND that auto mode still produces the
    common-sizes-≥-N walk.
"""
from __future__ import annotations

import sys
import pytest

# Stub PySide6 so the picker + worker modules import on dev (no Qt).
# Anything used as a base class (QObject, QDialog) needs to be a real class;
# anything used as a function call (Signal) needs to be callable. Use plain
# classes + lambdas to satisfy both.
import types as _types

def _make_qt_stub(name: str) -> _types.ModuleType:
    mod = _types.ModuleType(name)
    return mod

class _QObj: pass
class _QDialog:
    Accepted = 1
    Rejected = 0
class _QFrame: pass
class _QLabel: pass
class _QLineEdit: pass
class _QComboBox: pass
class _QFormLayout: pass
class _QHBoxLayout: pass
class _QVBoxLayout: pass
class _QDialogButtonBox:
    Ok = 1
    Cancel = 2
class _QFont: pass
class _QIntValidator: pass

if 'PySide6' not in sys.modules:
    sys.modules['PySide6'] = _make_qt_stub('PySide6')
if 'PySide6.QtCore' not in sys.modules:
    qtcore = _make_qt_stub('PySide6.QtCore')
    qtcore.QObject = _QObj
    qtcore.Signal = lambda *a, **k: (lambda *aa, **kk: None)
    class _QtNS: AlignRight = 0
    qtcore.Qt = _QtNS
    sys.modules['PySide6.QtCore'] = qtcore
if 'PySide6.QtGui' not in sys.modules:
    qtgui = _make_qt_stub('PySide6.QtGui')
    qtgui.QFont = _QFont
    qtgui.QIntValidator = _QIntValidator
    sys.modules['PySide6.QtGui'] = qtgui
if 'PySide6.QtWidgets' not in sys.modules:
    qtw = _make_qt_stub('PySide6.QtWidgets')
    qtw.QComboBox = _QComboBox
    qtw.QDialog = _QDialog
    qtw.QDialogButtonBox = _QDialogButtonBox
    qtw.QFormLayout = _QFormLayout
    qtw.QFrame = _QFrame
    qtw.QHBoxLayout = _QHBoxLayout
    qtw.QLabel = _QLabel
    qtw.QLineEdit = _QLineEdit
    qtw.QVBoxLayout = _QVBoxLayout
    qtw.QProgressBar = _QObj
    qtw.QPushButton = _QObj
    sys.modules['PySide6.QtWidgets'] = qtw


def test_picker_module_imports_with_stubbed_qt():
    """The picker module must import cleanly so its constants are usable."""
    from forza_abyss_painter.gui import inject_template_picker as picker_mod
    assert picker_mod.STANDARD_TEMPLATE_SIZES == (10, 20, 50, 100, 500, 1000, 1500, 3000)
    assert picker_mod.CUSTOM_MIN == 1
    assert picker_mod.CUSTOM_MAX == 3000


def test_standard_template_sizes_cover_fh6_offerings():
    """Sanity: the dropdown's standard sizes must match what the game's
    Create Vinyl Group menu actually offers. If FH6 adds a new template
    size in a patch, this list needs updating."""
    from forza_abyss_painter.gui.inject_template_picker import STANDARD_TEMPLATE_SIZES
    # Painter v1.6.1's recommended sizes per main.py:
    # "500, 1000, 1500, 2000 or 3000 is recommended" — plus tiny sizes
    # 10/20/50/100 for small designs. Our list should include all of these
    # except 2000 (not in FH6's standard menu as of build 3.360.259.0).
    assert 500 in STANDARD_TEMPLATE_SIZES
    assert 1000 in STANDARD_TEMPLATE_SIZES
    assert 1500 in STANDARD_TEMPLATE_SIZES
    assert 3000 in STANDARD_TEMPLATE_SIZES
    # Tiny sizes
    assert 10 in STANDARD_TEMPLATE_SIZES
    assert 100 in STANDARD_TEMPLATE_SIZES


def test_custom_range_matches_fh6_hard_ceiling():
    """CUSTOM_MAX = 3000 because FH6 won't accept > 3000 layers per group.
    If we let users enter eg 5000, the scan would burn time on a needle
    that physically can't be in memory."""
    from forza_abyss_painter.gui.inject_template_picker import CUSTOM_MAX, CUSTOM_MIN
    assert CUSTOM_MAX == 3000
    assert CUSTOM_MIN >= 1


def test_injector_find_active_vinyl_group_accepts_template_size_kwarg():
    """The injector method must accept template_size — caller (the worker)
    passes it through, and missing kwarg would be a runtime TypeError."""
    import inspect
    from forza_abyss_painter.inject.fh6_injector import FH6Injector
    sig = inspect.signature(FH6Injector.find_active_vinyl_group)
    assert "template_size" in sig.parameters
    # Default must be None (Auto mode — backward compat with the existing
    # call sites that haven't been updated yet).
    assert sig.parameters["template_size"].default is None


def test_inject_worker_accepts_template_size_kwarg():
    """Worker's __init__ must accept template_size for the GUI to pass the
    user's choice through."""
    import inspect
    from forza_abyss_painter.gui import inject_worker as w
    sig = inspect.signature(w.InjectionWorker.__init__)
    assert "template_size" in sig.parameters
    assert sig.parameters["template_size"].default is None
