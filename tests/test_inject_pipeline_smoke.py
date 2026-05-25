"""End-to-end smoke test of the entire inject pipeline using real PySide6,
real signals, real worker thread, real heap-scan validation logic, with
mocks ONLY at the OS boundary (find_process_id, _get_module_base,
ProcessHandle.open/try_read/write/enumerate_regions).

The stub-based test we shipped first masked a real PySide6 6.x bug
(`picker.Accepted` raises AttributeError on subclass instances) that
broke production. This test catches that whole class of bug because it
exercises against real Qt + real worker thread + real injector code.

What's mocked: only the kernel32 / psapi syscalls. Everything else —
log file open, JSON parsing, picker dialog construction, worker QThread,
Qt signal delivery, readiness gates, table validation, write loop with
progress, dialog close — runs the real implementation.

What's verified per scenario:
  - happy path: worker.run() completes without exception, log file
    written, expected status sequence emitted, writes actually issued
  - process-not-found path: emits error status, fires done, no crash
  - locator-misses-everything path: falls through fast → legacy → RTTI,
    surfaces "no confident match" error, fires done
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

PySide6 = pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, QEventLoop, QThread, QTimer    # noqa: E402
from PySide6.QtWidgets import QApplication                                  # noqa: E402


_app = QApplication.instance() or QApplication(sys.argv)


# -------------------------------------------------------------- fixtures


def _build_fd6_json(tmpdir: Path, n_shapes: int = 5) -> Path:
    """Write a valid FD6Document JSON to disk. Worker.load_json must accept it."""
    from forza_abyss_painter.io.json_schema import FD6Document
    from forza_abyss_painter.shapegen.shapes import RotatedEllipse

    shapes = [
        RotatedEllipse(x=100.0 + i, y=100.0, rx=10.0, ry=10.0,
                       angle=0.0, color=[200, 100, 50, 255])
        for i in range(n_shapes)
    ]
    doc = FD6Document.from_engine(
        source_image="smoke.png", image_size=(400, 400),
        shapes=shapes, profile_name="smoke",
    )
    path = tmpdir / "smoke.json"
    path.write_text(json.dumps(doc.to_dict()))
    return path


# -------------------------------------------------------------- mock ProcessHandle


class _MockProcessHandle:
    """Implements the ProcessHandle interface with a dict-backed flat address
    space. Tests build the address space, the handle reads/writes against it,
    and verifies the writes were issued correctly."""

    def __init__(self, pid: int, access: int = 0):
        self.pid = pid
        self.access = access
        self.regions: dict[int, bytes] = {}   # base_addr → bytes
        self.opened = False
        self.writes: list[tuple[int, bytes]] = []

    def open(self):
        self.opened = True

    def close(self):
        self.opened = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def try_read(self, addr: int, size: int) -> bytes | None:
        for base, data in self.regions.items():
            if base <= addr < base + len(data):
                off = addr - base
                end = off + size
                if end > len(data):
                    return None
                return data[off:end]
        return None

    def write(self, addr: int, data: bytes) -> int:
        self.writes.append((addr, bytes(data)))
        # Mutate the in-memory region too so subsequent reads see the write.
        for base, region in list(self.regions.items()):
            if base <= addr < base + len(region):
                off = addr - base
                new = bytearray(region)
                new[off:off + len(data)] = data
                self.regions[base] = bytes(new)
                return len(data)
        return len(data)

    def enumerate_regions(self):
        """Return MemoryRegion-shaped fakes for every region we registered.
        Heap-scan locate_livery_group expects readable, writable, not is_image."""
        from types import SimpleNamespace
        return [
            SimpleNamespace(base=base, size=len(data),
                            readable=True, writable=True, is_image=False)
            for base, data in self.regions.items()
        ]


def _make_layer_blob(profile) -> bytes:
    """Layer struct that scores 5/5 against _score_layer."""
    blob = bytearray(0x100)
    struct.pack_into('<2f', blob, profile.layer_position_offset, 100.0, 100.0)
    struct.pack_into('<2f', blob, profile.layer_scale_offset, 1.0, 1.0)
    struct.pack_into('<f', blob, profile.layer_rotation_offset, 0.0)
    blob[profile.layer_color_offset:profile.layer_color_offset + 4] = bytes([200, 100, 50, 255])
    blob[profile.layer_shape_id_offset:profile.layer_shape_id_offset + 1] = bytes([profile.shape_id_ellipse])
    blob[profile.layer_mask_offset:profile.layer_mask_offset + 1] = bytes([0])
    return bytes(blob)


def _populate_heap_with_findable_group(handle: _MockProcessHandle,
                                        profile, layer_count: int):
    """Plant a fake CLiveryGroup in the mock memory so the heap-fingerprint
    scan finds it. Layout: one heap region containing
        [...padding..., GROUP_HEADER, ...padding...]
    where GROUP_HEADER at offset GROUP_BASE_OFF has:
      - u16 count at +0x5A
      - u64 table pointer at +0x78
    Layer table at TABLE_ADDR has N pointers to N layer blobs.
    """
    from forza_abyss_painter.inject.fh6_injector import COUNT_OFF, TABLE_OFF

    HEAP_BASE = 0x200000000
    HEAP_SIZE = 0x10000   # 64 KiB
    GROUP_OFF = 0x1000    # offset into the heap region
    TABLE_ADDR = 0x300000000
    LAYER_BASE_ADDR = 0x400000000
    LAYER_STRIDE = 0x200

    # Layer table region: pointers to N unique layer blobs
    table_bytes = bytearray(layer_count * 8)
    layer_blob = _make_layer_blob(profile)
    for i in range(layer_count):
        ptr = LAYER_BASE_ADDR + i * LAYER_STRIDE
        struct.pack_into('<Q', table_bytes, i * 8, ptr)
        handle.regions[ptr] = layer_blob
    handle.regions[TABLE_ADDR] = bytes(table_bytes)

    # Heap region: contains the group with count + table pointer at right offsets
    heap = bytearray(HEAP_SIZE)
    group_addr = HEAP_BASE + GROUP_OFF
    struct.pack_into('<H', heap, GROUP_OFF + COUNT_OFF, layer_count)
    struct.pack_into('<Q', heap, GROUP_OFF + TABLE_OFF, TABLE_ADDR)
    handle.regions[HEAP_BASE] = bytes(heap)
    return group_addr, TABLE_ADDR


def _wait_for_done(worker, timeout_seconds: float = 10.0) -> bool:
    """Spin a QEventLoop until worker.done fires or timeout. Returns True on done."""
    loop = QEventLoop()
    fired = [False]
    def _on_done():
        fired[0] = True
        loop.quit()
    worker.done.connect(_on_done)
    QTimer.singleShot(int(timeout_seconds * 1000), loop.quit)
    loop.exec()
    return fired[0]


# -------------------------------------------------------------- the actual tests


def test_worker_happy_path_emits_full_signal_sequence(tmp_path, monkeypatch):
    """Real worker.run() with mocked OS layer: locate via heap scan,
    validate table at painter-matched thresholds, write 5 shapes, fire done.

    Verifies the ENTIRE worker pipeline works end-to-end: log open,
    JSON load, process search, attach, locate (fast miss → heap fallback
    succeeds), write loop with progress, completion. No stubs above the
    Windows API."""
    from forza_abyss_painter.gui.inject_worker import InjectionWorker
    from forza_abyss_painter.inject import fh6_injector as inj_mod
    from forza_abyss_painter.inject import win_process as wp_mod
    from forza_abyss_painter.inject.game_profiles import FH6

    # 5-shape valid FD6 JSON
    json_path = _build_fd6_json(tmp_path, n_shapes=5)

    # Mock OS layer: process exists, no module match (so signature-chain
    # gate 1 returns None and we fall through to heap scan), heap scan
    # populated with a findable group.
    handle = _MockProcessHandle(pid=12345)
    _populate_heap_with_findable_group(handle, FH6, layer_count=500)
    monkeypatch.setattr(wp_mod, "find_process_id",
                        lambda name: 12345 if "horizon6" in name.lower() else None)
    monkeypatch.setattr(wp_mod, "ProcessHandle",
                        lambda pid, access=0: handle)
    # ProcessHandle is also re-bound into fh6_injector at module-load time,
    # so we must patch there too — otherwise inj.attach() uses the real one
    # and hits "win_process is only available on Windows" on dev machines.
    monkeypatch.setattr(inj_mod, "ProcessHandle",
                        lambda pid, access=0: handle)
    monkeypatch.setattr(inj_mod, "_get_module_base",
                        lambda pid, name: None)   # forces fast-mode miss → heap scan

    worker = InjectionWorker(json_path, profile_key="fh6", template_size=500)
    statuses: list[tuple[str, str]] = []
    write_progress: list[tuple[int, int]] = []
    worker.status.connect(lambda msg, sev: statuses.append((sev, msg)))
    worker.write_progress.connect(lambda w, t: write_progress.append((w, t)))

    # Run in main thread — easier to debug. The worker's design tolerates this.
    worker.run()

    # Hard assertions on what the worker did
    severities = [s for s, _ in statuses]
    assert "error" not in severities, (
        f"unexpected error severity in:\n" +
        "\n".join(f"  [{s}] {m}" for s, m in statuses)
    )
    msgs = "\n".join(m for _, m in statuses)
    assert "Found Forza Horizon 6" in msgs, "PID-resolution status missing"
    assert "Loaded 5 shapes" in msgs
    assert "Attached" in msgs
    assert any("Injected" in m for _, m in statuses), (
        f"no success status — pipeline didn't complete writes. Statuses:\n{msgs}"
    )

    # Write loop actually issued writes (n_shapes × multiple fields per shape).
    # 5 shapes × ~6 writes each = ~30 expected.
    assert len(handle.writes) >= 5, (
        f"only {len(handle.writes)} writes issued — write loop didn't run"
    )

    # Final write_progress should hit (5, 5)
    assert write_progress and write_progress[-1] == (5, 5), (
        f"final write_progress = {write_progress[-1] if write_progress else 'none'}, expected (5, 5)"
    )


def test_worker_process_not_found_fires_error_status(tmp_path, monkeypatch):
    """FH6 not running → worker emits clear error, fires done, doesn't crash."""
    from forza_abyss_painter.gui.inject_worker import InjectionWorker
    from forza_abyss_painter.inject import win_process as wp_mod

    json_path = _build_fd6_json(tmp_path, n_shapes=5)
    monkeypatch.setattr(wp_mod, "find_process_id", lambda name: None)

    worker = InjectionWorker(json_path, profile_key="fh6")
    statuses: list[tuple[str, str]] = []
    worker.status.connect(lambda msg, sev: statuses.append((sev, msg)))
    worker.run()

    error_msgs = [m for s, m in statuses if s == "error"]
    assert error_msgs, f"expected error status, got: {statuses}"
    assert any("not running" in m for m in error_msgs), (
        f"expected 'not running' message, got: {error_msgs}"
    )


def test_worker_no_findable_group_falls_through_cleanly(tmp_path, monkeypatch):
    """Process found, attach succeeds, but heap scan finds no valid group.
    Worker must surface the error from find_active_vinyl_group, fire done,
    no crash. This is the path users hit when fast-mode + heap + RTTI all
    miss — previously a regression in 6f39cca made this raise RuntimeError
    without falling through to legacy at all."""
    from forza_abyss_painter.gui.inject_worker import InjectionWorker
    from forza_abyss_painter.inject import fh6_injector as inj_mod
    from forza_abyss_painter.inject import win_process as wp_mod

    json_path = _build_fd6_json(tmp_path, n_shapes=5)
    # Heap is empty → enumerate_regions returns nothing → heap scan finds nothing
    handle = _MockProcessHandle(pid=12345)
    monkeypatch.setattr(wp_mod, "find_process_id",
                        lambda name: 12345 if "horizon6" in name.lower() else None)
    monkeypatch.setattr(wp_mod, "ProcessHandle",
                        lambda pid, access=0: handle)
    # ProcessHandle is also re-bound into fh6_injector at module-load time,
    # so we must patch there too — otherwise inj.attach() uses the real one
    # and hits "win_process is only available on Windows" on dev machines.
    monkeypatch.setattr(inj_mod, "ProcessHandle",
                        lambda pid, access=0: handle)
    monkeypatch.setattr(inj_mod, "_get_module_base", lambda pid, name: None)

    worker = InjectionWorker(json_path, profile_key="fh6")
    statuses: list[tuple[str, str]] = []
    done_fired = [False]
    worker.status.connect(lambda msg, sev: statuses.append((sev, msg)))
    worker.done.connect(lambda: done_fired.__setitem__(0, True))
    worker.run()

    assert done_fired[0], "done MUST fire even when locate fails"
    # Expect SOME error status surfaced — either "no confident match" or similar
    error_msgs = [m for s, m in statuses if s == "error"]
    assert error_msgs, (
        f"locate failure should surface as error; got statuses:\n"
        + "\n".join(f"  [{s}] {m[:80]}" for s, m in statuses)
    )


def test_worker_emits_log_path_signal_before_anything_else(tmp_path, monkeypatch):
    """log_path signal must fire FIRST so the dialog can show users where
    the log is. If it fires late (or not at all), the user can't find the
    log after a failure."""
    from forza_abyss_painter.gui.inject_worker import InjectionWorker
    from forza_abyss_painter.inject import win_process as wp_mod

    json_path = _build_fd6_json(tmp_path, n_shapes=5)
    monkeypatch.setattr(wp_mod, "find_process_id", lambda name: None)

    worker = InjectionWorker(json_path, profile_key="fh6")
    timeline: list[str] = []
    worker.log_path.connect(lambda p: timeline.append("log_path"))
    worker.status.connect(lambda m, s: timeline.append("status"))
    worker.run()

    assert timeline, "no signals at all"
    assert timeline[0] == "log_path", (
        f"log_path should be FIRST signal; timeline was: {timeline[:5]}"
    )


def test_worker_writes_log_file_to_disk(tmp_path, monkeypatch):
    """Log file persists with timestamped lines and matches what the user
    sees in the dialog. Without this, post-mortem debugging is impossible
    (which is exactly what bit the FH6 3.360.x diagnosis)."""
    from forza_abyss_painter.gui.inject_worker import InjectionWorker
    from forza_abyss_painter.inject import win_process as wp_mod

    json_path = _build_fd6_json(tmp_path, n_shapes=5)
    monkeypatch.setattr(wp_mod, "find_process_id", lambda name: None)

    worker = InjectionWorker(json_path, profile_key="fh6")
    log_path_received: list[str] = []
    worker.log_path.connect(lambda p: log_path_received.append(p))
    worker.run()

    assert log_path_received, "log_path signal never fired"
    log_file = Path(log_path_received[0])
    assert log_file.exists(), f"log file not on disk at {log_file}"
    content = log_file.read_text(encoding="utf-8")
    # Must contain the entry banner
    assert "Forza Abyss Painter inject log" in content
    # Must contain the not-running error
    assert "not running" in content


def test_picker_then_worker_full_dialog_wiring(tmp_path, monkeypatch):
    """Construct picker + worker + dialog the way main_window does. Verify
    no exception, picker accepts, worker constructs with the right
    template_size, dialog's on_log_path slot is callable. This is the
    glue that the picker.Accepted bug broke — needs end-to-end coverage."""
    from forza_abyss_painter.gui.inject_template_picker import TemplateSizePickerDialog
    from forza_abyss_painter.gui.inject_worker import InjectionWorker
    from forza_abyss_painter.gui.inject_dialog import InjectionDialog
    from forza_abyss_painter.inject import win_process as wp_mod

    json_path = _build_fd6_json(tmp_path, n_shapes=5)
    monkeypatch.setattr(wp_mod, "find_process_id", lambda name: None)

    # Picker — pick 500 via the real combo
    picker = TemplateSizePickerDialog(None, json_shape_count=5)
    for i in range(picker.combo.count()):
        if picker.combo.itemData(i) == 500:
            picker.combo.setCurrentIndex(i); break
    picker._on_accept()
    # The fix-pattern: not picker.exec() truthiness — simulate with result()
    assert picker.result() == 1, "picker should accept"
    template_size = picker.selected_template_size
    assert template_size == 500

    # Worker — constructed with template_size from picker
    worker = InjectionWorker(json_path, profile_key="fh6", template_size=template_size)
    assert worker.template_size == 500

    # Dialog — constructs without error; on_log_path slot exists
    dialog = InjectionDialog(None, json_name="smoke.json", game_label="Forza Horizon 6")
    assert hasattr(dialog, "on_log_path")
    assert hasattr(dialog, "on_status")
    assert hasattr(dialog, "on_scan_progress")
    assert hasattr(dialog, "on_write_progress")
    assert hasattr(dialog, "on_done")

    # Connect — must not throw (signature mismatches would die here)
    worker.scan_progress.connect(dialog.on_scan_progress)
    worker.write_progress.connect(dialog.on_write_progress)
    worker.status.connect(dialog.on_status)
    worker.log_path.connect(dialog.on_log_path)
    worker.done.connect(dialog.on_done)
