"""Local GPU shape-generation workflow dialog.

Phase 2 of EXE GPU bundle (task #62) — the UI scaffolding. Phase 3 will
wire the Generate button to subprocess-invoke the embedded-Python runtime
that runs `forza_abyss_painter.shapegen.gpu.engine.run_gpu`. For now the
button reports "Phase 3 not yet shipped."

UX flow:
  1. User picks source image (PNG/JPG) via file picker
  2. User picks preset (Lineart 400 / Headshot 700 / Medium 1000 / HiRes 3000)
  3. Dialog shows estimated peak VRAM vs free VRAM (probe + warning if tight)
  4. Generate button enabled when source + preset both set
  5. Progress bar + cancel during run
  6. On success: auto-load the generated JSON into the main window's
     preview panel (caller wires this via the dialog's `output_path`
     attribute after exec returns Accepted)

Cancel mid-run cleanly terminates the subprocess (Phase 3 plumbing).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton, QVBoxLayout,
)


# Local-GPU preset table. Mirrors the Colab notebook lineup but with
# defaults tuned for consumer cards. Each entry: label, num_shapes,
# max_resolution, random_samples, est_peak_vram_gib.
# Tuned conservative — consumer cards often share VRAM with FH6 or
# other apps. Estimates assume bbox-local scoring (the production path).
LOCAL_PRESETS: list[dict] = [
    {
        "label": "Lineart — 400 shapes",
        "num_shapes": 400, "max_resolution": 480,
        "random_samples": 4096, "est_peak_vram_gib": 2.5,
        "description": "Logos, kanji, line art. Fast (~2 min on 30/40-series).",
    },
    {
        "label": "Headshot — 700 shapes",
        "num_shapes": 700, "max_resolution": 600,
        "random_samples": 6144, "est_peak_vram_gib": 3.5,
        "description": "Portraits. Balanced quality and speed.",
    },
    {
        "label": "Medium — 1000 shapes",
        "num_shapes": 1000, "max_resolution": 720,
        "random_samples": 8192, "est_peak_vram_gib": 5.0,
        "description": "General-purpose. Recommended default for 8+ GiB cards.",
    },
    {
        "label": "Hi-Res — 3000 shapes (FH6 closed only)",
        "num_shapes": 3000, "max_resolution": 1000,
        "random_samples": 12288, "est_peak_vram_gib": 12.0,
        "description": "Maximum detail. Needs 16+ GiB free — close FH6 first.",
    },
]


class GenerateLocallyDialog(QDialog):
    """Modal: pick source + preset, run shape-gen in a worker thread,
    return path to generated JSON via `self.output_path` after Accepted.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate shapes locally (GPU)")
        self.setModal(True)
        self.setMinimumWidth(620)

        self.source_path: Path | None = None
        self.output_path: Path | None = None     # set on successful run
        self._selected_preset_idx = 0            # default: Lineart

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # Header
        hdr = QLabel("Generate shapes locally")
        hf = QFont(); hf.setBold(True); hf.setPointSize(13)
        hdr.setFont(hf)
        root.addWidget(hdr)

        sub = QLabel(
            "Runs the GPU shape-generator on your local CUDA card. Output JSON "
            "loads automatically into the preview after generation."
        )
        sub.setStyleSheet("color: #999;")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # Form: source image picker, preset dropdown, output location.
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # Source image row.
        src_row = QHBoxLayout()
        self.source_field = QLineEdit(self)
        self.source_field.setPlaceholderText("Pick a PNG or JPG…")
        self.source_field.setReadOnly(True)
        src_row.addWidget(self.source_field)
        self.source_browse_btn = QPushButton("Browse…", self)
        self.source_browse_btn.clicked.connect(self._on_browse_source)
        src_row.addWidget(self.source_browse_btn)
        src_wrap = QFrame(self)
        src_wrap.setLayout(src_row)
        form.addRow("Source image:", src_wrap)

        # Preset dropdown.
        self.preset_combo = QComboBox(self)
        for p in LOCAL_PRESETS:
            self.preset_combo.addItem(
                f"{p['label']}  (~{p['est_peak_vram_gib']:.1f} GiB peak)",
                userData=p,
            )
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        form.addRow("Preset:", self.preset_combo)

        # Output location (defaults to source dir, same stem + _N.json).
        self.output_field = QLineEdit(self)
        self.output_field.setPlaceholderText("(defaults to source folder)")
        form.addRow("Output to:", self.output_field)

        root.addLayout(form)

        # Preset description box. Created BEFORE vram_info so the order of
        # widget addition reads top-to-bottom in the dialog layout. The
        # initial _on_preset_changed() populate call comes LAST (after all
        # widgets that the refresh touches are constructed) to avoid an
        # init-order AttributeError on self.vram_info.
        self.preset_desc = QLabel("")
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setStyleSheet(
            "background: #1a0a1f; border: 1px solid #3a2555; border-radius: 4px; "
            "padding: 8px; color: #cccccc; font-size: 11px;"
        )
        root.addWidget(self.preset_desc)

        # VRAM probe + warning area. Populated by _refresh_vram_estimate
        # at construction and every preset change.
        self.vram_info = QLabel("")
        self.vram_info.setWordWrap(True)
        self.vram_info.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(self.vram_info)

        # Both target widgets exist now — safe to populate.
        self._on_preset_changed(0)

        # Progress (hidden until Generate clicked).
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.status_label.setVisible(False)
        root.addWidget(self.status_label)

        # Buttons.
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)
        self.generate_btn = QPushButton("Generate", self)
        self.generate_btn.setDefault(True)
        self.generate_btn.setEnabled(False)   # gated on source-picked
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        btn_row.addWidget(self.generate_btn)
        root.addLayout(btn_row)

    # ----------------------------------------------------- ui event handlers

    def _on_browse_source(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Pick a source image",
            "", "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if path:
            self.source_path = Path(path)
            self.source_field.setText(path)
            self.generate_btn.setEnabled(True)
            # Suggest default output path next to source.
            if not self.output_field.text():
                preset = self.preset_combo.currentData()
                stem = self.source_path.stem
                suggested = (self.source_path.parent /
                             f"{stem}_{preset['num_shapes']}.json")
                self.output_field.setPlaceholderText(str(suggested))

    def _on_preset_changed(self, idx: int) -> None:
        self._selected_preset_idx = idx
        preset = self.preset_combo.currentData()
        if preset:
            self.preset_desc.setText(
                f"<b>{preset['label']}</b><br>"
                f"{preset['description']}<br><br>"
                f"<b>Settings:</b> "
                f"max_resolution={preset['max_resolution']}, "
                f"random_samples={preset['random_samples']}, "
                f"estimated peak VRAM: {preset['est_peak_vram_gib']:.1f} GiB"
            )
            self.preset_desc.setTextFormat(Qt.RichText)
        self._refresh_vram_estimate()
        # If source already picked, refresh the suggested output path.
        if self.source_path and preset:
            stem = self.source_path.stem
            suggested = (self.source_path.parent /
                         f"{stem}_{preset['num_shapes']}.json")
            self.output_field.setPlaceholderText(str(suggested))

    def _refresh_vram_estimate(self) -> None:
        """Probe free VRAM via the runtime (if installed) and compare to the
        preset's estimated peak. Surface a clear OK/tight/risky label."""
        preset = self.preset_combo.currentData()
        if not preset:
            self.vram_info.setText("")
            return
        # Phase 2: VRAM probe needs the installed runtime to import torch.
        # Phase 3 will subprocess-call torch_runner to do the probe. For
        # now show the estimated peak only, with a generic recommendation.
        est = preset["est_peak_vram_gib"]
        self.vram_info.setText(
            f"Estimated peak VRAM: <b>{est:.1f} GiB</b>. "
            f"Make sure your card has at least that much FREE (close FH6 + "
            f"other GPU apps if tight). Phase 3 will probe your card "
            f"directly and warn before launch."
        )
        self.vram_info.setTextFormat(Qt.RichText)

    def _on_generate_clicked(self) -> None:
        """Phase 3 stub. Real implementation: lock the form, show progress,
        spawn a QThread worker that subprocess-invokes the embedded-Python
        runtime running run_gpu with the chosen preset's settings, streams
        progress through a Qt signal."""
        self.source_browse_btn.setEnabled(False)
        self.preset_combo.setEnabled(False)
        self.output_field.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setText("Abort")
        self.progress.setVisible(True)
        self.status_label.setVisible(True)
        self.status_label.setText(
            "(Phase 2 scaffolding — Phase 3 will plug in the embedded-Python "
            "subprocess runner. Generation does not actually run yet.)"
        )
        # Don't auto-close — let the user dismiss after reading the stub note.


def open_generate_dialog_if_runtime_ready(parent) -> Path | None:
    """Convenience entry point used by the Tools menu. Checks runtime is
    installed; if not, prompts the install flow; if so (or after install
    completes), opens the generate dialog. Returns the generated JSON
    path on success, None on cancel/skip."""
    from forza_abyss_painter.gui.runtime_install_dialog import (
        prompt_install_or_use_existing,
    )
    if not prompt_install_or_use_existing(parent):
        return None
    dlg = GenerateLocallyDialog(parent)
    if dlg.exec() == QDialog.Accepted and dlg.output_path:
        return dlg.output_path
    return None
