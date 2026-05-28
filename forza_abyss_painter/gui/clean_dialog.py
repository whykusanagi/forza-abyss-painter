"""One-click Tools-menu entry for the fap-clean cleanup library.

Replaces the prior workflow of "drop to a terminal, run `fap-clean
input.json -o output.json`" with a single menu click + a summary dialog
the user can confirm before writing the cleaned file. Internally calls
the same `clean_doc()` function the CLI uses — no subprocess overhead,
direct access to the report dict for the summary, identical cleanup
semantics.

UX flow:
  1. Tools → Clean current JSON… opens a file picker (JSON files)
  2. We load + clean the picked file in-memory (fast: <1 sec for 3000
     shapes per the CLI's own timing notes)
  3. A summary dialog shows the before/after shape counts + a breakdown
     of WHY shapes were dropped (padding whites vs. dead shapes vs.
     both), plus a save-as field defaulting to `<stem>_cleaned.json`
     in the source directory. User can edit the destination or cancel.
  4. On confirm the cleaned doc lands at the chosen path and the
     returned path bubbles back to the caller so MainWindow can
     auto-load the cleaned file into the preview panel.

Non-destructive by default: never overwrites the input unless the user
explicitly retypes the same path. Matches the CLI's `--in-place` opt-in.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
)

from forza_abyss_painter.cli.clean import clean_doc
from forza_abyss_painter.io.exporter import load_json, save_json
from forza_abyss_painter.io.json_schema import FD6Document


def _format_report(report: dict, src_name: str) -> str:
    """Human-readable summary lines for the confirm dialog. Counts only
    — keep it factual; no editorializing about how much faster injection
    will be (we don't know the user's downstream choice)."""
    return (
        f"<b>{src_name}</b><br><br>"
        f"Input shapes: <b>{report['input_count']}</b><br>"
        f"Output shapes: <b>{report['output_count']}</b><br>"
        f"Dropped: <b>{report['dropped_total']}</b><br><br>"
        f"<i>Breakdown</i><br>"
        f"&nbsp;&nbsp;Padding-white shapes (only): "
        f"{report['dropped_padding_whites_only']}<br>"
        f"&nbsp;&nbsp;Dead / fully occluded shapes (only): "
        f"{report['dropped_dead_shapes_only']}<br>"
        f"&nbsp;&nbsp;Both conditions: {report['dropped_both_conditions']}"
    )


class CleanJsonDialog(QDialog):
    """Summary + save-as confirmation modal. Built after `clean_doc` has
    already run so we can show real counts up front — the user is
    confirming the WRITE, not the analyze step.

    Caller pattern (see `open_clean_json_dialog` below):
        cleaned_dict, report = clean_doc(load_json(src).to_dict())
        dlg = CleanJsonDialog(parent, src, cleaned_dict, report)
        if dlg.exec() == QDialog.Accepted:
            return dlg.output_path
    """

    def __init__(
        self,
        parent,
        src_path: Path,
        cleaned_doc: dict,
        report: dict,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Clean JSON — confirm")
        self.setModal(True)
        self.setMinimumWidth(540)

        self._src_path = Path(src_path)
        self._cleaned_doc = cleaned_doc
        self.output_path: Path | None = None    # set on Accept

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        hdr = QLabel("Cleanup ready to save")
        hf = QFont(); hf.setBold(True); hf.setPointSize(13)
        hdr.setFont(hf)
        root.addWidget(hdr)

        # Counts panel — read-only summary of what clean_doc found.
        self.summary = QLabel(_format_report(report, self._src_path.name))
        self.summary.setTextFormat(Qt.RichText)
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(
            "background: #1a0a1f; border: 1px solid #3a2555; "
            "border-radius: 4px; padding: 10px; color: #cccccc;"
        )
        root.addWidget(self.summary)

        # Save-as form — defaults to the same dir + `_cleaned` suffix,
        # mirroring the CLI's default output naming. User can retype to
        # overwrite the source (matches `--in-place`).
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.output_field = QLineEdit(self)
        default_out = (
            self._src_path.parent / f"{self._src_path.stem}_cleaned.json"
        )
        self.output_field.setText(str(default_out))
        browse_btn = QPushButton("Browse…", self)
        browse_btn.clicked.connect(self._on_browse)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_field)
        out_row.addWidget(browse_btn)
        out_wrap = QFrame(self)
        out_wrap.setLayout(out_row)
        form.addRow("Save as:", out_wrap)
        root.addLayout(form)

        # Buttons. Save is default + only enabled when output field is
        # non-empty (a user clearing the field shouldn't be able to
        # submit an empty-string path).
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        self.save_btn = QPushButton("Save cleaned JSON", self)
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self.save_btn)
        root.addLayout(btn_row)

    def _on_browse(self) -> None:
        path, _flt = QFileDialog.getSaveFileName(
            self, "Save cleaned JSON as",
            self.output_field.text(), "JSON (*.json)",
        )
        if path:
            self.output_field.setText(path)

    def _on_save(self) -> None:
        """Write the cleaned doc to the chosen path. On success we set
        `self.output_path` and call accept() so the caller can chain
        the auto-load step. Failure surfaces a modal error and leaves
        the dialog open so the user can pick a different destination."""
        out_text = self.output_field.text().strip()
        if not out_text:
            QMessageBox.warning(
                self, "Path missing",
                "Pick an output path before saving.",
            )
            return
        out_path = Path(out_text)
        try:
            cleaned = FD6Document.from_dict(self._cleaned_doc)
            save_json(cleaned, out_path)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.critical(
                self, "Save failed",
                f"Couldn't write cleaned JSON:\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                f"Pick a different output path and try again.",
            )
            return
        self.output_path = out_path
        self.accept()


def open_clean_json_dialog(parent) -> Path | None:
    """Tools menu → Clean current JSON entry point. Opens a file picker
    for the input JSON, runs `clean_doc()` synchronously (fast enough
    to skip a worker thread), then shows the confirm dialog. Returns
    the path of the saved cleaned JSON on Accept, None on Cancel or
    any error during load/clean.
    """
    src, _flt = QFileDialog.getOpenFileName(
        parent, "Pick a JSON to clean", "", "JSON (*.json)",
    )
    if not src:
        return None
    src_path = Path(src)
    try:
        doc = load_json(src_path)
        cleaned_dict, report = clean_doc(doc.to_dict())
    except (OSError, ValueError, KeyError) as exc:
        QMessageBox.critical(
            parent, "Couldn't clean",
            f"Failed to load or clean {src_path.name}:\n\n"
            f"{type(exc).__name__}: {exc}",
        )
        return None
    dlg = CleanJsonDialog(parent, src_path, cleaned_dict, report)
    if dlg.exec() == QDialog.Accepted and dlg.output_path is not None:
        return dlg.output_path
    return None
