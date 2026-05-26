"""Dialog helpers for surfacing fd6.shapes validator findings in the GUI.

Two surfaces:

  * `show_validation_dialog(parent, issues, *, title)` — full modal with
    a scrollable issue list and a copy-to-clipboard button. Used by the
    Tools → Validate current JSON action so users can review every
    finding.
  * `summarize_for_status_bar(issues)` — one-line "X errors, Y warnings"
    summary suitable for `statusBar().showMessage()`.

The dialog is intentionally read-only: validation never auto-fixes. The
user is the only authority on whether a non-injector-safe shape is
acceptable for their workflow.
"""
from __future__ import annotations

from typing import Iterable

from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QLabel, QPlainTextEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from forza_abyss_painter.io.validator import Issue, Severity


# Severity → human-readable + sort priority (errors first so they're
# at the top of the dialog). Keep in sync with Severity enum.
_SEVERITY_LABEL = {
    Severity.ERROR:   ("ERROR",   0),
    Severity.WARNING: ("WARNING", 1),
    Severity.INFO:    ("INFO",    2),
}


def _format_issue(issue: Issue) -> str:
    label, _ = _SEVERITY_LABEL[issue.severity]
    loc = f" at {issue.path}" if issue.path else ""
    return f"[{label}] [{issue.code}]{loc}\n    {issue.message}"


def _format_issues_as_text(issues: Iterable[Issue]) -> str:
    """Sort by severity (errors first), then by path so issues on the
    same shape cluster together. Plain text — usable in the dialog or
    on the clipboard."""
    ordered = sorted(issues, key=lambda i: (_SEVERITY_LABEL[i.severity][1], i.path))
    if not ordered:
        return "No findings — document validates clean."
    return "\n\n".join(_format_issue(i) for i in ordered)


def summarize_for_status_bar(issues: list[Issue]) -> str:
    """One-line summary, used as a transient status-bar message after
    auto-validation. Returns empty string if nothing worth reporting
    (no issues OR info-only) so callers can skip the message entirely."""
    errors = sum(1 for i in issues if i.severity is Severity.ERROR)
    warnings = sum(1 for i in issues if i.severity is Severity.WARNING)
    if errors and warnings:
        return f"JSON validation: {errors} error(s), {warnings} warning(s)"
    if errors:
        return f"JSON validation: {errors} error(s)"
    if warnings:
        return f"JSON validation: {warnings} warning(s)"
    return ""   # clean or info-only — don't clutter the status bar


def show_validation_dialog(
    parent: QWidget | None,
    issues: list[Issue],
    *,
    title: str = "JSON validation findings",
) -> None:
    """Modal dialog listing every finding. Read-only; user closes when
    they're done reviewing. Always shows even when issues is empty
    (so the Tools menu action gives clear feedback either way)."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(720, 480)

    layout = QVBoxLayout(dlg)

    header = QLabel(_header_text(issues))
    header.setWordWrap(True)
    layout.addWidget(header)

    body = QPlainTextEdit()
    body.setReadOnly(True)
    body.setPlainText(_format_issues_as_text(issues))
    # Monospace makes the path indentation line up — helpful when
    # scanning a long list of shape-level issues.
    body_font = body.font()
    body_font.setFamily("Menlo, Monaco, Consolas, monospace")
    body.setFont(body_font)
    layout.addWidget(body, stretch=1)

    btn_row = QDialogButtonBox(QDialogButtonBox.Close)

    if issues:
        # Only offer the copy button when there's something to copy —
        # avoids a misleadingly-active button on a clean run.
        copy_btn = QPushButton("Copy findings")
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(_format_issues_as_text(issues))
        )
        btn_row.addButton(copy_btn, QDialogButtonBox.ActionRole)

    btn_row.rejected.connect(dlg.reject)
    btn_row.accepted.connect(dlg.accept)
    layout.addWidget(btn_row)

    dlg.exec()


def _header_text(issues: list[Issue]) -> str:
    """One-paragraph summary at the top of the dialog. Reflects severity
    counts in plain English so users don't have to count rows."""
    if not issues:
        return ("This document validates clean against fd6.shapes v1. "
                "All shapes are well-formed and injector-safe.")
    errors = sum(1 for i in issues if i.severity is Severity.ERROR)
    warnings = sum(1 for i in issues if i.severity is Severity.WARNING)
    info = sum(1 for i in issues if i.severity is Severity.INFO)
    parts = []
    if errors:
        parts.append(f"{errors} ERROR-severity finding(s) — the document "
                     "will not inject cleanly until these are fixed")
    if warnings:
        parts.append(f"{warnings} WARNING(s) — the document will load, "
                     "but some shapes may be silently skipped by the injector")
    if info:
        parts.append(f"{info} informational note(s)")
    return ". ".join(parts) + "."
