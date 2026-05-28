"""Modal dialog shown before the inject worker starts. User picks which
vinyl-group template they have loaded in FH6 (10/20/50/100/500/1000/1500/3000
spheres, or a custom 1-3000 value), or leaves it on Auto and lets the heap
scan try a list of common sizes.

Painter parity: painter v1.6 makes the user enter the template count
explicitly. That's how it skips searching for wrong sizes in the heap. We
default to Auto (current behavior — scan for the JSON shape count plus
common larger sizes) so existing users see no UX change, but exposing the
explicit picker lets advanced users cut the scan time dramatically by
narrowing to the one size they actually have loaded.

UX rules enforced here:
  - Custom field only enabled when user picks "Custom"
  - Custom value validated as int in [1, 3000]
  - JSON shape count is shown for context — if user picks a template size
    smaller than the JSON shape count, the OK button is disabled with an
    inline warning (writes would overflow)
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIntValidator
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QVBoxLayout,
)


# Common FH6 vinyl-group template sizes. These are the EXACT counts the
# game's "Create Vinyl Group" templates offer. Anything off this list is
# probably custom-built and goes through the Custom path.
STANDARD_TEMPLATE_SIZES: tuple[int, ...] = (10, 20, 50, 100, 500, 1000, 1500, 1800, 3000)

# Hard ceiling on custom values. FH6 won't accept > 3000 layers per group.
CUSTOM_MIN = 1
CUSTOM_MAX = 3000


class TemplateSizePickerDialog(QDialog):
    """Pre-inject template selector. Returns either an int (specific size)
    or None (Auto mode — let the worker try common sizes) via
    `selected_template_size` after the user accepts.
    """

    AUTO_LABEL = "Auto-detect (try common sizes — slower if wrong)"
    CUSTOM_LABEL = "Custom (1–3000)"

    def __init__(self, parent=None, json_shape_count: int = 0) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select FH6 vinyl-group template")
        self.setModal(True)
        self.setMinimumWidth(480)
        self._json_shape_count = json_shape_count
        self._selected: int | None = None     # None = Auto, else the chosen int

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # Header
        header = QLabel("Which vinyl-group template did you load in FH6?")
        hf = QFont(); hf.setBold(True); hf.setPointSize(12)
        header.setFont(hf)
        root.addWidget(header)

        # Context: JSON shape count
        ctx = QLabel(
            f"Your JSON has {json_shape_count} shape{'s' if json_shape_count != 1 else ''}. "
            f"The template must have at least that many slots."
        )
        ctx.setStyleSheet("color: #888; font-size: 11px;")
        ctx.setWordWrap(True)
        root.addWidget(ctx)

        # Form: dropdown + custom field
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.combo = QComboBox(self)
        self.combo.addItem(self.AUTO_LABEL, userData=None)
        for size in STANDARD_TEMPLATE_SIZES:
            self.combo.addItem(f"{size} spheres", userData=size)
        self.combo.addItem(self.CUSTOM_LABEL, userData="custom")
        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        form.addRow("Template:", self.combo)

        self.custom_field = QLineEdit(self)
        self.custom_field.setPlaceholderText("1–3000")
        self.custom_field.setValidator(QIntValidator(CUSTOM_MIN, CUSTOM_MAX, self))
        self.custom_field.setEnabled(False)
        self.custom_field.textChanged.connect(self._update_ok_state)
        form.addRow("Custom value:", self.custom_field)
        root.addLayout(form)

        # Inline warning area — visible only when input is invalid (eg
        # template size < JSON shape count, or custom field empty/out-of-range).
        self.warn_box = QFrame(self)
        self.warn_box.setStyleSheet(
            "QFrame { background: #2a1414; border: 1px solid #b03030; border-radius: 4px; }"
            "QLabel { color: #ff8080; padding: 6px; font-size: 11px; }"
        )
        warn_layout = QVBoxLayout(self.warn_box)
        warn_layout.setContentsMargins(0, 0, 0, 0)
        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        warn_layout.addWidget(self.warn_label)
        self.warn_box.setVisible(False)
        root.addWidget(self.warn_box)

        # Buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self,
        )
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self._update_ok_state()

    # ---------------------------------------------------------- selection state

    @property
    def selected_template_size(self) -> int | None:
        """Returns the user's choice: None for Auto, else the specific size.
        Only valid after the dialog accepts (exec() returns QDialog.Accepted)."""
        return self._selected

    # ---------------------------------------------------------- UI internals

    def _on_combo_changed(self, _index: int) -> None:
        is_custom = self.combo.currentData() == "custom"
        self.custom_field.setEnabled(is_custom)
        if is_custom:
            self.custom_field.setFocus()
        self._update_ok_state()

    def _update_ok_state(self) -> None:
        """Enable OK only when the choice is valid AND can hold the JSON shapes."""
        size = self._current_choice()
        ok_button = self.buttons.button(QDialogButtonBox.Ok)
        # Custom mode with empty / out-of-range field = invalid
        if self.combo.currentData() == "custom" and size is None:
            self._show_warning("Enter a number between 1 and 3000 for the custom template size.")
            ok_button.setEnabled(False)
            return
        # Specific size smaller than JSON shape count = overflow
        if size is not None and size < self._json_shape_count:
            self._show_warning(
                f"Template size {size} is smaller than your JSON's {self._json_shape_count} "
                f"shapes. Load a larger template in FH6, OR pick a different template size."
            )
            ok_button.setEnabled(False)
            return
        # Auto = always valid (worker will fall back to common sizes ≥ JSON count)
        self._hide_warning()
        ok_button.setEnabled(True)

    def _current_choice(self) -> int | None:
        """Translate the current combo + custom-field state to either None
        (Auto) or a specific int. Returns None for Auto AND for invalid
        custom input; caller distinguishes via combo.currentData()."""
        data = self.combo.currentData()
        if data is None:
            return None   # Auto
        if data == "custom":
            text = self.custom_field.text().strip()
            if not text:
                return None
            try:
                value = int(text)
            except ValueError:
                return None
            if value < CUSTOM_MIN or value > CUSTOM_MAX:
                return None
            return value
        # Standard size — data IS the int
        return int(data)

    def _show_warning(self, message: str) -> None:
        self.warn_label.setText(message)
        self.warn_box.setVisible(True)

    def _hide_warning(self) -> None:
        self.warn_box.setVisible(False)

    def _on_accept(self) -> None:
        """Lock in the choice (or stay None for Auto) and close the dialog."""
        data = self.combo.currentData()
        if data is None:
            self._selected = None    # Auto
        elif data == "custom":
            self._selected = self._current_choice()    # validated to int by _update_ok_state
        else:
            self._selected = int(data)
        self.accept()
