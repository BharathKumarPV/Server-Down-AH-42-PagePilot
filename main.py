"""
PagePilot – Application entry point.

Usage:
    python main.py
"""

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from pagepilot.theme import apply_theme
from pagepilot.ui.main_window import MainWindow


def main() -> int:
    # High-DPI support (PySide6 ≥ 6.x handles this automatically; kept for clarity)
    app = QApplication(sys.argv)
    app.setApplicationName("PagePilot")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("PagePilot")

    apply_theme(app)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

