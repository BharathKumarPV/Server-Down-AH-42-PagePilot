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
    app.setApplicationVersion("0.2.0")
    app.setOrganizationName("PagePilot")

    apply_theme(app)

    print("Loading PagePilot components...", flush=True)
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    print("PagePilot Desktop is now open! (Check your taskbar or press Alt+Tab if behind terminal)", flush=True)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

