"""
PagePilot – Setup & Verification Script for Isolated OCR Worker Environment.

Detects Python 3.12, creates the .venv-ocr/ isolated environment, installs
pinned dependencies from requirements-ocr.txt, and verifies IPC communication.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
VENV_OCR_DIR = ROOT_DIR / ".venv-ocr"
REQUIREMENTS_OCR = ROOT_DIR / "requirements-ocr.txt"


def find_python_312() -> str | None:
    """Check for system-installed Python 3.12 executable."""
    # 1. py launcher
    try:
        res = subprocess.run(["py", "-3.12", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass

    # 2. Standard Windows install path
    candidates = [
        Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python312" / "python.exe",
        Path("C:/Python312/python.exe"),
        Path.home() / "AppData" / "Roaming" / "uv" / "python" / "cpython-3.12" / "python.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    return None


def main():
    print("=" * 70)
    print("PAGEPILOT STAGE 6B: OCR WORKER ENVIRONMENT SETUP")
    print("=" * 70)

    py312 = find_python_312()
    if not py312:
        print("[!] Python 3.12 was not detected on this system.")
        print("    PaddleOCR and PaddlePaddle wheels require Python 3.12 on Windows.")
        print("    To install Python 3.12 without touching your Python 3.14 main environment:")
        print("    Run in PowerShell:")
        print("        winget install Python.Python.3.12 --scope user")
        print("\n    The application will continue operating with its integrated simulation worker.")
        return 1

    print(f"[✓] Found Python 3.12 at: {py312}")

    # Create .venv-ocr if not exists
    py_bin = VENV_OCR_DIR / "Scripts" / "python.exe"
    if not py_bin.exists():
        print(f"[*] Creating isolated virtual environment at {VENV_OCR_DIR}...")
        subprocess.run([py312, "-m", "venv", str(VENV_OCR_DIR)], check=True)
        print("[✓] Virtual environment created.")
    else:
        print(f"[✓] Virtual environment already exists at {VENV_OCR_DIR}.")

    # Install pinned requirements
    if REQUIREMENTS_OCR.exists():
        print(f"[*] Installing pinned OCR requirements from {REQUIREMENTS_OCR.name}...")
        subprocess.run([str(py_bin), "-m", "pip", "install", "-r", str(REQUIREMENTS_OCR)], check=True)
        print("[✓] Pinned requirements successfully installed.")

    # Test worker IPC handshake
    print("[*] Testing worker communication protocol...")
    from pagepilot.ocr.worker_client import OCRWorkerClient
    client = OCRWorkerClient(python_path=str(py_bin))
    try:
        client.start()
        pong = client.ping()
        print(f"[✓] Worker responded successfully: {pong.get('status')}")
        print(f"    Info: {pong.get('payload')}")
        client.stop()
        print("[✓] Isolated OCR worker setup verified and ready.")
        return 0
    except Exception as exc:
        print(f"[!] Worker communication test failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

