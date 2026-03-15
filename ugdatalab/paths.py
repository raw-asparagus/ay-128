from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
LAB01_DIR = REPO_ROOT / "labs" / "01"
REPORT_DIR = LAB01_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"


def ensure_output_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
