from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
FIGURES_DIR = _HERE / "labs" / "02" / "figures"


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
