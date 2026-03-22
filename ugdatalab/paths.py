from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
LAB01_DIR = REPO_ROOT / "labs" / "01"
REPORT_DIR = LAB01_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"
LAB02_DIR = REPO_ROOT / "labs" / "02"
COURSE_MATERIALS_DIR = REPO_ROOT / "course_materials_sp2026"
CONTINUUM_PIXELS_PATH = COURSE_MATERIALS_DIR / "labs" / "lab_2" / "continuum_wavelengths.npz"


def ensure_output_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
