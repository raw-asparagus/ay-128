from pathlib import Path

import numpy as np
import pandas as pd

from ugdatalab.models.cache import _cache_stable

_MIST_DIR = Path(__file__).resolve().parents[1] / "data" / "MIST_v2.5_vvcrit0.0_full_isos"
_MIST_ALPHA_FE = 0.0
_MIST_VVCRIT = 0.0
_MIST_REQUIRED_COLUMNS = (
    "EEP",
    "log10_isochrone_age_yr",
    "initial_mass",
    "star_mass",
    "log_Teff",
    "log_g",
    "phase",
)


def _mist_member_path(feh: float) -> Path:
    """Resolve one metallicity member from the extracted MIST directory."""
    feh_scaled = int(round(abs(feh) * 100))
    alpha_scaled = int(round(abs(_MIST_ALPHA_FE) * 10))

    feh_tag = f"{'m' if feh < 0 else 'p'}{feh_scaled:03d}"
    alpha_tag = f"{'m' if _MIST_ALPHA_FE < 0 else 'p'}{alpha_scaled:d}"
    member_name = (
        f"feh_{feh_tag}_afe_{alpha_tag}_vvcrit{_MIST_VVCRIT:.1f}_full.iso"
    )

    return _MIST_DIR / member_name


def _read_mist_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("# EEP"):
                return line[2:].split()


def _load_mist_age_block(path: Path, age_gyr: float) -> pd.DataFrame:
    """Scan one extracted MIST file and return the nearest available age block."""
    columns = _read_mist_columns(path)
    col_indices = {name: columns.index(name) for name in _MIST_REQUIRED_COLUMNS}
    age_col_idx = columns.index("log10_isochrone_age_yr")

    target_log_age = np.log10(age_gyr * 1e9)
    best_rows = []
    best_distance = np.inf
    current_log_age = None
    current_rows = []

    with path.open("rb") as handle:
        for line in handle:
            if not line.strip() or line.startswith(b"#"):
                continue

            parts = line.split()
            log_age = float(parts[age_col_idx])

            if current_log_age is None:
                current_log_age = log_age
            elif log_age != current_log_age:
                distance = abs(current_log_age - target_log_age)
                if distance < best_distance:
                    best_rows = current_rows
                    best_distance = distance
                current_log_age = log_age
                current_rows = []

            current_rows.append(
                {
                    "EEP": int(parts[col_indices["EEP"]]),
                    "log10_isochrone_age_yr": float(
                        parts[col_indices["log10_isochrone_age_yr"]]
                    ),
                    "initial_mass": float(parts[col_indices["initial_mass"]]),
                    "star_mass": float(parts[col_indices["star_mass"]]),
                    "log_Teff": float(parts[col_indices["log_Teff"]]),
                    "log_g": float(parts[col_indices["log_g"]]),
                    "phase": float(parts[col_indices["phase"]]),
                }
            )

    if current_rows:
        distance = abs(current_log_age - target_log_age)
        if distance < best_distance:
            best_rows = current_rows

    return pd.DataFrame(best_rows)


@_cache_stable(module="ugdatalab.isochrones")
def _get_mist_isochrone(age_gyr: float, feh: float) -> pd.DataFrame:
    """Load one MIST isochrone directly from extracted local MIST files.

    This reader avoids the external ``isochrones`` package entirely. It works
    directly with the extracted MIST ``.iso`` files by:

    1. resolving the requested metallicity member in the local data directory,
    2. scanning the file for contiguous age blocks, and
    3. returning the nearest available age block.

    Parameters
    ----------
    age_gyr : float
        Requested stellar age in Gyr.
    feh : float
        Requested metallicity [Fe/H] in dex. This must match a metallicity
        available in the local archive naming scheme.

    Returns
    -------
    pd.DataFrame
        One isochrone with columns including ``Teff``, ``logg``,
        ``initial_mass``, ``star_mass``, ``logTeff``, and ``phase``.

    Notes
    -----
    The official MIST files store a discrete age grid. This loader returns the
    nearest available isochrone in that grid rather than interpolating between
    ages.
    """
    member_path = _mist_member_path(feh)
    isochrone = _load_mist_age_block(member_path, age_gyr)

    isochrone["Teff"] = np.power(10.0, isochrone["log_Teff"].to_numpy())
    isochrone["logg"] = isochrone["log_g"]
    isochrone["logTeff"] = isochrone["log_Teff"]
    isochrone["feh"] = float(feh)
    isochrone["alpha_fe"] = _MIST_ALPHA_FE

    isochrone = isochrone.sort_values("initial_mass").reset_index(drop=True)
    return isochrone[
        [
            "EEP",
            "log10_isochrone_age_yr",
            "initial_mass",
            "star_mass",
            "Teff",
            "logg",
            "logTeff",
            "log_Teff",
            "log_g",
            "phase",
            "feh",
            "alpha_fe",
        ]
    ]
