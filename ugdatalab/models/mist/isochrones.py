"""MIST v2.5 isochrone reader operating on the extracted ``.iso`` files."""

import numpy as np
import pandas as pd

from ugdatalab.data._fetch import mist_isochrones_dir
from ugdatalab.utils.cache import cache_stable

# Conversion from gigayears to years (MIST's internal unit).
_GYR_TO_YR = 1e9
_MIST_REQUIRED_COLUMNS = (
    "EEP",
    "log10_isochrone_age_yr",
    "initial_mass",
    "star_mass",
    "log_Teff",
    "log_g",
    "phase",
)


@cache_stable(module="ugdatalab.isochrones")
def get_mist_isochrone(
    age_gyr: float,
    feh: float,
    alpha_fe: float,
    vvcrit: float,
) -> pd.DataFrame:
    """Load the MIST isochrone nearest to ``age_gyr``, ``feh``, ``alpha_fe``, and ``vvcrit``.

    Resolves the file hierarchically — snapping ``vvcrit`` to the
    nearest available rotation rate, then ``alpha_fe`` to the nearest
    [α/Fe] *among files at that vvcrit*, then ``feh`` to the nearest
    [Fe/H] *among files at that (vvcrit, alpha_fe)* — so the chosen
    ``.iso`` file is guaranteed to exist regardless of grid completeness.
    Within the chosen file, picks the rows of the age block whose
    ``log10_isochrone_age_yr`` is closest to ``log10(age_gyr * 1e9)`` via
    a two-pass scan (first pass discovers the available age tokens,
    second pass collects only the rows in the chosen block). Adds derived
    columns (``Teff``, ``feh``, ``alpha_fe``, ``vvcrit``) and sorts by
    ``initial_mass``.

    Parameters
    ----------
    age_gyr : float
        Target isochrone age in gigayears.
    feh : float
        Requested [Fe/H]; snapped to the nearest grid value.
    alpha_fe : float
        Requested [α/Fe]; snapped to the nearest grid value (e.g., 0.0
        for solar-α, 0.4 for α-enhanced thick-disk / halo populations).
    vvcrit : float
        Requested rotation rate v/v_crit; snapped to the nearest grid
        value (e.g., 0.0 for non-rotating tracks).

    Returns
    -------
    pandas.DataFrame
        One row per EEP point, with columns ``EEP``,
        ``log10_isochrone_age_yr``, ``initial_mass``, ``star_mass``,
        ``Teff``, ``log_Teff``, ``log_g``, ``phase``,
        ``feh``, ``alpha_fe``, ``vvcrit`` (all snapped to grid).
    """
    cache_dir = mist_isochrones_dir()

    candidates = []
    for path in cache_dir.glob("feh_*_afe_*_vvcrit*_full.iso"):
        parts = path.name.split("_")
        feh_tok, alpha_tok, vvcrit_tok = parts[1], parts[3], parts[4]
        feh_val = (-1 if feh_tok[0] == "m" else 1) * int(feh_tok[1:]) / 100.0
        alpha_val = (-1 if alpha_tok[0] == "m" else 1) * int(alpha_tok[1:]) / 10.0
        vvcrit_val = float(vvcrit_tok[len("vvcrit"):])
        candidates.append((feh_val, alpha_val, vvcrit_val, path))

    vvcrit_snapped = min({v for _, _, v, _ in candidates}, key=lambda v: abs(v - vvcrit))
    candidates = [c for c in candidates if c[2] == vvcrit_snapped]

    alpha_snapped = min({a for _, a, _, _ in candidates}, key=lambda a: abs(a - alpha_fe))
    candidates = [c for c in candidates if c[1] == alpha_snapped]

    feh_snapped, _, _, member_path = min(candidates, key=lambda c: abs(c[0] - feh))

    target_log_age = np.log10(age_gyr * _GYR_TO_YR)

    # Pass 1: cache column layout, collect distinct age tokens (as bytes).
    col_indices = None
    age_col_idx = None
    age_tokens = set()
    with member_path.open("rb") as handle:
        for line in handle:
            if line.startswith(b"# EEP"):
                columns = line[2:].decode("utf-8", errors="replace").split()
                col_indices = {name: columns.index(name) for name in _MIST_REQUIRED_COLUMNS}
                age_col_idx = columns.index("log10_isochrone_age_yr")
                continue
            if not line.strip() or line.startswith(b"#"):
                continue
            age_tokens.add(line.split()[age_col_idx])

    best_age_token = min(age_tokens, key=lambda t: abs(float(t) - target_log_age))

    # Pass 2: collect rows whose age token matches (byte-exact, no float drift).
    rows = []
    with member_path.open("rb") as handle:
        for line in handle:
            if not line.strip() or line.startswith(b"#"):
                continue
            parts = line.split()
            if parts[age_col_idx] != best_age_token:
                continue
            rows.append([
                int(parts[col_indices["EEP"]]),
                float(parts[col_indices["log10_isochrone_age_yr"]]),
                float(parts[col_indices["initial_mass"]]),
                float(parts[col_indices["star_mass"]]),
                float(parts[col_indices["log_Teff"]]),
                float(parts[col_indices["log_g"]]),
                float(parts[col_indices["phase"]]),
            ])

    isochrone = pd.DataFrame(rows, columns=[
        "EEP", "log10_isochrone_age_yr", "initial_mass", "star_mass",
        "log_Teff", "log_g", "phase",
    ])

    isochrone["Teff"] = 10.0 ** isochrone["log_Teff"]
    isochrone["feh"] = feh_snapped
    isochrone["alpha_fe"] = alpha_snapped
    isochrone["vvcrit"] = vvcrit_snapped

    isochrone = isochrone.sort_values("initial_mass").reset_index(drop=True)
    return isochrone[
        [
            "EEP",
            "log10_isochrone_age_yr",
            "initial_mass",
            "star_mass",
            "Teff",
            "log_Teff",
            "log_g",
            "phase",
            "feh",
            "alpha_fe",
            "vvcrit",
        ]
    ]
