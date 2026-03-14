from __future__ import annotations

import numpy as np
from astropy import table

from ugdatalab.models.gaia import _as_float_array


# ---------------------------------------------------------------------------
# Private masked-array helpers
# ---------------------------------------------------------------------------

def _col_as_float(column) -> np.ndarray:
    return _as_float_array(column)


def _col_as_str(column) -> np.ndarray:
    arr = np.ma.asarray(column)
    if hasattr(arr, "filled"):
        arr = arr.filled("")
    return np.asarray(arr, dtype=str)


def _char_at(column, idx: int) -> np.ndarray:
    arr = _col_as_str(column)
    return np.array([s[idx] if len(s) > idx else "" for s in arr], dtype=str)


def _first_present_float(data: table.Table, names: list[str]) -> np.ndarray:
    for name in names:
        if name in data.colnames:
            return _as_float_array(data[name])
    return np.full(len(data), np.nan, dtype=float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attach_w2_absolute_magnitude(data: table.Table, *, copy: bool = True) -> table.Table:
    """Compute and attach mu, sigma_mu, M_W2, sigma_M_W2 from w2mpro and Gaia parallax."""
    out = data.copy() if copy else data

    parallax = _col_as_float(out["parallax"])
    parallax_error = _col_as_float(out["parallax_error"])
    w2 = _col_as_float(out["w2mpro"])
    w2_error = _first_present_float(out, ["w2mpro_error", "w2sigmpro"])

    mu = np.full(len(out), np.nan, dtype=float)
    sigma_mu = np.full(len(out), np.nan, dtype=float)
    positive_parallax = np.isfinite(parallax) & np.isfinite(parallax_error) & (parallax > 0)
    mu[positive_parallax] = 10.0 - 5.0 * np.log10(parallax[positive_parallax])
    sigma_mu[positive_parallax] = (
        5.0 * parallax_error[positive_parallax]
        / (parallax[positive_parallax] * np.log(10.0))
    )

    out["mu"] = mu
    out["sigma_mu"] = sigma_mu
    out["M_W2"] = w2 - mu
    out["sigma_M_W2"] = np.sqrt(w2_error ** 2 + sigma_mu ** 2)
    return out


class WISEQualityFilter:
    """Apply conservative WISE photometric quality cuts:
    - allwise_oid finite (matched best-neighbour)
    - number_of_mates == 0
    - number_of_neighbours == 1
    - ph_qual[1] in {'A', 'B'}
    - cc_flags[1] == '0'
    - ext_flag == 0 or 1
    """

    def __init__(self, data: table.Table):
        # Filter to finite allwise_oid (matched best-neighbour)
        allwise_oid = _col_as_float(data["allwise_oid"])
        matched = data[np.isfinite(allwise_oid)]

        # Restrict to RRab/RRc/RRd classes
        rr_classes = _col_as_str(matched["best_classification"])
        matched = matched[np.isin(rr_classes, ["RRab", "RRc", "RRd"])]

        # Apply WISE quality cuts
        number_of_mates = _col_as_float(matched["number_of_mates"])
        number_of_neighbours = _col_as_float(matched["number_of_neighbours"])
        w2 = _col_as_float(matched["w2mpro"])
        w2_error = _first_present_float(matched, ["w2mpro_error", "w2sigmpro"])
        ph_qual_w2 = _char_at(matched["ph_qual"], 1)
        cc_flags_w2 = _char_at(matched["cc_flags"], 1)
        ext_flg = _first_present_float(matched, ["ext_flag", "ext_flg"])

        quality_mask = (
            np.isfinite(number_of_mates)
            & np.isfinite(number_of_neighbours)
            & (number_of_mates == 0)
            & (number_of_neighbours == 1)
            & np.isfinite(w2)
            & np.isfinite(w2_error)
            & (w2_error > 0)
            & np.isin(ph_qual_w2, ["A", "B"])
            & (cc_flags_w2 == "0")
            & np.isfinite(ext_flg)
            & (ext_flg <= 1)
        )

        self.data: table.Table = matched[quality_mask]
