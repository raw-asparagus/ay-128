from dataclasses import dataclass

import numpy as np
from astropy import table

from ugdatalab.models.cache import _cache_stable
from ugdatalab.models.gaia.gaia import GaiaData, _get_gaia, _attach_rrlyrae_representative_period_column
from ugdatalab.models.utils import _char_at, _sanitize_table


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

_WISE_SCHEMA = {
    np.int64: ["source_id"],
    str: ["best_classification"],
    float: [
        "l", "b", "pf", "pf_error", "p1_o", "p1_o_error",
        "parallax", "parallax_error",
        "phot_g_mean_mag", "bp_rp", "w2mpro", "w2mpro_error",
    ],
}


def _add_wise_photometry_columns(data: table.Table) -> None:
    """Attach mu, sigma_mu, M_W2, sigma_M_W2 from w2mpro and Gaia parallax."""
    omega = data["parallax"]
    data["mu"] = 10 - 5 * np.log10(omega)
    data["sigma_mu"] = 5 * data["parallax_error"] / (omega * np.log(10))
    data["M_W2"] = data["w2mpro"] - data["mu"]
    data["sigma_M_W2"] = np.sqrt(data["w2mpro_error"] ** 2 + data["sigma_mu"] ** 2)


@_cache_stable(module="ugdatalab.wise")
def _get_wise_sample(query):
    raw = _get_gaia(query)
    poe = raw["parallax_over_error"]
    b = raw["b"]
    data = raw[(poe > 5) & (np.abs(b) > 30)]
    _add_wise_photometry_columns(data)
    return data


# ---------------------------------------------------------------------------
# WISEData and subclasses
# ---------------------------------------------------------------------------

@dataclass
class WISEData(GaiaData):
    """Fetches and caches the WISE quality-filtered sample with photometry-derived columns."""

    def __post_init__(self):
        data = _get_wise_sample(self.query)
        _sanitize_table(data, _WISE_SCHEMA)
        _attach_rrlyrae_representative_period_column(data)
        self.data = data


class WISESample(WISEData):
    """Conservative WISE photometric quality cuts.

    Accepts sources satisfying:
      allwise_oid finite (Gaia-AllWISE best-neighbour match)
      best_classification in {RRab, RRc, RRd}
      number_of_mates == 0 and number_of_neighbours == 1
        (one-to-one cross-match; Marrese et al. 2019, A&A 621, A144)
      ph_qual[W2] in {A, B} (SNR > 3)
      cc_flags[W2] == '0' (no artifact contamination)
      ext_flag <= 1 (point source)
        (AllWISE Explanatory Supplement, Cutri et al. 2013)
    """
    def __init__(self, source: WISEData):
        self.query = source.query
        self.include_lightcurve = False

        allwise_oid = np.asarray(source.data["allwise_oid"], dtype=float)
        matched = source.data[np.isfinite(allwise_oid)]

        rr_classes = matched["best_classification"]
        matched = matched[np.isin(rr_classes, ["RRab", "RRc", "RRd"])]

        number_of_mates = np.asarray(matched["number_of_mates"], dtype=float)
        number_of_neighbours = np.asarray(matched["number_of_neighbours"], dtype=float)
        w2 = matched["w2mpro"]
        w2_error = matched["w2mpro_error"]
        ph_qual_w2 = _char_at(matched["ph_qual"], 1)
        cc_flags_w2 = _char_at(matched["cc_flags"], 1)
        ext_flg = np.asarray(matched["ext_flag"], dtype=float)

        mask = (
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

        self.data = matched[mask]
        self.lightcurves = None
