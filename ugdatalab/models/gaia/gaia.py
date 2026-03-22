from dataclasses import dataclass, field

from astroquery.gaia import Gaia
from astropy import table
from ugdatalab.models.cache import cache_stable

import numpy as np

from ugdatalab.models.gaia.constants import ZP_ERR_G, ZP_G
from ugdatalab.models.gaia.lightcurves import (
    _fetch_joined_epoch_photometry, _attach_derived_epoch_columns,
    _attach_periodogram_periods, _attach_fourier_mean_magnitudes,
)
from ugdatalab.models.utils import _sanitize_table


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

def _attach_rrlyrae_representative_period_column(data: table.Table) -> None:
    cf = data["best_classification"]

    period = np.full(len(data), np.nan, dtype=float)
    fundamental = cf == "RRab"
    first_overtone = (cf == "RRc") | (cf == "RRd")

    period[fundamental] = data["pf"][fundamental]
    period[first_overtone] = data["p1_o"][first_overtone]
    data["rrlyrae_representative_period"] = period


_GAIA_SCHEMA = {
    np.int64: ["source_id"],
    int: ["num_clean_epochs_g"],
    str: ["best_classification"],
    float: [
        "l", "b", "pf", "pf_error", "p1_o", "p1_o_error", "int_average_g",
        "parallax", "parallax_error", "parallax_over_error",
        "phot_g_mean_flux", "phot_g_mean_flux_error", "phot_g_mean_mag",
        "bp_rp", "phot_g_mean_flux_over_error",
        "phot_bp_mean_flux_over_error",
        "phot_rp_mean_flux_over_error", "phot_bp_rp_excess_factor", "ruwe",
    ],
}


def _add_gaia_photometry_columns(data: table.Table) -> None:
    phot_g_mean_flux = data["phot_g_mean_flux"]
    phot_g_mean_flux_error = data["phot_g_mean_flux_error"]
    sigma_G_meas = (2.5 / np.log(10)) * np.abs(phot_g_mean_flux_error / phot_g_mean_flux)
    data["sigma_G"] = np.sqrt(sigma_G_meas**2 + ZP_ERR_G**2)

    omega = data["parallax"]
    data["mu"] = 10 - 5 * np.log10(omega)
    data["sigma_mu"] = 5 * data["parallax_error"] / (omega * np.log(10))
    data["M_G"] = data["phot_g_mean_mag"] - data["mu"]
    data["sigma_M"] = np.sqrt(data["sigma_G"]**2 + data["sigma_mu"]**2)


@cache_stable(module="ugdatalab.gaia")
def _get_gaia(query):
    job  = Gaia.launch_job_async(query)
    data = job.get_results()
    return data


@cache_stable(module="ugdatalab.gaia")
def _get_gaia_quality(query):
    raw = _get_gaia(query)
    poe = raw["parallax_over_error"]
    b = raw["b"]
    data = raw[(poe > 5) & (np.abs(b) > 30)]
    _add_gaia_photometry_columns(data)
    return data

# ---------------------------------------------------------------------------
# GaiaData and subclasses
# ---------------------------------------------------------------------------

@dataclass
class GaiaData:
    """Fetches and caches raw Gaia query results."""
    query: str
    include_lightcurve: bool = False
    data: table.Table = field(init=False, repr=False)
    lightcurves: table.Table | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        data = _get_gaia(self.query)
        _sanitize_table(data, _GAIA_SCHEMA)
        _attach_rrlyrae_representative_period_column(data)
        self.data = data
        self._load_lightcurves()

    def _load_lightcurves(self):
        if not self.include_lightcurve:
            self.lightcurves = None
            return

        lightcurves = _fetch_joined_epoch_photometry(self.data)
        _attach_derived_epoch_columns(lightcurves)
        _attach_periodogram_periods(lightcurves)
        _attach_fourier_mean_magnitudes(lightcurves)
        self.lightcurves = lightcurves


class GaiaQuality(GaiaData):
    """Fetches and caches the quality-filtered Gaia sample with photometry-derived columns."""

    def __post_init__(self):
        data = _get_gaia_quality(self.query)
        _sanitize_table(data, _GAIA_SCHEMA)
        _attach_rrlyrae_representative_period_column(data)
        self.data = data
        self._load_lightcurves()


class Local(GaiaQuality):
    def __init__(self, source: GaiaQuality):
        self.query = source.query
        self.include_lightcurve = False
        self.data = source.data[source.data["parallax"] > 0.25]
        self.lightcurves = None


class StrictGBPRP(GaiaQuality):
    """Strict BP/RP signal-to-noise cut.

    Accepts sources satisfying:
      phot_g_mean_flux_over_error > 5
      phot_bp_mean_flux_over_error > 5
      phot_rp_mean_flux_over_error > 5
    """
    def __init__(self, source: GaiaQuality):
        self.query = source.query
        self.include_lightcurve = False
        mask       = (
            (source.data["phot_g_mean_flux_over_error"] > 5) &
            (source.data["phot_bp_mean_flux_over_error"] > 5) &
            (source.data["phot_rp_mean_flux_over_error"] > 5)
        )
        self.data  = source.data[mask]
        self.lightcurves = None


class LindegrenC1(GaiaQuality):
    """RUWE quality cut (Lindegren et al. 2021, A&A 649, A2).

    Accepts sources satisfying:
      ruwe < 1.2 * max(1, exp(-0.2 * (G - 19.5)))
    """
    def __init__(self, source: GaiaQuality):
        self.query = source.query
        self.include_lightcurve = False
        G = source.data["phot_g_mean_mag"]
        u_max = 1.2 * np.maximum(1, np.exp(-0.2 * (G - 19.5)))
        mask = source.data["ruwe"] < u_max
        self.data = source.data[mask]
        self.lightcurves = None


class LindegrenC2(GaiaQuality):
    """BP/RP flux excess factor quality cut (Lindegren et al. 2021, A&A 649, A2).

    Accepts sources satisfying:
      1.0 + 0.015*(bp_rp)^2 < phot_bp_rp_excess_factor < 1.3 + 0.06*(bp_rp)^2
    """
    def __init__(self, source: GaiaQuality):
        self.query = source.query
        self.include_lightcurve = False
        bp_rp = source.data["bp_rp"]
        E = source.data["phot_bp_rp_excess_factor"]
        mask = (
            (E > 1.0 + 0.015 * bp_rp**2) &
            (E < 1.3 + 0.06  * bp_rp**2)
        )
        self.data = source.data[mask]
        self.lightcurves = None
