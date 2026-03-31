from dataclasses import dataclass, field

from astropy import table
from ugdatalab.models.cache import _cache_stable

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

    fundamental = cf == "RRab"
    first_overtone = (cf == "RRc") | (cf == "RRd")

    period = np.full(len(data), np.nan, dtype=float)
    period[fundamental] = data["pf"][fundamental]
    period[first_overtone] = data["p1_o"][first_overtone]
    data["rrlyrae_representative_period"] = period

    err = np.full(len(data), np.nan, dtype=float)
    err[fundamental] = data["pf_error"][fundamental]
    err[first_overtone] = data["p1_o_error"][first_overtone]
    data["rrlyrae_representative_period_error"] = err


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


@_cache_stable(module="ugdatalab.gaia")
def _get_gaia(query):
    from astroquery.gaia import Gaia

    job  = Gaia.launch_job_async(query)
    data = job.get_results()
    return data


@_cache_stable(module="ugdatalab.gaia")
def _get_gaia_sample(query):
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


class GaiaSample(GaiaData):
    """Fetches and caches the quality-filtered Gaia sample with photometry-derived columns."""

    def __post_init__(self):
        data = _get_gaia_sample(self.query)
        _sanitize_table(data, _GAIA_SCHEMA)
        _attach_rrlyrae_representative_period_column(data)

        self.data = data
        self._load_lightcurves()


class Local(GaiaData):
    def __init__(self, source: GaiaData):
        self.query = source.query
        self.include_lightcurve = False
        self.data = source.data[source.data["parallax"] > 0.25]
        self.lightcurves = None


class StrictG(GaiaData):
    """G-band signal-to-noise cut.

    Accepts sources satisfying:
      phot_g_mean_flux_over_error > 5
    """
    def __init__(self, source: GaiaData):
        self.query = source.query
        self.include_lightcurve = False
        self.data = source.data[source.data["phot_g_mean_flux_over_error"] > 5]
        self.lightcurves = None


class StrictBPRP(GaiaData):
    """BP/RP signal-to-noise cut.

    Accepts sources satisfying:
      phot_bp_mean_flux_over_error > 5
      phot_rp_mean_flux_over_error > 5
    """
    def __init__(self, source: GaiaData):
        self.query = source.query
        self.include_lightcurve = False
        mask = (
            (source.data["phot_bp_mean_flux_over_error"] > 5) &
            (source.data["phot_rp_mean_flux_over_error"] > 5)
        )
        self.data = source.data[mask]
        self.lightcurves = None


class LindegrenC1(GaiaData):
    """RUWE quality cut (Lindegren et al. 2021, A&A 649, A2).

    Accepts sources satisfying:
      ruwe < 1.2 * max(1, exp(-0.2 * (G - 19.5)))
    """
    def __init__(self, source: GaiaData):
        self.query = source.query
        self.include_lightcurve = False
        G = source.data["phot_g_mean_mag"]
        u_max = 1.2 * np.maximum(1, np.exp(-0.2 * (G - 19.5)))
        mask = source.data["ruwe"] < u_max
        self.data = source.data[mask]
        self.lightcurves = None


class LindegrenC2(GaiaData):
    """BP/RP flux excess factor quality cut (Lindegren et al. 2021, A&A 649, A2).

    Accepts sources satisfying:
      1.0 + 0.015*(bp_rp)^2 < phot_bp_rp_excess_factor < 1.3 + 0.06*(bp_rp)^2
    """
    def __init__(self, source: GaiaData):
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


class Deoutlier(GaiaSample):
    """Bayesian mixture-model outlier removal per RR Lyrae subclass.

    Fits a linear-Gaussian mixture contamination model to each subclass
    (RRab, RRc, RRd) in the period–luminosity plane, then keeps sources
    whose posterior inlier probability exceeds the threshold.
    """
    def __init__(self, source: GaiaSample, prob_threshold: float = 0.95):
        from ugdatalab.methods.bayesian.likelihoods import LinearGaussianLikelihood
        from ugdatalab.methods.bayesian.mixture import mixture_contamination

        self.query = source.query
        self.include_lightcurve = False

        inlier_probs = np.full(len(source.data), np.nan)
        for rr_class in ("RRab", "RRc", "RRd"):
            mask = source.data["best_classification"] == rr_class
            subset = source.data[mask]
            if len(subset) == 0:
                continue
            period = subset["rrlyrae_representative_period"]
            period_err = subset["rrlyrae_representative_period_error"]
            log_p = np.log10(period)
            sigma_logp = period_err / (period * np.log(10))
            likelihood = LinearGaussianLikelihood(
                x=log_p - np.mean(log_p),
                y=subset["M_G"],
                y_err=subset["sigma_M"],
                x_err=sigma_logp,
            )
            result = mixture_contamination(likelihood)
            inlier_probs[mask] = result.inlier_prob

        self.inlier_probs = inlier_probs
        self.data = source.data[inlier_probs >= prob_threshold]
        self.lightcurves = None


class GaiaReddening(GaiaData):
    """Compute empirical E(BP-RP) and A_G from period-color MCMC results.

    Adds columns ``E_bprp``, ``A_G``, and ``sigma_E`` to the data table
    using class-specific period-color fits for RRab and RRc.

    Parameters
    ----------
    source : GaiaData
        Input catalog with ``rrlyrae_representative_period``, ``bp_rp``,
        ``phot_bp_mean_flux_over_error``, ``phot_rp_mean_flux_over_error``,
        and ``best_classification`` columns.
    rrab_pc : MCMCResult
        Period-color NUTS result for RRab.
    rrc_pc : MCMCResult
        Period-color NUTS result for RRc.
    rrab_mean_log_p : float
        Mean log10(P/day) used to center the RRab fit.
    rrc_mean_log_p : float
        Mean log10(P/day) used to center the RRc fit.
    r_g : float
        Reddening-to-extinction ratio (default 2.0).
    """
    def __init__(
        self,
        source: GaiaData,
        rrab_pc,
        rrc_pc,
        rrab_mean_log_p: float,
        rrc_mean_log_p: float,
        r_g: float = 2.0,
    ):
        self.query = source.query
        self.include_lightcurve = False
        self.lightcurves = None

        pc_map = {"RRab": (rrab_pc, rrab_mean_log_p), "RRc": (rrc_pc, rrc_mean_log_p)}
        parts = []
        for rr_class, (pc_result, mean_lp) in pc_map.items():
            mask = source.data["best_classification"] == rr_class
            subset = source.data[mask]
            out = subset.copy()

            log_p = np.log10(subset["rrlyrae_representative_period"])
            bp_rp_obs = subset["bp_rp"]
            bp_rp_int = pc_result.predict(log_p - mean_lp)

            out["E_bprp"] = bp_rp_obs - bp_rp_int
            out["A_G"] = r_g * (bp_rp_obs - bp_rp_int)

            sigma_color = ((2.5 / np.log(10)) *
                           np.sqrt(1 / subset["phot_bp_mean_flux_over_error"]**2 +
                                   1 / subset["phot_rp_mean_flux_over_error"]**2))
            sigma_intrinsic = 10.0 ** pc_result.theta[2]
            out["sigma_E"] = np.sqrt(sigma_color**2 + sigma_intrinsic**2)

            parts.append(out)

        self.data = table.vstack(parts)


class StrictReddening(GaiaReddening):
    """Reddening quality cut on propagated uncertainty and physical bounds."""
    def __init__(self, source: GaiaReddening):
        self.query = source.query
        self.include_lightcurve = False
        self.lightcurves = None

        e = source.data["E_bprp"]
        sigma_e = source.data["sigma_E"]
        mask = (
            np.isfinite(e) & np.isfinite(sigma_e)
            & (sigma_e <= 0.15)
            & (e >= 0.0)
            & (e <= 10.0)
        )
        self.data = source.data[mask]
