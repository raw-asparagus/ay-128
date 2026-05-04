"""Gaia DR3 catalog loading, quality cuts, reddening, and outlier removal."""

from dataclasses import dataclass, field

import numpy as np
from astropy import table

from ugdatalab.methods.bayesian.likelihoods import LinearGaussianLikelihood
from ugdatalab.methods.bayesian.mixture import mixture_contamination
from ugdatalab.utils.cache import cache_stable
from ugdatalab.models.gaia.constants import ZP_ERR_G, _GAIA_SCHEMA
from ugdatalab.models.gaia.lightcurves import (
    _fetch_joined_epoch_photometry, _attach_derived_epoch_columns,
    _attach_periodogram_periods, _attach_fourier_mean_magnitudes,
)
from ugdatalab.utils.tables import _sanitize_table


# Sample-quality cuts for minimal line-of-sight dust applied inside ``_get_gaia_sample``.
_PARALLAX_OVER_ERROR_MIN = 5
_GALACTIC_LATITUDE_MIN_DEG = 30

# Local-volume parallax cut: 0.25 mas → distance ≲ 4 kpc.
_LOCAL_PARALLAX_MIN_MAS = 0.25

# Per-band SNR floor for StrictG / StrictBPRP (flux_over_error > this).
_FLUX_SNR_MIN = 5

# Default posterior-inlier probability threshold for Deoutlier.
_DEOUTLIER_PROB_THRESHOLD = 0.95

# StrictReddening cuts on the propagated reddening estimate.
_REDDENING_MAX_SIGMA = 0.15
_REDDENING_MIN = 0.0


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

def _attach_rrlyrae_representative_period_column(data: table.Table) -> None:
    """Attach ``rrlyrae_representative_period`` (and its error) using ``pf`` for RRab and ``p1_o`` for RRc/RRd."""
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


def _add_gaia_photometry_columns(data: table.Table) -> None:
    """Attach ``sigma_G``, ``mu``, ``sigma_mu``, ``M_G``, and ``sigma_M`` columns derived from G-band photometry and parallax."""
    phot_g_mean_flux = data["phot_g_mean_flux"]
    phot_g_mean_flux_error = data["phot_g_mean_flux_error"]
    sigma_G_meas = (2.5 / np.log(10.0)) * np.abs(phot_g_mean_flux_error / phot_g_mean_flux)
    data["sigma_G"] = np.sqrt(sigma_G_meas**2 + ZP_ERR_G**2)

    omega = data["parallax"]
    data["mu"] = 10 - 5 * np.log10(omega)
    data["sigma_mu"] = 5 * data["parallax_error"] / (omega * np.log(10))
    data["M_G"] = data["phot_g_mean_mag"] - data["mu"]
    data["sigma_M"] = np.sqrt(data["sigma_G"]**2 + data["sigma_mu"]**2)


@cache_stable(module="ugdatalab.gaia")
def _get_gaia(query):
    """Run an async Gaia ADQL query and return the result table (cached)."""
    from astroquery.gaia import Gaia

    job  = Gaia.launch_job_async(query)
    data = job.get_results()
    return data


@cache_stable(module="ugdatalab.gaia")
def _get_gaia_sample(query):
    """Run a Gaia query and apply the parallax quality and high-latitude sample cuts."""
    raw = _get_gaia(query)
    poe = raw["parallax_over_error"]
    b = raw["b"]
    data = raw[(poe > _PARALLAX_OVER_ERROR_MIN) & (np.abs(b) > _GALACTIC_LATITUDE_MIN_DEG)]
    _add_gaia_photometry_columns(data)
    return data

# ---------------------------------------------------------------------------
# GaiaData and subclasses
# ---------------------------------------------------------------------------

@dataclass
class GaiaData:
    """Fetch and cache raw Gaia query results, optionally with epoch photometry.

    Parameters
    ----------
    query : str
        Gaia ADQL query string.
    include_lightcurve : bool
        If True, also download epoch photometry and attach derived columns.

    Attributes
    ----------
    data : astropy.table.Table
        Sanitized query result with the RR Lyrae representative period column.
    lightcurves : astropy.table.Table or None
        Epoch photometry joined on ``source_id`` when ``include_lightcurve``.
    """
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
        """Fetch epoch photometry and attach derived columns when ``include_lightcurve`` is True."""
        if not self.include_lightcurve:
            self.lightcurves = None
            return

        lightcurves = _fetch_joined_epoch_photometry(self.data)
        _attach_derived_epoch_columns(lightcurves)
        _attach_periodogram_periods(lightcurves)
        _attach_fourier_mean_magnitudes(lightcurves)
        self.lightcurves = lightcurves


class GaiaSample(GaiaData):
    """Quality-filtered Gaia sample (parallax_over_error > 5, |b| > 30 deg) with photometry-derived columns."""

    def __post_init__(self):
        data = _get_gaia_sample(self.query)
        _sanitize_table(data, _GAIA_SCHEMA)
        _attach_rrlyrae_representative_period_column(data)

        self.data = data
        self._load_lightcurves()


class Local(GaiaData):
    """Local-volume parallax cut (parallax > 0.25 mas, distance <~ 4 kpc).

    Parameters
    ----------
    source : GaiaData
        Catalog to filter.
    """
    def __init__(self, source: GaiaData):
        self.query = source.query
        self.include_lightcurve = False
        self.data = source.data[source.data["parallax"] > _LOCAL_PARALLAX_MIN_MAS]
        self.lightcurves = None


class StrictG(GaiaData):
    """G-band signal-to-noise cut (``phot_g_mean_flux_over_error > 5``).

    Parameters
    ----------
    source : GaiaData
        Catalog to filter.
    """
    def __init__(self, source: GaiaData):
        self.query = source.query
        self.include_lightcurve = False
        self.data = source.data[source.data["phot_g_mean_flux_over_error"] > _FLUX_SNR_MIN]
        self.lightcurves = None


class StrictBPRP(GaiaData):
    """BP- and RP-band signal-to-noise cut (both flux_over_error > 5).

    Parameters
    ----------
    source : GaiaData
        Catalog to filter.
    """
    def __init__(self, source: GaiaData):
        self.query = source.query
        self.include_lightcurve = False
        mask = (
            (source.data["phot_bp_mean_flux_over_error"] > _FLUX_SNR_MIN) &
            (source.data["phot_rp_mean_flux_over_error"] > _FLUX_SNR_MIN)
        )
        self.data = source.data[mask]
        self.lightcurves = None


class LindegrenC1(GaiaData):
    """RUWE quality cut from Lindegren et al. 2021 (A&A 649, A2): ``ruwe < 1.2 * max(1, exp(-0.2 * (G - 19.5)))``.

    Parameters
    ----------
    source : GaiaData
        Catalog to filter.
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
    """BP/RP flux excess factor quality cut from Lindegren et al. 2021 (A&A 649, A2).

    Accepts sources satisfying
    ``1.0 + 0.015*(bp_rp)^2 < phot_bp_rp_excess_factor < 1.3 + 0.06*(bp_rp)^2``.

    Parameters
    ----------
    source : GaiaData
        Catalog to filter.
    """
    def __init__(self, source: GaiaData):
        self.query = source.query
        self.include_lightcurve = False
        bp_rp = source.data["bp_rp"]
        E = source.data["phot_bp_rp_excess_factor"]
        mask = (
            (E > 1.0 + 0.015 * bp_rp**2) &
            (E < 1.3 + 0.06 * bp_rp**2)
        )
        self.data = source.data[mask]
        self.lightcurves = None


class Deoutlier(GaiaSample):
    """Bayesian mixture-model outlier removal per RR Lyrae subclass.

    Fits a linear-Gaussian mixture contamination model to each subclass
    (RRab, RRc, RRd) in the period-luminosity plane and keeps sources
    whose posterior inlier probability exceeds ``prob_threshold``.

    Parameters
    ----------
    source : GaiaSample
        Catalog to filter.
    prob_threshold : float
        Minimum posterior inlier probability for a source to be kept.

    Attributes
    ----------
    inlier_probs : ndarray
        Posterior inlier probability for every row of ``source.data``
        (NaN for sources outside the three RR Lyrae subclasses).
    """
    def __init__(self, source: GaiaSample, prob_threshold: float = _DEOUTLIER_PROB_THRESHOLD):
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
    """
    def __init__(
        self,
        source: GaiaData,
        rrab_pc,
        rrc_pc,
        rrab_mean_log_p: float,
        rrc_mean_log_p: float,
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
            out["A_G"] = 2.0 * (bp_rp_obs - bp_rp_int)

            sigma_color = ((2.5 / np.log(10.0)) *
                           np.sqrt(1 / subset["phot_bp_mean_flux_over_error"]**2 +
                                   1 / subset["phot_rp_mean_flux_over_error"]**2))
            sigma_intrinsic = 10.0 ** pc_result.theta[2]
            out["sigma_E"] = np.sqrt(sigma_color**2 + sigma_intrinsic**2)

            parts.append(out)

        self.data = table.vstack(parts)


class StrictReddening(GaiaReddening):
    """Reddening quality cut on propagated uncertainty and physical bounds.

    Parameters
    ----------
    source : GaiaReddening
        Catalog to filter; must already carry ``E_bprp`` and ``sigma_E``.
    """
    def __init__(self, source: GaiaReddening):
        self.query = source.query
        self.include_lightcurve = False
        self.lightcurves = None

        e = source.data["E_bprp"]
        sigma_e = source.data["sigma_E"]
        mask = (
            np.isfinite(e) & np.isfinite(sigma_e)
            & (sigma_e <= _REDDENING_MAX_SIGMA)
            & (e >= _REDDENING_MIN)
        )
        self.data = source.data[mask]
