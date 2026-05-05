"""Gaia DR3 pipeline stages: row-filter cuts and column augmentations.

Each stage is a small ``__call__``-able (dataclass when parameterized)
that takes an ``astropy.table.Table`` and returns one. Compose with
:class:`~ugdatalab.utils.compose.Compose` to build pipelines that
:class:`~ugdatalab.models.gaia.gaia.GaiaData` runs immediately after
fetching and sanitizing.
"""

from dataclasses import dataclass

import numpy as np
from astropy import table

from ugdatalab.methods.bayesian.likelihoods import LinearGaussianLikelihood
from ugdatalab.methods.bayesian.mcmc import MCMCResult
from ugdatalab.methods.bayesian.mixture import mixture_contamination
from ugdatalab.models.gaia.constants import ZP_ERR_G


# ---------------------------------------------------------------------------
# Source-mandated column augmentations
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


class AttachRepresentativePeriod:
    """Attach the RR Lyrae representative period (and its error) column.

    Adds two columns to the table: ``rrlyrae_representative_period``
    (set to ``pf`` for RRab and ``p1_o`` for RRc/RRd; NaN otherwise) and
    its propagated error ``rrlyrae_representative_period_error``.

    Requires columns ``best_classification``, ``pf``, ``pf_error``,
    ``p1_o``, ``p1_o_error``.
    """

    def __call__(self, data: table.Table) -> table.Table:
        """Return a copy with the representative-period columns attached."""
        out = data.copy()
        _attach_rrlyrae_representative_period_column(out)
        return out


# ---------------------------------------------------------------------------
# Column-adding augmentations (same callable shape)
# ---------------------------------------------------------------------------


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


class AddPhotometryColumns:
    """Attach G-band and parallax-derived photometry columns.

    Adds ``sigma_G``, ``mu``, ``sigma_mu``, ``M_G``, ``sigma_M`` to the
    table. Mutates the table in place and returns it.
    """

    def __call__(self, data: table.Table) -> table.Table:
        """Attach photometry columns and return the same table."""
        _add_gaia_photometry_columns(data)
        return data


@dataclass
class AttachReddening:
    """Compute empirical E(BP-RP), A_G, sigma_E from period-color MCMC fits.

    For each of RRab and RRc, predicts intrinsic ``bp_rp`` from the
    period-color fit, computes ``E_bprp = bp_rp_obs - bp_rp_int``,
    ``A_G = 2 * E_bprp``, and propagates a per-source ``sigma_E`` that
    combines BP/RP photometric noise with the model's intrinsic scatter.
    Restricts the output to RRab and RRc subclasses; RRd rows are dropped.

    Parameters
    ----------
    rrab_pc, rrc_pc : MCMCResult
        Period-color NUTS results (provide ``predict`` and ``theta``).
    rrab_mean_log_p, rrc_mean_log_p : float
        Mean ``log10(P/day)`` used to center each fit.
    """
    rrab_pc: MCMCResult
    rrc_pc: MCMCResult
    rrab_mean_log_p: float
    rrc_mean_log_p: float

    def __call__(self, data: table.Table) -> table.Table:
        """Return a vstacked table of RRab and RRc rows with reddening columns."""
        pc_map = {
            "RRab": (self.rrab_pc, self.rrab_mean_log_p),
            "RRc": (self.rrc_pc, self.rrc_mean_log_p),
        }
        parts = []
        for rr_class, (pc_result, mean_lp) in pc_map.items():
            mask = data["best_classification"] == rr_class
            subset = data[mask]
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

        return table.vstack(parts)


# ---------------------------------------------------------------------------
# Row-filter cuts
# ---------------------------------------------------------------------------


class GaiaCompletenessCut:
    """Drop rows whose downstream-required Gaia values are non-finite or non-positive.

    Keeps rows where ``parallax``, ``parallax_error``,
    ``phot_g_mean_flux``, and ``phot_g_mean_flux_error`` are all finite,
    and where ``parallax``, ``parallax_error``, and ``phot_g_mean_flux_error``
    are strictly positive.
    """

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows with finite parallax/flux and positive parallax, parallax_error, and flux_error."""
        parallax = data["parallax"]
        parallax_error = data["parallax_error"]
        flux = data["phot_g_mean_flux"]
        flux_error = data["phot_g_mean_flux_error"]
        mask = (
            np.isfinite(parallax)
            & np.isfinite(parallax_error)
            & np.isfinite(flux)
            & np.isfinite(flux_error)
            & (parallax > 0)
            & (parallax_error > 0)
            & (flux > 0)
            & (flux_error > 0)
        )
        return data[mask]


@dataclass
class HighParallaxSnrCut:
    """Drop sources with low parallax signal-to-noise.

    Parameters
    ----------
    snr : float
        Minimum ``parallax_over_error``. Default ``5.0`` — the canonical
        Gaia "well-measured parallax" floor used across the package; lower
        values admit sources whose distance modulus is poorly constrained.
    """
    snr: float = 5.0

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows with ``parallax_over_error > self.snr``."""
        return data[data["parallax_over_error"] > self.snr]


@dataclass
class HighLatitudeCut:
    """Drop low-Galactic-latitude sources (high line-of-sight extinction).

    Parameters
    ----------
    b_min_deg : float
        Minimum ``|b|`` in degrees. Default ``30.0`` — keeps the off-plane
        regions where MW dust is mild; lowering it admits more of the disk
        and pulls in heavily reddened sources.
    """
    b_min_deg: float = 30.0

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows with ``|b| > self.b_min_deg``."""
        return data[np.abs(data["b"]) > self.b_min_deg]


@dataclass
class LocalCut:
    """Restrict to the local volume by a parallax floor.

    Parameters
    ----------
    parallax_min_mas : float
        Minimum parallax in mas. Default ``0.25`` — corresponds to a
        distance ``≲ 4 kpc``; raise to study only the Solar neighborhood,
        lower to extend toward the Galactic halo.
    """
    parallax_min_mas: float = 0.25

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows with ``parallax > self.parallax_min_mas``."""
        return data[data["parallax"] > self.parallax_min_mas]


@dataclass
class StrictGCut:
    """G-band photometric SNR cut.

    Parameters
    ----------
    snr : float
        Minimum ``phot_g_mean_flux_over_error``. Default ``5.0`` — matches
        the BP/RP cuts; loosening admits noisier G-band photometry.
    """
    snr: float = 5.0

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows with ``phot_g_mean_flux_over_error > self.snr``."""
        return data[data["phot_g_mean_flux_over_error"] > self.snr]


@dataclass
class StrictBPRPCut:
    """BP- and RP-band photometric SNR cut (both must pass).

    Parameters
    ----------
    snr : float
        Minimum ``phot_{bp,rp}_mean_flux_over_error``. Default ``5.0`` —
        matches :class:`StrictGCut`. BP/RP are noisier than G, so this
        rejects more sources than the G cut alone.
    """
    snr: float = 5.0

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows where both BP and RP flux SNR exceed ``self.snr``."""
        mask = (
            (data["phot_bp_mean_flux_over_error"] > self.snr)
            & (data["phot_rp_mean_flux_over_error"] > self.snr)
        )
        return data[mask]


@dataclass
class LindegrenC1Cut:
    """RUWE astrometric-quality cut from Lindegren et al. 2021 (A&A 649, A2).

    Accepts ``ruwe < multiplier * max(1, exp(decay * (G - g_pivot)))``.

    Parameters
    ----------
    multiplier : float
        Coefficient on the empirical envelope. Default ``1.2`` — matches
        Lindegren+21 Eq. C.1; the standard "1.2 RUWE" criterion.
    g_pivot : float
        G-magnitude pivot at which the envelope departs from constant.
        Default ``19.5`` — published Lindegren+21 value.
    decay : float
        Slope in ``exp``. Default ``-0.2`` — published Lindegren+21 value
        controlling envelope growth at faint magnitudes.
    """
    multiplier: float = 1.2
    g_pivot: float = 19.5
    decay: float = -0.2

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows passing the magnitude-dependent RUWE envelope."""
        G = data["phot_g_mean_mag"]
        u_max = self.multiplier * np.maximum(1, np.exp(self.decay * (G - self.g_pivot)))
        return data[data["ruwe"] < u_max]


@dataclass
class LindegrenC2Cut:
    """BP/RP flux excess factor cut from Lindegren et al. 2021 (A&A 649, A2).

    Accepts sources satisfying
    ``1.0 + 0.015 * bp_rp**2 < phot_bp_rp_excess_factor < 1.3 + 0.06 * bp_rp**2``.
    """

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows inside the BP/RP excess-factor envelope."""
        bp_rp = data["bp_rp"]
        E = data["phot_bp_rp_excess_factor"]
        mask = (
            (E > 1.0 + 0.015 * bp_rp**2)
            & (E < 1.3 + 0.06 * bp_rp**2)
        )
        return data[mask]


def _compute_inlier_probs(data: table.Table) -> np.ndarray:
    """Return per-row posterior inlier probabilities from a P-L mixture fit per RR Lyrae subclass."""
    inlier_probs = np.full(len(data), np.nan)
    for rr_class in ("RRab", "RRc", "RRd"):
        mask = data["best_classification"] == rr_class
        subset = data[mask]
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
    return inlier_probs


@dataclass
class AttachInlierProbColumn:
    """Attach a per-row Bayesian outlier ``inlier_prob`` column, optionally filtering.

    Adds an ``inlier_prob`` column with the posterior inlier probability
    for every row (NaN for rows outside RRab/RRc/RRd). When
    ``prob_threshold`` is provided, also drops rows whose inlier
    probability falls below it (NaN-valued rows are always dropped in
    that case).

    Requires columns ``best_classification``,
    ``rrlyrae_representative_period``,
    ``rrlyrae_representative_period_error``, ``M_G``, ``sigma_M``.

    Parameters
    ----------
    prob_threshold : float or None
        Minimum posterior inlier probability for a row to be kept.
        Default ``None`` — only attach the column, do not filter (use
        when you want to visualize which rows would be dropped before
        committing to a threshold). Set to a float (e.g. ``0.95``) to
        attach and filter in one step.
    """
    prob_threshold: float | None = None

    def __call__(self, data: table.Table) -> table.Table:
        """Return a copy with ``inlier_prob`` attached and (optionally) low-probability rows dropped."""
        out = data.copy()
        out["inlier_prob"] = _compute_inlier_probs(data)
        if self.prob_threshold is None:
            return out
        return out[out["inlier_prob"] >= self.prob_threshold]


@dataclass
class StrictReddeningCut:
    """Reddening quality cut on propagated uncertainty and physical bounds.

    Requires ``E_bprp`` and ``sigma_E`` columns — must follow
    :class:`AttachReddening`.

    Parameters
    ----------
    sigma_max : float
        Maximum allowed ``sigma_E``. Default ``0.15`` — cuts sources with
        poorly-constrained extinction; loosening admits noisier reddening
        estimates.
    e_min : float
        Minimum allowed ``E_bprp``. Default ``0.0`` — physical (negative
        reddening is unphysical); keep at zero unless you're explicitly
        modeling photometric noise that scatters into negative E.
    """
    sigma_max: float = 0.15
    e_min: float = 0.0

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows with finite, small-uncertainty, non-negative reddening."""
        e = data["E_bprp"]
        sigma_e = data["sigma_E"]
        mask = (
            np.isfinite(e) & np.isfinite(sigma_e)
            & (sigma_e <= self.sigma_max)
            & (e >= self.e_min)
        )
        return data[mask]
