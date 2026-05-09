"""Gaia DR3 epoch-photometry pipeline stages: finite-cut and derived-column attachers.

Each stage is a small ``__call__``-able that takes an
``astropy.table.Table`` of joined epoch photometry and returns one.
Compose with :class:`~ugdatalab.utils.compose.Compose` to build pipelines
that :class:`~ugdatalab.models.gaia.lightcurves.GaiaLightcurves` runs
immediately after fetching and joining.
"""

import numpy as np
from astropy import table

from ugdatalab.models.gaia.constants import (
    ZP_ERR_G, ZP_G, RRLYRAE_PERIOD_MIN, RRLYRAE_PERIOD_MAX, _EPOCH_SCHEMA,
)
from ugdatalab.methods.periodogram import lomb_scargle
from ugdatalab.methods.fourier import FourierFit, fourier_fit, build_design_matrix


# Number of phase-grid points used when integrating Fourier fits to compute
# flux-space mean magnitudes; chosen for negligible quadrature error at the
# RR Lyrae cadence we work with.
_EPOCH_GRID_N = 1000

# Upper bound on the harmonic-order grid swept when selecting the best
# Fourier order per source.
_MAX_HARMONIC_K = 25

# Held-out fraction and seed for the per-source Fourier-order CV.
_FOURIER_CV_FRACTION = 0.2
_FOURIER_CV_SEED = 346


def _select_fourier_order(
    times: np.ndarray,
    mags: np.ndarray,
    mag_err: np.ndarray,
    period: float,
    k_grid: np.ndarray,
) -> int:
    """Pick the Fourier order minimizing held-out χ² on a single light curve.

    A single 80/20 train/holdout split scored by mean χ² per held-out
    point.
    """
    rng = np.random.default_rng(_FOURIER_CV_SEED)
    idx = rng.permutation(len(times))
    n_cv = round(_FOURIER_CV_FRACTION * len(times))
    train_idx, cv_idx = idx[n_cv:], idx[:n_cv]

    best_k, best_score = k_grid[0], np.inf
    for k in k_grid:
        fit = fourier_fit(
            times[train_idx], mags[train_idx], mag_err[train_idx],
            period, k,
        )
        resid = mags[cv_idx] - fit.predict(times[cv_idx])
        score = np.mean((resid / mag_err[cv_idx]) ** 2)
        if score < best_score:
            best_k, best_score = k, score
    return int(best_k)


# ---------------------------------------------------------------------------
# Lightcurve column augmentation
# ---------------------------------------------------------------------------


def _attach_derived_epoch_columns(data: table.Table) -> None:
    """Attach per-epoch mag errors and per-source flux-mean magnitudes."""
    flux = data["g_transit_flux"]
    flux_err = data["g_transit_flux_error"]

    # Per-epoch magnitude error
    meas_err = (2.5 / np.log(10.0)) * np.abs(flux_err / flux)
    data["g_transit_mag_err"] = np.sqrt(meas_err**2 + ZP_ERR_G**2)

    # Per-source flux-space mean magnitude via weighted bincount
    _, inv = np.unique(data["source_id"], return_inverse=True)
    n = np.bincount(inv)
    mean_flux = np.bincount(inv, weights=flux) / n
    mean_flux_err = np.sqrt(np.bincount(inv, weights=flux_err**2)) / n

    mean_mag = -2.5 * np.log10(mean_flux) + ZP_G
    mean_meas_err = (2.5 / np.log(10.0)) * (mean_flux_err / mean_flux)
    total_err = np.sqrt(mean_meas_err**2 + ZP_ERR_G**2)

    data["mean_g_transit_mag"] = mean_mag[inv]
    data["mean_g_transit_mag_err"] = total_err[inv]


class AttachDerivedEpochColumns:
    """Attach per-epoch magnitude error and per-source flux-mean magnitude columns.

    Adds ``g_transit_mag_err``, ``mean_g_transit_mag``, and
    ``mean_g_transit_mag_err``.
    """

    def __call__(self, data: table.Table) -> table.Table:
        """Attach derived epoch columns and return the same table."""
        _attach_derived_epoch_columns(data)
        return data


def _attach_periodogram_periods(data: table.Table) -> None:
    """Attach per-source Lomb-Scargle best periods as ``period_ls``."""
    grouped = data.group_by("source_id")
    periods = np.array([
        lomb_scargle(
            g["g_transit_time"],
            g["g_transit_flux"],
            g["g_transit_flux_error"],
            RRLYRAE_PERIOD_MIN,
            RRLYRAE_PERIOD_MAX,
        ).best_period
        for g in grouped.groups
    ])
    _, inv = np.unique(data["source_id"], return_inverse=True)
    data["period_ls"] = periods[inv]


class AttachPeriodogramPeriods:
    """Attach the per-source Lomb-Scargle best period as ``period_ls``."""

    def __call__(self, data: table.Table) -> table.Table:
        """Attach the periodogram period column and return the same table."""
        _attach_periodogram_periods(data)
        return data


def _fourier_mean_mag_with_err(fit: FourierFit) -> tuple[float, float]:
    """Flux-space mean magnitude and its propagated error from a fitted Fourier model."""
    epoch_grid = np.linspace(0.0, fit.period, _EPOCH_GRID_N, endpoint=False)
    omega = 2.0 * np.pi / fit.period
    X_grid = build_design_matrix(epoch_grid, omega, fit.k)
    mag_grid = X_grid @ fit.beta
    flux_grid = 10.0 ** (-0.4 * (mag_grid - ZP_G))
    mean_flux = np.mean(flux_grid)

    mean_mag = -2.5 * np.log10(mean_flux) + ZP_G
    grad = np.mean(X_grid * flux_grid[:, None], axis=0) / mean_flux
    mean_mag_var = grad @ fit.beta_cov @ grad
    return mean_mag, np.sqrt(np.clip(mean_mag_var, 0.0, None))


def _attach_fourier_mean_magnitudes(data: table.Table) -> None:
    """Fit a per-source Fourier model and attach ``fourier_mean_g_mag`` and ``fourier_mean_g_mag_err``."""
    grouped = data.group_by("source_id")
    means = np.empty(len(grouped.groups))
    errs = np.empty(len(grouped.groups))
    for i, g in enumerate(grouped.groups):
        period = g["period_ls"][0]
        best_k = _select_fourier_order(
            g["g_transit_time"], g["g_transit_mag"], g["g_transit_mag_err"],
            period,
            np.arange(1, _MAX_HARMONIC_K + 1),
        )
        fit = fourier_fit(
            g["g_transit_time"], g["g_transit_mag"], g["g_transit_mag_err"],
            period, best_k,
        )
        means[i], errs[i] = _fourier_mean_mag_with_err(fit)

    _, inv = np.unique(data["source_id"], return_inverse=True)
    data["fourier_mean_g_mag"] = means[inv]
    data["fourier_mean_g_mag_err"] = errs[inv]


class AttachFourierMeanMagnitudes:
    """Attach Fourier-fit mean G-band magnitude and its propagated error.

    Adds ``fourier_mean_g_mag`` and ``fourier_mean_g_mag_err``. Runs
    per-source cross-validation over Fourier order then evaluates the
    mean magnitude in flux space.
    """

    def __call__(self, data: table.Table) -> table.Table:
        """Attach Fourier mean-magnitude columns and return the same table."""
        _attach_fourier_mean_magnitudes(data)
        return data


# ---------------------------------------------------------------------------
# Lightcurve row-filter cuts
# ---------------------------------------------------------------------------


class LightcurveCompletenessCut:
    """Drop epochs whose required float columns are non-finite.

    Keeps rows where every column listed in ``_EPOCH_SCHEMA[float]``
    (``g_transit_time``, ``g_transit_flux``, ``g_transit_flux_error``,
    ``g_transit_mag``) is finite.
    """

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows whose epoch float columns are all finite."""
        mask = np.all(
            [np.isfinite(data[name]) for name in _EPOCH_SCHEMA[float]],
            axis=0,
        )
        return data[mask]
