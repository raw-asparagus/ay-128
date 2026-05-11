from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

from ugdatalab.models.apogee.constants import APOGEE_BAD_PIXMASK_BITS, LABEL_LATEX
from ugdatalab.models.apogee.spectra_pipeline import _apply_bitmask
from ugdatalab.plotters.bayesian import plot_corner
from ugdatalab.plotting import (
    LW_FINE,
    LW_LIGHT,
    LW_MEDIUM,
    LW_STANDARD,
    MS_MICRO,
    SS_MICRO,
    ALPHA_FAINT,
    ALPHA_LIGHT,
    ALPHA_STANDARD,
    NEUTRAL_COLOR,
    FILL_STYLE,
    GUIDE_STYLE,
    FIT_STYLE,
    MODEL_STYLE,
    ERRORBAR_STYLE,
    LABEL_SIZE,
    LEGEND_SIZE,
    textwidth_figure,
    columnwidth_figure,
    subpanels,
    zero_line,
)

_FIGURES_DIR = Path(__file__).parent / "report" / "figures"

# --- Corner / contour ---
# 1σ and 2σ enclosed-mass contour levels
_CORNER_CONTOUR_LEVELS = (0.393, 0.865)

# --- Pixel-mask sentinels & error gates ---
# Error magnitude marking masked APOGEE pixels
_BAD_PIXEL_ERROR_SENTINEL = 1e5
# Max σ for "well-measured" training pixel comparisons
_ERROR_THRESHOLD_TRAINING = 1.0
# Stricter σ cap used in mystery-spectrum overlay
_ERROR_THRESHOLD_MYSTERY = 0.25

# --- Padding / margins ---
# Padding fraction for label-axis ranges
_PAD_FRACTION_LABELS = 0.05
# Mask-frequency y-axis headroom multiplier
_MASK_FRACTION_PAD_FACTOR = 1.1
# Percentile cap when sizing the error-axis limit
_ERROR_PERCENTILE_CAP = 94

# --- Continuum normalization ---
# Per-chip extrapolation guard as fraction of pixels
_CONTINUUM_EXTRAP_FRAC = 0.05

# --- Sequence offsets ---
# Vertical offset between stacked spectra in sequence plots
_SPECTRUM_OFFSET_STEP = 0.2

# --- Layout ---
# Columns in the per-label recovery scatter grid
_LABEL_RECOVERY_NCOLS = 3

# --- MCMC effective-sample-size thresholds ---
# ESS at or above which posteriors are "well-mixed"
_NEFF_HIGH_THRESHOLD = 1000
# ESS below which posteriors are flagged as poorly-mixed
_NEFF_LOW_THRESHOLD = 200

# --- MIST isochrone selection window ---
# Lower Teff bound for plotted isochrone tracks (K)
_TEFF_MIN_ISOCHRONE = 3500
# Upper Teff bound for plotted isochrone tracks (K)
_TEFF_MAX_ISOCHRONE = 6000
# Upper log g bound for plotted isochrone tracks
_LOGG_MAX_ISOCHRONE = 4.5


def savefig(fig, name):
    """Write *fig* to ``report/figures/<name>``, creating the directory if needed."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES_DIR / name)


# ---------------------------------------------------------------------------
# 1. Label corner plot (Problem 1)
# ---------------------------------------------------------------------------

def plot_label_corner(labels, label_names):
    """5D corner plot of stellar labels (Teff, logg, [Fe/H], [Mg/Fe], [Si/Fe]).

    Parameters
    ----------
    labels : ndarray, shape (N, 5)
    label_names : list of str
        LaTeX-formatted label names.
    """
    fig, ax = textwidth_figure(16)
    ax.remove()
    fig = corner.corner(
        labels,
        labels=label_names,
        fig=fig,
        color="C0",
        fill_contours=False,
        plot_datapoints=True,
        plot_density=True,
        levels=_CORNER_CONTOUR_LEVELS,
        smooth=1.0,
        hist_kwargs=dict(
            density=True, histtype="step", linewidth=LW_STANDARD,
            alpha=ALPHA_LIGHT,
        ),
        contour_kwargs=dict(linewidths=LW_STANDARD, alpha=ALPHA_LIGHT),
        data_kwargs=dict(ms=MS_MICRO, alpha=ALPHA_FAINT),
    )

    savefig(fig, "fig_label_corner.pdf")
    return fig


def plot_label_corner_by_field(labels, label_names, fields):
    """Overlaid corner plots colored by APOGEE field.

    Parameters
    ----------
    labels : ndarray, shape (N, 5)
    label_names : list of str
        LaTeX-formatted label names.
    fields : array-like of str, shape (N,)
        Field name for each star.
    """
    unique_fields = sorted(set(fields))
    colors = [f"C{i}" for i in range(len(unique_fields))]

    fig, ax = textwidth_figure(16)
    ax.remove()

    for field_name, color in zip(unique_fields, colors):
        mask = np.array(fields) == field_name
        fig = corner.corner(
            labels[mask],
            labels=label_names,
            fig=fig,
            color=color,
            fill_contours=False,
            plot_datapoints=True,
            plot_density=False,
            levels=_CORNER_CONTOUR_LEVELS,
            smooth=1.0,
            hist_kwargs=dict(
                density=True, histtype="step", linewidth=LW_STANDARD,
                alpha=ALPHA_LIGHT, label=field_name,
            ),
            contour_kwargs=dict(linewidths=LW_STANDARD, alpha=ALPHA_LIGHT),
            data_kwargs=dict(ms=MS_MICRO, alpha=ALPHA_FAINT),
        )

    fig.legend(
        *fig.axes[0].get_legend_handles_labels(),
        loc="upper right", frameon=False,
    )

    savefig(fig, "fig_label_corner_by_field.pdf")
    return fig


# ---------------------------------------------------------------------------
# 2. Example raw spectrum (Problem 2)
# ---------------------------------------------------------------------------

def _raw_spectrum_panels(wavelength, flux_raw, error_raw, apogee_id):
    """Build the shared 2-panel raw spectrum layout and return ``(fig, ax_flux, ax_err)``."""
    fig, _ = textwidth_figure(4.5)
    _.remove()
    ax_flux, ax_err = subpanels(fig, 2, height_ratios=(3, 1), sharex=True)

    ax_flux.plot(wavelength, flux_raw, lw=LW_FINE, alpha=ALPHA_STANDARD,
                 color="C0", rasterized=True)
    ax_flux.set_ylabel("Flux\n" r"[$10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]")
    if apogee_id is not None:
        ax_flux.set_title(apogee_id, loc="left", fontsize=LABEL_SIZE)

    ax_err.plot(wavelength, error_raw, lw=LW_FINE, alpha=ALPHA_STANDARD,
                color="C0", rasterized=True)
    finite_err = error_raw[np.isfinite(error_raw)]
    ax_err.set_ylim(0, np.percentile(finite_err, _ERROR_PERCENTILE_CAP))
    ax_err.set_xlabel(r"Wavelength [\AA]")
    ax_err.set_ylabel(r"$\sigma$")

    return fig, ax_flux, ax_err


def plot_example_spectrum(wavelength, flux_raw, error_raw, apogee_id):
    """Plot a single raw (un-normalized) APOGEE spectrum with errors.

    Parameters
    ----------
    wavelength : ndarray, shape (8575,)
    flux_raw : ndarray, shape (8575,)
    error_raw : ndarray, shape (8575,)
    apogee_id : str, optional
        Star ID for the panel title.
    """
    fig, ax_flux, ax_err = _raw_spectrum_panels(wavelength, flux_raw, error_raw, apogee_id)
    return ax_flux, ax_err


# ---------------------------------------------------------------------------
# 3. Bitmask diagnostic (Problem 3)
# ---------------------------------------------------------------------------

def plot_bitmask_diagnostic(wavelength, flux_raw, error_raw, bitmask, apogee_id):
    """Raw spectrum with bad pixels highlighted.

    Parameters
    ----------
    wavelength : ndarray, shape (n_pixels,)
    flux_raw : ndarray, shape (n_pixels,)
    error_raw : ndarray, shape (n_pixels,)
    bitmask : ndarray, shape (n_pixels,)
    apogee_id : str, optional
    """
    _, error_masked = _apply_bitmask(flux_raw, error_raw, bitmask)
    affected = (error_masked >= _BAD_PIXEL_ERROR_SENTINEL) | ~np.isfinite(error_masked)

    # Pixel edges for axvspan shading
    dw = np.diff(wavelength)
    edges = np.empty(len(wavelength) + 1)
    edges[1:-1] = wavelength[:-1] + dw / 2
    edges[0] = wavelength[0] - dw[0] / 2
    edges[-1] = wavelength[-1] + dw[-1] / 2

    fig, ax_flux, ax_err = _raw_spectrum_panels(wavelength, flux_raw, error_raw, apogee_id)

    affected_idx = np.where(affected)[0]
    for i, idx in enumerate(affected_idx):
        ax_flux.axvspan(edges[idx], edges[idx + 1], color="C3", alpha=ALPHA_FAINT,
                        zorder=0, linewidth=0, label="Affected" if i == 0 else None)
        ax_err.axvspan(edges[idx], edges[idx + 1], color="C3", alpha=ALPHA_FAINT,
                       zorder=0, linewidth=0)

    ax_flux.legend(loc="upper right", fontsize=LABEL_SIZE,
                   title=f"{np.sum(affected)}/{len(affected)} pixels affected",
                   title_fontsize=LABEL_SIZE)

    savefig(fig, "fig_bitmask_diagnostic.pdf")
    return ax_flux, ax_err


# ---------------------------------------------------------------------------
# 3b. Bitmask frequency (Problem 3)
# ---------------------------------------------------------------------------

def plot_bitmask_frequency(wavelength, flux_raw, bitmask_raw):
    """Per-pixel mask frequency across all training spectra, broken down by cause.

    Parameters
    ----------
    wavelength : ndarray, shape (n_pixels,)
    flux_raw : ndarray, shape (N, n_pixels)
        Raw flux before bitmask application.
    bitmask_raw : ndarray, shape (N, n_pixels)
        Raw APOGEE pixel bitmask.
    """
    # Category 1: NaN flux (inter-chip gaps, already missing in FITS)
    nan_flux = np.isnan(flux_raw)

    # Category 2: bitmask-flagged (bits 0-7, 12)
    bm_flagged = np.zeros_like(flux_raw, dtype=bool)
    for bit in APOGEE_BAD_PIXMASK_BITS:
        bm_flagged |= (bitmask_raw & (1 << bit)).astype(bool)
    bm_flagged &= ~nan_flux  # don't double-count

    # Category 3: negative flux (unflagged cosmic rays / artifacts)
    neg_flux = (flux_raw < 0) & ~nan_flux & ~bm_flagged

    n_stars = flux_raw.shape[0]
    nan_count = np.sum(nan_flux, axis=0)
    bm_count = np.sum(bm_flagged, axis=0)
    neg_count = np.sum(neg_flux, axis=0)

    # Plot as fraction of training set
    nan_frac = nan_count / n_stars
    bm_frac = bm_count / n_stars
    neg_frac = neg_count / n_stars

    fig, ax = textwidth_figure(2.5)

    # Stack: bitmask on bottom, negative flux, NaN on top
    cum1 = bm_frac
    cum2 = cum1 + neg_frac
    cum3 = cum2 + nan_frac

    ax.fill_between(wavelength, 0, cum1, step="mid",
                    color="C0", **FILL_STYLE)
    ax.step(wavelength, cum1, where="mid",
            lw=LW_FINE, color="C0", label="Bitmask-flagged")

    ax.fill_between(wavelength, cum1, cum2, step="mid",
                    color="C3", **FILL_STYLE)
    ax.step(wavelength, cum2, where="mid",
            lw=LW_FINE, color="C3", label="Negative flux")

    ax.fill_between(wavelength, cum2, cum3, step="mid",
                    color=NEUTRAL_COLOR, **FILL_STYLE)
    ax.step(wavelength, cum3, where="mid",
            lw=LW_FINE, color=NEUTRAL_COLOR, label="Missing (NaN)")

    ax.set_ylim(0, min(1.0, cum3.max() * _MASK_FRACTION_PAD_FACTOR))
    ax.set_xlabel(r"Wavelength [\AA]")
    ax.set_ylabel("Fraction of\nstars masked")
    ax.set_title(
        f"{n_stars} stars, {flux_raw.shape[1]} pixels",
        loc="left", fontsize=LABEL_SIZE,
    )
    ax.legend(fontsize=LABEL_SIZE, loc="upper left")

    savefig(fig, "fig_bitmask_frequency.pdf")
    return ax


# ---------------------------------------------------------------------------
# 4. Normalization diagnostic (Problem 4)
# ---------------------------------------------------------------------------

def plot_normalization_diagnostic(
    wavelength, flux_masked, error_masked, continuum_fit, flux_norm, error_norm,
    continuum_mask, apogee_id,
):
    """Three-panel normalization diagnostic: masked flux, continuum fit, normalized.

    Parameters
    ----------
    wavelength : ndarray, shape (8575,)
    flux_masked, error_masked : ndarray, shape (8575,)
        Spectrum after bitmask application (bad pixels have error=1e6).
    continuum_fit : ndarray, shape (8575,)
        Fitted continuum polynomial.
    flux_norm, error_norm : ndarray, shape (8575,)
        Normalized spectrum.
    continuum_mask : ndarray, shape (8575,)
        Boolean mask of continuum pixels.
    apogee_id : str, optional
    """
    fig, ax = textwidth_figure(6)
    ax.remove()
    ax_top, ax_bot = subpanels(fig, 2, height_ratios=(4, 3), sharex=True)

    good = np.isfinite(error_masked) & (error_masked < _BAD_PIXEL_ERROR_SENTINEL)

    # Use nan to break lines at bad/gap pixels instead of boolean indexing
    flux_plot = np.where(good, flux_masked, np.nan)
    cont_plot = np.where(np.isfinite(continuum_fit), continuum_fit, np.nan)
    norm_plot = np.where(
        np.isfinite(flux_norm) & (error_norm < _BAD_PIXEL_ERROR_SENTINEL), flux_norm, np.nan,
    )

    # Per-chip continuum with 5% extrapolation
    valid_cont_idx = np.where(np.isfinite(cont_plot))[0]
    if len(valid_cont_idx) > 0:
        breaks = np.where(np.diff(valid_cont_idx) > 1)[0]
        chip_ranges = np.split(valid_cont_idx, breaks + 1)
    else:
        chip_ranges = []

    chip_colors = ["C0", "C2", "C3"]
    chip_names = ["Blue", "Green", "Red"]

    # Top: masked spectrum + continuum fit
    ax_top.plot(wavelength, flux_plot, lw=LW_FINE,
                alpha=ALPHA_LIGHT, color="C1", rasterized=True,
                label="Masked flux")

    for i, chip_idx in enumerate(chip_ranges):
        if len(chip_idx) < 2:
            continue
        color = chip_colors[i % len(chip_colors)]
        n_ext = max(1, int(_CONTINUUM_EXTRAP_FRAC * len(chip_idx)))

        # Build per-chip array with extrapolation
        lo = max(0, chip_idx[0] - n_ext)
        hi = min(len(wavelength), chip_idx[-1] + 1 + n_ext)
        chip_cont = np.full(len(wavelength), np.nan)
        chip_cont[chip_idx] = cont_plot[chip_idx]

        # Left extrapolation
        slope_l = cont_plot[chip_idx[1]] - cont_plot[chip_idx[0]]
        dw_l = wavelength[chip_idx[1]] - wavelength[chip_idx[0]]
        for j in range(lo, chip_idx[0]):
            chip_cont[j] = cont_plot[chip_idx[0]] + slope_l / dw_l * (wavelength[j] - wavelength[chip_idx[0]])

        # Right extrapolation
        slope_r = cont_plot[chip_idx[-1]] - cont_plot[chip_idx[-2]]
        dw_r = wavelength[chip_idx[-1]] - wavelength[chip_idx[-2]]
        for j in range(chip_idx[-1] + 1, hi):
            chip_cont[j] = cont_plot[chip_idx[-1]] + slope_r / dw_r * (wavelength[j] - wavelength[chip_idx[-1]])

        ax_top.plot(wavelength, chip_cont,
                    lw=LW_STANDARD, alpha=ALPHA_STANDARD, color=color,
                    zorder=4, label=f"{chip_names[i]} chip")
    # Highlight continuum pixels used in the fit with filled bands
    cont_px_idx = np.where(continuum_mask & good & (error_masked < _BAD_PIXEL_ERROR_SENTINEL))[0]
    dw = np.diff(wavelength)
    edges = np.empty(len(wavelength) + 1)
    edges[1:-1] = wavelength[:-1] + dw / 2
    edges[0] = wavelength[0] - dw[0] / 2
    edges[-1] = wavelength[-1] + dw[-1] / 2
    for j, idx in enumerate(cont_px_idx):
        ax_top.axvspan(edges[idx], edges[idx + 1], color="C2", alpha=ALPHA_FAINT,
                       zorder=0, linewidth=0,
                       label="Continuum pixels" if j == 0 else None)
    ax_top.set_ylabel("Flux\n" r"[$10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]")
    ax_top.legend(fontsize=LEGEND_SIZE, ncols=3)
    if apogee_id is not None:
        ax_top.set_title(apogee_id, loc="left", fontsize=LABEL_SIZE)

    # Bottom: normalized spectrum
    ax_bot.plot(wavelength, norm_plot, lw=LW_FINE,
                alpha=ALPHA_STANDARD, color="C1", rasterized=True,
                label="Normalized flux")
    ax_bot.axhline(1.0, **GUIDE_STYLE)
    ax_bot.set_ylabel("Normalized\nflux")
    ax_bot.set_xlabel(r"Wavelength [\AA]")

    savefig(fig, "fig_normalization.pdf")
    return ax_top, ax_bot


# ---------------------------------------------------------------------------
# 4. Similar stars comparison (Problem 4)
# ---------------------------------------------------------------------------

def plot_similar_stars_comparison(wavelength, flux_array, error_array, labels,
                                 apogee_ids, ref_idx, n_similar):
    """Overlay normalized spectra of stars with similar labels.

    Finds the closest stars in label space (Euclidean on standardized labels)
    and overplots their spectra.

    Parameters
    ----------
    wavelength : ndarray, shape (8575,)
    flux_array : ndarray, shape (N, 8575)
    error_array : ndarray, shape (N, 8575)
    labels : ndarray, shape (N, 5)
    apogee_ids : ndarray, shape (N,)
    ref_idx : int
        Index of the reference star.
    n_similar : int
        Number of similar stars to overlay.
    """
    # Standardize labels for distance calculation
    std = np.std(labels, axis=0)
    std[std == 0] = 1.0
    scaled = (labels - labels[ref_idx]) / std
    dist = np.sqrt(np.sum(scaled ** 2, axis=1))
    dist[ref_idx] = np.inf
    neighbors = np.argsort(dist)[:n_similar]

    all_idx = [ref_idx] + list(neighbors)

    fig, ax = textwidth_figure(5)
    ax.remove()
    ax_top, ax_bot = subpanels(fig, 2, height_ratios=(3, 2), sharex=True)

    # Collect good-pixel fluxes for scatter computation
    all_flux = []
    for i, idx in enumerate(all_idx):
        good = np.isfinite(flux_array[idx]) & (error_array[idx] < _BAD_PIXEL_ERROR_SENTINEL)
        flux_plot = np.where(good, flux_array[idx], np.nan)
        color = "C0" if i == 0 else f"C{i}"
        alpha = ALPHA_STANDARD if i == 0 else ALPHA_FAINT
        ax_top.plot(wavelength, flux_plot,
                    lw=LW_FINE, alpha=alpha, color=color, rasterized=True,
                    label=str(apogee_ids[idx]))
        all_flux.append(flux_plot)

    ax_top.axhline(1.0, **GUIDE_STYLE)
    ax_top.set_ylabel("Normalized\nflux")
    ax_top.legend(fontsize=LEGEND_SIZE, ncols=3)

    # Bottom panel: per-pixel standard deviation across the group
    all_flux = np.array(all_flux)
    with np.errstate(invalid="ignore"):
        pixel_scatter = np.nanstd(all_flux, axis=0)
    ax_bot.plot(wavelength, pixel_scatter, lw=LW_FINE, color="C0",
                alpha=ALPHA_STANDARD, rasterized=True)
    ax_bot.set_xlabel(r"Wavelength [\AA]")
    ax_bot.set_ylabel(r"$\sigma_{\rm pix}$")

    savefig(fig, "fig_similar_stars.pdf")
    return ax_top, ax_bot


# ---------------------------------------------------------------------------
# 5. Training prediction (Problem 7)
# ---------------------------------------------------------------------------

def plot_training_prediction(wavelength, flux_obs, error_obs, flux_pred,
                             apogee_id, wl_min, wl_max):
    """Observed vs Cannon-predicted spectrum in a wavelength window.

    Parameters
    ----------
    wavelength : ndarray, shape (n_pixels,)
    flux_obs : ndarray, shape (n_pixels,)
        Observed normalized flux.
    error_obs : ndarray, shape (n_pixels,)
        Observed errors (1e6 sentinel for masked pixels).
    flux_pred : ndarray, shape (n_pixels,)
        Model-predicted flux.
    apogee_id : str, optional
    wl_min, wl_max : float
        Wavelength window in Angstroms.
    """
    mask = (wavelength >= wl_min) & (wavelength <= wl_max)
    wl = wavelength[mask]
    obs = flux_obs[mask]
    err = error_obs[mask]
    pred = flux_pred[mask]

    good = np.isfinite(err) & (err < _ERROR_THRESHOLD_TRAINING) & np.isfinite(pred)
    obs_plot = np.where(good, obs, np.nan)
    err_plot = np.where(good, err, np.nan)
    pred_plot = np.where(good, pred, np.nan)
    resid = obs - pred
    resid_plot = np.where(good, resid, np.nan)

    fig, ax = textwidth_figure(8)
    ax.remove()
    axes = subpanels(fig, 2, height_ratios=(3, 1))

    axes[0].errorbar(wl, obs_plot, yerr=err_plot, fmt=".", **ERRORBAR_STYLE,
                     color="C0", alpha=ALPHA_FAINT, zorder=2, label="Observed")
    axes[0].plot(wl, pred_plot, **FIT_STYLE, color="C1", zorder=3, label="Cannon model")
    axes[0].set_ylabel("Normalized\nflux")
    axes[0].legend(fontsize=LEGEND_SIZE)
    if apogee_id is not None:
        axes[0].set_title(
            rf"{apogee_id}, {wl_min:.0f}--{wl_max:.0f}\,\AA",
            loc="left", fontsize=LABEL_SIZE,
        )

    axes[1].errorbar(wl, resid_plot, yerr=err_plot, fmt=".", **ERRORBAR_STYLE,
                     color="C0", alpha=ALPHA_FAINT, zorder=2)
    zero_line(axes[1])
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel(r"Wavelength [\AA]")

    savefig(fig, "fig_training_prediction.pdf")
    return axes


def plot_mystery_prediction(wavelength, flux_obs, error_obs, flux_pred,
                            scatter, wl_min, wl_max):
    """Observed mystery spectrum vs MCMC-fitted Cannon prediction.

    Two panels matching plot_training_prediction style: scatter with error
    bars (top) and residuals (bottom).

    Parameters
    ----------
    wavelength : ndarray, shape (n_pixels,)
    flux_obs : ndarray, shape (n_pixels,)
        Observed normalized flux.
    error_obs : ndarray, shape (n_pixels,)
        Observed errors (1e6 sentinel for masked pixels).
    flux_pred : ndarray, shape (n_pixels,)
        Cannon prediction at MCMC posterior median labels.
    scatter : ndarray, shape (n_pixels,)
        Cannon per-pixel intrinsic scatter (variance, s²_λ).
    wl_min, wl_max : float
        Wavelength window in Angstroms.
    """
    mask = (wavelength >= wl_min) & (wavelength <= wl_max)
    wl = wavelength[mask]
    obs = flux_obs[mask]
    err = error_obs[mask]
    pred = flux_pred[mask]

    good = np.isfinite(err) & (err < _ERROR_THRESHOLD_MYSTERY) & np.isfinite(pred)
    obs_plot = np.where(good, obs, np.nan)
    err_plot = np.where(good, err, np.nan)
    pred_plot = np.where(good, pred, np.nan)
    resid = obs - pred
    resid_plot = np.where(good, resid, np.nan)

    fig, ax = textwidth_figure(8)
    ax.remove()
    axes = subpanels(fig, 2, height_ratios=(3, 1))

    axes[0].errorbar(wl, obs_plot, yerr=err_plot, fmt=".", **ERRORBAR_STYLE,
                     color="C0", alpha=ALPHA_FAINT, zorder=2, label="Mystery spectrum")
    axes[0].plot(wl, pred_plot, **FIT_STYLE, color="C1", zorder=3, label="MCMC fit")
    axes[0].set_ylabel("Normalized\nflux")
    axes[0].legend(fontsize=LEGEND_SIZE)
    axes[0].set_title(
        rf"Mystery spectrum, {wl_min:.0f}--{wl_max:.0f}\,\AA",
        loc="left", fontsize=LABEL_SIZE,
    )

    axes[1].errorbar(wl, resid_plot, yerr=err_plot, fmt=".", **ERRORBAR_STYLE,
                     color="C0", alpha=ALPHA_FAINT, zorder=2)
    zero_line(axes[1])
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel(r"Wavelength [\AA]")

    savefig(fig, "fig_mystery_prediction.pdf")
    return axes


# ---------------------------------------------------------------------------
# 6. Gradient spectra (Problem 8)
# ---------------------------------------------------------------------------

# Air wavelengths from Smith et al. (2013), Table 5, via jobovy/apogee
_MG_LINES = [15740.716, 15748.9, 15765.8, 15879.5, 15886.2, 15954.477]
_SI_LINES = [15361.161, 15376.831, 15833.602, 15960.063, 16060.009,
             16094.787, 16215.670, 16680.770, 16828.159]


def plot_gradient_spectra(model):
    """Five-panel gradient spectra df/dl for each label.

    Marks known Mg I and Si I lines on the [Mg/Fe] and [Si/Fe] panels.

    Parameters
    ----------
    model : CannonModel
        Trained Cannon model.
    """
    n_labels = len(model.label_names)
    fig, ax = textwidth_figure(14)
    ax.remove()
    axes = subpanels(fig, n_labels, sharex=True)

    for i in range(n_labels):
        # Gradient in scaled units (per 1-sigma change) for comparable amplitudes
        grad = model.theta[:, 1 + i]
        grad_plot = np.where(np.isfinite(model.scatter), grad, np.nan)
        axes[i].plot(model.wavelength, grad_plot, lw=LW_FINE,
                     alpha=ALPHA_STANDARD, color=f"C{i}", rasterized=True,
                     zorder=3)
        zero_line(axes[i])
        label_bare = LABEL_LATEX[model.label_names[i]].strip("$")
        axes[i].set_ylabel(
            rf"$\dfrac{{\partial f_\lambda}}{{\partial {label_bare}}}$",
            fontsize=LABEL_SIZE,
        )

        # Mark known spectral lines
        lines = []
        if model.label_names[i] == "mg_fe":
            lines = _MG_LINES
        elif model.label_names[i] == "si_fe":
            lines = _SI_LINES

        for wl in lines:
            axes[i].axvline(wl, color=NEUTRAL_COLOR, ls=":", lw=LW_LIGHT,
                            alpha=ALPHA_LIGHT, zorder=1)

    axes[-1].set_xlabel(r"Wavelength [\AA]")

    savefig(fig, "fig_gradient_spectra.pdf")
    return axes


# ---------------------------------------------------------------------------
# 7. Scatter spectrum (Problem 8)
# ---------------------------------------------------------------------------

# Known H-band absorption features — air wavelengths from Smith et al. (2013)
# and jobovy/apogee (https://github.com/jobovy/apogee/blob/main/apogee/spec/plot.py)
# Brackett series: vacuum wavelengths from NIST ASD
_BRACKETT = {
    r"Br$\,20$": 15196, r"Br$\,19$": 15265, r"Br$\,18$": 15346,
    r"Br$\,17$": 15443, r"Br$\,16$": 15561, r"Br$\,15$": 15705,
    r"Br$\,14$": 15884, r"Br$\,13$": 16113, r"Br$\,12$": 16411,
    r"Br$\,11$": 16811,
}
# CO first-overtone bandheads: Smith et al. (2013) Table 4
_CO_BANDHEADS = {
    r"CO$\,3\text{--}1$": 15582, r"CO$\,4\text{--}2$": 15780.5,
    r"CO$\,5\text{--}3$": 15988, r"CO$\,6\text{--}4$": 16189.5,
    r"CO$\,7\text{--}5$": 16398, r"CO$\,8\text{--}6$": 16614,
}
# Atomic lines: Smith et al. (2013) Tables 2 & 5, via jobovy/apogee
_ATOMIC = {
    "Fe I": [15194.492, 15207.526, 15395.718, 15490.339, 15648.510,
             15964.867, 16040.657, 16153.247, 16165.032, 16697.635],
    "Mg I": _MG_LINES,
    "Si I": _SI_LINES,
    "Al I": [16718.957, 16750.564, 16763.359],
    "Ca I": [16136.823, 16150.763, 16155.236, 16157.364],
}
_ATOMIC_COLORS = {
    "Fe I": "C4", "Mg I": "C0", "Si I": "C1", "Al I": "C5",
    "Ca I": "C6",
}


def plot_scatter_spectrum(model):
    """Per-pixel intrinsic scatter s_lambda vs wavelength with absorption line guides.

    Parameters
    ----------
    model : CannonModel
        Trained Cannon model.
    """
    fig, ax = textwidth_figure(3)

    scatter_plot = np.where(
        np.isfinite(model.scatter), np.sqrt(model.scatter), np.nan,
    )
    ax.plot(model.wavelength, scatter_plot,
            lw=LW_FINE, alpha=ALPHA_STANDARD, color="C0", rasterized=True,
            zorder=3)

    # Guide lines for known features
    ylo, yhi = ax.get_ylim()
    label_y = yhi * 0.92

    for j, (name, wl) in enumerate(_BRACKETT.items()):
        ax.axvline(wl, color="C3", ls=":", lw=LW_LIGHT, alpha=ALPHA_LIGHT, zorder=1,
                   label="H I (Brackett)" if j == 0 else None)

    for j, (name, wl) in enumerate(_CO_BANDHEADS.items()):
        ax.axvline(wl, color="C2", ls=":", lw=LW_LIGHT, alpha=ALPHA_LIGHT, zorder=1,
                   label="CO" if j == 0 else None)

    for species, wls in _ATOMIC.items():
        color = _ATOMIC_COLORS.get(species, NEUTRAL_COLOR)
        for j, wl in enumerate(wls):
            ax.axvline(wl, color=color, ls=":", lw=LW_LIGHT,
                       alpha=ALPHA_FAINT, zorder=1,
                       label=species if j == 0 else None)

    ax.legend(fontsize=LEGEND_SIZE, ncols=4, loc="upper center")

    ax.set_xlabel(r"Wavelength [\AA]")
    ax.set_ylabel(r"Intrinsic scatter $s_\lambda$")
    ax.set_ylim(ylo, yhi)

    savefig(fig, "fig_scatter_spectrum.pdf")
    return ax


# ---------------------------------------------------------------------------
# 8. Label recovery (Problem 9)
# ---------------------------------------------------------------------------

def _plot_label_recovery_impl(true_labels, fitted_labels, label_names, output_name):
    """Core implementation for label recovery plots."""
    n_labels = true_labels.shape[1]
    ncols = _LABEL_RECOVERY_NCOLS
    nrows = (n_labels + ncols - 1) // ncols
    fig, subfigs = textwidth_figure(6 * nrows, subfigures=(nrows, ncols))
    subfigs = np.atleast_2d(subfigs)

    all_axes = np.empty((n_labels, 2), dtype=object)
    col_last = {}
    for i in range(n_labels):
        col_last[i % ncols] = i

    for i in range(n_labels):
        row, col = i // ncols, i % ncols
        ax_c, ax_r = subpanels(subfigs[row, col], 2, height_ratios=(4, 1),
                               sharex=True)
        all_axes[i, 0] = ax_c
        all_axes[i, 1] = ax_r

    for i in range(n_labels):
        ax_c, ax_r = all_axes[i]

        t = true_labels[:, i]
        f = fitted_labels[:, i]
        resid = f - t
        bias = np.mean(resid)
        scatter_val = np.std(resid)

        # Main panel: 1:1 scatter
        ax_c.scatter(t, f, s=SS_MICRO, color="C0", alpha=ALPHA_FAINT,
                     zorder=2, rasterized=True)
        lo = min(np.min(t), np.min(f))
        hi = max(np.max(t), np.max(f))
        margin = _PAD_FRACTION_LABELS * (hi - lo)
        ax_c.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                  color=NEUTRAL_COLOR, lw=LW_MEDIUM, alpha=ALPHA_STANDARD,
                  zorder=3)
        ax_c.set_xlim(lo - margin, hi + margin)
        ax_c.set_ylim(lo - margin, hi + margin)
        ax_c.set_ylabel(rf"Cannon {label_names[i]}")
        ax_c.set_title(
            rf"bias$={bias:+.2f}$, $\sigma={scatter_val:.2f}$",
            loc="left", fontsize=LABEL_SIZE,
        )

        # Residual panel
        ax_r.scatter(t, resid, s=SS_MICRO, color="C0", alpha=ALPHA_FAINT,
                     zorder=2, rasterized=True)
        zero_line(ax_r)
        ax_r.set_ylabel("Res.")
        ax_r.set_xlabel(rf"ASPCAP {label_names[i]}")

    # Hide unused subfigures if n_labels is odd
    if n_labels % ncols != 0:
        subfigs[-1, -1].set_visible(False)

    savefig(fig, output_name)
    return all_axes


def plot_label_recovery(true_labels, fitted_labels, label_names):
    """Five-panel 1:1 comparison of true vs fitted labels with residual sub-panels.

    Each panel shows the main 1:1 scatter plot (top) and residuals (bottom),
    annotated with bias (mean offset) and scatter (std of residuals).

    Parameters
    ----------
    true_labels : ndarray, shape (N, 5)
    fitted_labels : ndarray, shape (N, 5)
    label_names : list of str
        LaTeX-formatted label names.
    """
    return _plot_label_recovery_impl(
        true_labels, fitted_labels, label_names, "fig_label_recovery.pdf",
    )


# ---------------------------------------------------------------------------
# 9. Outlier spectra (Problem 10)
# ---------------------------------------------------------------------------

def plot_outlier_spectra(wavelength, flux_obs_worst, error_obs_worst,
                         flux_pred_worst, ids_worst,
                         flux_obs_best, error_obs_best,
                         flux_pred_best, ids_best,
                         wl_min, wl_max):
    """Plot residuals (obs - model) for worst and best-fit stars side by side.

    Left column: 3 worst-fit stars. Right column: 3 best-fit stars.

    Parameters
    ----------
    wavelength : ndarray, shape (n_pixels,)
    flux_obs_worst, flux_obs_best : ndarray, shape (3, n_pixels)
    error_obs_worst, error_obs_best : ndarray, shape (3, n_pixels)
    flux_pred_worst, flux_pred_best : ndarray, shape (3, n_pixels)
    ids_worst, ids_best : array-like, length 3
    wl_min, wl_max : float
    """
    mask = (wavelength >= wl_min) & (wavelength <= wl_max)
    wl = wavelength[mask]

    fig, _ = columnwidth_figure(15)
    _.remove()
    axes = subpanels(fig, 6, hspace=0.35, sharex=True)

    def _plot_resid(ax, obs, pred, err, star_id, color):
        """Plot the (obs - pred) residual on *ax*, masking bad-error pixels."""
        good = np.isfinite(err) & (err < _BAD_PIXEL_ERROR_SENTINEL)
        resid = np.where(good, obs - pred, np.nan)
        ax.plot(wl, resid, lw=LW_FINE, color=color,
                alpha=ALPHA_STANDARD, zorder=2)
        zero_line(ax)
        ax.set_title(str(star_id), loc="left", fontsize=LABEL_SIZE)
        ax.set_ylabel("Residual")

    # Top 3 rows: worst fit
    for row in range(3):
        ax = axes[row]
        _plot_resid(ax, flux_obs_worst[row, mask], flux_pred_worst[row, mask],
                    error_obs_worst[row, mask], ids_worst[row], "C3")
        if row == 0:
            ax.set_title("Worst fit", loc="right", fontsize=LEGEND_SIZE,
                         fontstyle="italic", color=NEUTRAL_COLOR)

    # Bottom 3 rows: best fit
    for row in range(3):
        ax = axes[row + 3]
        _plot_resid(ax, flux_obs_best[row, mask], flux_pred_best[row, mask],
                    error_obs_best[row, mask], ids_best[row], "C0")
        if row == 0:
            ax.set_title("Best fit", loc="right", fontsize=LEGEND_SIZE,
                         fontstyle="italic", color=NEUTRAL_COLOR)
    axes[-1].set_xlabel(r"Wavelength [\AA]")

    savefig(fig, "fig_outlier_spectra.pdf")
    return fig


def plot_outlier_residuals(wavelength, flux_obs, error_obs, flux_pred,
                           scatter, apogee_ids, wl_min, wl_max):
    """Per-star residual plots for outlier investigation.

    Each panel shows (obs - model) / sqrt(sigma^2 + s^2) in a wavelength
    window, with +/-1 and +/-2 sigma guides.

    Parameters
    ----------
    wavelength : ndarray, shape (n_pixels,)
    flux_obs : ndarray, shape (n_stars, n_pixels)
    error_obs : ndarray, shape (n_stars, n_pixels)
    flux_pred : ndarray, shape (n_stars, n_pixels)
    scatter : ndarray, shape (n_pixels,)
        Cannon per-pixel intrinsic scatter (variance, s^2_lambda).
    apogee_ids : array-like
    wl_min, wl_max : float
    """
    n_stars = flux_obs.shape[0]
    mask = (wavelength >= wl_min) & (wavelength <= wl_max)
    wl = wavelength[mask]
    s2 = scatter[mask]

    fig, ax = textwidth_figure(3 * n_stars)
    ax.remove()
    axes = subpanels(fig, n_stars, sharex=True, hspace=0.14)
    if n_stars == 1:
        axes = [axes]

    for i in range(n_stars):
        obs = flux_obs[i, mask]
        pred = flux_pred[i, mask]
        err = error_obs[i, mask]
        good = np.isfinite(err) & (err < _BAD_PIXEL_ERROR_SENTINEL) & np.isfinite(pred) & np.isfinite(s2)

        sigma_tot = np.sqrt(np.where(good, err ** 2 + s2, np.nan))
        pull = np.where(good, (obs - pred) / sigma_tot, np.nan)

        axes[i].plot(wl, pull, lw=LW_FINE, color="C0",
                     alpha=ALPHA_STANDARD, rasterized=True)
        axes[i].axhline(0, **GUIDE_STYLE)
        for sigma_level in [1, -1, 2, -2]:
            axes[i].axhline(sigma_level, ls=":", lw=LW_LIGHT,
                            color=NEUTRAL_COLOR, alpha=ALPHA_FAINT, zorder=1)
        axes[i].set_ylabel("Pull")
        axes[i].set_ylim(-5, 5)
        axes[i].set_title(str(apogee_ids[i]), loc="left", fontsize=LABEL_SIZE)

    axes[-1].set_xlabel(r"Wavelength [\AA]")

    return axes


# ---------------------------------------------------------------------------
# 10. Kiel diagram (Problem 11)
# ---------------------------------------------------------------------------

def plot_kiel_diagram(fitted_labels, isochrone_tracks):
    """Kiel diagram (log g vs Teff) colored by [Fe/H] with MIST isochrones.

    Parameters
    ----------
    fitted_labels : ndarray, shape (N, 5)
        Fitted stellar labels [Teff, logg, [Fe/H], [Mg/Fe], [Si/Fe]].
    isochrone_tracks : list of (label_str, DataFrame), optional
        Each entry is (label, isochrone_df) with columns ``Teff`` and ``log_g``.
    """
    teff = fitted_labels[:, 0]
    logg = fitted_labels[:, 1]
    feh = fitted_labels[:, 2]

    fig, ax = textwidth_figure(8)

    sc = ax.scatter(teff, logg, c=feh, s=SS_MICRO, alpha=ALPHA_STANDARD,
                    cmap="coolwarm", zorder=2, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$[\mathrm{Fe/H}]$")

    if isochrone_tracks is not None:
        for label, iso_df in isochrone_tracks:
            iso_mask = (iso_df["Teff"] > _TEFF_MIN_ISOCHRONE) & (iso_df["Teff"] < _TEFF_MAX_ISOCHRONE) & (iso_df["log_g"] < _LOGG_MAX_ISOCHRONE)
            ax.plot(iso_df["Teff"][iso_mask], iso_df["log_g"][iso_mask],
                    **MODEL_STYLE, color=NEUTRAL_COLOR, zorder=4,
                    label=label)

    # Inverted axes: hot→cold left→right, low-g at top
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r"$T_{\rm eff}$ [K]")
    ax.set_ylabel(r"$\log g$ [dex]")

    if isochrone_tracks is not None:
        ax.legend(fontsize=LEGEND_SIZE)

    savefig(fig, "fig_kiel_diagram.pdf")
    return ax


# ---------------------------------------------------------------------------
# 11. Metallicity sequence (Problem 13)
# ---------------------------------------------------------------------------

def plot_metallicity_sequence(model, feh_values, teff, logg,
                              mg_fe, si_fe, wl_min, wl_max):
    """Synthetic spectra at varying [Fe/H] with vertical offsets.

    Parameters
    ----------
    model : CannonModel
    feh_values : array-like
        [Fe/H] values to plot.
    teff, logg, mg_fe, si_fe : float
        Fixed label values.
    wl_min, wl_max : float
        Wavelength window in Angstroms.
    """
    mask = (model.wavelength >= wl_min) & (model.wavelength <= wl_max)
    wl = model.wavelength[mask]

    fig, ax = textwidth_figure(12)

    for i, feh in enumerate(feh_values):
        labels = np.array([teff, logg, feh, mg_fe, si_fe])
        spec = model.predict(labels)
        offset = -_SPECTRUM_OFFSET_STEP * i
        ax.plot(wl, spec[mask] + offset, lw=LW_FINE, alpha=ALPHA_STANDARD,
                color=f"C{i % 10}", zorder=3,
                label=rf"$[\mathrm{{Fe/H}}]={feh:+.2f}$")

    ax.set_xlabel(r"Wavelength [\AA]")
    ax.set_ylabel("Normalized\n" r"flux $+$ offset")
    ax.set_title(
        rf"$T_{{\rm eff}}={teff}$ K, $\log g={logg}$",
        loc="left", fontsize=LABEL_SIZE,
    )
    ax.legend(fontsize=LEGEND_SIZE, ncols=2)

    savefig(fig, "fig_metallicity_sequence.pdf")
    return ax


# ---------------------------------------------------------------------------
# 12. RGB evolution (Problem 14)
# ---------------------------------------------------------------------------

def plot_rgb_evolution(model, teff_track, logg_track, feh,
                       mg_fe, si_fe, wl_min, wl_max):
    """Synthetic spectra along the RGB evolutionary track with vertical offsets.

    Parameters
    ----------
    model : CannonModel
    teff_track, logg_track : array-like
        (Teff, logg) points along the RGB.
    feh, mg_fe, si_fe : float
        Fixed abundance labels.
    wl_min, wl_max : float
        Wavelength window.
    """
    mask = (model.wavelength >= wl_min) & (model.wavelength <= wl_max)
    wl = model.wavelength[mask]

    fig, ax = textwidth_figure(12)

    for i, (t, g) in enumerate(zip(teff_track, logg_track)):
        labels = np.array([t, g, feh, mg_fe, si_fe])
        spec = model.predict(labels)
        offset = -_SPECTRUM_OFFSET_STEP * i
        ax.plot(wl, spec[mask] + offset, lw=LW_FINE, alpha=ALPHA_STANDARD,
                color=f"C{i % 10}", zorder=3,
                label=rf"$T_{{\rm eff}}={t:.0f}$, $\log g={g:.1f}$")

    ax.set_xlabel(r"Wavelength [\AA]")
    ax.set_ylabel("Normalized\n" r"flux $+$ offset")
    ax.set_title(
        rf"RGB track, $[\mathrm{{Fe/H}}]={feh:+.1f}$",
        loc="left", fontsize=LABEL_SIZE,
    )
    ax.legend(fontsize=LEGEND_SIZE, ncols=2)

    savefig(fig, "fig_rgb_evolution.pdf")
    return ax


# ---------------------------------------------------------------------------
# 13. Neural network loss curves (Problem 16)
# ---------------------------------------------------------------------------

def plot_nn_loss(train_losses, val_losses):
    """Training and validation loss vs epoch.

    Parameters
    ----------
    train_losses, val_losses : array-like
        Loss per epoch.
    """
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = columnwidth_figure(3)

    ax.plot(epochs, train_losses, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
            color="C0", label="Training", zorder=3)
    ax.plot(epochs, val_losses, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
            color="C1", label="Validation", zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.legend()

    savefig(fig, "fig_nn_loss.pdf")
    return ax


# ---------------------------------------------------------------------------
# 14. Neural network label recovery (Problem 16)
# ---------------------------------------------------------------------------

def plot_nn_label_recovery(true_labels, fitted_labels, label_names):
    """Same format as ``plot_label_recovery`` for neural network predictions.

    Parameters
    ----------
    true_labels : ndarray, shape (N, 5)
    fitted_labels : ndarray, shape (N, 5)
    label_names : list of str
        LaTeX-formatted label names.
    """
    return _plot_label_recovery_impl(
        true_labels, fitted_labels, label_names, "fig_nn_label_recovery.pdf",
    )


# ---------------------------------------------------------------------------
# 15. MCMC corner plot (Problem 12)
# ---------------------------------------------------------------------------

def plot_methods_corner(result):
    """Corner plot of MCMC posterior samples for the Cannon label fit.

    Parameters
    ----------
    result : Posterior
        NUTS result with ``.samples`` and ``.labels``.
    """
    fig = plot_corner(result)
    savefig(fig, "fig_methods_corner.pdf")
    return fig
