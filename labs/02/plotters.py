from pathlib import Path

import corner
import numpy as np

from ugdatalab.plotting import (
    LW_FINE,
    LW_LIGHT,
    LW_STANDARD,
    MS_MICRO,
    SS_MICRO,
    SS_FINE,
    ALPHA_FAINT,
    ALPHA_LIGHT,
    ALPHA_STANDARD,
    NEUTRAL_COLOR,
    GUIDE_STYLE,
    FIT_STYLE,
    MODEL_STYLE,
    ERRORBAR_STYLE,
    SCATTER_STYLE,
    textwidth_figure,
    subpanels,
    zero_line,
    corner_figure,
)

_FIGURES_DIR = Path(__file__).parent / "report" / "figures"


def savefig(fig, name):
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
    fig = corner_figure()
    fig = corner.corner(
        labels,
        labels=label_names,
        fig=fig,
        color="C0",
        fill_contours=False,
        plot_datapoints=True,
        plot_density=True,
        levels=(0.393, 0.865),
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


# ---------------------------------------------------------------------------
# 2. Example raw spectrum (Problem 2)
# ---------------------------------------------------------------------------

def plot_example_spectrum(wavelength, flux_raw, apogee_id=None):
    """Plot a single raw (un-normalized) APOGEE spectrum.

    Parameters
    ----------
    wavelength : ndarray, shape (8575,)
    flux_raw : ndarray, shape (8575,)
    apogee_id : str, optional
        Star ID for the panel title.
    """
    fig, ax = textwidth_figure(5)

    ax.plot(wavelength, flux_raw, lw=LW_FINE, alpha=ALPHA_STANDARD,
            color="C0", rasterized=True)
    ax.set_xlabel(r"Wavelength [\AA]")
    ax.set_ylabel(r"Flux [$10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]")
    if apogee_id is not None:
        ax.set_title(apogee_id, loc="left", fontsize="small")

    savefig(fig, "fig_example_spectrum.pdf")
    return ax


# ---------------------------------------------------------------------------
# 3. Normalization diagnostic (Problem 4)
# ---------------------------------------------------------------------------

def plot_normalization_diagnostic(
    wavelength, flux_raw, error_raw, continuum_fit, flux_norm, error_norm,
    continuum_mask, apogee_id=None,
):
    """Three-panel normalization diagnostic: raw, continuum fit, normalized.

    Parameters
    ----------
    wavelength : ndarray, shape (8575,)
    flux_raw, error_raw : ndarray, shape (8575,)
        Raw spectrum before normalization.
    continuum_fit : ndarray, shape (8575,)
        Fitted continuum polynomial.
    flux_norm, error_norm : ndarray, shape (8575,)
        Normalized spectrum.
    continuum_mask : ndarray, shape (8575,)
        Boolean mask of continuum pixels.
    apogee_id : str, optional
    """
    fig, ax = textwidth_figure(12)
    ax.remove()
    axes = subpanels(fig, 3, sharex=True, hspace=0.08)

    # Top: raw spectrum + continuum fit
    axes[0].plot(wavelength, flux_raw, lw=LW_FINE, alpha=ALPHA_FAINT,
                 color="C0", rasterized=True, label="Raw flux")
    valid_cont = np.isfinite(continuum_fit)
    axes[0].plot(wavelength[valid_cont], continuum_fit[valid_cont],
                 lw=LW_STANDARD, alpha=ALPHA_STANDARD, color="C1",
                 label="Continuum fit")
    cont_px = continuum_mask & np.isfinite(error_raw) & (error_raw < np.inf)
    axes[0].scatter(wavelength[cont_px], flux_raw[cont_px], s=SS_MICRO,
                    alpha=ALPHA_FAINT, color="C2", zorder=3,
                    label="Continuum pixels", rasterized=True)
    axes[0].set_ylabel(r"Flux")
    axes[0].legend(fontsize="x-small")
    if apogee_id is not None:
        axes[0].set_title(apogee_id, loc="left", fontsize="small")

    # Middle: continuum fit alone (zoomed)
    axes[1].plot(wavelength[valid_cont], continuum_fit[valid_cont],
                 lw=LW_STANDARD, alpha=ALPHA_STANDARD, color="C1")
    axes[1].set_ylabel(r"Continuum")

    # Bottom: normalized spectrum
    valid_norm = np.isfinite(flux_norm) & (error_norm < np.inf)
    axes[2].plot(wavelength[valid_norm], flux_norm[valid_norm], lw=LW_FINE,
                 alpha=ALPHA_STANDARD, color="C0", rasterized=True,
                 label="Normalized flux")
    axes[2].axhline(1.0, **GUIDE_STYLE)
    axes[2].set_ylabel(r"Normalized flux")
    axes[2].set_xlabel(r"Wavelength [\AA]")

    savefig(fig, "fig_normalization.pdf")
    return axes


# ---------------------------------------------------------------------------
# 4. Similar stars comparison (Problem 4)
# ---------------------------------------------------------------------------

def plot_similar_stars_comparison(wavelength, flux_array, error_array, labels,
                                 apogee_ids, ref_idx, n_similar=5):
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

    fig, ax = textwidth_figure(5)

    valid_ref = np.isfinite(flux_array[ref_idx]) & (error_array[ref_idx] < np.inf)
    ax.plot(wavelength[valid_ref], flux_array[ref_idx][valid_ref],
            lw=LW_FINE, alpha=ALPHA_STANDARD, color="C0", rasterized=True,
            label=str(apogee_ids[ref_idx]))

    for i, idx in enumerate(neighbors):
        valid = np.isfinite(flux_array[idx]) & (error_array[idx] < np.inf)
        ax.plot(wavelength[valid], flux_array[idx][valid],
                lw=LW_FINE, alpha=ALPHA_FAINT, color=f"C{i + 1}",
                rasterized=True, label=str(apogee_ids[idx]))

    ax.axhline(1.0, **GUIDE_STYLE)
    ax.set_xlabel(r"Wavelength [\AA]")
    ax.set_ylabel(r"Normalized flux")
    ax.legend(fontsize="x-small", ncols=2)

    savefig(fig, "fig_similar_stars.pdf")
    return ax


# ---------------------------------------------------------------------------
# 5. Training prediction (Problem 7)
# ---------------------------------------------------------------------------

def plot_training_prediction(wavelength, flux_obs, error_obs, flux_pred,
                             apogee_id=None, wl_min=16000, wl_max=16100):
    """Observed vs Cannon-predicted spectrum in a wavelength window.

    Parameters
    ----------
    wavelength : ndarray, shape (n_pixels,)
    flux_obs : ndarray, shape (n_pixels,)
        Observed normalized flux.
    error_obs : ndarray, shape (n_pixels,)
        Observed errors (inf for masked pixels).
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

    fig, ax = textwidth_figure(8)
    ax.remove()
    axes = subpanels(fig, 2, height_ratios=(3, 1), hspace=0.05)

    valid = np.isfinite(err) & (err < np.inf)
    axes[0].errorbar(wl[valid], obs[valid], yerr=err[valid], **ERRORBAR_STYLE,
                     color="C0", alpha=ALPHA_FAINT, zorder=2, label="Observed")
    axes[0].plot(wl, pred, **FIT_STYLE, color="C1", zorder=3, label="Cannon model")
    axes[0].set_ylabel(r"Normalized flux")
    axes[0].legend(fontsize="x-small")
    if apogee_id is not None:
        axes[0].set_title(
            rf"{apogee_id}, {wl_min:.0f}--{wl_max:.0f}\,\AA",
            loc="left", fontsize="small",
        )

    resid = obs - pred
    axes[1].errorbar(wl[valid], resid[valid], yerr=err[valid], **ERRORBAR_STYLE,
                     color="C0", alpha=ALPHA_FAINT, zorder=2)
    zero_line(axes[1])
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel(r"Wavelength [\AA]")

    savefig(fig, "fig_training_prediction.pdf")
    return axes


# ---------------------------------------------------------------------------
# 6. Gradient spectra (Problem 8)
# ---------------------------------------------------------------------------

_MG_LINES = [15740, 15748, 15765]
_SI_LINES = [15888, 16060, 16094]


def plot_gradient_spectra(model):
    """Five-panel gradient spectra df/dl for each label.

    Marks known Mg I and Si I lines on the [Mg/Fe] and [Si/Fe] panels.

    Parameters
    ----------
    model : CannonModel
        Trained Cannon model.
    """
    from ugdatalab.models.apogee.constants import LABEL_LATEX

    n_labels = len(model.label_names)
    fig, ax = textwidth_figure(14)
    ax.remove()
    axes = subpanels(fig, n_labels, sharex=True, hspace=0.12)

    for i in range(n_labels):
        grad = model.gradient(i)
        valid = np.isfinite(model.scatter)
        axes[i].plot(model.wavelength[valid], grad[valid], lw=LW_FINE,
                     alpha=ALPHA_STANDARD, color=f"C{i}", rasterized=True,
                     zorder=3)
        zero_line(axes[i])
        axes[i].set_ylabel(LABEL_LATEX[i], fontsize="small")
        axes[i].set_title(LABEL_LATEX[i], loc="left", fontsize="small")

        # Mark known spectral lines
        lines = []
        if model.label_names[i] == "MG_FE":
            lines = _MG_LINES
        elif model.label_names[i] == "SI_FE":
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

def plot_scatter_spectrum(model):
    """Per-pixel intrinsic scatter s² vs wavelength.

    Parameters
    ----------
    model : CannonModel
        Trained Cannon model.
    """
    fig, ax = textwidth_figure(5)

    valid = np.isfinite(model.scatter) & (model.scatter < np.inf)
    ax.plot(model.wavelength[valid], np.sqrt(model.scatter[valid]),
            lw=LW_FINE, alpha=ALPHA_STANDARD, color="C0", rasterized=True,
            zorder=3)
    ax.set_xlabel(r"Wavelength [\AA]")
    ax.set_ylabel(r"Intrinsic scatter $s_\lambda$")
    ax.set_title("Per-pixel intrinsic scatter", loc="left", fontsize="small")

    savefig(fig, "fig_scatter_spectrum.pdf")
    return ax


# ---------------------------------------------------------------------------
# 8. Label recovery (Problem 9)
# ---------------------------------------------------------------------------

def _plot_label_recovery_impl(true_labels, fitted_labels, label_names, output_name):
    """Core implementation for label recovery plots."""
    n_labels = true_labels.shape[1]
    fig, subfigs = textwidth_figure(14, subfigures=(1, n_labels))

    all_axes = []
    for i in range(n_labels):
        axes = subpanels(subfigs[i], 2, height_ratios=(3, 1), hspace=0.05)
        all_axes.append(axes)

        t = true_labels[:, i]
        f = fitted_labels[:, i]
        resid = f - t
        bias = np.mean(resid)
        scatter = np.std(resid)

        # Main panel: 1:1 scatter
        axes[0].scatter(t, f, **SCATTER_STYLE, color="C0", zorder=3,
                        rasterized=True)
        lo = min(np.min(t), np.min(f))
        hi = max(np.max(t), np.max(f))
        margin = 0.05 * (hi - lo)
        axes[0].plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                     **MODEL_STYLE, color=NEUTRAL_COLOR, zorder=1)
        axes[0].set_xlim(lo - margin, hi + margin)
        axes[0].set_ylim(lo - margin, hi + margin)
        axes[0].set_ylabel(rf"Fitted {label_names[i]}")
        axes[0].set_title(
            rf"bias$={bias:+.2f}$, $\sigma={scatter:.2f}$",
            loc="left", fontsize="x-small",
        )

        # Residual panel
        axes[1].scatter(t, resid, **SCATTER_STYLE, color="C0", zorder=3,
                        rasterized=True)
        zero_line(axes[1])
        axes[1].set_xlabel(rf"True {label_names[i]}")
        axes[1].set_ylabel("Res.")

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

def plot_outlier_spectra(wavelength, flux_obs, error_obs, flux_pred, apogee_ids,
                         wl_min=15800, wl_max=16200):
    """Overlay observed and model spectra for outlier stars.

    Parameters
    ----------
    wavelength : ndarray, shape (n_pixels,)
    flux_obs : ndarray, shape (n_stars, n_pixels)
        Observed spectra of outlier stars.
    error_obs : ndarray, shape (n_stars, n_pixels)
    flux_pred : ndarray, shape (n_stars, n_pixels)
        Cannon-predicted spectra for the outlier stars.
    apogee_ids : array-like
        IDs for each star (used for titles).
    wl_min, wl_max : float
        Wavelength window.
    """
    n_stars = flux_obs.shape[0]
    mask = (wavelength >= wl_min) & (wavelength <= wl_max)
    wl = wavelength[mask]

    fig, ax = textwidth_figure(4 * n_stars)
    ax.remove()
    axes = subpanels(fig, n_stars, sharex=True, hspace=0.15)
    if n_stars == 1:
        axes = [axes]

    for i in range(n_stars):
        obs = flux_obs[i, mask]
        pred = flux_pred[i, mask]
        err = error_obs[i, mask]
        valid = np.isfinite(err) & (err < np.inf)

        axes[i].errorbar(wl[valid], obs[valid], yerr=err[valid],
                         **ERRORBAR_STYLE, color="C0", alpha=ALPHA_FAINT,
                         zorder=2)
        axes[i].plot(wl, pred, **FIT_STYLE, color="C1", zorder=3)
        axes[i].set_ylabel("Flux")
        axes[i].set_title(str(apogee_ids[i]), loc="left", fontsize="small")

    axes[-1].set_xlabel(r"Wavelength [\AA]")

    savefig(fig, "fig_outlier_spectra.pdf")
    return axes


# ---------------------------------------------------------------------------
# 10. Kiel diagram (Problem 11)
# ---------------------------------------------------------------------------

def plot_kiel_diagram(fitted_labels, label_names, isochrone_tracks=None):
    """Kiel diagram (log g vs Teff) colored by [Fe/H] with MIST isochrones.

    Parameters
    ----------
    fitted_labels : ndarray, shape (N, 5)
        Fitted stellar labels [Teff, logg, [Fe/H], [Mg/Fe], [Si/Fe]].
    label_names : list of str
        Label column names (to identify which column is which).
    isochrone_tracks : list of (label_str, DataFrame), optional
        Each entry is (label, isochrone_df) with columns ``Teff`` and ``logg``.
    """
    teff = fitted_labels[:, 0]
    logg = fitted_labels[:, 1]
    feh = fitted_labels[:, 2]

    fig, ax = textwidth_figure(8)

    sc = ax.scatter(teff, logg, c=feh, s=SS_FINE, alpha=ALPHA_STANDARD,
                    cmap="coolwarm", zorder=3, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$[\mathrm{Fe/H}]$")

    if isochrone_tracks is not None:
        for label, iso_df in isochrone_tracks:
            iso_mask = (iso_df["Teff"] > 3500) & (iso_df["Teff"] < 6000) & (iso_df["logg"] < 4.5)
            ax.plot(iso_df["Teff"][iso_mask], iso_df["logg"][iso_mask],
                    **MODEL_STYLE, color=NEUTRAL_COLOR, zorder=1,
                    label=label)

    # Inverted axes: hot→cold left→right, low-g at top
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r"$T_{\rm eff}$ [K]")
    ax.set_ylabel(r"$\log g$ [dex]")

    if isochrone_tracks is not None:
        ax.legend(fontsize="x-small")

    savefig(fig, "fig_kiel_diagram.pdf")
    return ax


# ---------------------------------------------------------------------------
# 11. Metallicity sequence (Problem 13)
# ---------------------------------------------------------------------------

def plot_metallicity_sequence(model, feh_values, teff=4800, logg=2.5,
                              mg_fe=0.0, si_fe=0.0,
                              wl_min=16000, wl_max=16200):
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

    fig, ax = textwidth_figure(8)

    for i, feh in enumerate(feh_values):
        labels = np.array([teff, logg, feh, mg_fe, si_fe])
        spec = model.predict(labels)
        offset = -0.15 * i
        ax.plot(wl, spec[mask] + offset, lw=LW_FINE, alpha=ALPHA_STANDARD,
                color=f"C{i % 10}", zorder=3,
                label=rf"$[\mathrm{{Fe/H}}]={feh:+.2f}$")

    ax.set_xlabel(r"Wavelength [\AA]")
    ax.set_ylabel(r"Normalized flux $+$ offset")
    ax.set_title(
        rf"$T_{{\rm eff}}={teff}$ K, $\log g={logg}$",
        loc="left", fontsize="small",
    )
    ax.legend(fontsize="x-small", ncols=2)

    savefig(fig, "fig_metallicity_sequence.pdf")
    return ax


# ---------------------------------------------------------------------------
# 12. RGB evolution (Problem 14)
# ---------------------------------------------------------------------------

def plot_rgb_evolution(model, teff_track, logg_track, feh=0.0,
                       mg_fe=0.0, si_fe=0.0,
                       wl_min=16000, wl_max=16200):
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

    fig, ax = textwidth_figure(8)

    for i, (t, g) in enumerate(zip(teff_track, logg_track)):
        labels = np.array([t, g, feh, mg_fe, si_fe])
        spec = model.predict(labels)
        offset = -0.15 * i
        ax.plot(wl, spec[mask] + offset, lw=LW_FINE, alpha=ALPHA_STANDARD,
                color=f"C{i % 10}", zorder=3,
                label=rf"$T_{{\rm eff}}={t:.0f}$, $\log g={g:.1f}$")

    ax.set_xlabel(r"Wavelength [\AA]")
    ax.set_ylabel(r"Normalized flux $+$ offset")
    ax.set_title(
        rf"RGB track, $[\mathrm{{Fe/H}}]={feh:+.1f}$",
        loc="left", fontsize="small",
    )
    ax.legend(fontsize="x-small", ncols=2)

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

    fig, ax = textwidth_figure(5)

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
