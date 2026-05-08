from pathlib import Path

import corner
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np

from ugdatalab.methods.fourier import fourier_fit, phase_fold
from cross_validate import cross_validate
from ugdatalab.plotters.bayesian import predict_posterior
from ugdatalab.models.gaia import GaiaData, GaiaLightcurves
from ugdatalab.plotting import (
    TEXTWIDTH_IN,
    LW_NONE,
    LW_FINE,
    LW_LIGHT,
    LW_STANDARD,
    LW_MEDIUM,
    MS_MICRO,
    MS_FINE,
    MS_STANDARD,
    MS_MEDIUM,
    SS_MICRO,
    SS_FINE,
    SS_STANDARD,
    ALPHA_EXTRA_LIGHT,
    ALPHA_FAINT,
    ALPHA_LIGHT,
    ALPHA_STANDARD,
    NEUTRAL_COLOR,
    ERRORBAR_STYLE,
    GRID_STYLE,
    GUIDE_STYLE,
    FILL_STYLE,
    FIT_STYLE,
    MODEL_STYLE,
    LABEL_SIZE,
    LEGEND_SIZE,
    textwidth_figure,
    columnwidth_figure,
    subpanels,
    zero_line,
    corner_figure,
    landscapewidth_figure,
)

_FIGURES_DIR = Path(__file__).parent / "report" / "figures"
_CLASS_COLORS = {"RRab": "C0", "RRc": "C1", "RRd": "C2"}

# --- Padding / margins ---
# Generic axis padding as fraction of data range
_PAD_FRACTION_DATA = 0.05
# Symmetric residual-axis padding multiplier
_RESIDUAL_PAD_FACTOR = 1.1

# --- Phase grid resolution ---
# Sample count for phase-folded curve evaluation
_PHASE_GRID_POINTS = 1000

# --- Subplot geometry ---
# Vertical gap between raw and phase-folded panels
_HSPACE_RAW_PHASED = 0.42
# Horizontal gap for two-column row layouts
_WSPACE_TWO_COL = 0.28
# Main / residual panel height ratio
_HEIGHT_RATIO_FOURIER_RESID = (2.5, 1)

# --- Normalized residual histogram ---
# Normalized-residual histogram range in σ
_NORM_RESIDUAL_RANGE = (-5.0, 5.0)
# Histogram bins across the residual range
_NORM_RESIDUAL_BINS = 31
# Smooth-curve sample count over the same range
_NORM_RESIDUAL_GRID_POINTS = 400

# --- Lightcurve extrapolation window ---
# Days before last epoch shown in extrapolation panel
_EXTRAP_LOOKBACK_DAYS = 5.0
# Days after last epoch shown
_EXTRAP_FORWARD_DAYS = 12.0
# Offset for the predicted-magnitude marker
_EXTRAP_PREDICTION_OFFSET_DAYS = 10.0

# --- Sky / Galactic plotting ---
# Half-width of the |b| band drawn on Aitoff plots
_GALACTIC_LATITUDE_BAND_DEG = 30

# --- Period / log-scale ticks ---
# Minor-tick fractions on log-period axes
_LOG_LOCATOR_SUBS_PERIOD = [0.2, 0.4, 0.6, 0.8, 1.0]

# --- Corner / contour ---
# 1σ and 2σ enclosed-mass contour levels
_CORNER_CONTOUR_LEVELS = (0.393, 0.865)

# --- Reddening / SFD comparison ---
# Y-axis ceiling for the full empirical-A_G panel
_EXTINCTION_CEILING_MAG = 20.0
# Y-axis ceiling for the large-SFD regime panel
_EXTINCTION_LARGE_REGIME_MAG = 10.0
# Lower colorbar percentile for reddening maps
_PERCENTILE_LOW = 0.5
# Upper colorbar percentile for reddening maps
_PERCENTILE_HIGH = 99.5
# Minimum points in a bin to compute a median trend
_MIN_POINTS_PER_BIN = 10


def savefig(fig, name):
    """Write *fig* to ``report/figures/<name>``, creating the directory if needed."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES_DIR / name)


# ---------------------------------------------------------------------------
# 1. Raw + phase-folded lightcurve
# ---------------------------------------------------------------------------

def plot_raw_phase_folded_lightcurve(rrlyrae, source_id):
    """Two-panel plot: raw time series (top) and phase-folded lightcurve (bottom)."""
    lc = rrlyrae.data
    data = lc[lc["source_id"] == source_id]

    period = data["rrlyrae_representative_period"][0]
    epoch = data["g_transit_time"]
    mag = data["g_transit_mag"]
    mag_err = data["g_transit_mag_err"]
    phase = phase_fold(epoch, period)

    fig, ax = columnwidth_figure(33 / 4)
    ax.remove()
    axes = subpanels(fig, 2, hspace=_HSPACE_RAW_PHASED, sharex=False, sharey=True)

    axes[0].errorbar(epoch, mag, yerr=mag_err, ecolor="C0", fmt="none", alpha=ALPHA_FAINT, zorder=1, **ERRORBAR_STYLE)
    axes[0].scatter(epoch, mag, s=SS_MICRO, alpha=ALPHA_STANDARD, color="C0",
                    zorder=2, rasterized=True, label="Raw light curve")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Time [days]")
    axes[0].set_ylabel(r"$G$ [mag]")
    axes[0].legend(loc="best")

    axes[1].errorbar(phase, mag, yerr=mag_err, ecolor="C1", fmt="none", alpha=ALPHA_FAINT, zorder=1, **ERRORBAR_STYLE)
    axes[1].scatter(phase, mag, s=SS_MICRO, alpha=ALPHA_STANDARD, color="C1",
                    zorder=2, rasterized=True, label="Phase-folded light curve")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Phase")
    axes[1].set_ylabel(r"$G$ [mag]")
    axes[1].legend(loc="best")

    savefig(fig, "fig_lc_raw_phased.pdf")
    return axes


# ---------------------------------------------------------------------------
# 2. Lomb-Scargle periodogram
# ---------------------------------------------------------------------------

def plot_lomb_scargle_periodogram(result):
    """Plot periodogram power vs period with best-period marker."""
    order = np.argsort(result.periods)
    periods = result.periods[order]
    power = result.power[order]

    fig, ax = columnwidth_figure(4)

    ax.plot(periods, power, color="C0", lw=LW_STANDARD, alpha=ALPHA_STANDARD)
    ax.axvline(result.best_period, color="C1", ls="--", lw=LW_LIGHT,
               alpha=ALPHA_STANDARD, label=rf"$P={result.best_period:.4f}$ d")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$P_{\rm LS}$ [days]")
    ax.set_ylabel("Lomb-Scargle power")
    ax.legend()

    savefig(fig, "fig_periodogram.pdf")
    return ax


# ---------------------------------------------------------------------------
# 3. Period comparison (P_F vs P_LS and P_1O vs P_LS)
# ---------------------------------------------------------------------------

def plot_vari_rrlyrae_period_comparison(rrlyrae):
    """Two-panel comparison of catalog periods to Lomb-Scargle periods."""
    lc = rrlyrae.data
    _, first_idx = np.unique(lc["source_id"], return_index=True)
    rows = lc[first_idx]

    classifications = rows["best_classification"]
    pf = rows["pf"]
    pf_err = rows["pf_error"]
    p_ls = rows["period_ls"]
    p1o = rows["p1_o"]
    p1o_err = rows["p1_o_error"]

    finite_f = np.isfinite(pf) & np.isfinite(p_ls)

    fig, ax = textwidth_figure(10)
    ax.remove()
    axes = subpanels(fig, 1, 2, sharex=False, wspace=_WSPACE_TWO_COL)
    ax_f, ax_o = axes

    # --- Left panel: P_F vs P_LS ---
    for label, color in _CLASS_COLORS.items():
        if label == "RRc":
            continue
        mask = classifications == label
        m = finite_f & mask
        m_err = m & np.isfinite(pf_err)
        ax_f.errorbar(pf[m_err], p_ls[m_err], xerr=pf_err[m_err],
                      fmt="none", ecolor=color, elinewidth=LW_FINE,
                      alpha=ALPHA_FAINT, zorder=2)
        ax_f.scatter(pf[m], p_ls[m], s=SS_FINE, alpha=ALPHA_STANDARD,
                     color=color, label=label, zorder=3)

    ax_f.autoscale_view()
    lo = min(ax_f.get_xlim()[0], ax_f.get_ylim()[0])
    hi = max(ax_f.get_xlim()[1], ax_f.get_ylim()[1])
    ax_f.plot([lo, hi], [lo, hi], color="C3", **MODEL_STYLE,
              label=r"$P_{\rm LS}=P_{\rm F}$", zorder=1)

    p1o_over_pf = np.where(np.isfinite(p1o) & (pf > 0), p1o / pf, np.nan)
    rrd_ratios = p1o_over_pf[np.isfinite(p1o_over_pf)]
    ratio = np.median(rrd_ratios)
    ax_f.plot([lo, hi], [ratio * lo, ratio * hi], color="C4",
              **MODEL_STYLE, zorder=1,
              label=rf"$P_{{\rm LS}}\approx({ratio:.3f})\,P_{{\rm F}}$")

    ax_f.set_xlabel(r"\texttt{vari\_rrlyrae} fundamental period $P_{\rm F}$ [days]")
    ax_f.set_ylabel(r"L-S period $P_{\rm LS}$ [days]")
    ax_f.set_xlim(lo, hi)
    ax_f.set_ylim(lo, hi)
    ax_f.set_aspect("equal", adjustable="box")
    ax_f.legend(ncols=2)

    # --- Right panel: P_1O vs P_LS (RRd only) ---
    finite_rrd = (classifications == "RRd") & np.isfinite(p1o) & np.isfinite(p_ls)
    rrd_color = _CLASS_COLORS["RRd"]
    m_err = finite_rrd & np.isfinite(p1o_err)
    ax_o.errorbar(p1o[m_err], p_ls[m_err], xerr=p1o_err[m_err],
                  fmt="none", ecolor=rrd_color, elinewidth=LW_FINE,
                  alpha=ALPHA_FAINT, zorder=2)
    ax_o.scatter(p1o[finite_rrd], p_ls[finite_rrd], s=SS_FINE,
                 alpha=ALPHA_STANDARD, color=rrd_color, zorder=3)
    ax_o.autoscale_view()
    lo_r = min(ax_o.get_xlim()[0], ax_o.get_ylim()[0])
    hi_r = max(ax_o.get_xlim()[1], ax_o.get_ylim()[1])
    ax_o.plot([lo_r, hi_r], [lo_r, hi_r], color="C5", **MODEL_STYLE,
              label=r"$P_{\rm LS}=P_{1\rm O}$", zorder=1)
    ax_o.set_xlim(lo_r, hi_r)
    ax_o.set_ylim(lo_r, hi_r)
    ax_o.legend()

    ax_o.set_xlabel(r"\texttt{vari\_rrlyrae} first-overtone period $P_{1\rm O}$ [days]")
    ax_o.set_ylabel(r"L-S period $P_{\rm LS}$ [days]")
    ax_o.set_aspect("equal", adjustable="box")

    savefig(fig, "fig_period_comparison.pdf")
    return axes


# ---------------------------------------------------------------------------
# 4. Fourier harmonic fits grid
# ---------------------------------------------------------------------------

def plot_fourier_harmonic_fits(rrlyrae, source_id, K_values):
    """Grid of phase-folded Fourier fits at different harmonic orders."""
    lc = rrlyrae.data
    data = lc[lc["source_id"] == source_id]

    period = data["period_ls"][0]
    epoch = data["g_transit_time"]
    mag = data["g_transit_mag"]
    mag_err = data["g_transit_mag_err"]

    phase = phase_fold(epoch, period)
    order = np.argsort(phase)
    epoch, phase, mag, mag_err = epoch[order], phase[order], mag[order], mag_err[order]

    phase_grid = np.linspace(0.0, 1.0, _PHASE_GRID_POINTS, endpoint=False)
    epoch_grid = phase_grid * period
    ncols = 2
    nrows = (len(K_values) + 1) // 2
    fig, subfigs = textwidth_figure(11, subfigures=(nrows, ncols))
    subfigs = np.atleast_2d(subfigs)

    axes = np.empty((len(K_values), 2), dtype=object)
    col_last = {}
    for i in range(len(K_values)):
        col_last[i % ncols] = i

    for i in range(len(K_values)):
        row, col = i // ncols, i % ncols
        ax_c, ax_r = subpanels(subfigs[row, col], 2, height_ratios=_HEIGHT_RATIO_FOURIER_RESID)
        axes[i, 0] = ax_c
        axes[i, 1] = ax_r

    for i, K in enumerate(K_values):
        col = i % ncols
        is_bottom = i == col_last[col]

        fit = fourier_fit(epoch, mag, mag_err, period=period, k=K)
        model_mag = fit.predict(epoch_grid)
        residuals = mag - fit.predict(epoch)

        ax_c, ax_r = axes[i]

        ax_c.errorbar(phase, mag, yerr=mag_err, fmt="o", ms=MS_MICRO,
                      elinewidth=LW_FINE, color="C0", alpha=ALPHA_FAINT, zorder=2)
        ax_c.plot(phase_grid, model_mag, color="C1", lw=LW_MEDIUM, alpha=ALPHA_STANDARD, zorder=3)
        ax_c.invert_yaxis()
        ax_c.set_ylabel(r"$G$ [mag]")
        ax_c.set_title(rf"$K={K}$, $\chi_r^2={fit.chi2_r:.2f}$",
                       loc="left", fontsize=LEGEND_SIZE)
        ax_c.tick_params(axis="x", bottom=False, labelbottom=False)

        ax_r.errorbar(phase, residuals, yerr=mag_err, fmt="o", ms=MS_MICRO,
                      elinewidth=LW_FINE, color="C0", alpha=ALPHA_FAINT, zorder=2)
        zero_line(ax_r)
        ax_r.set_ylabel("Res.")
        if is_bottom:
            ax_r.set_xlabel("Phase")
        else:
            ax_r.tick_params(axis="x", bottom=False, labelbottom=False)

    savefig(fig, "fig_fourier_harmonics.pdf")
    return axes


# ---------------------------------------------------------------------------
# 5. Cross-validation chi2 vs K
# ---------------------------------------------------------------------------

def plot_fourier_cross_validation(result):
    """Plot training reduced chi-squared and CV mean chi-squared vs harmonic order."""
    Ks = result.param_values
    chi2r_train = result.chi2r_train
    mean_chi2_cv = result.mean_chi2_cv
    best_K = result.best_param

    valid = np.isfinite(chi2r_train) & np.isfinite(mean_chi2_cv)
    cv_best = mean_chi2_cv[Ks == best_K][0]

    fig, ax = columnwidth_figure(21 / 4)

    ax.plot(Ks[valid], chi2r_train[valid], marker="o", ls="none", ms=MS_FINE,
            alpha=ALPHA_STANDARD, label=r"Training $\chi_r^2$")
    ax.plot(Ks[valid], mean_chi2_cv[valid], marker="o", ls="none", ms=MS_FINE,
            alpha=ALPHA_STANDARD, label=r"Cross-validation $\langle\chi^2\rangle$")
    ax.axvline(best_K, color="C2", ls=":", lw=LW_LIGHT, alpha=ALPHA_LIGHT,
               label=rf"Best $K={best_K}$")
    ax.axhline(cv_best, color="C2", ls=":", lw=LW_LIGHT, alpha=ALPHA_LIGHT,
               label=rf"Best CV $\langle\chi^2\rangle={cv_best:.3f}$")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel(r"$K$ (number of Fourier harmonics)")
    ax.set_ylabel(r"$\chi_r^2$")
    ax.set_title(rf"20\% CV, train={len(result.train_idx)}, CV={len(result.cv_idx)}",
                 loc="left", fontsize=LEGEND_SIZE)
    ax.legend()

    savefig(fig, "fig_crossval.pdf")
    return ax


# ---------------------------------------------------------------------------
# 6. CV normalized residual histograms
# ---------------------------------------------------------------------------

def plot_fourier_cv_residual_histograms(rrlyrae, source_id, result, low_fit, best_fit):
    """Two-panel histogram of normalized residuals for low-K and best-K fits."""
    lc = rrlyrae.data
    lc = lc[lc["source_id"] == source_id]
    epochs = lc["g_transit_time"]
    mags = lc["g_transit_mag"]
    errs = lc["g_transit_mag_err"]

    train, cv = result.train_idx, result.cv_idx

    bins = np.linspace(*_NORM_RESIDUAL_RANGE, _NORM_RESIDUAL_BINS)
    x_gauss = np.linspace(*_NORM_RESIDUAL_RANGE, _NORM_RESIDUAL_GRID_POINTS)
    gaussian = np.exp(-0.5 * x_gauss**2) / np.sqrt(2.0 * np.pi)

    fig, ax = textwidth_figure(3)
    ax.remove()
    axes = subpanels(fig, 1, 2, wspace=_WSPACE_TWO_COL, sharex=True)

    panels = [
        (axes[0], low_fit, rf"Low $K = {low_fit.k}$"),
        (axes[1], best_fit, rf"Best $K = {best_fit.k}$"),
    ]

    for ax, fit, label in panels:
        train_norm = (mags[train] - fit.predict(epochs[train])) / errs[train]
        cv_norm = (mags[cv] - fit.predict(epochs[cv])) / errs[cv]
        train_norm = train_norm[np.isfinite(train_norm)]
        cv_norm = cv_norm[np.isfinite(cv_norm)]

        ax.hist(cv_norm, bins=bins, histtype="stepfilled", density=True,
                color="C1", alpha=ALPHA_FAINT, lw=LW_FINE, label="CV", zorder=1)
        ax.hist(train_norm, bins=bins, histtype="step", density=True,
                color="C0", alpha=ALPHA_STANDARD, lw=LW_MEDIUM, label="Training", zorder=2)
        ax.plot(x_gauss, gaussian, color=NEUTRAL_COLOR, ls="--", lw=LW_STANDARD,
                alpha=ALPHA_STANDARD, label=r"$\mathcal{N}(0,1)$", zorder=3)
        ax.axvline(0.0, color=NEUTRAL_COLOR, ls=":", lw=LW_STANDARD)
        ax.set_title(label, loc="left", fontsize=LEGEND_SIZE)
        ax.set_xlabel(r"$(G - G_{\rm model})/\sigma$")

    axes[0].set_ylabel("Density")
    axes[1].set_ylabel("Density")
    axes[1].legend(loc="best")

    savefig(fig, "fig_fourier_cv_residuals.pdf")
    return axes


# ---------------------------------------------------------------------------
# 7. CV phase comparison (best K vs high K)
# ---------------------------------------------------------------------------

def plot_fourier_cv_phase_comparison(rrlyrae, source_id, result, best_fit, high_fit):
    """Two-panel phase-folded comparison showing train/CV split for best and high K."""
    lc = rrlyrae.data
    lc = lc[lc["source_id"] == source_id]
    epochs = lc["g_transit_time"]
    mags = lc["g_transit_mag"]
    errs = lc["g_transit_mag_err"]

    train, cv = result.train_idx, result.cv_idx
    period = best_fit.period
    train_phase = phase_fold(epochs[train], period)
    cv_phase = phase_fold(epochs[cv], period)
    phase_grid = np.linspace(0.0, 1.0, _PHASE_GRID_POINTS, endpoint=False)
    epoch_grid = phase_grid * period

    all_mags = np.concatenate([mags[train], mags[cv]])
    pad = _PAD_FRACTION_DATA * (np.max(all_mags) - np.min(all_mags))
    y_lim = (np.max(all_mags) + pad, np.min(all_mags) - pad)

    fig, ax = textwidth_figure(3)
    ax.remove()
    axes = subpanels(fig, 1, 2, wspace=_WSPACE_TWO_COL, sharex=False)

    for ax, fit, K in [(axes[0], best_fit, best_fit.k), (axes[1], high_fit, high_fit.k)]:
        ax.scatter(train_phase, mags[train], s=SS_MICRO, alpha=ALPHA_EXTRA_LIGHT,
                   color="C0", rasterized=True, label="Training")
        ax.scatter(cv_phase, mags[cv], s=SS_MICRO, alpha=ALPHA_FAINT,
                   color="C1", rasterized=True, label="CV")
        ax.errorbar(train_phase, mags[train], yerr=errs[train],
                    ecolor="C0", fmt="none", alpha=ALPHA_FAINT, zorder=1, **ERRORBAR_STYLE)
        ax.errorbar(cv_phase, mags[cv], yerr=errs[cv],
                    ecolor="C1", fmt="none", alpha=ALPHA_FAINT, zorder=1, **ERRORBAR_STYLE)
        ax.plot(phase_grid, fit.predict(epoch_grid), color=NEUTRAL_COLOR,
                lw=LW_STANDARD, alpha=ALPHA_STANDARD, label="Model")
        ax.set_ylim(y_lim)
        ax.invert_yaxis()
        ax.set_xlabel("Phase")
        tag = "Best" if K == best_fit.k else "High"
        ax.set_title(rf"{tag} $K = {K}$", loc="left", fontsize=LEGEND_SIZE)

    axes[0].set_ylabel(r"$G$ [mag]")
    axes[1].set_ylabel(r"$G$ [mag]")
    axes[0].legend(loc="best")

    savefig(fig, "fig_fourier_cv_phase.pdf")
    return axes


# ---------------------------------------------------------------------------
# 8. Mean G catalog comparison
# ---------------------------------------------------------------------------

def plot_mean_g_catalog_comparison(rrlyrae):
    """Two-column comparison of epoch-mean and Fourier-mean G vs Gaia int_average_g."""
    lc = rrlyrae.data
    _, first_idx = np.unique(lc["source_id"], return_index=True)
    rows = lc[first_idx]

    simple_g = rows["mean_g_transit_mag"]
    simple_g_err = rows["mean_g_transit_mag_err"]
    fourier_g = rows["fourier_mean_g_mag"]
    fourier_g_err = rows["fourier_mean_g_mag_err"]
    int_g = rows["int_average_g"]

    resid_s = simple_g - int_g
    resid_f = fourier_g - int_g
    resid_s_err = np.hypot(np.nan_to_num(simple_g_err), 0.0)
    resid_f_err = np.hypot(np.nan_to_num(fourier_g_err), 0.0)

    fig, (sf_l, sf_r) = textwidth_figure(19 / 2, subfigures=(1, 2))
    ax_s, ax_sr = subpanels(sf_l, 2, height_ratios=(5.5, 1))
    ax_f, ax_fr = subpanels(sf_r, 2, height_ratios=(5.5, 1))
    axes = np.array([ax_s, ax_sr, ax_f, ax_fr])

    # Shared limits across both columns
    valid_s = np.isfinite(simple_g) & np.isfinite(int_g) & np.isfinite(resid_s)
    valid_f = np.isfinite(fourier_g) & np.isfinite(int_g) & np.isfinite(resid_f)
    all_mag = np.concatenate([int_g[valid_s], simple_g[valid_s],
                              int_g[valid_f], fourier_g[valid_f]])
    pad = _PAD_FRACTION_DATA * (np.max(all_mag) - np.min(all_mag))
    mag_lo = np.min(all_mag) - pad
    mag_hi = np.max(all_mag) + pad

    all_resid = np.concatenate([
        resid_s[valid_s] - resid_s_err[valid_s], resid_s[valid_s] + resid_s_err[valid_s],
        resid_f[valid_f] - resid_f_err[valid_f], resid_f[valid_f] + resid_f_err[valid_f],
    ])
    resid_max = _RESIDUAL_PAD_FACTOR * np.nanmax(np.abs(all_resid))

    panels = [
        (ax_s, ax_sr, simple_g, simple_g_err, resid_s, resid_s_err, valid_s,
         "C0", r"$\langle G \rangle_{\rm epoch}$ [mag]"),
        (ax_f, ax_fr, fourier_g, fourier_g_err, resid_f, resid_f_err, valid_f,
         "C1", r"$\langle G \rangle_{\rm Fourier}$ [mag]"),
    ]

    for ax_m, ax_r, y, y_err, resid, r_err, valid, color, ylabel in panels:
        ax_m.errorbar(int_g[valid], y[valid], yerr=y_err[valid], ms=MS_FINE,
                      ecolor=color, fmt="none", alpha=ALPHA_FAINT, zorder=1, **ERRORBAR_STYLE)
        ax_m.scatter(int_g[valid], y[valid], s=SS_FINE, alpha=ALPHA_STANDARD,
                     color=color, rasterized=True)
        ax_m.plot([mag_lo, mag_hi], [mag_lo, mag_hi], **GUIDE_STYLE)
        ax_m.set_xlim(mag_hi, mag_lo)
        ax_m.set_ylim(mag_hi, mag_lo)
        ax_m.set_ylabel(ylabel)
        ax_m.tick_params(labelbottom=False)

        ax_r.errorbar(int_g[valid], resid[valid], yerr=r_err[valid], ms=MS_FINE,
                      ecolor=color, fmt="none", alpha=ALPHA_FAINT, zorder=1, **ERRORBAR_STYLE)
        ax_r.scatter(int_g[valid], resid[valid], s=SS_FINE, alpha=ALPHA_STANDARD,
                     color=color, rasterized=True)
        zero_line(ax_r)
        ax_r.set_xlim(mag_hi, mag_lo)
        ax_r.set_ylim(-resid_max, resid_max)
        ax_r.set_xlabel(r"Gaia $\mathtt{int\_average\_g}$ [mag]")
        ax_r.set_ylabel("Res.")

    savefig(fig, "fig_mean_g_comparison.pdf")
    return axes


# ---------------------------------------------------------------------------
# 9. Fourier extrapolation
# ---------------------------------------------------------------------------

def plot_fourier_extrapolation(fit):
    """Plot last few days of data with 12-day extrapolation and 10-day prediction."""
    epochs = fit.x
    mags = fit.y
    mag_errs = fit.y_err

    epoch_last = np.max(epochs)
    epoch_start = epoch_last - _EXTRAP_LOOKBACK_DAYS
    epoch_end = epoch_last + _EXTRAP_FORWARD_DAYS
    epoch_grid = np.linspace(epoch_start, epoch_end, 2000)
    mag_grid = fit.predict(epoch_grid)
    mag_grid_err = fit.predict_std(epoch_grid)

    epoch_pred = epoch_last + _EXTRAP_PREDICTION_OFFSET_DAYS
    mag_pred = fit.predict(np.array([epoch_pred]))[0]
    mag_pred_err = fit.predict_std(np.array([epoch_pred]))[0]

    observed_mask = epochs >= epoch_start
    model_observed = epoch_grid <= epoch_last

    fig, ax = textwidth_figure(5)

    ax.errorbar(epochs[observed_mask], mags[observed_mask], yerr=mag_errs[observed_mask],
                fmt="o", ms=MS_STANDARD, elinewidth=LW_FINE, color="C0",
                alpha=ALPHA_STANDARD, label="Gaia data")
    ax.plot(epoch_grid[model_observed], mag_grid[model_observed], color="C1",
            lw=LW_STANDARD, alpha=ALPHA_STANDARD, label=rf"Fourier fit ($K={fit.k}$)")
    ax.plot(epoch_grid[~model_observed], mag_grid[~model_observed], color="C1",
            lw=LW_STANDARD, alpha=ALPHA_STANDARD, ls="--", label="12-day extrapolation")

    ax.fill_between(epoch_grid, mag_grid - mag_grid_err, mag_grid + mag_grid_err,
                    color="C1", **FILL_STYLE, label=r"Model $\pm1\sigma$")

    ax.axvline(epoch_last, color=NEUTRAL_COLOR, ls=":", lw=LW_STANDARD, alpha=ALPHA_LIGHT)
    ax.plot(epoch_pred, mag_pred, marker="*", ms=MS_MEDIUM, color="k", lw=LW_NONE,
            alpha=ALPHA_STANDARD,
            label=rf"10-day prediction: $G={mag_pred:.3f}\pm{mag_pred_err:.3f}$")
    ax.invert_yaxis()
    ax.set_xlim(epoch_start, epoch_end)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(r"$G$ [mag]")
    ax.legend(ncols=2, loc="upper right")

    savefig(fig, "fig_fourier_extrapolation.pdf")
    return ax


# ---------------------------------------------------------------------------
# 10. RRab vs RRc shape comparison
# ---------------------------------------------------------------------------

def _prepare_shape_panels(rrlyrae, rr_class):
    """Build per-source panel data for shape comparison."""
    lc = rrlyrae.data
    period_col = "p1_o" if rr_class == "RRc" else "pf"
    phase_grid = np.linspace(0.0, 1.0, _PHASE_GRID_POINTS, endpoint=False)

    panels = []
    for row in rrlyrae.source.data:
        sid = row["source_id"]
        period = row[period_col]
        star = lc[lc["source_id"] == sid]

        cv_period = star["period_ls"][0]
        cv = cross_validate(
            star["g_transit_time"], star["g_transit_mag"], star["g_transit_mag_err"],
            lambda x, y, ye, k, p=cv_period: fourier_fit(x, y, ye, p, k),
            np.arange(1, 26),
        )
        best_K = cv.best_param
        fit = fourier_fit(
            star["g_transit_time"], star["g_transit_mag"], star["g_transit_mag_err"],
            period, best_K,
        )
        mean_g = star["fourier_mean_g_mag"][0]

        phase = phase_fold(star["g_transit_time"], period)
        mag_centered = star["g_transit_mag"] - mean_g
        model_centered = fit.predict(star["g_transit_time"]) - mean_g
        residuals = mag_centered - model_centered
        order = np.argsort(phase)

        panels.append(dict(
            source_id=sid, rr_class=rr_class,
            phase=phase[order], mag_centered=mag_centered[order],
            mag_err=star["g_transit_mag_err"][order], residuals=residuals[order],
            phase_grid=phase_grid,
            model_centered=fit.predict(phase_grid * period) - mean_g,
            best_K=best_K, period=period, chi2_r=fit.chi2_r, mean_g=mean_g,
        ))
    return panels


def plot_rrlyrae_shape_comparison(rrab, rrc):
    """Grid comparing RRab and RRc phase-folded lightcurve shapes."""
    rrab_panels = _prepare_shape_panels(rrab, "RRab")
    rrc_panels = _prepare_shape_panels(rrc, "RRc")
    all_panels = rrab_panels + rrc_panels

    # Shared y-limits across all panels
    all_mag, all_resid = [], []
    for p in all_panels:
        finite = np.isfinite(p["mag_centered"])
        all_mag.append(p["mag_centered"][finite])
        all_mag.append(p["model_centered"][np.isfinite(p["model_centered"])])
        r_finite = np.isfinite(p["residuals"])
        all_resid.append(p["residuals"][r_finite])

    mag_vals = np.concatenate(all_mag)
    pad = max(_PAD_FRACTION_DATA * (np.max(mag_vals) - np.min(mag_vals)), _PAD_FRACTION_DATA)
    y_lim = (np.min(mag_vals) - pad, np.max(mag_vals) + pad)

    resid_vals = np.concatenate(all_resid)
    resid_max = max(np.max(np.abs(resid_vals)), _PAD_FRACTION_DATA) * _RESIDUAL_PAD_FACTOR
    resid_lim = (-resid_max, resid_max)

    nrows = max(len(rrab_panels), len(rrc_panels))
    fig, subfigs = textwidth_figure(56 * nrows / TEXTWIDTH_IN, subfigures=(nrows, 2))
    subfigs = np.atleast_2d(subfigs)

    axes = np.empty((nrows, 2, 2), dtype=object)
    columns = [
        (rrab_panels, _CLASS_COLORS["RRab"], "RRab"),
        (rrc_panels, _CLASS_COLORS["RRc"], "RRc"),
    ]

    for col, (panel_list, color, rr_class) in enumerate(columns):
        for row in range(nrows):
            ax_c, ax_r = subpanels(subfigs[row, col], 2, height_ratios=(3, 1))
            axes[row, col, 0] = ax_c
            axes[row, col, 1] = ax_r

            if row >= len(panel_list):
                ax_c.axis("off")
                ax_r.axis("off")
                continue

            p = panel_list[row]
            ax_c.errorbar(p["phase"], p["mag_centered"], yerr=p["mag_err"],
                          fmt="o", ms=MS_MICRO, elinewidth=LW_FINE, color=color,
                          alpha=ALPHA_FAINT, zorder=2,
                          label=rr_class if row == 0 else "_nolegend_")
            ax_c.plot(p["phase_grid"], p["model_centered"], color=NEUTRAL_COLOR,
                      lw=LW_MEDIUM, alpha=ALPHA_STANDARD, zorder=3,
                      label="Fourier model" if row == 0 else "_nolegend_")
            ax_c.set_xlim(0.0, 1.0)
            ax_c.set_ylim(y_lim)
            ax_c.invert_yaxis()
            ax_c.set_ylabel(r"$G - \langle G \rangle_{\rm Fourier}$")
            ax_c.set_title(
                rf"{rr_class}: Gaia DR3 {p['source_id']}"
                "\n"
                rf"$P={p['period']:.4f}\,\mathrm{{d}}$, $K={p['best_K']}$, "
                rf"$\chi_r^2={p['chi2_r']:.2f}$, "
                rf"$\langle G \rangle_{{\rm Fourier}}={p['mean_g']:.2f}$",
                loc="left", fontsize=LEGEND_SIZE)
            ax_c.tick_params(axis="x", bottom=False, labelbottom=False)

            ax_r.errorbar(p["phase"], p["residuals"], yerr=p["mag_err"],
                          fmt="o", ms=MS_MICRO, elinewidth=LW_FINE, color=color,
                          alpha=ALPHA_FAINT, zorder=2)
            zero_line(ax_r)
            ax_r.set_xlim(0.0, 1.0)
            ax_r.set_ylim(resid_lim)
            ax_r.set_ylabel("Res.")
            ax_r.set_xlabel("Phase")

        axes[0, col, 0].legend(loc="best")

    savefig(fig, "fig_rrab_rrc.pdf")
    return axes


# ---------------------------------------------------------------------------
# 13. Mollweide sky map: removed vs kept
# ---------------------------------------------------------------------------

def _to_rad(data):
    """Return galactic (l, b) wrapped to [-180, 180) deg and converted to radians."""
    l_wrap = np.where(data["l"] > 180, data["l"] - 360, data["l"])
    return np.deg2rad(l_wrap), np.deg2rad(data["b"])


def plot_aitoff_diff(source: GaiaData, filtered: GaiaData):
    """Aitoff projection showing removed vs kept stars after quality cuts."""
    full_data = source.data
    kept_data = filtered.data
    removed_mask = ~np.isin(full_data["source_id"], kept_data["source_id"])
    removed = full_data[removed_mask]

    fig, ax = textwidth_figure(9)
    ax.remove()
    ax = fig.add_subplot(111, projection="aitoff")

    l_rm, b_rm = _to_rad(removed)
    ax.scatter(l_rm, b_rm, marker="x", color="C1", s=SS_STANDARD,
               linewidths=LW_LIGHT, alpha=ALPHA_STANDARD, rasterized=True,
               zorder=1, label=f"Removed ($N$={len(removed)})")

    l_k, b_k = _to_rad(kept_data)
    ax.scatter(l_k, b_k, color="C0", s=SS_FINE, alpha=ALPHA_FAINT,
               rasterized=True, zorder=2, label=f"Kept ($N$={len(kept_data)})")

    longs = np.linspace(-np.pi, np.pi, 1000)
    ax.fill_between(longs, np.full_like(longs, np.deg2rad(-_GALACTIC_LATITUDE_BAND_DEG)),
                    np.full_like(longs, np.deg2rad(_GALACTIC_LATITUDE_BAND_DEG)),
                    color="C3", alpha=ALPHA_EXTRA_LIGHT,
                    label=r"$|b| < 30^\circ$ band")

    ax.set_xlabel(r"Galactic longitude $l$")
    ax.set_ylabel(r"Galactic latitude $b$")
    ax.legend(loc="upper right")

    savefig(fig, "fig_calibration_sky_c12.pdf")
    return ax


# ---------------------------------------------------------------------------
# 14. Period–absolute magnitude scatter
# ---------------------------------------------------------------------------

def plot_period_abs_mag(data):
    """Period vs absolute G-band magnitude, colored by RR Lyrae subclass."""
    periods = data["rrlyrae_representative_period"]
    m_g = data["M_G"]
    sigma_m = data["sigma_M"]
    classifications = data["best_classification"]

    fig, ax = textwidth_figure(29 / 4)

    for label, color in _CLASS_COLORS.items():
        mask = classifications == label
        if not np.any(mask):
            continue
        ax.errorbar(periods[mask], m_g[mask], yerr=sigma_m[mask], fmt="none",
                    color=color, alpha=ALPHA_LIGHT, zorder=1)
        ax.scatter(periods[mask], m_g[mask], color=color, label=label,
                   s=SS_MICRO, alpha=ALPHA_STANDARD, rasterized=True, zorder=2)

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=_LOG_LOCATOR_SUBS_PERIOD))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.autoscale_view()
    y0, y1 = ax.get_ylim()
    ax.set_ylim(max(y0, y1), min(y0, y1))
    ax.set_xlabel(r"Catalog period $P$ [days]")
    ax.set_ylabel(r"$M_{\rm G}$ [mag]")
    ax.legend(loc="lower right")

    savefig(fig, "fig_clean_period_abs_mag.pdf")
    return ax


# ---------------------------------------------------------------------------
# 15. Period–abs mag before/after C1/C2 comparison
# ---------------------------------------------------------------------------

def plot_period_abs_mag_c12_comparison(source: GaiaData, filtered: GaiaData):
    """Period-luminosity diagram showing kept points and removed x-markers."""

    post_c12_data = filtered.data
    pre_c12_data = source.data

    periods = post_c12_data["rrlyrae_representative_period"]
    m_g = post_c12_data["M_G"]
    sigma_m = post_c12_data["sigma_M"]
    classifications = post_c12_data["best_classification"]

    fig, ax = columnwidth_figure(11 / 2)

    # Plot kept points
    for label, color in _CLASS_COLORS.items():
        mask = classifications == label
        if not np.any(mask):
            continue
        ax.errorbar(periods[mask], m_g[mask], yerr=sigma_m[mask], fmt="none",
                    color=color, alpha=ALPHA_LIGHT, zorder=1)
        ax.scatter(periods[mask], m_g[mask], color=color, label=label,
                   s=SS_MICRO, alpha=ALPHA_STANDARD, rasterized=True, zorder=2)

    # Overlay removed as x markers, colored by classification
    removed_mask = ~np.isin(pre_c12_data["source_id"], post_c12_data["source_id"])
    removed = pre_c12_data[removed_mask]
    if len(removed):
        rm_periods = removed["rrlyrae_representative_period"]
        rm_mg = removed["M_G"]
        rm_class = removed["best_classification"]
        for label, color in _CLASS_COLORS.items():
            m = rm_class == label
            if not np.any(m):
                continue
            ax.scatter(rm_periods[m], rm_mg[m], marker="x", s=SS_STANDARD,
                       linewidths=LW_FINE, color=color, alpha=ALPHA_FAINT,
                       rasterized=True, zorder=3)

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=_LOG_LOCATOR_SUBS_PERIOD))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.autoscale_view()
    y0, y1 = ax.get_ylim()
    ax.set_ylim(max(y0, y1), min(y0, y1))
    ax.set_xlabel(r"Catalog period $P$ [days]")
    ax.set_ylabel(r"$M_{\rm G}$ [mag]")

    handles = [Line2D([0], [0], marker="o", ls="none", markerfacecolor=c,
                       markeredgecolor="none", ms=MS_FINE, label=l)
               for l, c in _CLASS_COLORS.items()]
    handles.append(Line2D([0], [0], marker="x", ls="none", color=NEUTRAL_COLOR,
                          ms=MS_FINE, label="Removed"))
    ax.legend(handles=handles, loc="lower right")

    savefig(fig, "fig_period_abs_mag_c12_comparison.pdf")
    return ax


# ---------------------------------------------------------------------------
# 16. Inlier probability period-luminosity comparison
# ---------------------------------------------------------------------------

def plot_inlier_prob_period_luminosity_comparison(source: GaiaData, clean: GaiaData):
    """P-M_G colored by inlier probability, with removed points marked.

    *source* must already carry the ``inlier_prob`` column (built via a
    pipeline that includes ``AttachInlierProbColumn()``); *clean* is the
    same pipeline with ``AttachInlierProbColumn(prob_threshold=...)``
    in place of the bare attach call.
    """
    pre_data = source.data
    inlier_prob = pre_data["inlier_prob"]

    periods = pre_data["rrlyrae_representative_period"]
    m_g = pre_data["M_G"]

    kept = np.isin(pre_data["source_id"], clean.data["source_id"])
    removed = ~kept

    fig, ax = columnwidth_figure(29 / 4)

    sc = ax.scatter(periods[kept], m_g[kept], c=inlier_prob[kept],
                    cmap="RdYlGn", vmin=0.0, vmax=1.0, s=SS_MICRO,
                    alpha=ALPHA_STANDARD, rasterized=True, zorder=2)
    ax.scatter(periods[removed], m_g[removed], c=inlier_prob[removed],
               cmap="RdYlGn", vmin=0.0, vmax=1.0, marker="x", s=SS_STANDARD,
               linewidths=LW_LIGHT, alpha=ALPHA_FAINT, rasterized=True, zorder=1,
               label=rf"Removed ($N={removed.sum()}$)")
    fig.colorbar(sc, ax=ax, label="Posterior inlier probability")

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=_LOG_LOCATOR_SUBS_PERIOD))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.autoscale_view()
    y0, y1 = ax.get_ylim()
    ax.set_ylim(max(y0, y1), min(y0, y1))
    ax.set_xlabel(r"$P$ [days]")
    ax.set_ylabel(r"$M_{\rm G}$ [mag]")
    ax.legend(loc="lower right")

    savefig(fig, "fig_inlier_prob_period_luminosity_comparison.pdf")
    return ax


# ---------------------------------------------------------------------------
# 17. Multi-sampler comparison corner plot
# ---------------------------------------------------------------------------

def plot_sampler_comparison_corner(mh, nuts_potential, nuts_native, output_name):
    """Overlay corner contours from MH, NUTS+Potential, and native NUTS.

    Parameters
    ----------
    mh : MHResult
        Metropolis-Hastings result.
    nuts_potential : MCMCResult
        NUTS with pm.Potential result.
    nuts_native : MCMCResult
        Native PyMC NUTS result.
    output_name : str
        Filename for the saved figure (e.g., ``"fig_methods_corner_rrab.pdf"``).
    """
    labels = [r"$a$", r"$b$", r"$\log_{10}\,\sigma_s$"]
    samplers = {
        "Metropolis-Hastings": mh,
        "NUTS + Potential": nuts_potential,
        "Native PyMC NUTS": nuts_native,
    }

    fig = corner_figure()

    handles = []
    for i, (name, result) in enumerate(samplers.items()):
        color = f"C{i}"
        fig = corner.corner(
            result.samples,
            labels=labels,
            fig=fig,
            color=color,
            fill_contours=False,
            plot_datapoints=False,
            plot_density=False,
            levels=_CORNER_CONTOUR_LEVELS,
            smooth=1.0,
            hist_kwargs=dict(
                density=True, histtype="step", linewidth=LW_STANDARD, alpha=ALPHA_LIGHT,
            ),
            contour_kwargs=dict(linewidths=LW_STANDARD, alpha=ALPHA_LIGHT),
        )
        handles.append(Line2D([0], [0], color=color, lw=LW_STANDARD, label=name))

    fig.legend(handles=handles, loc="upper right", frameon=False)
    savefig(fig, output_name)
    return fig


# ---------------------------------------------------------------------------
# 18. Optical G vs WISE W2 posterior predictive comparison
# ---------------------------------------------------------------------------

def plot_optical_vs_w2(rrab_optical, rrc_optical, rrab_w2, rrc_w2,
                       rrab_mean_log_p, rrc_mean_log_p):
    """Two-panel (top/bottom) comparison of optical G and WISE W2 PL relations.

    Top panel: RRab with G (C0) and W2 (C1) posterior predictives overlaid.
    Bottom panel: RRc with G (C0) and W2 (C1) posterior predictives overlaid.
    X-axis shows catalog period in days (log scale).

    Parameters
    ----------
    rrab_optical, rrc_optical : MCMCResult
        Optical G-band NUTS results for RRab and RRc.
    rrab_w2, rrc_w2 : MCMCResult
        WISE W2-band NUTS results for RRab and RRc.
    rrab_mean_log_p, rrc_mean_log_p : float
        Mean log10(P/day) used to center the predictor for RRab and RRc.
    """
    fig, ax_dummy = textwidth_figure(45 / 2)
    ax_dummy.remove()
    axes = subpanels(fig, 2, sharex=True, hspace=0.07)

    for ax, optical, w2, mean_lp in [
        (axes[0], rrab_optical, rrab_w2, rrab_mean_log_p),
        (axes[1], rrc_optical, rrc_w2, rrc_mean_log_p),
    ]:
        for result, color, band_label in [
            (optical, "C0", "Gaia $G$"),
            (w2, "C1", r"WISE $W\!2$"),
        ]:
            x_c, y, y_err = result.x, result.y, result.y_err
            pp = predict_posterior(result)

            periods = 10.0 ** (x_c + mean_lp)
            p_grid = 10.0 ** (pp["x_grid"] + mean_lp)

            ax.errorbar(periods, y, yerr=y_err, fmt="none", color=NEUTRAL_COLOR,
                        alpha=ALPHA_LIGHT, lw=LW_FINE, zorder=1)
            ax.scatter(periods, y, s=SS_MICRO, color=color, alpha=ALPHA_FAINT,
                       rasterized=True, zorder=2,
                       label=f"{band_label} data")
            ax.fill_between(p_grid, pp["q025"], pp["q975"], color=color,
                            **FILL_STYLE, zorder=3)
            ax.fill_between(p_grid, pp["q16"], pp["q84"], color=color,
                            alpha=ALPHA_FAINT, lw=LW_NONE, zorder=4)
            ax.plot(p_grid, pp["median"], color=color, **FIT_STYLE, zorder=5)

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=_LOG_LOCATOR_SUBS_PERIOD))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.autoscale_view()
        y0, y1 = ax.get_ylim()
        ax.set_ylim(max(y0, y1), min(y0, y1))
        ax.set_ylabel("Absolute magnitude [mag]")

    handles, _ = axes[0].get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color=NEUTRAL_COLOR, **FIT_STYLE, label="Median"),
        plt.Rectangle((0, 0), 1, 1, fc=NEUTRAL_COLOR, alpha=ALPHA_FAINT, lw=0,
                       label=r"68\% predictive"),
        plt.Rectangle((0, 0), 1, 1, fc=NEUTRAL_COLOR, **FILL_STYLE,
                       label=r"95\% predictive"),
    ])
    axes[0].legend(handles=handles, loc="best")

    axes[-1].set_xlabel(r"Catalog period $P$ [days]")

    savefig(fig, "fig_optical_ir_comparison.pdf")
    return axes


# ---------------------------------------------------------------------------
# 19. Period-color posterior predictive comparison (RRab vs RRc)
# ---------------------------------------------------------------------------

def plot_period_color_comparison(rrab_result, rrc_result,
                                rrab_mean_log_p, rrc_mean_log_p):
    """Single-panel period-color posterior predictive with RRab and RRc overlaid.

    Parameters
    ----------
    rrab_result, rrc_result : MCMCResult
        Period-color NUTS results for RRab and RRc.
    rrab_mean_log_p, rrc_mean_log_p : float
        Mean log10(P/day) used to center the predictor.
    """
    fig, ax = textwidth_figure(33 / 4)

    for result, mean_lp, color, class_label in [
        (rrab_result, rrab_mean_log_p, "C0", "RRab"),
        (rrc_result, rrc_mean_log_p, "C1", "RRc"),
    ]:
        x_c, y, y_err = result.x, result.y, result.y_err
        pp = predict_posterior(result)

        periods = 10.0 ** (x_c + mean_lp)
        p_grid = 10.0 ** (pp["x_grid"] + mean_lp)

        ax.errorbar(periods, y, yerr=y_err, fmt="none", color=NEUTRAL_COLOR,
                    alpha=ALPHA_LIGHT, lw=LW_FINE, zorder=1)
        ax.scatter(periods, y, s=SS_MICRO, color=color, alpha=ALPHA_FAINT,
                   rasterized=True, zorder=2, label=f"{class_label} data")
        ax.fill_between(p_grid, pp["q025"], pp["q975"], color=color,
                        **FILL_STYLE, zorder=3)
        ax.fill_between(p_grid, pp["q16"], pp["q84"], color=color,
                        alpha=ALPHA_FAINT, lw=LW_NONE, zorder=4)
        ax.plot(p_grid, pp["median"], color=color, **FIT_STYLE, zorder=5)

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=_LOG_LOCATOR_SUBS_PERIOD))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.autoscale_view()
    ax.set_xlabel(r"Catalog period $P$ [days]")
    ax.set_ylabel(r"$G_{\rm BP} - G_{\rm RP}$ [mag]")

    handles, _ = ax.get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color=NEUTRAL_COLOR, **FIT_STYLE, label="Median"),
        plt.Rectangle((0, 0), 1, 1, fc=NEUTRAL_COLOR, alpha=ALPHA_FAINT, lw=0,
                       label=r"68\% predictive"),
        plt.Rectangle((0, 0), 1, 1, fc=NEUTRAL_COLOR, **FILL_STYLE,
                       label=r"95\% predictive"),
    ])
    ax.legend(handles=handles, loc="best")

    savefig(fig, "fig_period_color.pdf")
    return ax


# ---------------------------------------------------------------------------
# 20. Empirical vs catalog extinction comparison
# ---------------------------------------------------------------------------

def plot_empirical_vs_catalog_extinction(population):
    """Square scatter comparison of empirical A_G vs Gaia g_absorption.

    Parameters
    ----------
    population : Table
        Reddening population table with ``g_absorption`` and ``A_G`` columns.
    """
    catalog_ag = population["g_absorption"]
    empirical_ag = population["A_G"]
    finite = np.isfinite(catalog_ag) & np.isfinite(empirical_ag)
    catalog_ag = catalog_ag[finite]
    empirical_ag = empirical_ag[finite]

    fig, ax = columnwidth_figure(10)

    ax.scatter(
        catalog_ag, empirical_ag,
        s=SS_MICRO, alpha=ALPHA_EXTRA_LIGHT, color=_CLASS_COLORS["RRab"],
        rasterized=True, zorder=2, label="RRab",
    )
    lo = min(catalog_ag.min(), empirical_ag.min())
    margin = _PAD_FRACTION_DATA * (_EXTINCTION_CEILING_MAG - lo)
    ax.plot([lo - margin, _EXTINCTION_CEILING_MAG], [lo - margin, _EXTINCTION_CEILING_MAG], **GUIDE_STYLE, label="1:1")
    ax.set_xlim(lo - margin, _EXTINCTION_CEILING_MAG)
    ax.set_ylim(lo - margin, _EXTINCTION_CEILING_MAG)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Gaia DR3 $g_{\mathrm{absorption}}$ [mag]")
    ax.set_ylabel(r"Empirical $A_G$ [mag]")
    ax.legend(loc="upper left")

    savefig(fig, "fig_extinction_comparison.pdf")
    return ax


# ---------------------------------------------------------------------------
# 21. Aitoff reddening map
# ---------------------------------------------------------------------------

def _aitoff_coords(l_deg, b_deg):
    """Convert Galactic (l, b) in degrees to Aitoff-ready radians."""
    l_wrap = np.where(l_deg > 180, l_deg - 360, l_deg)
    return np.deg2rad(l_wrap), np.deg2rad(b_deg)


def _aitoff_figure():
    """Create a landscape-width Aitoff projection figure."""
    fig, ax = landscapewidth_figure(6)
    ax.remove()
    ax = fig.add_subplot(111, projection="aitoff")
    return fig, ax


def plot_aitoff_reddening(l_deg, b_deg, values):
    """Aitoff projection of RR Lyrae color excess.

    Parameters
    ----------
    l_deg, b_deg : array-like
        Galactic coordinates in degrees.
    values : array-like
        Color excess values to map.
    """
    vmin = float(min(0.0, np.nanpercentile(values, _PERCENTILE_LOW)))
    vmax = float(np.nanpercentile(values, _PERCENTILE_HIGH))

    fig, ax = _aitoff_figure()

    l_rad, b_rad = _aitoff_coords(l_deg, b_deg)
    sc = ax.scatter(
        l_rad, b_rad, c=values,
        s=SS_MICRO, alpha=ALPHA_FAINT, cmap="magma",
        vmin=vmin, vmax=vmax, rasterized=True, lw=LW_NONE,
    )
    fig.colorbar(
        sc, ax=ax, orientation="vertical", shrink=0.6,
        label=r"$E(G_{\mathrm{BP}} - G_{\mathrm{RP}})$ [mag]",
    )
    ax.grid(True, **GRID_STYLE)
    ax.set_xlabel(r"Galactic longitude $l$")
    ax.set_ylabel(r"Galactic latitude $b$")

    savefig(fig, "fig_reddening_map.pdf")
    return fig, ax


def plot_aitoff_sfd(l_deg, b_deg, sfd_ebv):
    """Aitoff projection of SFD E(B-V) sampled at RR Lyrae positions.

    Parameters
    ----------
    l_deg, b_deg : array-like
        Galactic coordinates in degrees.
    sfd_ebv : array-like
        SFD E(B-V) values.
    """
    vmin = float(max(0.0, np.nanpercentile(sfd_ebv, _PERCENTILE_LOW)))
    vmax = float(np.nanpercentile(sfd_ebv, _PERCENTILE_HIGH))

    fig, ax = _aitoff_figure()

    l_rad, b_rad = _aitoff_coords(l_deg, b_deg)
    sc = ax.scatter(
        l_rad, b_rad, c=sfd_ebv,
        s=SS_MICRO, alpha=ALPHA_FAINT, cmap="viridis",
        vmin=vmin, vmax=vmax, rasterized=True, lw=LW_NONE,
    )
    fig.colorbar(
        sc, ax=ax, orientation="vertical", shrink=0.6,
        label=r"SFD $E(B-V)$ [mag]",
    )
    ax.grid(True, **GRID_STYLE)
    ax.set_xlabel(r"Galactic longitude $l$")
    ax.set_ylabel(r"Galactic latitude $b$")

    savefig(fig, "fig_sfd_map.pdf")
    return fig, ax


# ---------------------------------------------------------------------------
# 22. SFD regime decomposition (similar-scale vs large-SFD)
# ---------------------------------------------------------------------------

def plot_sfd_regime_decomposition(
    sfd_ebv, empirical, similar_mask, large_mask,
    slope, intercept,
    similar_scale_max, large_sfd_min,
    bin_count,
):
    """Two-panel: similar-scale hexbin with fit + median, large-SFD scatter.

    Parameters
    ----------
    sfd_ebv, empirical : array-like
        Full arrays (masks select subsets).
    similar_mask, large_mask : array-like of bool
        Boolean masks for each regime.
    slope, intercept : float
        Linear fit parameters for the similar-scale regime.
    similar_scale_max, large_sfd_min : float
        Regime boundaries in SFD E(B-V) mag.
    bin_count : int
        Number of bins for the median trend line.
    """
    fig, ax_dummy = textwidth_figure(33 / 4)
    ax_dummy.remove()
    axes = subpanels(fig, 1, ncols=2, wspace=_WSPACE_TWO_COL, sharex=False, sharey=False)

    # --- left: similar scale (hexbin + fit + binned median) ---
    x_sim = sfd_ebv[similar_mask]
    y_sim = empirical[similar_mask]
    hb = axes[0].hexbin(
        x_sim, y_sim, gridsize=80, cmap="viridis",
        mincnt=1, rasterized=True, norm=LogNorm(),
    )
    fig.colorbar(hb, ax=axes[0], orientation="horizontal", label="Stars per bin")
    x_fit = np.linspace(0, similar_scale_max, 100)
    axes[0].plot(x_fit, slope * x_fit + intercept, color="C1", **FIT_STYLE,
                 label=rf"Fit: slope $= {slope:.2f}$")

    # binned median
    edges = np.linspace(0, similar_scale_max, bin_count + 1)
    centers, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x_sim >= lo) & (x_sim < hi)
        if sel.sum() >= _MIN_POINTS_PER_BIN:
            centers.append(0.5 * (lo + hi))
            medians.append(float(np.nanmedian(y_sim[sel])))
    axes[0].plot(centers, medians, color="C3", **MODEL_STYLE, label="Binned median")

    axes[0].set_xlabel(r"SFD $E(B-V)$ [mag]")
    axes[0].set_ylabel(r"$E(G_{\mathrm{BP}} - G_{\mathrm{RP}})$ [mag]")
    axes[0].legend(loc="upper left")

    # --- right: large SFD (full y range + quality ceiling) ---
    x_lg = sfd_ebv[large_mask]
    y_lg = empirical[large_mask]
    axes[1].scatter(
        x_lg, y_lg,
        s=SS_MICRO, alpha=ALPHA_FAINT, color="C0", rasterized=True,
    )
    axes[1].axhline(_EXTINCTION_LARGE_REGIME_MAG, **GUIDE_STYLE, label=r"$E = 10$ ceiling")
    lo_y = min(y_lg.min(), 0) if len(y_lg) else -1
    margin = _PAD_FRACTION_DATA * (_EXTINCTION_LARGE_REGIME_MAG - lo_y)
    axes[1].set_ylim(lo_y - margin, _EXTINCTION_LARGE_REGIME_MAG + margin)
    axes[1].set_xlabel(r"SFD $E(B-V)$ [mag]")
    axes[1].set_ylabel(r"$E(G_{\mathrm{BP}} - G_{\mathrm{RP}})$ [mag]")
    axes[1].legend(loc="upper right")

    savefig(fig, "fig_regime_decomposition.pdf")
    return axes
