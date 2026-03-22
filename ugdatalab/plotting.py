import matplotlib as mpl
import matplotlib.pyplot as plt

TEXTWIDTH_IN = 7.59
COLUMNWIDTH_IN = 3.73
A4_WIDTH_IN = 8.27
A4_HEIGHT_IN = 11.69
A4_MARGIN_IN = 0.75
A4_USABLE_WIDTH_IN = A4_WIDTH_IN - 2.0 * A4_MARGIN_IN
A4_USABLE_HEIGHT_IN = A4_HEIGHT_IN - 2.0 * A4_MARGIN_IN
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8
ANNOTATION_SIZE = 9
EMPHASIS_SIZE = 10
TITLE_SIZE = 9

# Visual weight scale tuned for the figure sizes defined above.
LW_NONE = 0.0
LW_GRID = 0.4
LW_FINE = 0.6
LW_GUIDE = 0.8
LW_LIGHT = 0.9
LW_STANDARD = 1.0
LW_MEDIUM = 1.1
LW_STRONG = 1.3
LW_FIT = 1.5
LW_MODEL = 1.6
LW_EMPHASIS = 1.8
LW_CALLOUT = 2.2
LW_LEVEL = 2.6

MARKER_MS_MICRO  = 2.0    # micro      — extra-fine, reduced dense scatter
MARKER_MS_FINE   = 2.5    # fine       — dense scatter / many-point plots
MARKER_MS_SMALL  = 3.5    # small      — RR Lyrae light-curve data
MARKER_MS_MEDIUM = 5.0    # medium     — moderate emphasis
MARKER_MS_LARGE  = 8.0    # large      — prominent markers
MARKER_MS_BIG    = 12.0   # big        — callout / special annotation
RRLYRAE_POINT_ALPHA = 0.55

ALPHA_SHADE = 0.1
ALPHA_EXTRA_LIGHT = 0.2
ALPHA_DIM = 0.3
ALPHA_MUTED = 0.4
ALPHA_FAINT = 0.5
ALPHA_LIGHT = 0.6
ALPHA_STANDARD = 0.7
ALPHA_DENSE = 0.75
ALPHA_GUIDE = 0.8
ALPHA_EMPHASIS = 0.9

PRIMARY_COLOR = "C0"
SECONDARY_COLOR = "C1"
TERTIARY_COLOR = "C2"
QUATERNARY_COLOR = "C3"
QUINARY_COLOR = "C4"
SENARY_COLOR = "C5"
SEPTENARY_COLOR = "C6"
NEUTRAL_COLOR = "C7"
LIGHT_NEUTRAL_COLOR = "C8"
NONARY_COLOR = "C9"
COMPONENT_COLORS = (QUINARY_COLOR, SENARY_COLOR, SEPTENARY_COLOR, LIGHT_NEUTRAL_COLOR, NONARY_COLOR)

MCMC_SAMPLER_COLORS = {
    "native_nuts":         {"RRab": PRIMARY_COLOR,    "RRc": SECONDARY_COLOR},
    "metropolis_hastings": {"RRab": QUATERNARY_COLOR, "RRc": QUINARY_COLOR},
    "nuts_potential":      {"RRab": SENARY_COLOR,     "RRc": SEPTENARY_COLOR},
}
_MCMC_SAMPLER_ALIASES = {
    "native": "native_nuts",
    "nuts": "native_nuts",
    "native_pymc_nuts": "native_nuts",
    "metropolis_hastings": "metropolis_hastings",
    "metropolis_hastings_sampler": "metropolis_hastings",
    "metropolis_hastings_fit": "metropolis_hastings",
    "mh": "metropolis_hastings",
    "nuts_potential": "nuts_potential",
    "nuts_with_potential": "nuts_potential",
    "potential": "nuts_potential",
}
_REPORT_FIGURE_FILENAMES = {
    "plot_raw_phase_folded_lightcurve": "fig_lc_raw_phased.pdf",
    "plot_lomb_scargle_periodogram": "fig_periodogram.pdf",
    "plot_fourier_harmonic_fits": "fig_fourier_harmonics.pdf",
    "plot_fourier_cross_validation": "fig_crossval.pdf",
    "plot_fourier_cv_normalized_residual_histograms": "fig_fourier_cv_residuals.pdf",
    "plot_fourier_cv_phase_comparison": "fig_fourier_cv_phase.pdf",
    "plot_vari_rrlyrae_period_comparison": "fig_period_comparison.pdf",
    "plot_rrlyrae_shape_comparison": "fig_rrab_rrc.pdf",
    "plot_mollweide": "fig_calibration_sky.pdf",
    "plot_mollweide_diff": "fig_calibration_sky_c12.pdf",
    "plot_calibration_sky_distribution": "fig_calibration_sky.pdf",
    "plot_period_abs_mag_stage_comparison": "fig_pl_stages.pdf",
    "plot_period_abs_mag_c12_comparison": "fig_period_abs_mag_c12_comparison.pdf",
    "plot_inlier_prob_period_luminosity_comparison": "fig_inlier_prob_period_luminosity_comparison.pdf",
    "plot_pl_posterior_predictive": "fig_pl_posterior.pdf",
    "plot_pl_sampler_comparison_corner": "fig_methods_corner.pdf",
    "plot_pc_posterior_predictive_comparison": "fig_period_color.pdf",
    "plot_empirical_vs_catalog_extinction_comparison": "fig_extinction_comparison.pdf",
    "plot_mean_g_catalog_comparison": "fig_mean_g_comparison.pdf",
    "plot_period_mean_g": "fig_period_mean_g.pdf",
    "plot_fourier_extrapolation": "fig_fourier_extrapolation.pdf",
    "plot_aitoff_reddening_map": "fig_reddening_map.pdf",
    "plot_sfd_empirical_hexbin_comparison": "fig_sfd_comparison.pdf",
    "plot_sfd_all_sky_hexbin": "fig_sfd_all_sky_hexbin.pdf",
    "plot_aitoff_sfd_map": "fig_sfd_map.pdf",
    "plot_regime_decomposition": "fig_regime_decomposition.pdf",
    "plot_reddening_distribution": "fig_reddening_distribution.pdf",
    "plot_optical_vs_w2_comparison": "fig_optical_ir_comparison.pdf",
    "plot_aitoff_reddening_dark": "fig_reddening_map_dark.pdf",
    "plot_quality_diagnostics": "fig_quality_diagnostics.pdf",
    "plot_inlier_prob_map": "fig_inlier_prob_map.pdf",
}

mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": LABEL_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.grid": True,
        "axes.titlesize": EMPHASIS_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "grid.linewidth": LW_GRID,
        "grid.alpha": ALPHA_FAINT,
        "legend.fontsize": LEGEND_SIZE,
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    }
)



# ---------------------------------------------------------------------------
# Backward-compatible helpers re-exported for lab2_plotting (which imports
# ``ugdatalab.plotting as ugplt`` and calls these directly).
# We define them here instead of re-importing from lab1_plotter to avoid a
# circular import (lab1_plotter imports constants from this module).
# ---------------------------------------------------------------------------

def _textwidth_figsize(height_out_of_8: float) -> tuple[float, float]:
    return (TEXTWIDTH_IN, height_out_of_8 / 8 * TEXTWIDTH_IN)


def _apply_grid(ax) -> None:
    ax.grid(True)


def _tight_layout(fig, *, use_pyplot: bool = False, **kwargs) -> None:
    if use_pyplot:
        plt.tight_layout(**kwargs)
    else:
        fig.tight_layout(**kwargs)
