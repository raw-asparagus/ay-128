from __future__ import annotations

import functools
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from astropy.table import Table
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .lightcurves import (
    FourierFit,
    HarmonicCrossValidationResult,
    attach_flux_mean_magnitudes,
    attach_periodogram_periods,
    cross_validate_harmonics,
    fourier_fit,
    fourier_mean_magnitude,
    phase_fold,
)
from .models.gaia import rrlyrae_representative_period
from .paths import FIGURES_DIR, ensure_output_dirs

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

MARKER_MS_FINE   = 2.5    # very fine  — dense scatter / many-point plots
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
    "plot_regime_decomposition": "fig_regime_decomposition.pdf",
    "plot_reddening_distribution": "fig_reddening_distribution.pdf",
    "plot_optical_vs_w2_comparison": "fig_optical_ir_comparison.pdf",
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


def _escape_latex_text(text: str) -> str:
    safe = text
    safe = re.sub(r"(?<!\\)&", r"\\&", safe)
    safe = re.sub(r"(?<!\\)%", r"\\%", safe)
    safe = re.sub(r"(?<!\\)#", r"\\#", safe)
    return safe


def _tight_layout(fig: Figure, *, use_pyplot: bool = False, **kwargs) -> None:
    if use_pyplot:
        plt.tight_layout(**kwargs)
    else:
        fig.tight_layout(**kwargs)


def _save_figure(fig: Figure, path: Path, **kwargs) -> None:
    fig.savefig(path, **kwargs)


def _save_lab01_figure(fig: Figure, filename: str, **kwargs) -> None:
    ensure_output_dirs()
    _save_figure(fig, FIGURES_DIR / filename, **kwargs)


# Backwards-compatible alias for older callers/tests.
_save_lab02_figure = _save_lab01_figure


def _as_table(source, attr: str = "data") -> Table:
    if isinstance(source, Table):
        return source
    if hasattr(source, attr):
        return getattr(source, attr)
    raise TypeError(f"Expected an astropy Table or an object with `{attr}`.")


def _optional_float_array(data: Table, *names: str) -> np.ndarray:
    for name in names:
        if name in data.colnames:
            return np.asarray(data[name], dtype=float)
    return np.full(len(data), np.nan, dtype=float)


def _plot_period_values(data: Table) -> np.ndarray:
    return rrlyrae_representative_period(data)


def _plot_class_masks(classifications) -> list[tuple[str, np.ndarray]]:
    classifications = np.asarray(classifications, dtype=str)
    ordered = ["RRab", "RRc", "RRd"]
    labels = [label for label in ordered if np.any(classifications == label)]
    labels.extend(
        label
        for label in np.unique(classifications)
        if label not in ordered
    )
    return [(label, classifications == label) for label in labels]


def _labels(source, override=None):
    if override is not None:
        return override
    ndim = source.samples.shape[1]
    return getattr(source, "param_labels", [rf"$\theta_{i}$" for i in range(ndim)])


def _apply_grid(ax: Any) -> None:
    ax.grid(True)

def _figure_from_plot_result(result: Any) -> Figure:
    if isinstance(result, Figure):
        return result
    figure = getattr(result, "figure", None)
    if isinstance(figure, Figure):
        return figure
    if isinstance(result, tuple):
        for item in result:
            try:
                return _figure_from_plot_result(item)
            except TypeError:
                continue
    if isinstance(result, (list, np.ndarray)):
        items = np.asarray(result, dtype=object)
        if items.size:
            return _figure_from_plot_result(items.flat[0])
    raise TypeError("Could not resolve a matplotlib Figure from plot result.")


def _resolve_report_filename(function_name: str, *, save: bool | str, save_name: str | None) -> str | None:
    if save_name is not None:
        return save_name
    if isinstance(save, str):
        return save
    if save:
        if function_name not in _REPORT_FIGURE_FILENAMES:
            raise ValueError(
                f"No default report filename registered for {function_name!r}; "
                "pass save_name='your-file.pdf' explicitly."
            )
        return _REPORT_FIGURE_FILENAMES[function_name]
    return None


def _wrap_public_plot(fn):
    @functools.wraps(fn)
    def wrapped(*args, save: bool | str = False, save_name: str | None = None, save_kwargs: dict[str, Any] | None = None, **kwargs):
        result = fn(*args, **kwargs)
        filename = _resolve_report_filename(fn.__name__, save=save, save_name=save_name)
        if filename is not None:
            _save_lab01_figure(_figure_from_plot_result(result), filename, **(save_kwargs or {}))
        return result

    return wrapped


def _single_panel(figsize: tuple[float, float], *, constrained_layout: bool = False):
    return plt.subplots(figsize=figsize, constrained_layout=constrained_layout)


def _textwidth_figsize(height_out_of_8: float) -> tuple[float, float]:
    return (TEXTWIDTH_IN, height_out_of_8 / 8 * TEXTWIDTH_IN)


def _columnwidth_figsize(height_out_of_3_5: float) -> tuple[float, float]:
    return (COLUMNWIDTH_IN, height_out_of_3_5 / 3.5 * COLUMNWIDTH_IN)


def _stacked_panels(
    nrows: int,
    *,
    figsize: tuple[float, float],
    height_ratios: list[int] | tuple[int, ...],
    hspace: float = 0.0,
):
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": list(height_ratios), "hspace": hspace},
    )
    if hspace == 0.0:
        fig.subplots_adjust(hspace=0.0)
    return fig, axes


def _grid_1x2(
    *,
    figsize: tuple[float, float],
    sharex: str | bool = False,
    sharey: str | bool = False,
):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=sharex, sharey=sharey)
    fig.subplots_adjust(wspace=0.28)
    return fig, axes


def _grid_nx1(
    nrows: int,
    *,
    figsize: tuple[float, float],
    hspace: float,
):
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(nrows, 1, hspace=hspace)
    return fig, grid


def _zero_line(ax: Any) -> None:
    ax.axhline(0.0, color=NEUTRAL_COLOR, lw=LW_GUIDE, ls="--")


def _unity_line(ax: Any, *, label: str | None = None) -> None:
    ax.axhline(1.0, color=NEUTRAL_COLOR, lw=LW_GUIDE, ls="--", alpha=ALPHA_STANDARD, label=label)


def _reference_vline(
    ax: Any,
    x: float,
    *,
    label: str | None = None,
    color: str = NEUTRAL_COLOR,
    lw: float = LW_LIGHT,
    ls: str = "--",
    alpha: float = ALPHA_GUIDE,
) -> None:
    ax.axvline(x, color=color, lw=lw, ls=ls, alpha=alpha, label=label)


def _set_descending_magnitude_yaxis(ax) -> None:
    ax.autoscale_view()
    y0, y1 = ax.get_ylim()
    ax.set_ylim(max(y0, y1), min(y0, y1))


def _set_log_period_xaxis(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=[0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _source_header(source_id: int, classification: str | None = None) -> str:
    if classification:
        return f"{classification}: Gaia DR3 {int(source_id)}"
    return f"Gaia DR3 {int(source_id)}"


def _set_left_title(ax: Any, header: str | None = None, detail: str | None = None) -> None:
    lines = []
    if header:
        lines.append(_escape_latex_text(header))
    if detail:
        lines.append(detail)
    if lines:
        ax.set_title("\n".join(lines), loc="left", y=1.05, pad=2.0, fontsize=TITLE_SIZE)


def _set_title(ax: Any, title: str | None) -> None:
    if title is not None:
        _set_left_title(ax, title)


def _default_scatter_kwargs(**kwargs):
    kwargs.setdefault("s", MARKER_MS_FINE**2)
    kwargs.setdefault("alpha", ALPHA_STANDARD)
    kwargs.setdefault("rasterized", True)
    return kwargs


def _rrlyrae_class_color(label: str, fallback_index: int = 0) -> str:
    class_colors = {
        "RRab": PRIMARY_COLOR,
        "RRc": SECONDARY_COLOR,
        "RRd": TERTIARY_COLOR,
    }
    if label in class_colors:
        return class_colors[label]
    return COMPONENT_COLORS[fallback_index % len(COMPONENT_COLORS)]


def mcmc_sampler_color(class_label: str, sampler_kind: str) -> str:
    rr_class = str(class_label).strip()
    sampler_key = re.sub(r"[^a-z0-9]+", "_", str(sampler_kind).strip().lower()).strip("_")
    sampler_key = _MCMC_SAMPLER_ALIASES.get(sampler_key, sampler_key)

    if sampler_key not in MCMC_SAMPLER_COLORS:
        valid = ", ".join(sorted(MCMC_SAMPLER_COLORS))
        raise ValueError(f"Unsupported sampler kind {sampler_kind!r}. Expected one of: {valid}.")
    if rr_class not in MCMC_SAMPLER_COLORS[sampler_key]:
        valid = ", ".join(sorted(MCMC_SAMPLER_COLORS[sampler_key]))
        raise ValueError(f"Unsupported RR Lyrae class {class_label!r}. Expected one of: {valid}.")
    return MCMC_SAMPLER_COLORS[sampler_key][rr_class]


def plot_calibration_sky_distribution(initial_source, cleaned_source):
    initial_data = _as_table(initial_source)
    cleaned_data = _as_table(cleaned_source)

    fig = plt.figure(figsize=_textwidth_figsize(19 / 4))
    ax = fig.add_subplot(111, projection="aitoff")

    l_initial = np.deg2rad(_wrap_longitude(np.asarray(initial_data["l"], dtype=float)))
    b_initial = np.deg2rad(np.asarray(initial_data["b"], dtype=float))
    ax.scatter(
        l_initial,
        b_initial,
        s=MARKER_MS_FINE**2,
        color=LIGHT_NEUTRAL_COLOR,
        alpha=ALPHA_LIGHT,
        rasterized=True,
        label=f"Initial sample ($N={len(initial_data)}$)",
        zorder=1,
    )

    l_clean = np.deg2rad(_wrap_longitude(np.asarray(cleaned_data["l"], dtype=float)))
    b_clean = np.deg2rad(np.asarray(cleaned_data["b"], dtype=float))
    classifications = np.asarray(cleaned_data["best_classification"], dtype=str)
    for i, (label, mask) in enumerate(_plot_class_masks(classifications)):
        ax.scatter(
            l_clean[mask],
            b_clean[mask],
            s=MARKER_MS_FINE**2,
            color=_rrlyrae_class_color(label, i),
            alpha=ALPHA_DENSE,
            rasterized=True,
            label=f"{label} cleaned",
            zorder=2,
        )

    longs = np.linspace(-np.pi, np.pi, 512)
    for latitude in (-30.0, 30.0):
        ax.plot(
            longs,
            np.full_like(longs, np.deg2rad(latitude)),
            color=QUATERNARY_COLOR,
            lw=LW_LIGHT,
            ls="--",
            alpha=ALPHA_GUIDE,
        )

    ax.set_title("Calibration sample in Galactic coordinates")
    ax.set_xlabel(r"Galactic longitude $l$")
    ax.set_ylabel(r"Galactic latitude $b$")
    ax.legend(loc="lower left", ncol=2)
    _apply_grid(ax)
    _tight_layout(fig)
    return fig


def plot_period_abs_mag_stage_comparison(raw_source, c12_source, clean_source):
    stage_specs = (
        ("Initial sample", _as_table(raw_source)),
        ("After C1/C2", _as_table(c12_source)),
        ("Mixture-cleaned", _as_table(clean_source)),
    )

    fig, axes = plt.subplots(
        1,
        len(stage_specs),
        figsize=_textwidth_figsize(7 / 2),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes, dtype=object)

    for index, (title, data) in enumerate(stage_specs):
        periods = _plot_period_values(data)
        abs_mag = np.asarray(data["M_G"], dtype=float)
        abs_mag_err = _optional_float_array(data, "sigma_M")
        classifications = np.asarray(data["best_classification"], dtype=str)
        plot_period_abs_mag(
            periods,
            abs_mag,
            abs_mag_err,
            classifications,
            ax=axes[index],
            show_legend=index == 0,
        )
        _add_running_median_line(axes[index], periods, abs_mag)
        if index > 0 and axes[index].legend_ is not None:
            axes[index].legend_.remove()

    axes[0].set_ylabel(r"$M_{\rm G}$ [mag]")
    fig.tight_layout()
    return axes


def plot_pl_sampler_comparison_corner(
    sample_map: dict[str, np.ndarray],
    *,
    class_label: str | None = None,
    param_labels: list[str] | tuple[str, ...] | None = None,
):
    if not sample_map:
        raise ValueError("sample_map must contain at least one sampler.")

    first_samples = np.asarray(next(iter(sample_map.values())), dtype=float)
    if first_samples.ndim != 2:
        raise ValueError("Each sampler entry must be a two-dimensional sample array.")
    labels = list(param_labels) if param_labels is not None else _labels(
        SimpleNamespace(samples=first_samples, param_labels=[r"$a$", r"$b$", r"$\log_{10}\sigma_{\rm scatter}$"])
    )

    fig = None
    handles = []
    for label, samples in sample_map.items():
        color = _sampler_overlay_color(label, class_label)
        fig = corner.corner(
            np.asarray(samples, dtype=float),
            labels=labels,
            fig=fig if fig is not None else plt.figure(figsize=_textwidth_figsize(7)),
            color=color,
            fill_contours=False,
            plot_datapoints=False,
            plot_density=False,
            levels=(0.393, 0.865),
            smooth=1.0,
            hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.6, "alpha": ALPHA_LIGHT},
            contour_kwargs={"linewidths": 1.3, "alpha": ALPHA_LIGHT},
        )
        handles.append(Line2D([0], [0], color=color, lw=2.0, label=label))

    fig.legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def _draw_relation_line_draws_layer(
    ax: Any,
    ctx: Any,
    samples: np.ndarray,
    *,
    color: str,
    data_label: str,
    mean_label: str,
    n_draws: int,
    seed: int,
    y_label: str,
) -> None:
    samples = np.asarray(samples, dtype=float)
    if len(samples) == 0:
        raise ValueError("No post-burn samples available for posterior-line plotting.")

    rng = np.random.default_rng(seed)
    draw_idx = rng.choice(len(samples), size=min(n_draws, len(samples)), replace=False)
    x_obs = np.asarray(ctx.x_centered, dtype=float)
    y_obs = np.asarray(ctx.y, dtype=float)
    sigma = np.asarray(ctx.sigma, dtype=float)
    x_grid = np.linspace(np.nanmin(x_obs), np.nanmax(x_obs), 256)

    ax.errorbar(
        x_obs,
        y_obs,
        yerr=sigma,
        fmt="o",
        ms=MARKER_MS_SMALL,
        elinewidth=LW_FINE,
        color=color,
        alpha=ALPHA_FAINT,
        label=data_label,
        zorder=3,
    )
    for j, sample in enumerate(samples[draw_idx]):
        slope, intercept, _ = sample
        ax.plot(
            x_grid,
            slope * x_grid + intercept,
            color=color,
            alpha=ALPHA_SHADE + 0.02,
            lw=LW_FINE,
            label=f"{n_draws} posterior draws" if j == 0 else "_nolegend_",
            zorder=2,
        )

    mean_params = np.mean(samples, axis=0)
    ax.plot(
        x_grid,
        mean_params[0] * x_grid + mean_params[1],
        color=SECONDARY_COLOR,
        lw=LW_EMPHASIS,
        label=mean_label,
        zorder=4,
    )
    ax.set_xlabel(r"$\log_{10}(P/\mathrm{day}) - \langle \log_{10}P \rangle_{\rm class}$")
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    _apply_grid(ax)


def plot_pl_posterior_line_draws(
    ctx: Any,
    sampler_like: Any,
    *,
    color: str = PRIMARY_COLOR,
    title: str | None = None,
    n_draws: int = 50,
    seed: int = 7,
):
    samples = _post_burn_samples(sampler_like)
    _, ax = _single_panel(_columnwidth_figsize(5 / 2))
    _draw_relation_line_draws_layer(
        ax,
        ctx,
        samples,
        color=color,
        data_label=f"{ctx.class_label} data",
        mean_label="Posterior mean",
        n_draws=n_draws,
        seed=seed,
        y_label=r"$M_{\rm G}$ [mag]",
    )
    if title is not None:
        ax.set_title(title)
    _set_descending_magnitude_yaxis(ax)
    _tight_layout(ax.figure)
    return ax


def plot_pl_posterior_line_draws_comparison(
    primary_ctx: Any,
    primary_sampler_like: Any,
    secondary_ctx: Any,
    secondary_sampler_like: Any,
    *,
    primary_color: str = PRIMARY_COLOR,
    secondary_color: str = TERTIARY_COLOR,
    y_label: str = r"$M_{\rm G}$ [mag]",
    title: str | None = None,
    n_draws: int = 50,
    seed: int = 7,
):
    _, ax = _single_panel(_textwidth_figsize(19 / 4))
    _draw_relation_line_draws_layer(
        ax,
        primary_ctx,
        _post_burn_samples(primary_sampler_like),
        color=primary_color,
        data_label=f"{primary_ctx.class_label} data",
        mean_label=f"{primary_ctx.class_label} posterior mean",
        n_draws=n_draws,
        seed=seed,
        y_label=y_label,
    )
    _draw_relation_line_draws_layer(
        ax,
        secondary_ctx,
        _post_burn_samples(secondary_sampler_like),
        color=secondary_color,
        data_label=f"{secondary_ctx.class_label} data",
        mean_label=f"{secondary_ctx.class_label} posterior mean",
        n_draws=n_draws,
        seed=seed + 1,
        y_label=y_label,
    )
    if title is not None:
        ax.set_title(title)
    _set_descending_magnitude_yaxis(ax)
    _tight_layout(ax.figure)
    return ax


def plot_w2_posterior_line_draws(
    ctx: Any,
    sampler_like: Any,
    *,
    color: str = PRIMARY_COLOR,
    title: str | None = None,
    n_draws: int = 50,
    seed: int = 11,
):
    samples = _post_burn_samples(sampler_like)
    _, ax = _single_panel(_columnwidth_figsize(5 / 2))
    _draw_relation_line_draws_layer(
        ax,
        ctx,
        samples,
        color=color,
        data_label=f"{ctx.class_label} data",
        mean_label="Posterior mean",
        n_draws=n_draws,
        seed=seed,
        y_label=r"$M_{W2}$ [mag]",
    )
    if title is not None:
        ax.set_title(title)
    _set_descending_magnitude_yaxis(ax)
    _tight_layout(ax.figure)
    return ax


def figure(result, name: str):
    return result.figures[name]


def figure_names(result) -> list[str]:
    return sorted(result.figures)


def _normalized_sampler_key(label: str) -> str:
    sampler_key = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")
    return _MCMC_SAMPLER_ALIASES.get(sampler_key, sampler_key)


def _sampler_overlay_color(label: str, class_label: str | None = None) -> str:
    sampler_key = _normalized_sampler_key(label)
    if class_label is not None:
        return mcmc_sampler_color(class_label, sampler_key)
    generic_colors = {
        "metropolis_hastings": PRIMARY_COLOR,
        "nuts_potential": SECONDARY_COLOR,
        "native_nuts": TERTIARY_COLOR,
    }
    return generic_colors.get(sampler_key, NEUTRAL_COLOR)


def _post_burn_samples(source: Any) -> np.ndarray:
    if hasattr(source, "samples"):
        n_burn = int(getattr(source, "n_burn", 0))
        return np.asarray(source.samples[n_burn:], dtype=float)
    return np.asarray(source, dtype=float)


def _add_running_median_line(ax: Any, x_values: np.ndarray, y_values: np.ndarray, *, bins: int = 12) -> None:
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(finite) < 3:
        return
    x = np.asarray(x_values[finite], dtype=float)
    y = np.asarray(y_values[finite], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_chunks = np.array_split(x, min(bins, len(x)))
    y_chunks = np.array_split(y, min(bins, len(y)))
    centers = np.asarray([np.median(chunk) for chunk in x_chunks if len(chunk)], dtype=float)
    medians = np.asarray([np.median(chunk) for chunk in y_chunks if len(chunk)], dtype=float)
    if centers.size:
        ax.plot(centers, medians, color=NEUTRAL_COLOR, lw=LW_MEDIUM, ls="--", label="Running median")


def _draw_mollweide(ax, l_deg, b_deg):
    l_deg = np.asarray(l_deg, dtype=float)
    b_deg = np.asarray(b_deg, dtype=float)
    l_wrap = np.where(l_deg > 180, l_deg - 360, l_deg)
    l_rad = np.deg2rad(l_wrap)
    b_rad = np.deg2rad(b_deg)
    ax.scatter(
        l_rad,
        b_rad,
        s=MARKER_MS_FINE**2,
        color=PRIMARY_COLOR,
        alpha=ALPHA_DENSE,
        rasterized=True,
    )

    longs = np.linspace(-np.pi, np.pi, 1000)
    lat_lo = np.full_like(longs, np.deg2rad(-30))
    lat_hi = np.full_like(longs, np.deg2rad(30))
    ax.fill_between(
        longs,
        lat_lo,
        lat_hi,
        color=QUATERNARY_COLOR,
        alpha=ALPHA_SHADE,
        label=r"$|b| < 30^\circ$ band",
    )

    ax.set_xlabel(r"Galactic longitude $l$")
    ax.set_ylabel(r"Galactic latitude $b$")
    ax.legend(loc="lower right")
    _apply_grid(ax)
    return ax


def plot_mollweide(l_deg, b_deg):
    fig = plt.figure(figsize=_textwidth_figsize(9 / 2))
    ax = fig.add_subplot(111, projection="aitoff")
    _draw_mollweide(ax, l_deg, b_deg)
    return ax


def plot_mollweide_diff(source, subset, ax=None):
    data = _as_table(source)
    subset_data = _as_table(subset)
    diff_mask = ~np.isin(data["source_id"], subset_data["source_id"])

    if ax is None:
        fig = plt.figure(figsize=_textwidth_figsize(9 / 2))
        ax = fig.add_subplot(111, projection="aitoff")

    def to_rad(table):
        l_wrap = np.where(table["l"] > 180, table["l"] - 360, table["l"])
        return np.deg2rad(l_wrap), np.deg2rad(table["b"])

    diff_data = data[diff_mask]
    l_diff, b_diff = to_rad(diff_data)
    ax.scatter(
        l_diff,
        b_diff,
        marker="x",
        color=SECONDARY_COLOR,
        label=f"Removed ($N$={len(diff_data)})",
        s=MARKER_MS_SMALL**2,
        linewidths=LW_LIGHT,
        alpha=ALPHA_EMPHASIS,
        rasterized=True,
        zorder=1,
    )

    l_sub, b_sub = to_rad(subset_data)
    ax.scatter(
        l_sub,
        b_sub,
        color=PRIMARY_COLOR,
        label=f"Kept ($N$={len(subset_data)})",
        s=MARKER_MS_FINE**2,
        alpha=ALPHA_DENSE,
        rasterized=True,
        zorder=2,
    )

    longs = np.linspace(-np.pi, np.pi, 1000)
    lat_lo = np.full_like(longs, np.deg2rad(-30))
    lat_hi = np.full_like(longs, np.deg2rad(30))
    ax.fill_between(
        longs,
        lat_lo,
        lat_hi,
        color=QUATERNARY_COLOR,
        alpha=ALPHA_SHADE,
        label=r"$|b| < 30^\circ$ band",
    )

    ax.set_xlabel(r"Galactic longitude $l$")
    ax.set_ylabel(r"Galactic latitude $b$")
    ax.legend(loc="upper right")
    _apply_grid(ax)
    return ax


def plot_lomb_scargle_periodogram(source_id: int, data: Table):
    data = _as_table(data)
    data = data[data["source_id"] == int(source_id)]
    if len(data) == 0:
        raise ValueError("No light-curve rows available for plotting.")

    from ugdatalab.lightcurves import lomb_scargle_periodogram

    periods, power, _ = lomb_scargle_periodogram(data)
    periods = np.asarray(periods, dtype=float)
    power = np.asarray(power, dtype=float)

    order = np.argsort(periods)
    periods = periods[order]
    power = power[order]

    period = float(data["period_ls"][0])
    classification = str(data["best_classification"][0])

    _, ax = _single_panel(_columnwidth_figsize(3 / 2))

    ax.plot(periods, power, color=PRIMARY_COLOR, lw=LW_MEDIUM)
    ax.axvline(
        period,
        color=SECONDARY_COLOR,
        ls="--",
        lw=LW_LIGHT,
        alpha=ALPHA_EMPHASIS,
        label=rf"$P={period:.4f}$ d",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$P_{\rm LS}$ [days]")
    ax.set_ylabel("Lomb-Scargle power")
    ax.legend()
    _apply_grid(ax)
    return ax


def _draw_period_abs_mag(
    ax,
    periods,
    abs_mag,
    abs_mag_err,
    classifications,
    *,
    show_legend: bool = True,
    legend_kwargs: dict[str, Any] | None = None,
):
    periods = np.asarray(periods, dtype=float)
    m_g = np.asarray(abs_mag, dtype=float)
    sigma_m = np.asarray(abs_mag_err, dtype=float)
    classifications = np.asarray(classifications, dtype=str)
    class_masks = _plot_class_masks(classifications)
    if class_masks:
        for i, (label, mask) in enumerate(class_masks):
            color = _rrlyrae_class_color(label, i)
            ax.errorbar(
                periods[mask],
                m_g[mask],
                yerr=sigma_m[mask],
                fmt="none",
                color=color,
                alpha=ALPHA_LIGHT,
                zorder=1,
            )
            ax.scatter(
                periods[mask],
                m_g[mask],
                color=color,
                label=label,
                s=MARKER_MS_FINE**2,
                alpha=ALPHA_DENSE,
                rasterized=True,
                zorder=2,
            )
    else:
        ax.errorbar(
            periods,
            m_g,
            yerr=sigma_m,
            fmt="none",
            color=PRIMARY_COLOR,
            alpha=ALPHA_LIGHT,
            zorder=1,
        )
        ax.scatter(
            periods,
            m_g,
            color=PRIMARY_COLOR,
            label="RR Lyrae",
            s=MARKER_MS_FINE**2,
            alpha=ALPHA_DENSE,
            rasterized=True,
            zorder=2,
        )

    _set_log_period_xaxis(ax)
    _set_descending_magnitude_yaxis(ax)
    ax.set_ylabel(r"$M_{\rm G}$ [mag]")
    if show_legend:
        ax.legend(loc="lower right", **(legend_kwargs or {}))
    _apply_grid(ax)
    return ax


def plot_period_abs_mag(
    periods,
    abs_mag,
    abs_mag_err,
    classifications,
    *,
    ax=None,
    show_legend: bool = True,
    legend_kwargs: dict[str, Any] | None = None,
):
    if ax is None:
        _, ax = _single_panel(_textwidth_figsize(7 / 2))
    _draw_period_abs_mag(
        ax,
        periods,
        abs_mag,
        abs_mag_err,
        classifications,
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
    )
    ax.set_xlabel(r"Catalog period $P$ [days]")
    return ax


def plot_period_abs_mag_ls(
    periods,
    abs_mag,
    abs_mag_err,
    classifications,
    *,
    ax=None,
    show_legend: bool = True,
    legend_kwargs: dict[str, Any] | None = None,
):
    if ax is None:
        _, ax = _single_panel(_columnwidth_figsize(5 / 2))
    _draw_period_abs_mag(
        ax,
        periods,
        abs_mag,
        abs_mag_err,
        classifications,
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
    )
    ax.set_xlabel(r"L-S period $P_{\rm LS}$ [days]")
    return ax


def _class_count_map(classifications) -> dict[str, int]:
    return {
        label: int(np.count_nonzero(mask))
        for label, mask in _plot_class_masks(classifications)
    }


def _class_cut_label_map(reference_classifications, current_classifications) -> dict[str, str]:
    reference_counts = _class_count_map(reference_classifications)
    current_counts = _class_count_map(current_classifications)
    labels = list(reference_counts)
    labels.extend(label for label in current_counts if label not in reference_counts)
    return {
        label: f"{label} ({max(reference_counts.get(label, 0) - current_counts.get(label, 0), 0)} cut)"
        for label in labels
    }


def _period_abs_mag_legend_handles(
    labels,
    *,
    label_map: dict[str, str] | None = None,
) -> list[Line2D]:
    handles = []
    for i, label in enumerate(labels):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=_rrlyrae_class_color(label, i),
                markeredgecolor="none",
                markersize=MARKER_MS_MEDIUM,
                alpha=ALPHA_DENSE,
                label=label_map.get(label, label) if label_map is not None else label,
            )
        )
    return handles


def _period_abs_mag_comparison_figure(*, height_out_of_8: float = 3):
    return _grid_1x2(
        figsize=_textwidth_figsize(height_out_of_8),
        sharex=False,
        sharey=True,
    )


def plot_period_abs_mag_comparison(
    left_periods,
    left_abs_mag,
    left_abs_mag_err,
    left_classifications,
    right_periods,
    right_abs_mag,
    right_abs_mag_err,
    right_classifications,
):
    fig, axes = _grid_1x2(
        figsize=_textwidth_figsize(19 / 4),
        sharex=True,
        sharey=True,
    )
    plot_period_abs_mag(
        left_periods,
        left_abs_mag,
        left_abs_mag_err,
        left_classifications,
        ax=axes[0],
    )
    plot_period_abs_mag(
        right_periods,
        right_abs_mag,
        right_abs_mag_err,
        right_classifications,
        ax=axes[1],
    )
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()
    _tight_layout(fig)
    return axes


def _overlay_removed_x_markers(ax, full_data, kept_data, reference_labels) -> None:
    """Overlay x markers on points in full_data that are absent from kept_data."""
    kept_ids = set(np.asarray(kept_data["source_id"], dtype=int))
    removed_mask = ~np.isin(np.asarray(full_data["source_id"], dtype=int), list(kept_ids))
    removed = full_data[removed_mask]
    if len(removed) == 0:
        return
    removed_classes = np.asarray(removed["best_classification"], dtype=str)
    removed_periods = _plot_period_values(removed)
    removed_m_g = np.asarray(removed["M_G"], dtype=float)
    for idx, label in enumerate(reference_labels):
        mask = removed_classes == label
        if not mask.any():
            continue
        ax.scatter(
            removed_periods[mask],
            removed_m_g[mask],
            marker="x",
            s=MARKER_MS_SMALL**2,
            linewidths=LW_FINE,
            color=_rrlyrae_class_color(label, idx),
            alpha=ALPHA_MUTED,
            rasterized=True,
            zorder=3,
        )


def plot_period_abs_mag_c12_comparison(pre_c12_source, post_c12_source):
    pre_c12_data = _as_table(pre_c12_source)
    post_c12_data = _as_table(post_c12_source)

    pre_c12_classes = np.asarray(pre_c12_data["best_classification"], dtype=str)
    post_c12_classes = np.asarray(post_c12_data["best_classification"], dtype=str)
    reference_labels = list(_class_count_map(pre_c12_classes))

    _, ax = _single_panel(_columnwidth_figsize(5 / 2))
    plot_period_abs_mag(
        _plot_period_values(post_c12_data),
        np.asarray(post_c12_data["M_G"], dtype=float),
        np.asarray(post_c12_data["sigma_M"], dtype=float),
        post_c12_classes,
        ax=ax,
        show_legend=False,
    )
    _overlay_removed_x_markers(ax, pre_c12_data, post_c12_data, reference_labels)

    label_map = _class_cut_label_map(pre_c12_classes, post_c12_classes)
    handles = _period_abs_mag_legend_handles(reference_labels, label_map=label_map)
    n_removed = len(pre_c12_data) - len(post_c12_data)
    handles.append(
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color="grey",
            markersize=MARKER_MS_MEDIUM,
            markeredgewidth=LW_FINE,
            alpha=ALPHA_MUTED,
            label=rf"Removed",
        )
    )
    ax.legend(handles=handles, loc="lower right")
    return ax


def plot_period_abs_mag_clean_vs_astrometric_comparison(
    reference_source,
    clean_source,
    refined_source,
):
    reference_data = _as_table(reference_source)
    clean_data = _as_table(clean_source)
    refined_data = _as_table(refined_source)

    reference_classes = np.asarray(reference_data["best_classification"], dtype=str)
    clean_classes = np.asarray(clean_data["best_classification"], dtype=str)
    refined_classes = np.asarray(refined_data["best_classification"], dtype=str)
    reference_labels = list(_class_count_map(reference_classes))

    fig, axes = _period_abs_mag_comparison_figure()
    plot_period_abs_mag(
        _plot_period_values(clean_data),
        np.asarray(clean_data["M_G"], dtype=float),
        np.asarray(clean_data["sigma_M"], dtype=float),
        clean_classes,
        ax=axes[0],
        show_legend=False,
    )
    axes[0].legend(
        handles=_period_abs_mag_legend_handles(
            reference_labels,
            label_map=_class_cut_label_map(reference_classes, clean_classes),
        ),
        loc="lower right",
    )

    plot_period_abs_mag(
        _plot_period_values(refined_data),
        np.asarray(refined_data["M_G"], dtype=float),
        np.asarray(refined_data["sigma_M"], dtype=float),
        refined_classes,
        ax=axes[1],
        show_legend=False,
    )
    axes[1].legend(
        handles=_period_abs_mag_legend_handles(
            reference_labels,
            label_map=_class_cut_label_map(reference_classes, refined_classes),
        ),
        loc="lower right",
    )
    axes[1].set_ylabel("")
    _tight_layout(fig)
    return axes


def plot_mollweide_period_abs_mag_overview(
    l_deg,
    b_deg,
    periods,
    abs_mag,
    abs_mag_err,
    classifications,
):
    fig = plt.figure(figsize=_textwidth_figsize(19 / 4))
    gs = fig.add_gridspec(1, 2)
    ax_sky = fig.add_subplot(gs[0], projection="aitoff")
    ax_pl = fig.add_subplot(gs[1])
    _draw_mollweide(ax_sky, l_deg, b_deg)
    plot_period_abs_mag(periods, abs_mag, abs_mag_err, classifications, ax=ax_pl)
    _tight_layout(fig)
    return np.array([ax_sky, ax_pl], dtype=object)


def plot_period_luminosity_diff(source, subset, ax=None):
    data = _as_table(source)
    subset_data = _as_table(subset)
    diff_mask = ~np.isin(data["source_id"], subset_data["source_id"])

    if ax is None:
        _, ax = _single_panel(_columnwidth_figsize(5 / 2))

    diff_data = data[diff_mask]
    diff_period = _plot_period_values(diff_data)
    subset_period = _plot_period_values(subset_data)
    ax.scatter(
        diff_period,
        np.asarray(diff_data["M_G"], dtype=float),
        color=SECONDARY_COLOR,
        label=f"Removed ($N$={len(diff_data)})",
        s=MARKER_MS_FINE**2,
        alpha=ALPHA_DENSE,
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        subset_period,
        np.asarray(subset_data["M_G"], dtype=float),
        color=PRIMARY_COLOR,
        label=f"Kept ($N$={len(subset_data)})",
        s=MARKER_MS_FINE**2,
        alpha=ALPHA_DENSE,
        rasterized=True,
        zorder=2,
    )

    _set_log_period_xaxis(ax)
    _set_descending_magnitude_yaxis(ax)
    ax.set_xlabel(r"$P$ [days]")
    ax.set_ylabel(r"$M_{\rm G}$ [mag]")
    ax.legend()
    _apply_grid(ax)
    return ax


def plot_inlier_prob_period_luminosity_comparison(
    prob_source,
    period_source,
    subset,
):
    data = _as_table(prob_source, attr="all_data")
    periods = _plot_period_values(data)
    m_g = np.asarray(data["M_G"], dtype=float)
    inlier_prob = np.asarray(data["inlier_prob"], dtype=float)

    threshold = 0.95
    kept = inlier_prob >= threshold
    removed = ~kept

    _, ax = _single_panel(_columnwidth_figsize(5 / 2))

    sc = ax.scatter(
        periods[kept],
        m_g[kept],
        c=inlier_prob[kept],
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        s=MARKER_MS_FINE**2,
        alpha=ALPHA_DENSE,
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        periods[removed],
        m_g[removed],
        c=inlier_prob[removed],
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        marker="x",
        s=MARKER_MS_SMALL**2,
        linewidths=LW_LIGHT,
        alpha=ALPHA_MUTED,
        rasterized=True,
        zorder=1,
        label=rf"Removed ($p_{{\rm in}} < {threshold}$, $N={int(removed.sum())}$)",
    )
    plt.colorbar(sc, ax=ax, label="Posterior inlier probability")
    _set_log_period_xaxis(ax)
    _set_descending_magnitude_yaxis(ax)
    ax.set_xlabel(r"$P$ [days]")
    ax.set_ylabel(r"$M_{\rm G}$ [mag]")
    ax.legend(loc="lower right")
    _apply_grid(ax)
    return ax


def plot_period_mean_g(data: Table):
    _, ax = _single_panel(_columnwidth_figsize(3 / 2))

    classifications = np.asarray(data["best_classification"]).astype(str)
    periods = np.asarray(data["best_period"], dtype=float)
    mean_g = np.asarray(data["mean_apparent_g"], dtype=float)
    mean_g_err = np.asarray(data["mean_apparent_g_err"], dtype=float)
    finite = np.isfinite(periods) & np.isfinite(mean_g) & np.isfinite(mean_g_err)

    ax.errorbar(
        periods[finite],
        mean_g[finite],
        yerr=mean_g_err[finite],
        fmt="none",
        color=NEUTRAL_COLOR,
        alpha=ALPHA_STANDARD,
        zorder=1,
    )

    for i, label in enumerate(np.unique(classifications[finite])):
        mask = finite & (classifications == label)
        ax.scatter(
            periods[mask],
            mean_g[mask],
            color=_rrlyrae_class_color(label, i),
            label=label,
            zorder=2,
            s=MARKER_MS_SMALL**2,
            alpha=RRLYRAE_POINT_ALPHA,
            rasterized=True,
        )

    _set_log_period_xaxis(ax)
    ax.invert_yaxis()
    ax.set_xlabel(r"L-S period $P_{\rm LS}$ [days]")
    ax.set_ylabel(r"$\langle G \rangle_{\rm epoch}$ [mag]")
    ax.autoscale_view()
    ax.legend()
    _apply_grid(ax)
    _tight_layout(ax.figure)
    return ax


def plot_vari_rrlyrae_period_comparison(data: Table):
    classifications = np.asarray(data["best_classification"]).astype(str)
    fundamental_period = np.asarray(data["pf"], dtype=float)
    best_period = np.asarray(data["best_period"], dtype=float)
    first_overtone_period = np.asarray(data["p1_o"], dtype=float)
    fundamental_period_err = np.asarray(data["pf_error"], dtype=float)
    first_overtone_period_err = np.asarray(data["p1_o_error"], dtype=float)
    p1_o_over_pf = np.full(len(data), np.nan, dtype=float)
    np.divide(
        first_overtone_period,
        fundamental_period,
        out=p1_o_over_pf,
        where=np.isfinite(first_overtone_period) & np.isfinite(fundamental_period) & (fundamental_period > 0),
    )

    finite_fundamental = np.isfinite(fundamental_period) & np.isfinite(best_period)
    if not np.any(finite_fundamental):
        raise ValueError("No finite fundamental-period comparison points are available for plotting.")

    fig, axes = plt.subplots(1, 2, figsize=_textwidth_figsize(4))

    ax_fundamental, ax_first_overtone = axes

    unique_classes = np.unique(classifications[finite_fundamental])
    rr_d_color = PRIMARY_COLOR
    for i, label in enumerate(unique_classes):
        color = _rrlyrae_class_color(label, i)
        mask = finite_fundamental & (classifications == label)
        finite_err = mask & np.isfinite(fundamental_period_err)
        ax_fundamental.errorbar(
            fundamental_period[finite_err],
            best_period[finite_err],
            xerr=fundamental_period_err[finite_err],
            fmt="none",
            ecolor=color,
            elinewidth=LW_FINE,
            alpha=ALPHA_FAINT,
            zorder=1,
        )
        ax_fundamental.scatter(
            fundamental_period[mask],
            best_period[mask],
            s=MARKER_MS_SMALL**2,
            alpha=RRLYRAE_POINT_ALPHA,
            color=color,
            label=label,
            zorder=2,
        )
        if label == "RRd":
            rr_d_color = color

    ax_fundamental.autoscale_view()
    x_lo, x_hi = ax_fundamental.get_xlim()
    y_lo, y_hi = ax_fundamental.get_ylim()
    lo = float(min(x_lo, y_lo))
    hi = float(max(x_hi, y_hi))

    ax_fundamental.plot(
        [lo, hi],
        [lo, hi],
        color=NEUTRAL_COLOR,
        ls="--",
        lw=LW_STANDARD,
        alpha=0.35,
        label=r"$P_{\rm LS}=P_{\rm F}$",
        zorder=1,
    )

    rrd_ratios = p1_o_over_pf[np.isfinite(p1_o_over_pf)]
    if len(rrd_ratios):
        ratio = float(np.median(rrd_ratios))
        ax_fundamental.plot(
            [lo, hi],
            [ratio * lo, ratio * hi],
            color=QUATERNARY_COLOR,
            ls=":",
            lw=LW_MEDIUM,
            label=rf"$P_{{\rm LS}}\approx({ratio:.3f})\,P_{{\rm F}}$",
            zorder=1,
        )

    ax_fundamental.set_xlabel(r"\texttt{vari\_rrlyrae} fundamental period $P_{\rm F}$ [days]")
    ax_fundamental.set_ylabel(r"L-S period $P_{\rm LS}$ [days]")
    ax_fundamental.set_xlim(lo, hi)
    ax_fundamental.set_ylim(lo, hi)
    ax_fundamental.set_aspect("equal", adjustable="box")
    ax_fundamental.legend(ncols=2)
    _apply_grid(ax_fundamental)

    finite_rrd = (classifications == "RRd") & np.isfinite(first_overtone_period) & np.isfinite(best_period)
    if np.any(finite_rrd):
        lo_rrd = float(
            np.nanmin(np.concatenate([first_overtone_period[finite_rrd], best_period[finite_rrd]]))
        )
        hi_rrd = float(
            np.nanmax(np.concatenate([first_overtone_period[finite_rrd], best_period[finite_rrd]]))
        )
        finite_rrd_err = finite_rrd & np.isfinite(first_overtone_period_err)
        ax_first_overtone.errorbar(
            first_overtone_period[finite_rrd_err],
            best_period[finite_rrd_err],
            xerr=first_overtone_period_err[finite_rrd_err],
            fmt="none",
            ecolor=rr_d_color,
            elinewidth=LW_FINE,
            alpha=ALPHA_FAINT,
            zorder=1,
        )
        ax_first_overtone.scatter(
            first_overtone_period[finite_rrd],
            best_period[finite_rrd],
            s=MARKER_MS_SMALL**2,
            alpha=RRLYRAE_POINT_ALPHA,
            color=rr_d_color,
            zorder=2,
        )
        ax_first_overtone.plot(
            [lo_rrd, hi_rrd],
            [lo_rrd, hi_rrd],
            color=NEUTRAL_COLOR,
            ls="--",
            lw=LW_STANDARD,
            alpha=0.35,
            label=r"$P_{\rm LS}=P_{1\rm O}$",
            zorder=1,
        )
        ax_first_overtone.legend()
    else:
        ax_first_overtone.text(
            0.5,
            0.5,
            "No RRd stars with finite $P_{1\\rm O}$",
            ha="center",
            va="center",
            transform=ax_first_overtone.transAxes,
        )

    ax_first_overtone.set_xlabel(r"\texttt{vari\_rrlyrae} first-overtone period $P_{1\rm O}$ [days]")
    ax_first_overtone.set_ylabel(r"L-S period $P_{\rm LS}$ [days]")
    _apply_grid(ax_first_overtone)

    _tight_layout(fig)
    return axes


def plot_hr(source, ax=None):
    data = _as_table(source)
    if ax is None:
        _, ax = _single_panel(_columnwidth_figsize(7 / 2))

    class_masks = _plot_class_masks(data["best_classification"])
    if class_masks:
        for i, (label, mask) in enumerate(class_masks):
            ax.scatter(
                data["bp_rp"][mask],
                np.asarray(data["M_G"], dtype=float)[mask],
                color=_rrlyrae_class_color(label, i),
                label=label,
                s=MARKER_MS_FINE**2,
                alpha=ALPHA_DENSE,
                rasterized=True,
            )
    else:
        ax.scatter(
            data["bp_rp"],
            np.asarray(data["M_G"], dtype=float),
            color=PRIMARY_COLOR,
            label="RR Lyrae",
            s=MARKER_MS_FINE**2,
            alpha=ALPHA_DENSE,
            rasterized=True,
        )

    ax.invert_yaxis()
    ax.set_xlabel(r"$G_\mathrm{BP} - G_\mathrm{RP}$ [mag]")
    ax.set_ylabel(r"$M_{\rm G}$ [mag]")
    ax.legend()
    _apply_grid(ax)
    return ax


def plot_raw_phase_folded_lightcurve(source_id: int, data: Table):
    data = _as_table(data)
    data = data[data["source_id"] == int(source_id)]
    if len(data) == 0:
        raise ValueError("No light-curve rows available for plotting.")
    classification = str(data["best_classification"][0])
    period = float(rrlyrae_representative_period(data)[0])

    epoch = np.asarray(data["g_transit_time"], dtype=float)
    mag = np.asarray(data["g_transit_mag"], dtype=float)
    mag_err = np.asarray(data["g_transit_mag_err"], dtype=float)
    phase = (epoch % period) / period

    fig, axes = _grid_1x2(figsize=_textwidth_figsize(2))

    errorbar_alpha = RRLYRAE_POINT_ALPHA * (ALPHA_LIGHT / ALPHA_STANDARD)
    raw_errorbar_kwargs = {
        "fmt": "none",
        "ecolor": PRIMARY_COLOR,
        "elinewidth": LW_FINE,
        "alpha": errorbar_alpha,
        "zorder": 1,
    }
    phase_errorbar_kwargs = {
        "fmt": "none",
        "ecolor": SECONDARY_COLOR,
        "elinewidth": LW_FINE,
        "alpha": errorbar_alpha,
        "zorder": 1,
    }
    axes[0].errorbar(epoch, mag, yerr=mag_err, **raw_errorbar_kwargs)
    axes[1].errorbar(phase, mag, yerr=mag_err, **phase_errorbar_kwargs)

    axes[0].scatter(
        epoch,
        mag,
        s=MARKER_MS_FINE**2,
        alpha=RRLYRAE_POINT_ALPHA,
        color=PRIMARY_COLOR,
        zorder=2,
        rasterized=True,
        label="Raw light curve",
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Time [days]")
    axes[0].set_ylabel(r"$G$ [mag]")
    axes[0].legend(loc="best")
    _apply_grid(axes[0])

    axes[1].scatter(
        phase,
        mag,
        s=MARKER_MS_FINE**2,
        alpha=RRLYRAE_POINT_ALPHA,
        color=SECONDARY_COLOR,
        zorder=2,
        rasterized=True,
        label="Phase-folded light curve",
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Phase")
    axes[1].set_ylabel(r"$G$ [mag]")
    axes[1].legend(loc="best")
    _apply_grid(axes[1])

    axes[0].autoscale_view()
    axes[1].autoscale_view()

    return axes


def plot_fourier_harmonic_fits(
    data: Table,
    K_values: list[int] | tuple[int, ...],
):
    data = _as_table(data)

    from ugdatalab.lightcurves import fourier_fit, phase_fold

    period = float(data["period_ls"][0])

    epoch = np.asarray(data["g_transit_time"], dtype=float)
    mag = np.asarray(data["g_transit_mag"], dtype=float)
    mag_err = np.asarray(data["g_transit_mag_err"], dtype=float)
    phase = phase_fold(epoch, period)
    order = np.argsort(phase)
    epoch = epoch[order]
    phase = phase[order]
    mag = mag[order]
    mag_err = mag_err[order]

    phase_grid = np.linspace(0.0, 1.0, 1000, endpoint=False)
    epoch_grid = phase_grid * period
    K_values = tuple(int(K) for K in K_values)
    if len(K_values) == 0:
        raise ValueError("K_values must contain at least one harmonic order.")

    fig, outer_grid = _grid_nx1(
        len(K_values),
        figsize=(TEXTWIDTH_IN, A4_USABLE_HEIGHT_IN),
        hspace=0.24,
    )
    axes = np.empty((len(K_values), 2), dtype=object)
    first_curve_ax = None
    for i in range(len(K_values)):
        inner_grid = outer_grid[i, 0].subgridspec(2, 1, height_ratios=[2.5, 1], hspace=0.0)
        if first_curve_ax is None:
            ax_curve = fig.add_subplot(inner_grid[0, 0])
            first_curve_ax = ax_curve
        else:
            ax_curve = fig.add_subplot(inner_grid[0, 0], sharex=first_curve_ax)
        ax_resid = fig.add_subplot(inner_grid[1, 0], sharex=first_curve_ax)
        axes[i, 0] = ax_curve
        axes[i, 1] = ax_resid

    all_curve_values = [mag]
    all_residual_values = []
    for i, K in enumerate(K_values):
        fit = fourier_fit(data, period, K)
        model_mag = fit.predict(epoch_grid)
        fitted_mag = fit.predict(epoch)
        residuals = mag - fitted_mag
        all_curve_values.append(model_mag)
        all_residual_values.append(residuals)

        ax_curve, ax_resid = axes[i]
        is_bottom = i == len(K_values) - 1

        ax_curve.errorbar(
            phase,
            mag,
            yerr=mag_err,
            fmt="o",
            ms=MARKER_MS_FINE,
            elinewidth=LW_FINE,
            color=PRIMARY_COLOR,
            alpha=ALPHA_MUTED,
            zorder=2,
        )
        ax_curve.plot(phase_grid, model_mag, color=SECONDARY_COLOR, lw=LW_MEDIUM, zorder=3)
        ax_curve.invert_yaxis()
        ax_curve.set_ylabel(r"$G$ [mag]")
        _set_left_title(
            ax_curve,
            None,
            rf"$K={K}$, $\chi_\nu^2={fit.chi2_r:.2f}$",
        )
        ax_curve.tick_params(axis="x", bottom=False, labelbottom=False)
        _apply_grid(ax_curve)

        ax_resid.errorbar(
            phase,
            residuals,
            yerr=mag_err,
            fmt="o",
            ms=MARKER_MS_FINE,
            elinewidth=LW_FINE,
            color=PRIMARY_COLOR,
            alpha=ALPHA_MUTED,
            zorder=2,
        )
        _zero_line(ax_resid)
        ax_resid.set_ylabel("Res.")
        if is_bottom:
            ax_resid.set_xlabel("Phase")
        else:
            ax_resid.tick_params(axis="x", bottom=False, labelbottom=False)
        _apply_grid(ax_resid)

    curve_values = np.concatenate(all_curve_values)
    curve_values = curve_values[np.isfinite(curve_values)]
    if len(curve_values):
        curve_min = float(np.min(curve_values))
        curve_max = float(np.max(curve_values))
        curve_pad = max(0.05 * (curve_max - curve_min), 0.02)
        for ax_curve in axes[:, 0]:
            ax_curve.set_ylim(curve_max + curve_pad, curve_min - curve_pad)

    if all_residual_values:
        residual_values = np.concatenate(all_residual_values)
        residual_values = residual_values[np.isfinite(residual_values)]
        if len(residual_values):
            resid_max = float(np.max(np.abs(residual_values)))
            resid_max = resid_max if resid_max > 0 else 0.1
            resid_max *= 1.1
            for ax_resid in axes[:, 1]:
                ax_resid.set_ylim(-resid_max, resid_max)

    return axes


def _prepare_rrlyrae_shape_panels(source, rr_class: str) -> list[dict[str, Any]]:
    sample_data = _as_table(source)
    lightcurves = attach_periodogram_periods(attach_flux_mean_magnitudes(_as_table(source, attr="lightcurves").copy()))
    lightcurve_source_ids = lightcurves["source_id"]
    period_column = "p1_o" if rr_class == "RRc" else "pf"
    phase_grid = np.linspace(0.0, 1.0, 1000, endpoint=False)

    panels = []
    for row in sample_data:
        source_id = int(row["source_id"])
        period = float(row[period_column])
        star = lightcurves[lightcurve_source_ids == source_id]

        cross_validation_res = cross_validate_harmonics(star)
        best_K = int(cross_validation_res.best_K)
        fit = fourier_fit(star, period, best_K)
        mean_fourier_g = fourier_mean_magnitude(fit)

        phase = phase_fold(star["g_transit_time"], period)
        model_centered_data = fit.predict(star["g_transit_time"]) - mean_fourier_g
        mag_centered = np.asarray(star["g_transit_mag"], dtype=float) - mean_fourier_g
        residuals = mag_centered - model_centered_data
        mag_err = np.asarray(star["g_transit_mag_err"], dtype=float)
        order = np.argsort(phase)

        panels.append(
            {
                "source_id": source_id,
                "rr_class": rr_class,
                "phase_data": phase[order],
                "mag_centered": mag_centered[order],
                "mag_err": mag_err[order],
                "residuals": residuals[order],
                "phase_grid": phase_grid,
                "model_centered": fit.predict(phase_grid * period) - mean_fourier_g,
                "best_K": best_K,
                "period": period,
                "chi2_r": float(fit.chi2_r),
                "n_epochs": len(star),
            }
        )

    if len(panels) == 0:
        raise ValueError(f"{rr_class} sample must contain at least one source.")
    return panels


def plot_rrlyrae_shape_comparison(rrab_source, rrc_source):
    rrab_panels = _prepare_rrlyrae_shape_panels(rrab_source, "RRab")
    rrc_panels = _prepare_rrlyrae_shape_panels(rrc_source, "RRc")
    panels = rrab_panels + rrc_panels

    nrows = max(len(rrc_panels), len(rrab_panels))
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 3.25 * nrows))
    outer_grid = fig.add_gridspec(
        nrows,
        2,
        hspace=0.48,
        wspace=0.28,
    )
    axes = np.empty((2 * nrows, 2), dtype=object)

    all_values = []
    all_residual_values = []
    for panel in panels:
        mag_centered = np.asarray(panel["mag_centered"], dtype=float)
        mag_err = np.asarray(panel["mag_err"], dtype=float)
        mag_err = np.where(np.isfinite(mag_err), mag_err, 0.0)
        mag_finite = np.isfinite(mag_centered)
        if np.any(mag_finite):
            all_values.append((mag_centered - mag_err)[mag_finite])
            all_values.append((mag_centered + mag_err)[mag_finite])

        model_centered = np.asarray(panel["model_centered"], dtype=float)
        finite = model_centered[np.isfinite(model_centered)]
        if len(finite):
            all_values.append(finite)
        residuals = np.asarray(panel["residuals"], dtype=float)
        residual_finite = np.isfinite(residuals)
        if np.any(residual_finite):
            all_residual_values.append((residuals - mag_err)[residual_finite])
            all_residual_values.append((residuals + mag_err)[residual_finite])
    if len(all_values) == 0:
        raise ValueError("No finite centered magnitudes are available for plotting.")

    values = np.concatenate(all_values)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    pad = max(0.05 * (value_max - value_min), 0.05)
    y_limits = (value_min - pad, value_max + pad)
    residual_limits = (-1.0, 1.0)
    if len(all_residual_values):
        residual_values = np.concatenate(all_residual_values)
        residual_values = residual_values[np.isfinite(residual_values)]
        if len(residual_values):
            resid_max = float(np.max(np.abs(residual_values)))
            resid_max = max(resid_max, 0.05) * 1.1
            residual_limits = (-resid_max, resid_max)

    panel_columns = (
        (rrab_panels, PRIMARY_COLOR, "RRab"),
        (rrc_panels, SECONDARY_COLOR, "RRc"),
    )
    ylabel = r"$G - \langle G \rangle_{\rm Fourier}$"

    for col, (panel_list, point_color, rr_class) in enumerate(panel_columns):
        first_curve_ax = None
        for row in range(nrows):
            inner_grid = outer_grid[row, col].subgridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.0)
            if first_curve_ax is None:
                ax = fig.add_subplot(inner_grid[0, 0])
                first_curve_ax = ax
            else:
                ax = fig.add_subplot(inner_grid[0, 0], sharex=first_curve_ax)
            ax_resid = fig.add_subplot(inner_grid[1, 0], sharex=ax)
            axes[2 * row, col] = ax
            axes[2 * row + 1, col] = ax_resid
            if row >= len(panel_list):
                ax.axis("off")
                ax_resid.axis("off")
                continue

            panel = panel_list[row]
            phase_data = np.asarray(panel["phase_data"], dtype=float)
            mag_centered = np.asarray(panel["mag_centered"], dtype=float)
            mag_err = np.asarray(panel["mag_err"], dtype=float)
            residuals = np.asarray(panel["residuals"], dtype=float)
            phase_grid = np.asarray(panel["phase_grid"], dtype=float)
            model_centered = np.asarray(panel["model_centered"], dtype=float)
            source_id = int(panel["source_id"])
            best_K = int(panel["best_K"])
            period = float(panel["period"])
            chi2_r = float(panel["chi2_r"])

            ax.errorbar(
                phase_data,
                mag_centered,
                yerr=mag_err,
                fmt="o",
                ms=MARKER_MS_SMALL,
                elinewidth=LW_FINE,
                color=point_color,
                alpha=ALPHA_MUTED,
                zorder=2,
                label=rr_class if row == 0 else "_nolegend_",
            )
            ax.plot(
                phase_grid,
                model_centered,
                color=NEUTRAL_COLOR,
                lw=LW_MEDIUM,
                zorder=3,
                label="Fourier model" if row == 0 else "_nolegend_",
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(y_limits)
            ax.invert_yaxis()
            ax.set_ylabel(ylabel)
            _set_left_title(
                ax,
                _source_header(source_id, rr_class),
                rf"$P={period:.4f}\,\mathrm{{d}}$, $K={best_K}$, $\chi_\nu^2={chi2_r:.2f}$",
            )
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            _apply_grid(ax)

            ax_resid.errorbar(
                phase_data,
                residuals,
                yerr=mag_err,
                fmt="o",
                ms=MARKER_MS_SMALL,
                elinewidth=LW_FINE,
                color=point_color,
                alpha=ALPHA_MUTED,
                zorder=2,
            )
            _zero_line(ax_resid)
            ax_resid.set_xlim(0.0, 1.0)
            ax_resid.set_ylim(residual_limits)
            ax_resid.set_ylabel("Res.")
            ax_resid.set_xlabel("Phase")
            _apply_grid(ax_resid)

        axes[0, col].legend(loc="best")

    return axes


def plot_fourier_cross_validation(result: HarmonicCrossValidationResult):
    Ks = np.asarray(result.Ks, dtype=int)
    chi2r_train = np.asarray(result.chi2r_train, dtype=float)
    chi2r_cv = np.asarray(result.chi2r_cv, dtype=float)
    best_K = int(result.best_K)
    target_id = int(result.source_id)
    n_train = len(result.train_idx)
    n_cv = len(result.cv_idx)
    valid = np.isfinite(chi2r_train) & np.isfinite(chi2r_cv)
    if not np.any(valid):
        raise ValueError("No finite cross-validation values are available for plotting.")

    best_mask = Ks == best_K
    if not np.any(best_mask):
        raise ValueError("best_K must be present in Ks.")
    cv_best = float(chi2r_cv[best_mask][0])

    _, ax = _single_panel(_columnwidth_figsize(3 / 2))
    ax.plot(
        Ks[valid],
        chi2r_train[valid],
        marker="o",
        ls="none",
        ms=MARKER_MS_FINE,
        alpha=ALPHA_DENSE,
        label=r"Training $\chi_r^2$",
    )
    ax.plot(
        Ks[valid],
        chi2r_cv[valid],
        marker="o",
        ls="none",
        ms=MARKER_MS_FINE,
        alpha=ALPHA_DENSE,
        label=r"Cross-validation $\chi_r^2$",
    )
    ax.axvline(best_K, color=TERTIARY_COLOR, ls=":", lw=LW_LIGHT, alpha=ALPHA_GUIDE, label=rf"Best $K={best_K}$")
    ax.axhline(cv_best, color=TERTIARY_COLOR, ls=":", lw=LW_LIGHT, alpha=ALPHA_GUIDE, label=rf"Best CV $\chi_r^2={cv_best:.3f}$")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel(r"$K$ (number of Fourier harmonics)")
    ax.set_ylabel(r"$\chi_r^2$")
    _set_left_title(
        ax,
        None,
        rf"20\% CV, train={n_train}, CV={n_cv}",
    )
    ax.legend()
    _apply_grid(ax)
    return ax


def plot_fourier_cv_normalized_residual_histograms(
    train_lightcurve: Table,
    cv_lightcurve: Table,
    low_fit: FourierFit,
    best_fit: FourierFit,
):
    train_lightcurve = _as_table(train_lightcurve)
    cv_lightcurve = _as_table(cv_lightcurve)

    train_epochs = np.asarray(train_lightcurve["g_transit_time"], dtype=float)
    train_mags = np.asarray(train_lightcurve["g_transit_mag"], dtype=float)
    train_errs = np.asarray(train_lightcurve["g_transit_mag_err"], dtype=float)
    cv_epochs = np.asarray(cv_lightcurve["g_transit_time"], dtype=float)
    cv_mags = np.asarray(cv_lightcurve["g_transit_mag"], dtype=float)
    cv_errs = np.asarray(cv_lightcurve["g_transit_mag_err"], dtype=float)

    train_norm_low = (train_mags - low_fit.predict(train_epochs)) / train_errs
    cv_norm_low = (cv_mags - low_fit.predict(cv_epochs)) / cv_errs
    train_norm_best = (train_mags - best_fit.predict(train_epochs)) / train_errs
    cv_norm_best = (cv_mags - best_fit.predict(cv_epochs)) / cv_errs

    return plot_fourier_normalized_residual_histograms(
        train_norm_low,
        cv_norm_low,
        train_norm_best,
        cv_norm_best,
        int(low_fit.K),
        int(best_fit.K),
    )


def plot_fourier_normalized_residual_histograms(
    train_norm_low: np.ndarray,
    cv_norm_low: np.ndarray,
    train_norm_best: np.ndarray,
    cv_norm_best: np.ndarray,
    low_K: int,
    best_K: int,
):
    fig, axes = _grid_1x2(figsize=_textwidth_figsize(3 / 2))
    bins = np.linspace(-5.0, 5.0, 31)
    x_gauss = np.linspace(-5.0, 5.0, 400)
    gaussian = np.exp(-0.5 * x_gauss**2) / np.sqrt(2.0 * np.pi)

    panels = (
        (axes[0], np.asarray(train_norm_low, dtype=float), np.asarray(cv_norm_low, dtype=float), rf"Low $K = {low_K}$"),
        (axes[1], np.asarray(train_norm_best, dtype=float), np.asarray(cv_norm_best, dtype=float), rf"Best $K = {best_K}$"),
    )

    for ax, train_norm, cv_norm, label in panels:
        train_norm = train_norm[np.isfinite(train_norm)]
        cv_norm = cv_norm[np.isfinite(cv_norm)]
        _, _, cv_patches = ax.hist(
            cv_norm,
            bins=bins,
            histtype="stepfilled",
            density=True,
            color=SECONDARY_COLOR,
            alpha=ALPHA_DIM,
            lw=LW_FINE,
            label="CV",
            zorder=1,
        )
        train_artist = ax.hist(
            train_norm,
            bins=bins,
            histtype="step",
            density=True,
            color=PRIMARY_COLOR,
            alpha=ALPHA_DENSE,
            lw=LW_MEDIUM,
            label="Training",
            zorder=2,
        )[2][0]
        gaussian_artist = ax.plot(
            x_gauss,
            gaussian,
            color=NEUTRAL_COLOR,
            ls="--",
            lw=LW_STANDARD,
            label=r"$\mathcal{N}(0,1)$",
            zorder=3,
        )[0]
        ax.axvline(0.0, color=NEUTRAL_COLOR, ls=":", lw=LW_STANDARD)
        _set_left_title(
            ax,
            None,
            label,
        )
        ax.set_xlabel(r"$(G - G_{\rm model})/\sigma$")
        _apply_grid(ax)
        ax._histogram_legend_handles = [train_artist, cv_patches[0], gaussian_artist]

    axes[0].set_ylabel("Density")
    axes[1].set_ylabel("Density")
    axes[0].legend(handles=axes[0]._histogram_legend_handles, loc="best")
    return axes


def plot_fourier_cv_phase_comparison(
    train_lightcurve: Table,
    cv_lightcurve: Table,
    best_fit: FourierFit,
    high_fit: FourierFit,
):
    train_lightcurve = _as_table(train_lightcurve)
    cv_lightcurve = _as_table(cv_lightcurve)

    from ugdatalab.lightcurves import phase_fold

    period = float(best_fit.period)

    train_epochs = np.asarray(train_lightcurve["g_transit_time"], dtype=float)
    train_mags = np.asarray(train_lightcurve["g_transit_mag"], dtype=float)
    train_errs = np.asarray(train_lightcurve["g_transit_mag_err"], dtype=float)
    cv_epochs = np.asarray(cv_lightcurve["g_transit_time"], dtype=float)
    cv_mags = np.asarray(cv_lightcurve["g_transit_mag"], dtype=float)
    cv_errs = np.asarray(cv_lightcurve["g_transit_mag_err"], dtype=float)

    train_phase = phase_fold(train_epochs, period)
    cv_phase = phase_fold(cv_epochs, period)
    phase_grid = np.linspace(0.0, 1.0, 1000, endpoint=False)
    epoch_grid = phase_grid * period

    model_mag_best = best_fit.predict(epoch_grid)
    model_mag_high = high_fit.predict(epoch_grid)

    return plot_fourier_train_cv_phase_comparison(
        train_phase,
        train_mags,
        cv_phase,
        cv_mags,
        phase_grid,
        model_mag_best,
        model_mag_high,
        int(best_fit.K),
        int(high_fit.K),
        train_errs=train_errs,
        cv_errs=cv_errs,
    )


def plot_fourier_train_cv_phase_comparison(
    train_phase: np.ndarray,
    train_mags: np.ndarray,
    cv_phase: np.ndarray,
    cv_mags: np.ndarray,
    phase_grid: np.ndarray,
    model_mag_best: np.ndarray,
    model_mag_high: np.ndarray,
    best_K: int,
    high_K: int,
    *,
    train_errs: np.ndarray | None = None,
    cv_errs: np.ndarray | None = None,
):
    train_phase = np.asarray(train_phase, dtype=float)
    train_mags = np.asarray(train_mags, dtype=float)
    train_errs = None if train_errs is None else np.asarray(train_errs, dtype=float)
    cv_phase = np.asarray(cv_phase, dtype=float)
    cv_mags = np.asarray(cv_mags, dtype=float)
    cv_errs = None if cv_errs is None else np.asarray(cv_errs, dtype=float)
    phase_grid = np.asarray(phase_grid, dtype=float)
    model_mag_best = np.asarray(model_mag_best, dtype=float)
    model_mag_high = np.asarray(model_mag_high, dtype=float)

    fig, axes = _grid_1x2(figsize=_textwidth_figsize(3 / 2))
    all_mags = [train_mags[np.isfinite(train_mags)], cv_mags[np.isfinite(cv_mags)]]
    if train_errs is not None:
        train_errs_safe = np.where(np.isfinite(train_errs), train_errs, 0.0)
        all_mags.extend(
            [
                (train_mags - train_errs_safe)[np.isfinite(train_mags)],
                (train_mags + train_errs_safe)[np.isfinite(train_mags)],
            ]
        )
    if cv_errs is not None:
        cv_errs_safe = np.where(np.isfinite(cv_errs), cv_errs, 0.0)
        all_mags.extend(
            [
                (cv_mags - cv_errs_safe)[np.isfinite(cv_mags)],
                (cv_mags + cv_errs_safe)[np.isfinite(cv_mags)],
            ]
        )
    all_mags = np.concatenate(all_mags)
    pad = 0.05 * (np.max(all_mags) - np.min(all_mags))
    y_limits = (np.max(all_mags) + pad, np.min(all_mags) - pad)

    panels = (
        (axes[0], model_mag_best, best_K),
        (axes[1], model_mag_high, high_K),
    )

    for ax, model_mag, K in panels:
        ax.scatter(
            train_phase,
            train_mags,
            s=MARKER_MS_SMALL**2,
            alpha=ALPHA_EXTRA_LIGHT,
            color=PRIMARY_COLOR,
            rasterized=True,
            label="Training",
        )
        ax.scatter(
            cv_phase,
            cv_mags,
            s=MARKER_MS_SMALL**2,
            alpha=ALPHA_MUTED,
            color=SECONDARY_COLOR,
            rasterized=True,
            label="CV",
        )
        if train_errs is not None:
            ax.errorbar(
                train_phase,
                train_mags,
                yerr=train_errs,
                fmt="none",
                ecolor=PRIMARY_COLOR,
                elinewidth=LW_FINE,
                alpha=ALPHA_DIM,
                zorder=1,
            )
        if cv_errs is not None:
            ax.errorbar(
                cv_phase,
                cv_mags,
                yerr=cv_errs,
                fmt="none",
                ecolor=SECONDARY_COLOR,
                elinewidth=LW_FINE,
                alpha=ALPHA_DIM,
                zorder=1,
            )
        ax.plot(phase_grid, model_mag, color=NEUTRAL_COLOR, lw=LW_STANDARD, label="Model")
        ax.set_ylim(y_limits)
        ax.invert_yaxis()
        ax.set_xlabel("Phase")
        detail = rf"Best $K = {K}$" if K == best_K else rf"High $K = {K}$"
        _set_left_title(
            ax,
            None,
            detail,
        )
        _apply_grid(ax)

    axes[0].set_ylabel(r"$G$ [mag]")
    axes[1].set_ylabel(r"$G$ [mag]")
    axes[0].legend(loc="best")
    return axes


def plot_fourier_extrapolation(
    result: HarmonicCrossValidationResult,
    fit,
):
    from ugdatalab.lightcurves import predict_future_magnitude

    epochs = np.asarray(fit.epochs, dtype=float)
    mags = np.asarray(fit.mags, dtype=float)
    mag_errs = np.asarray(fit.mag_errs, dtype=float)
    epoch_last = float(np.max(epochs))
    epoch_window_start = epoch_last - 5.0
    epoch_window_end = epoch_last + 12.0
    epoch_grid = np.linspace(epoch_window_start, epoch_window_end, 2000)
    mag_grid = fit.predict(epoch_grid)
    mag_grid_err = fit.predict_std(epoch_grid)
    epoch_pred, mag_pred, mag_pred_err = predict_future_magnitude(fit)
    time_unit = getattr(fit.epochs, "unit", None)
    time_label = f"Time [{time_unit}]" if time_unit is not None else "Time [days]"

    observed_mask = epochs >= float(np.min(epoch_grid))
    model_observed = epoch_grid <= float(epoch_last)

    _, ax = _single_panel(_textwidth_figsize(5 / 2))
    ax.errorbar(
        epochs[observed_mask],
        mags[observed_mask],
        yerr=mag_errs[observed_mask],
        fmt="o",
        ms=MARKER_MS_SMALL,
        elinewidth=LW_FINE,
        color=PRIMARY_COLOR,
        alpha=RRLYRAE_POINT_ALPHA,
        label="Gaia data",
    )
    ax.plot(
        epoch_grid[model_observed],
        mag_grid[model_observed],
        color=SECONDARY_COLOR,
        lw=LW_STANDARD,
        label=rf"Fourier fit ($K={fit.K}$)",
    )
    ax.plot(
        epoch_grid[~model_observed],
        mag_grid[~model_observed],
        color=SECONDARY_COLOR,
        lw=LW_STANDARD,
        ls="--",
        label="12-day extrapolation",
    )
    if np.any(np.isfinite(mag_grid_err)) and np.nanmax(mag_grid_err) > 0.0:
        ax.fill_between(
            epoch_grid,
            mag_grid - mag_grid_err,
            mag_grid + mag_grid_err,
            color=SECONDARY_COLOR,
            alpha=ALPHA_SHADE,
            lw=LW_NONE,
            label=r"Model $\pm1\sigma$",
        )
    ax.axvline(epoch_last, color=NEUTRAL_COLOR, ls=":", lw=LW_STANDARD, alpha=ALPHA_GUIDE)
    ax.plot(
        epoch_pred,
        mag_pred,
        marker="*",
        ms=MARKER_MS_BIG,
        color="k",
        lw=LW_NONE,
        label=rf"10-day prediction: $G={mag_pred:.3f}\pm{mag_pred_err:.3f}$",
    )
    ax.invert_yaxis()
    ax.set_xlim(float(np.min(epoch_grid)), float(np.max(epoch_grid)))
    ax.set_xlabel(time_label)
    ax.set_ylabel(r"$G$ [mag]")
    ax.legend(ncols=2, loc="upper right")
    _apply_grid(ax)
    return ax


def plot_mean_g_catalog_comparison(
    summary: Table,
):
    summary = _as_table(summary)
    simple_mean_g = np.asarray(summary["mean_apparent_g"], dtype=float)
    simple_mean_g_err = _optional_float_array(summary, "mean_apparent_g_err", "mean_apparent_g_error")
    fourier_mean_g = np.asarray(summary["fourier_mean_apparent_g"], dtype=float)
    fourier_mean_g_err = _optional_float_array(
        summary,
        "fourier_mean_apparent_g_err",
        "fourier_mean_apparent_g_error",
    )
    int_average_g = np.asarray(summary["int_average_g"], dtype=float)
    int_average_g_err = _optional_float_array(summary, "int_average_g_err", "int_average_g_error")

    resid_simple = simple_mean_g - int_average_g
    resid_fourier = fourier_mean_g - int_average_g
    simple_resid_err = np.hypot(
        np.where(np.isfinite(simple_mean_g_err), simple_mean_g_err, 0.0),
        np.where(np.isfinite(int_average_g_err), int_average_g_err, 0.0),
    )
    fourier_resid_err = np.hypot(
        np.where(np.isfinite(fourier_mean_g_err), fourier_mean_g_err, 0.0),
        np.where(np.isfinite(int_average_g_err), int_average_g_err, 0.0),
    )

    simple_valid = np.isfinite(simple_mean_g) & np.isfinite(int_average_g) & np.isfinite(resid_simple)
    fourier_valid = np.isfinite(fourier_mean_g) & np.isfinite(int_average_g) & np.isfinite(resid_fourier)

    int_err_safe    = np.where(np.isfinite(int_average_g_err),    int_average_g_err,    0.0)
    simple_err_safe = np.where(np.isfinite(simple_mean_g_err),  simple_mean_g_err,  0.0)
    fourier_err_safe = np.where(np.isfinite(fourier_mean_g_err), fourier_mean_g_err, 0.0)

    def _lims(y, y_err, valid):
        vals = np.concatenate([
            int_average_g[valid], y[valid],
            (int_average_g - int_err_safe)[valid], (int_average_g + int_err_safe)[valid],
            (y - y_err)[valid], (y + y_err)[valid],
        ])
        pad = 0.05 * (np.max(vals) - np.min(vals))
        return np.array([np.min(vals) - pad, np.max(vals) + pad])

    simple_lims  = _lims(simple_mean_g,  simple_err_safe,  simple_valid)
    fourier_lims = _lims(fourier_mean_g, fourier_err_safe, fourier_valid)

    simple_resid_max  = 1.1 * np.nanmax(np.abs(np.concatenate([
        resid_simple[simple_valid]  - simple_resid_err[simple_valid],
        resid_simple[simple_valid]  + simple_resid_err[simple_valid],
    ])))
    fourier_resid_max = 1.1 * np.nanmax(np.abs(np.concatenate([
        resid_fourier[fourier_valid] - fourier_resid_err[fourier_valid],
        resid_fourier[fourier_valid] + fourier_resid_err[fourier_valid],
    ])))

    # Two side-by-side columns at textwidth — simple (left) and Fourier (right) —
    # each with a main panel and a touching residual panel via a nested GridSpec.
    fig = plt.figure(figsize=_textwidth_figsize(4))
    gs = fig.add_gridspec(1, 2, wspace=0.30)
    gs_left  = gs[0].subgridspec(2, 1, height_ratios=[4, 1], hspace=0)
    gs_right = gs[1].subgridspec(2, 1, height_ratios=[4, 1], hspace=0)
    ax_simple        = fig.add_subplot(gs_left[0])
    ax_simple_resid  = fig.add_subplot(gs_left[1],  sharex=ax_simple)
    ax_fourier       = fig.add_subplot(gs_right[0])
    ax_fourier_resid = fig.add_subplot(gs_right[1], sharex=ax_fourier)
    axes = np.array([ax_simple, ax_simple_resid, ax_fourier, ax_fourier_resid])

    xerr = np.where(np.isfinite(int_average_g_err), int_average_g_err, 0.0)

    for ax_main, ax_resid, y, y_err, y_err_safe, resid, resid_err, lims, resid_max, color, ylabel in [
        (ax_simple,  ax_simple_resid,  simple_mean_g,  simple_mean_g_err,  simple_err_safe,
         resid_simple,  simple_resid_err,  simple_lims,  simple_resid_max,
         PRIMARY_COLOR,   r"$\langle G \rangle_{\rm epoch}$ [mag]"),
        (ax_fourier, ax_fourier_resid, fourier_mean_g, fourier_mean_g_err, fourier_err_safe,
         resid_fourier, fourier_resid_err, fourier_lims, fourier_resid_max,
         SECONDARY_COLOR, r"$\langle G \rangle_{\rm Fourier}$ [mag]"),
    ]:
        valid = np.isfinite(y) & np.isfinite(int_average_g) & np.isfinite(resid)
        yerr_safe = np.where(np.isfinite(y_err), y_err, 0.0)

        ax_main.errorbar(
            int_average_g[valid], y[valid],
            xerr=xerr[valid], yerr=yerr_safe[valid],
            fmt="none", ecolor=color, elinewidth=LW_FINE, alpha=ALPHA_DIM, zorder=1,
        )
        ax_main.scatter(
            int_average_g[valid], y[valid],
            s=MARKER_MS_SMALL**2, alpha=RRLYRAE_POINT_ALPHA, color=color, rasterized=True,
        )
        ax_main.plot(lims, lims, color=NEUTRAL_COLOR, ls="--", lw=LW_STANDARD)
        ax_main.set_xlim(lims[1], lims[0])
        ax_main.set_ylim(lims)
        ax_main.invert_yaxis()
        ax_main.set_ylabel(ylabel)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        _apply_grid(ax_main)

        ax_resid.errorbar(
            int_average_g[valid], resid[valid],
            xerr=xerr[valid], yerr=resid_err[valid],
            fmt="none", ecolor=color, elinewidth=LW_FINE, alpha=ALPHA_DIM, zorder=1,
        )
        ax_resid.scatter(
            int_average_g[valid], resid[valid],
            s=MARKER_MS_SMALL**2, alpha=RRLYRAE_POINT_ALPHA, color=color, rasterized=True,
        )
        ax_resid.axhline(0.0, color=NEUTRAL_COLOR, ls="--", lw=LW_STANDARD)
        ax_resid.set_xlim(lims[1], lims[0])
        ax_resid.set_ylim(-resid_max, resid_max)
        ax_resid.set_xlabel(r"Gaia $\mathtt{int\_average\_g}$ [mag]")
        ax_resid.set_ylabel("Res.")
        _apply_grid(ax_resid)

    return axes


def plot_inlier_prob_map(source, ax=None):
    data = _as_table(source, attr="all_data")
    if ax is None:
        _, ax = _single_panel(_columnwidth_figsize(5 / 2))

    periods = _plot_period_values(data)
    sc = ax.scatter(
        periods,
        np.asarray(data["M_G"], dtype=float),
        c=np.asarray(data["inlier_prob"], dtype=float),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        s=MARKER_MS_FINE**2,
        alpha=ALPHA_DENSE,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="Posterior inlier probability")
    _set_log_period_xaxis(ax)
    _set_descending_magnitude_yaxis(ax)
    ax.set_xlabel(r"$P$ [days]")
    ax.set_ylabel(r"$M_{\rm G}$ [mag]")
    _apply_grid(ax)
    return ax


def _relation_predictive_envelope(ctx: Any, samples: np.ndarray):
    x_grid = np.linspace(float(np.min(ctx.x_centered)), float(np.max(ctx.x_centered)), 300)
    order = np.argsort(ctx.x_centered)
    sigma_obs_grid = np.interp(x_grid, ctx.x_centered[order], ctx.sigma[order])
    sigma_logp = np.asarray(getattr(ctx, "sigma_logp", np.zeros_like(ctx.x_centered)), dtype=float)
    sigma_x_grid = np.interp(x_grid, ctx.x_centered[order], sigma_logp[order])

    rng = np.random.default_rng(42)
    step = max(len(samples) // 400, 1)
    sample_pool = np.asarray(samples[::step], dtype=float)
    if len(sample_pool) > 400:
        keep_idx = rng.choice(len(sample_pool), size=400, replace=False)
        sample_pool = sample_pool[keep_idx]

    mean_draws = np.empty((len(sample_pool), len(x_grid)), dtype=float)
    predictive_draws = np.empty_like(mean_draws)
    for i, (a_s, b_s, sigma_s) in enumerate(sample_pool):
        mu_grid = a_s * x_grid + b_s
        sigma_pred = np.sqrt(sigma_obs_grid**2 + sigma_s**2 + (a_s * sigma_x_grid) ** 2)
        mean_draws[i] = mu_grid
        predictive_draws[i] = rng.normal(mu_grid, sigma_pred)

    return SimpleNamespace(
        x_grid=x_grid,
        median_mean=np.quantile(mean_draws, 0.50, axis=0),
        q16=np.quantile(predictive_draws, 0.16, axis=0),
        q84=np.quantile(predictive_draws, 0.84, axis=0),
        q025=np.quantile(predictive_draws, 0.025, axis=0),
        q975=np.quantile(predictive_draws, 0.975, axis=0),
    )


def _pl_predictive_envelope(ctx: Any, samples: np.ndarray):
    return _relation_predictive_envelope(ctx, samples)


def _draw_pl_posterior_predictive_layer(
    ax: Any,
    ctx: Any,
    samples: np.ndarray,
    color: str,
    *,
    data_label: str | None,
    median_label: str | None,
    show_interval_labels: bool,
) -> None:
    env = _pl_predictive_envelope(ctx, samples)

    ax.errorbar(
        np.asarray(ctx.x_centered, dtype=float),
        np.asarray(ctx.y, dtype=float),
        yerr=np.asarray(ctx.sigma, dtype=float),
        fmt="none",
        color=NEUTRAL_COLOR,
        alpha=ALPHA_LIGHT,
        lw=LW_FINE,
        zorder=1,
    )
    ax.scatter(
        np.asarray(ctx.x_centered, dtype=float),
        np.asarray(ctx.y, dtype=float),
        s=MARKER_MS_SMALL**2,
        color=color,
        alpha=ALPHA_MUTED,
        rasterized=True,
        zorder=2,
        label=data_label,
    )
    ax.fill_between(
        env.x_grid,
        env.q025,
        env.q975,
        color=color,
        alpha=ALPHA_SHADE,
        lw=LW_NONE,
        zorder=3,
        label="95% predictive envelope" if show_interval_labels else None,
    )
    ax.fill_between(
        env.x_grid,
        env.q16,
        env.q84,
        color=color,
        alpha=ALPHA_EXTRA_LIGHT,
        lw=LW_NONE,
        zorder=4,
        label="68% predictive envelope" if show_interval_labels else None,
    )
    ax.plot(
        env.x_grid,
        env.median_mean,
        color=color,
        lw=LW_EMPHASIS,
        zorder=5,
        label=median_label,
    )


def plot_pl_posterior_predictive(
    ctx: Any,
    samples: np.ndarray,
):
    _, ax = _single_panel(_textwidth_figsize(19 / 4))
    color = _rrlyrae_class_color(getattr(ctx, "class_label", ""), 0)
    _draw_pl_posterior_predictive_layer(
        ax,
        ctx,
        samples,
        color,
        data_label=f"{ctx.class_label} data",
        median_label="Posterior median",
        show_interval_labels=True,
    )
    ax.set_xlabel(r"$\log_{10}(P/\mathrm{day}) - \langle \log_{10}P \rangle_{\rm class}$")
    ax.set_ylabel(r"$M_{\rm G}$ [mag]")
    _set_descending_magnitude_yaxis(ax)
    ax.legend(loc="best")
    _apply_grid(ax)
    _tight_layout(ax.figure)
    return ax


def plot_pl_posterior_predictive_comparison(
    primary_ctx: Any,
    primary_samples: np.ndarray,
    secondary_ctx: Any,
    secondary_samples: np.ndarray,
):
    _, ax = _single_panel(_textwidth_figsize(5))
    primary_color = _rrlyrae_class_color(getattr(primary_ctx, "class_label", ""), 0)
    secondary_color = _rrlyrae_class_color(getattr(secondary_ctx, "class_label", ""), 1)
    _draw_pl_posterior_predictive_layer(
        ax,
        primary_ctx,
        primary_samples,
        primary_color,
        data_label=f"{primary_ctx.class_label} data",
        median_label=f"{primary_ctx.class_label} posterior median",
        show_interval_labels=False,
    )
    _draw_pl_posterior_predictive_layer(
        ax,
        secondary_ctx,
        secondary_samples,
        secondary_color,
        data_label=f"{secondary_ctx.class_label} data",
        median_label=f"{secondary_ctx.class_label} posterior median",
        show_interval_labels=False,
    )
    ax.set_xlabel(r"$\log_{10}(P/\mathrm{day}) - \langle \log_{10}P \rangle_{\rm class}$")
    ax.set_ylabel(r"$M_{\rm G}$ [mag]")
    _set_descending_magnitude_yaxis(ax)
    ax.legend(loc="best")
    _apply_grid(ax)
    _tight_layout(ax.figure)
    return ax


def _draw_pc_posterior_predictive_layer(
    ax: Any,
    ctx: Any,
    samples: np.ndarray,
    color: str,
    *,
    data_label: str | None,
    median_label: str | None,
    show_interval_labels: bool,
) -> None:
    env = _relation_predictive_envelope(ctx, samples)

    ax.errorbar(
        np.asarray(ctx.x_centered, dtype=float),
        np.asarray(ctx.y, dtype=float),
        yerr=np.asarray(ctx.sigma, dtype=float),
        fmt="none",
        color=NEUTRAL_COLOR,
        alpha=ALPHA_LIGHT,
        lw=LW_FINE,
        zorder=1,
    )
    ax.scatter(
        np.asarray(ctx.x_centered, dtype=float),
        np.asarray(ctx.y, dtype=float),
        s=MARKER_MS_SMALL**2,
        color=color,
        alpha=ALPHA_MUTED,
        rasterized=True,
        zorder=2,
        label=data_label,
    )
    ax.fill_between(
        env.x_grid,
        env.q025,
        env.q975,
        color=color,
        alpha=ALPHA_SHADE,
        lw=LW_NONE,
        zorder=3,
        label="95% predictive envelope" if show_interval_labels else None,
    )
    ax.fill_between(
        env.x_grid,
        env.q16,
        env.q84,
        color=color,
        alpha=ALPHA_EXTRA_LIGHT,
        lw=LW_NONE,
        zorder=4,
        label="68% predictive envelope" if show_interval_labels else None,
    )
    ax.plot(
        env.x_grid,
        env.median_mean,
        color=color,
        lw=LW_EMPHASIS,
        zorder=5,
        label=median_label,
    )


def plot_pc_posterior_predictive(
    ctx: Any,
    samples: np.ndarray,
):
    _, ax = _single_panel(_textwidth_figsize(19 / 4))
    color = _rrlyrae_class_color(getattr(ctx, "class_label", ""), 0)
    _draw_pc_posterior_predictive_layer(
        ax,
        ctx,
        samples,
        color,
        data_label=f"{ctx.class_label} data",
        median_label="Posterior median",
        show_interval_labels=True,
    )
    ax.set_xlabel(r"$\log_{10}(P/\mathrm{day}) - \langle \log_{10}P \rangle_{\rm class}$")
    ax.set_ylabel(getattr(ctx, "y_label", r"$G_\mathrm{BP} - G_\mathrm{RP}$ [mag]"))
    ax.legend(loc="best")
    _apply_grid(ax)
    _tight_layout(ax.figure)
    return ax


def plot_pc_posterior_predictive_comparison(
    primary_ctx: Any,
    primary_samples: np.ndarray,
    secondary_ctx: Any,
    secondary_samples: np.ndarray,
):
    _, ax = _single_panel(_textwidth_figsize(5))
    primary_color = _rrlyrae_class_color(getattr(primary_ctx, "class_label", ""), 0)
    secondary_color = _rrlyrae_class_color(getattr(secondary_ctx, "class_label", ""), 1)
    _draw_pc_posterior_predictive_layer(
        ax,
        primary_ctx,
        primary_samples,
        primary_color,
        data_label=f"{primary_ctx.class_label} data",
        median_label=f"{primary_ctx.class_label} posterior median",
        show_interval_labels=False,
    )
    _draw_pc_posterior_predictive_layer(
        ax,
        secondary_ctx,
        secondary_samples,
        secondary_color,
        data_label=f"{secondary_ctx.class_label} data",
        median_label=f"{secondary_ctx.class_label} posterior median",
        show_interval_labels=False,
    )
    ax.set_xlabel(r"$\log_{10}(P/\mathrm{day}) - \langle \log_{10}P \rangle_{\rm class}$")
    ax.set_ylabel(getattr(primary_ctx, "y_label", r"$G_\mathrm{BP} - G_\mathrm{RP}$ [mag]"))
    ax.legend(loc="best")
    _apply_grid(ax)
    _tight_layout(ax.figure)
    return ax


def _draw_optical_pl_literature_panel(
    ax: Any,
    comparison: Any,
    *,
    color: str,
    show_legend: bool,
) -> None:
    ax.errorbar(
        np.asarray(comparison.x_obs, dtype=float),
        np.asarray(comparison.y_obs, dtype=float),
        yerr=np.asarray(comparison.sigma_obs, dtype=float),
        fmt="o",
        ms=MARKER_MS_FINE,
        alpha=ALPHA_DIM,
        color=color,
        elinewidth=LW_GRID,
        capsize=0,
        label=f"{comparison.rr_class} Gaia data",
    )
    ax.plot(
        np.asarray(comparison.x_grid, dtype=float),
        np.asarray(comparison.median_mean, dtype=float),
        color="k",
        lw=LW_MODEL,
        label=r"This work: median $M_{\rm G}$ fit",
    )
    ax.fill_between(
        np.asarray(comparison.x_grid, dtype=float),
        np.asarray(comparison.predictive_q16, dtype=float),
        np.asarray(comparison.predictive_q84, dtype=float),
        color=color,
        alpha=ALPHA_EXTRA_LIGHT,
        lw=LW_NONE,
        label="This work: 68% predictive band",
    )
    ax.text(
        0.03,
        0.97,
        "Klein+Bloom 2014: optical PL is weak;\nIR PL is steeper and tighter.\n\nBeaton+2018: bandpass, reddening,\nmetallicity, and calibration choices matter.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=TITLE_SIZE,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": LIGHT_NEUTRAL_COLOR,
            "alpha": ALPHA_SHADE + 0.35,
        },
    )
    ax.set_title(f"{comparison.rr_class}: Gaia $G$ fit in optical-literature context")
    ax.set_xlabel(r"$\log_{10}(P/\mathrm{day})$")
    _apply_grid(ax)
    _set_descending_magnitude_yaxis(ax)
    if show_legend:
        ax.legend(loc="best")


def plot_optical_pl_literature_comparison(
    rrab_comparison: Any,
    rrc_comparison: Any,
):
    fig, axes = _grid_1x2(
        figsize=_textwidth_figsize(19 / 4),
        sharey=True,
    )
    _draw_optical_pl_literature_panel(
        axes[0],
        rrab_comparison,
        color=PRIMARY_COLOR,
        show_legend=True,
    )
    _draw_optical_pl_literature_panel(
        axes[1],
        rrc_comparison,
        color=SECONDARY_COLOR,
        show_legend=False,
    )
    axes[0].set_ylabel(r"Absolute magnitude [mag]")
    _tight_layout(fig)
    return fig


def _draw_pc_posterior_draws_panel(
    ax: Any,
    data: Any,
    samples: np.ndarray,
    *,
    rr_class: str,
    color: str,
    seed: int,
) -> None:
    x_values = np.asarray(data.x, dtype=float)
    y_values = np.asarray(data.y, dtype=float)
    sigma_values = np.asarray(data.sigma, dtype=float)
    x_range = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 300)
    rng = np.random.default_rng(seed)
    n_draws = min(50, len(samples))
    draw_idx = rng.choice(len(samples), size=n_draws, replace=False)

    ax.errorbar(
        x_values,
        y_values,
        yerr=sigma_values,
        fmt=".",
        color="k",
        alpha=ALPHA_FAINT,
        ms=MARKER_MS_FINE,
        lw=LW_GRID,
        zorder=1,
        label=f"{rr_class} data",
    )
    for slope, intercept, _ in np.asarray(samples, dtype=float)[draw_idx]:
        ax.plot(x_range, slope * x_range + intercept, lw=LW_GRID, alpha=ALPHA_MUTED, color=color, zorder=2)
    ax.plot([], [], color=color, lw=LW_FIT, label="50 posterior draws")
    ax.set_title(f"{rr_class} period-color relation")
    ax.set_xlabel(data.x_label)
    _apply_grid(ax)
    ax.legend(fontsize=LEGEND_SIZE)


def plot_pc_posterior_draws_comparison(
    rrab_data: Any,
    rrab_samples: np.ndarray,
    rrab_seed: int,
    rrc_data: Any,
    rrc_samples: np.ndarray,
    rrc_seed: int,
):
    fig, axes = _grid_1x2(
        figsize=_textwidth_figsize(19 / 4),
        sharey=True,
    )
    _draw_pc_posterior_draws_panel(
        axes[0],
        rrab_data,
        rrab_samples,
        rr_class="RRab",
        color=PRIMARY_COLOR,
        seed=rrab_seed,
    )
    _draw_pc_posterior_draws_panel(
        axes[1],
        rrc_data,
        rrc_samples,
        rr_class="RRc",
        color=SECONDARY_COLOR,
        seed=rrc_seed,
    )
    axes[0].set_ylabel(rrab_data.y_label)
    _tight_layout(fig)
    return fig


def plot_empirical_vs_catalog_extinction_comparison(
    catalog_ag: np.ndarray,
    empirical_ag: np.ndarray,
    residuals: np.ndarray,
):
    fig, axes = _grid_1x2(
        figsize=_textwidth_figsize(7 / 2),
        sharey=False,
    )

    x_values = np.asarray(catalog_ag, dtype=float)
    y_values = np.asarray(empirical_ag, dtype=float)
    delta_values = np.asarray(residuals, dtype=float)

    finite_scatter = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(finite_scatter):
        raise ValueError("catalog_ag and empirical_ag must contain at least one finite pair.")

    x_plot = x_values[finite_scatter]
    y_plot = y_values[finite_scatter]
    line_min = float(min(np.min(x_plot), np.min(y_plot)))
    line_max = float(max(np.max(x_plot), np.max(y_plot)))
    if not np.isfinite(line_min) or not np.isfinite(line_max):
        raise ValueError("catalog_ag and empirical_ag must contain finite values.")

    axes[0].scatter(
        x_plot,
        y_plot,
        s=MARKER_MS_FINE**2,
        alpha=ALPHA_FAINT,
        color=PRIMARY_COLOR,
        rasterized=True,
        label="RR Lyrae sample",
    )
    axes[0].plot(
        [line_min, line_max],
        [line_min, line_max],
        linestyle="--",
        color=NEUTRAL_COLOR,
        lw=LW_MEDIUM,
        label="1:1 line",
    )
    axes[0].set_title("Catalog vs. empirical $A_G$")
    axes[0].set_xlabel(r"Gaia DR3 $g_{\mathrm{absorption}}$ [mag]")
    axes[0].set_ylabel(r"Empirical $A_G$ [mag]")
    axes[0].legend(loc="best")
    _apply_grid(axes[0])

    finite_residuals = delta_values[np.isfinite(delta_values)]
    if finite_residuals.size == 0:
        raise ValueError("residuals must contain at least one finite value.")

    residual_min = float(np.min(finite_residuals))
    residual_max = float(np.max(finite_residuals))
    if residual_min == residual_max:
        residual_min -= 0.5
        residual_max += 0.5

    axes[1].hist(
        finite_residuals,
        bins=80,
        range=(residual_min, residual_max),
        color=SECONDARY_COLOR,
        alpha=ALPHA_LIGHT,
        label="Residual count",
    )
    residual_median = float(np.median(finite_residuals))
    axes[1].axvline(
        residual_median,
        linestyle="--",
        color=QUATERNARY_COLOR,
        lw=LW_MEDIUM,
        label=rf"Median = {residual_median:.3f} mag",
    )
    axes[1].set_title(r"$A_G^{\mathrm{calc}} - g_{\mathrm{absorption}}$")
    axes[1].set_xlabel(r"Residual [mag]")
    axes[1].set_ylabel("Count")
    axes[1].legend(loc="best")
    _apply_grid(axes[1])

    _tight_layout(fig)
    return fig, axes


def plot_posterior(
    samples: np.ndarray,
    labels: list[str] | tuple[str, ...],
    n_burn: int,
    *,
    param_idx: int,
    pdf_fn=None,
):
    _, ax = _single_panel(_columnwidth_figsize(5 / 2))

    post_burn = np.asarray(samples, dtype=float)[n_burn:, param_idx]
    xlabel = list(labels)[param_idx]
    ax.hist(post_burn, bins=50, density=True, alpha=ALPHA_LIGHT, color=PRIMARY_COLOR, label="MCMC samples")

    if pdf_fn is not None:
        margin = 0.5 * (post_burn.max() - post_burn.min())
        grid = np.linspace(post_burn.min() - margin, post_burn.max() + margin, 500)
        legend_param = xlabel
        if legend_param.startswith("$") and legend_param.endswith("$"):
            legend_param = legend_param[1:-1]
        ax.plot(
            grid,
            pdf_fn(grid),
            lw=LW_MEDIUM,
            color=SECONDARY_COLOR,
            label=rf"Analytic $p({legend_param}\mid x)$",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.legend()
    _apply_grid(ax)
    return ax


def plot_trace(
    samples: np.ndarray,
    log_probs: np.ndarray,
    labels: list[str] | tuple[str, ...],
    n_burn: int,
    *,
    color: str = PRIMARY_COLOR,
    log_prob_color: str | None = None,
):
    post_burn_samples = np.asarray(samples, dtype=float)[n_burn:]
    post_burn_log_probs = np.asarray(log_probs, dtype=float)[n_burn:]
    steps = np.arange(len(post_burn_samples))
    ndim = post_burn_samples.shape[1]
    lbls = list(labels)
    trace_color = color
    logp_color = trace_color if log_prob_color is None else log_prob_color

    trace_height_out_of_8 = 8 * (31 / 20) * (ndim + 1) / TEXTWIDTH_IN
    _, axes = _stacked_panels(
        ndim + 1,
        figsize=_textwidth_figsize(trace_height_out_of_8),
        height_ratios=[1] * (ndim + 1),
    )

    for i, lbl in enumerate(lbls):
        axes[i].plot(steps, post_burn_samples[:, i], lw=LW_FINE, alpha=ALPHA_EMPHASIS, color=trace_color)
        axes[i].set_ylabel(lbl)
        _apply_grid(axes[i])

    axes[-1].plot(steps, post_burn_log_probs, lw=LW_FINE, alpha=ALPHA_EMPHASIS, color=logp_color)
    axes[-1].set_ylabel(r"$\ln P$")
    axes[-1].set_xlabel("Step")
    _apply_grid(axes[-1])
    return axes


def plot_corner(source, labels=None, *, color: str = PRIMARY_COLOR):
    n_burn = getattr(source, "n_burn", 0)
    samples = source.samples[n_burn:]
    lbls = _labels(source, labels)

    fig = corner.corner(
        samples,
        labels=lbls,
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        color=color,
        fig=plt.figure(figsize=_textwidth_figsize(7)),
    )
    return fig


# ---------------------------------------------------------------------------
# Aitoff projection maps (Lab 1 notebooks 07, 08)
# ---------------------------------------------------------------------------

def _wrap_longitude(l_deg: np.ndarray) -> np.ndarray:
    """Wrap Galactic longitude to [-180, +180] degrees for Aitoff projection."""
    l_deg = np.asarray(l_deg, dtype=float)
    return ((l_deg + 180.0) % 360.0) - 180.0


def plot_aitoff_reddening_map(
    data: Table,
    mask: np.ndarray,
    *,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "magma",
    alpha: float = 0.22,
    size: float = 2.0,
) -> tuple:
    """Aitoff projection scatter plot of empirical E(BP-RP) reddening."""
    l_plot = _wrap_longitude(np.asarray(data["l"], dtype=float))
    b_deg = np.asarray(data["b"], dtype=float)
    e_bprp = np.asarray(data["E_bprp"], dtype=float)
    mask = np.asarray(mask, dtype=bool)

    fig = plt.figure(figsize=_textwidth_figsize(4), dpi=180)
    ax = fig.add_subplot(111, projection="aitoff")
    sc = ax.scatter(
        np.deg2rad(l_plot[mask]),
        np.deg2rad(b_deg[mask]),
        c=e_bprp[mask],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=size,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    ax.grid(True, alpha=0.35)
    ax.set_title(f"{title}\nN = {int(mask.sum()):,}")
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, shrink=0.85)
    cbar.set_label(r"$E(G_{\mathrm{BP}} - G_{\mathrm{RP}})$ [mag]")
    plt.tight_layout()
    return fig, ax


def plot_aitoff_value_map(
    ax,
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    cmap: str = "magma",
    size: float = 2.0,
    alpha: float = 0.24,
) -> tuple:
    """Lower-level per-panel Aitoff helper (takes a pre-created ax)."""
    sc = ax.scatter(
        np.deg2rad(_wrap_longitude(np.asarray(l_deg, dtype=float))),
        np.deg2rad(np.asarray(b_deg, dtype=float)),
        c=np.asarray(values, dtype=float),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=size,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    ax.grid(True, alpha=0.35)
    ax.set_title(title)
    return sc, colorbar_label


def plot_optical_vs_w2_comparison(optical_map: dict, wise_map: dict) -> np.ndarray:
    """Two-panel Gaia G vs WISE W2 PL comparison (RRab top, RRc bottom)."""
    band_colors = {"G": PRIMARY_COLOR, "W2": SECONDARY_COLOR}

    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, 1.20 * TEXTWIDTH_IN), sharex=True)

    for i, (ax, class_label) in enumerate(zip(axes, ("RRab", "RRc"))):
        optical = optical_map[class_label]
        wise = wise_map[class_label]

        ax.errorbar(
            optical.x_obs, optical.y_obs, yerr=optical.sigma_obs,
            fmt="o", ms=MARKER_MS_FINE, alpha=0.12, color=band_colors["G"],
            elinewidth=0.4, capsize=0, label=r"Gaia $G$ data",
        )
        ax.fill_between(
            optical.x_grid, optical.predictive_q16, optical.predictive_q84,
            color=band_colors["G"], alpha=0.16, label=r"Gaia $G$ 68\% predictive",
        )
        ax.plot(optical.x_grid, optical.median_mean, color=band_colors["G"], lw=LW_FIT, label=r"Gaia $G$ median")

        ax.errorbar(
            wise.x_obs, wise.y_obs, yerr=wise.sigma_obs,
            fmt="s", ms=MARKER_MS_FINE, alpha=0.16, color=band_colors["W2"],
            elinewidth=0.4, capsize=0, label=r"WISE $W2$ data",
        )
        ax.fill_between(
            wise.x_grid, wise.predictive_q16, wise.predictive_q84,
            color=band_colors["W2"], alpha=0.18, label=r"WISE $W2$ 68\% predictive",
        )
        ax.plot(wise.x_grid, wise.median_mean, color=band_colors["W2"], lw=LW_FIT, label=r"WISE $W2$ median")

        y_all = np.concatenate([
            np.asarray(optical.y_obs, dtype=float),
            np.asarray(optical.predictive_q16, dtype=float),
            np.asarray(optical.predictive_q84, dtype=float),
            np.asarray(wise.y_obs, dtype=float),
            np.asarray(wise.predictive_q16, dtype=float),
            np.asarray(wise.predictive_q84, dtype=float),
        ])
        finite = np.isfinite(y_all)
        if np.any(finite):
            y_min = float(np.min(y_all[finite]))
            y_max = float(np.max(y_all[finite]))
            pad = 0.15 * max(y_max - y_min, 0.5)
            ax.set_ylim(y_max + pad, y_min - pad)

        ax.set_title(class_label)
        ax.set_ylabel("Absolute magnitude [mag]")
        ax.grid(True, lw=LW_GRID, alpha=ALPHA_FAINT)

    axes[0].legend(loc="best", fontsize=LEGEND_SIZE)
    axes[1].set_xlabel(r"$\log_{10}(P/\mathrm{day})$")
    fig.tight_layout()
    return axes


def plot_w2_posterior_predictive(ctx, samples: np.ndarray):
    """Thin wrapper around plot_pl_posterior_predictive with W2 axis labels."""
    ax = plot_pl_posterior_predictive(ctx, samples)
    ax.set_ylabel(r"$M_{W2}$ [mag]")
    ax.set_title(rf"{ctx.class_label}: posterior predictive fit in WISE $W2$")
    return ax


def plot_quality_diagnostics(
    data: Table,
    components: dict,
    *,
    sample_size: int = 25000,
    seed: int = 7,
    max_sigma_e: float = 0.15,
) -> tuple:
    """BP/RP excess scatter + sigma_E histogram diagnostic panels."""
    rng = np.random.default_rng(seed)
    finite = components["finite"]
    sigma_e = np.asarray(data["sigma_E"], dtype=float)
    bp_rp = np.asarray(data["bp_rp"], dtype=float)
    excess = np.asarray(data["phot_bp_rp_excess_factor"], dtype=float)
    bad_excess = finite & ~components["bp_rp_excess"]
    good_excess = finite & components["bp_rp_excess"]

    good_idx = np.flatnonzero(good_excess)
    bad_idx = np.flatnonzero(bad_excess)
    if len(good_idx) > sample_size:
        good_idx = rng.choice(good_idx, size=sample_size, replace=False)
    if len(bad_idx) > sample_size:
        bad_idx = rng.choice(bad_idx, size=sample_size, replace=False)

    x_grid = np.linspace(
        np.nanpercentile(bp_rp[finite], 0.5),
        np.nanpercentile(bp_rp[finite], 99.5),
        300,
    )
    lower = 1.0 + 0.015 * x_grid ** 2
    upper = 1.3 + 0.06 * x_grid ** 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)

    axes[0].hist(sigma_e[finite], bins=80, color="0.45", alpha=0.8)
    axes[0].axvline(
        max_sigma_e, color="C3", linestyle="--", linewidth=2,
        label=rf"$\sigma_E = {max_sigma_e:.2f}$ mag",
    )
    axes[0].set_xlabel(r"$\sigma_E$ [mag]")
    axes[0].set_ylabel("Number of stars")
    axes[0].set_title("Propagated reddening uncertainty")
    axes[0].legend()

    axes[1].scatter(
        bp_rp[good_idx], excess[good_idx], s=4, alpha=0.08, color="0.25",
        rasterized=True, label="inside envelope",
    )
    axes[1].scatter(
        bp_rp[bad_idx], excess[bad_idx], s=6, alpha=0.35, color="C3",
        rasterized=True, label="rejected by excess cut",
    )
    axes[1].plot(x_grid, lower, color="C0", linestyle="--", linewidth=2)
    axes[1].plot(x_grid, upper, color="C0", linestyle="--", linewidth=2, label="Gaia excess envelope")
    axes[1].set_xlabel(r"$G_{\mathrm{BP}} - G_{\mathrm{RP}}$ [mag]")
    axes[1].set_ylabel("phot\\_bp\\_rp\\_excess\\_factor")
    axes[1].set_title("BP/RP excess diagnostic")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    return fig, axes


def plot_sfd_empirical_hexbin_comparison(
    data: Table,
    subset_specs: list,
    *,
    gridsize: int = 55,
    cmap: str = "cividis",
) -> tuple:
    """Hexbin density plot of empirical E(BP-RP) vs SFD E(B-V) for latitude subsets.

    subset_specs: list of (mask, label) tuples.
    """
    from ugdatalab.dust import binned_median_trend

    n_panels = len(subset_specs)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(TEXTWIDTH_IN, 0.40 * TEXTWIDTH_IN), dpi=180,
        sharex=True, sharey=True, constrained_layout=True,
    )
    if n_panels == 1:
        axes = np.asarray([axes])

    e_bprp = np.asarray(data["E_bprp"], dtype=float)
    sfd_ebv = np.asarray(data["sfd_ebv"], dtype=float)

    last_hb = None
    for ax, (mask, label) in zip(axes, subset_specs):
        x = sfd_ebv[mask]
        y = e_bprp[mask]
        last_hb = ax.hexbin(
            x, y,
            gridsize=gridsize,
            mincnt=1,
            bins="log",
            cmap=cmap,
            rasterized=True,
        )
        centers, medians = binned_median_trend(x, y)
        ax.plot(centers, medians, color="white", linewidth=2.5)
        ax.plot(centers, medians, color="C3", linewidth=1.25)
        ax.set_title(f"{label}\nN = {int(np.count_nonzero(mask)):,}")
        ax.set_xlabel(r"SFD $E(B-V)$ [mag]")
        ax.set_ylabel(r"RR Lyrae $E(G_{\mathrm{BP}}-G_{\mathrm{RP}})$ [mag]")

    if last_hb is not None:
        fig.colorbar(last_hb, ax=list(axes), label="log10(count)")

    return fig, np.asarray(axes)


def plot_reddening_distribution(
    e_bprp: np.ndarray,
    *,
    min_ebprp: float = 0.0,
    max_ebprp: float = 3.0,
) -> tuple:
    """Histogram of E(BP-RP) distribution with lower-bound and upper-ceiling markers."""
    e_bprp = np.asarray(e_bprp, dtype=float)
    finite = np.isfinite(e_bprp)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.hist(e_bprp[finite], bins=120, color="0.4", alpha=0.8)
    ax.axvline(
        min_ebprp, color=QUATERNARY_COLOR, linestyle="--", linewidth=LW_FIT,
        label=rf"Lower bound $E = {min_ebprp:.1f}$ mag",
    )
    ax.axvline(
        max_ebprp, color=SECONDARY_COLOR, linestyle="--", linewidth=LW_FIT,
        label=rf"Upper ceiling $E = {max_ebprp:.1f}$ mag",
    )
    ax.set_xlabel(r"$E(G_{\mathrm{BP}} - G_{\mathrm{RP}})$ [mag]")
    ax.set_ylabel("Number of stars")
    ax.set_title(
        r"$E(G_{\mathrm{BP}}-G_{\mathrm{RP}})$ distribution"
        "\n(quality bounds highlighted)"
    )
    ax.legend()
    ax.set_yscale("log")
    _apply_grid(ax)
    plt.tight_layout()
    return fig, ax


def plot_sfd_all_sky_hexbin(
    data: Table,
    *,
    gridsize: int = 70,
    cmap: str = "viridis",
) -> tuple:
    """Single-panel hexbin of empirical E(BP-RP) vs SFD E(B-V) for the full cleaned sample."""
    from ugdatalab.dust import binned_median_trend, rank_spearman

    e_bprp = np.asarray(data["E_bprp"], dtype=float)
    sfd_ebv = np.asarray(data["sfd_ebv"], dtype=float)
    finite = np.isfinite(e_bprp) & np.isfinite(sfd_ebv)
    x = sfd_ebv[finite]
    y = e_bprp[finite]

    centers, medians = binned_median_trend(x, y)
    rho = rank_spearman(x, y)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=180)
    hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=1, bins="log", cmap=cmap, rasterized=True)
    ax.plot(centers, medians, color="white", linewidth=2.5, label="Binned median trend")
    ax.plot(centers, medians, color=QUATERNARY_COLOR, linewidth=LW_STANDARD)
    ax.set_xlabel(r"SFD $E(B-V)$ [mag]")
    ax.set_ylabel(r"RR Lyrae $E(G_{\mathrm{BP}}-G_{\mathrm{RP}})$ [mag]")
    ax.set_title(
        rf"Matched-sightline density plot"
        "\n"
        rf"Spearman $\rho = {rho:.3f}$, $N = {int(finite.sum()):,}$"
    )
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="log10(count)")
    _apply_grid(ax)
    plt.tight_layout()
    return fig, ax


def plot_regime_decomposition(
    sfd_ebv: np.ndarray,
    empirical: np.ndarray,
    *,
    similar_mask: np.ndarray,
    large_mask: np.ndarray,
    slope: float,
    intercept: float,
    rho_sim: float,
    similar_scale_max: float = 2.0,
    large_sfd_min: float = 10.0,
    quality_ceiling: float = 3.0,
) -> tuple:
    """Two-panel regime decomposition: similar-scale hexbin+fit (left) and large-SFD scatter (right)."""
    from ugdatalab.dust import binned_median_trend

    sfd_ebv = np.asarray(sfd_ebv, dtype=float)
    empirical = np.asarray(empirical, dtype=float)
    similar_mask = np.asarray(similar_mask, dtype=bool)
    large_mask = np.asarray(large_mask, dtype=bool)

    x_sim = sfd_ebv[similar_mask]
    y_sim = empirical[similar_mask]
    x_fit = np.linspace(0, similar_scale_max, 200)
    y_fit = slope * x_fit + intercept
    centers_sim, medians_sim = binned_median_trend(x_sim, y_sim)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Left: similar-scale hexbin with linear fit
    ax = axes[0]
    hb = ax.hexbin(x_sim, y_sim, gridsize=60, mincnt=1, bins="log", cmap="viridis", rasterized=True)
    ax.plot(centers_sim, medians_sim, color="white", linewidth=2.5, label="Binned median")
    ax.plot(centers_sim, medians_sim, color=QUATERNARY_COLOR, linewidth=LW_STANDARD)
    ax.plot(
        x_fit, y_fit, "--", color=SECONDARY_COLOR, linewidth=LW_FIT,
        label=rf"Linear fit: slope $= {slope:.3f}$",
    )
    ax.set_xlabel(r"SFD $E(B-V)$ [mag]")
    ax.set_ylabel(r"RR Lyrae $E(G_{\mathrm{BP}}-G_{\mathrm{RP}})$ [mag]")
    ax.set_title(
        rf"Similar-scale regime ($\leq {similar_scale_max}$ mag)"
        "\n"
        rf"$N = {int(similar_mask.sum()):,}$, Spearman $\rho = {rho_sim:.3f}$"
    )
    ax.legend(loc="upper left", fontsize=LEGEND_SIZE)
    fig.colorbar(hb, ax=ax, label="log10(count)")
    _apply_grid(ax)

    # Right: large-SFD scatter with quality ceiling
    ax = axes[1]
    x_large = sfd_ebv[large_mask]
    y_large = empirical[large_mask]
    ax.scatter(
        x_large, y_large,
        s=MARKER_MS_SMALL**2, alpha=ALPHA_LIGHT, color=PRIMARY_COLOR,
        edgecolors="none", rasterized=True,
    )
    ax.axhline(
        quality_ceiling, color=QUATERNARY_COLOR, linestyle="--", linewidth=LW_FIT,
        label=rf"Quality ceiling $E = {quality_ceiling:.1f}$ mag",
    )
    ax.set_xlabel(r"SFD $E(B-V)$ [mag]")
    ax.set_ylabel(r"RR Lyrae $E(G_{\mathrm{BP}}-G_{\mathrm{RP}})$ [mag]")
    ax.set_title(
        rf"Large-SFD regime ($> {large_sfd_min}$ mag)"
        "\n"
        rf"$N = {int(large_mask.sum()):,}$"
    )
    ax.legend(loc="upper right", fontsize=LEGEND_SIZE)
    _apply_grid(ax)

    plt.tight_layout()
    return fig, np.asarray(axes)


_REPORT_SAVABLE_PLOTS = (
    "plot_mollweide",
    "plot_mollweide_diff",
    "plot_w2_posterior_line_draws",
    "plot_lomb_scargle_periodogram",
    "plot_period_abs_mag",
    "plot_period_abs_mag_ls",
    "plot_period_abs_mag_comparison",
    "plot_period_abs_mag_c12_comparison",
    "plot_period_abs_mag_clean_vs_astrometric_comparison",
    "plot_mollweide_period_abs_mag_overview",
    "plot_period_luminosity_diff",
    "plot_inlier_prob_period_luminosity_comparison",
    "plot_period_mean_g",
    "plot_vari_rrlyrae_period_comparison",
    "plot_hr",
    "plot_raw_phase_folded_lightcurve",
    "plot_fourier_harmonic_fits",
    "plot_rrlyrae_shape_comparison",
    "plot_fourier_cross_validation",
    "plot_fourier_cv_normalized_residual_histograms",
    "plot_fourier_normalized_residual_histograms",
    "plot_fourier_cv_phase_comparison",
    "plot_fourier_train_cv_phase_comparison",
    "plot_fourier_extrapolation",
    "plot_mean_g_catalog_comparison",
    "plot_inlier_prob_map",
    "plot_pl_posterior_predictive",
    "plot_pl_posterior_predictive_comparison",
    "plot_pl_sampler_comparison_corner",
    "plot_pl_posterior_line_draws",
    "plot_pl_posterior_line_draws_comparison",
    "plot_pc_posterior_predictive",
    "plot_pc_posterior_predictive_comparison",
    "plot_pc_posterior_draws_comparison",
    "plot_optical_pl_literature_comparison",
    "plot_empirical_vs_catalog_extinction_comparison",
    "plot_posterior",
    "plot_trace",
    "plot_corner",
    "plot_aitoff_reddening_map",
    "plot_optical_vs_w2_comparison",
    "plot_w2_posterior_predictive",
    "plot_w2_posterior_line_draws",
    "plot_quality_diagnostics",
    "plot_sfd_empirical_hexbin_comparison",
    "plot_sfd_all_sky_hexbin",
    "plot_regime_decomposition",
    "plot_reddening_distribution",
    "plot_calibration_sky_distribution",
    "plot_period_abs_mag_stage_comparison",
)

for _plot_name in _REPORT_SAVABLE_PLOTS:
    globals()[_plot_name] = _wrap_public_plot(globals()[_plot_name])
