from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from astropy.table import Table
from matplotlib.figure import Figure

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

SCATTER_S_FINE = 6
SCATTER_S_STANDARD = 14
SCATTER_S_EMPHASIS = 30
SCATTER_S_CALLOUT = 60

MARKER_MS_FINE = 3.0
MARKER_MS_STANDARD = 8.0
MARKER_MS_MEDIUM = 10.5
MARKER_MS_LARGE = 16.0

RRLYRAE_SCATTER_S = 4
RRLYRAE_MARKER_MS = 2.5
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


def _plot_class_masks(data: Table) -> list[tuple[str, np.ndarray]]:
    classifications = np.asarray(data["best_classification"], dtype=str)
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


def _save_lab02_figure(fig: Figure, filename: str, **kwargs) -> None:
    ensure_output_dirs()
    _save_figure(fig, FIGURES_DIR / filename, **kwargs)


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
    kwargs.setdefault("s", SCATTER_S_FINE)
    kwargs.setdefault("alpha", ALPHA_STANDARD)
    kwargs.setdefault("rasterized", True)
    return kwargs


def _rrlyrae_class_color(label: str, fallback_index: int = 0) -> str:
    class_colors = {
        "RRab": PRIMARY_COLOR,
        "RRd": SECONDARY_COLOR,
        "RRc": TERTIARY_COLOR,
    }
    if label in class_colors:
        return class_colors[label]
    return COMPONENT_COLORS[fallback_index % len(COMPONENT_COLORS)]


def figure(result, name: str):
    return result.figures[name]


def figure_names(result) -> list[str]:
    return sorted(result.figures)


def plot_mollweide(source, ax=None, title=None, **scatter_kwargs):
    data = _as_table(source)
    l_wrap = np.where(data["l"] > 180, data["l"] - 360, data["l"])
    l_rad = np.deg2rad(l_wrap)
    b_rad = np.deg2rad(data["b"])

    if ax is None:
        fig = plt.figure(figsize=_textwidth_figsize(22 / 5))
        ax = fig.add_subplot(111, projection="mollweide")

    scatter_kwargs = _default_scatter_kwargs(**scatter_kwargs)
    scatter_kwargs.setdefault("color", PRIMARY_COLOR)
    ax.scatter(l_rad, b_rad, **scatter_kwargs)

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
    _set_title(ax, title)
    ax.legend(loc="lower right")
    _apply_grid(ax)
    return ax


def plot_mollweide_diff(source, subset, ax=None, title=None, **scatter_kwargs):
    data = _as_table(source)
    subset_data = _as_table(subset)
    diff_mask = ~np.isin(data["source_id"], subset_data["source_id"])

    if ax is None:
        fig = plt.figure(figsize=_textwidth_figsize(22 / 5))
        ax = fig.add_subplot(111, projection="mollweide")

    scatter_kwargs = _default_scatter_kwargs(**scatter_kwargs)

    def to_rad(table):
        l_wrap = np.where(table["l"] > 180, table["l"] - 360, table["l"])
        return np.deg2rad(l_wrap), np.deg2rad(table["b"])

    diff_data = data[diff_mask]
    l_diff, b_diff = to_rad(diff_data)
    ax.scatter(
        l_diff,
        b_diff,
        color=SECONDARY_COLOR,
        label=f"Removed ($N$={len(diff_data)})",
        zorder=1,
        **scatter_kwargs,
    )

    l_sub, b_sub = to_rad(subset_data)
    ax.scatter(
        l_sub,
        b_sub,
        color=PRIMARY_COLOR,
        label=f"Kept ($N$={len(subset_data)})",
        zorder=2,
        **scatter_kwargs,
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
    _set_title(ax, title)
    ax.legend(loc="lower right")
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

    _, ax = _single_panel(_columnwidth_figsize(35 / 16))

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


def plot_period_abs_mag(
    source,
    ax=None,
    title=None,
    *,
    use_periodogram: bool = False,
    period_column: str | None = None,
    abs_mag_column: str | None = None,
    abs_mag_err_column: str | None = None,
    **scatter_kwargs,
):
    data = _as_table(source)
    if period_column is None:
        period_column = "period_ls" if use_periodogram else None
    if abs_mag_column is None:
        abs_mag_column = "M_G_ls" if use_periodogram else "M_G"
    if abs_mag_err_column is None:
        abs_mag_err_column = "sigma_M_ls" if use_periodogram else "sigma_M"

    if ax is None:
        _, ax = _single_panel(_columnwidth_figsize(5 / 2))

    scatter_kwargs = _default_scatter_kwargs(**scatter_kwargs)

    periods = (
        np.asarray(data[period_column], dtype=float)
        if period_column is not None
        else _plot_period_values(data)
    )
    m_g = np.asarray(data[abs_mag_column], dtype=float)
    sigma_m = np.asarray(data[abs_mag_err_column], dtype=float)
    ax.errorbar(
        periods,
        m_g,
        yerr=sigma_m,
        fmt="none",
        color=LIGHT_NEUTRAL_COLOR,
        alpha=ALPHA_LIGHT,
        zorder=1,
    )
    class_masks = _plot_class_masks(data)
    if class_masks:
        for i, (label, mask) in enumerate(class_masks):
            ax.scatter(
                periods[mask],
                m_g[mask],
                color=_rrlyrae_class_color(label, i),
                label=label,
                zorder=2,
                **scatter_kwargs,
            )
    else:
        ax.scatter(periods, m_g, color=PRIMARY_COLOR, label="RR Lyrae", zorder=2, **scatter_kwargs)

    ax.set_xscale("log")
    _set_descending_magnitude_yaxis(ax)
    if period_column == "period_ls":
        ax.set_xlabel(r"L-S period $P_{\rm LS}$ [days]")
    else:
        ax.set_xlabel(r"Catalog period $P$ [days]")
    ax.set_ylabel(r"$M_G$ [mag]")
    _set_title(ax, title)
    ax.legend()
    _apply_grid(ax)
    return ax


def plot_period_luminosity_diff(source, subset, ax=None, title=None, **scatter_kwargs):
    data = _as_table(source)
    subset_data = _as_table(subset)
    diff_mask = ~np.isin(data["source_id"], subset_data["source_id"])

    if ax is None:
        _, ax = _single_panel(_columnwidth_figsize(5 / 2))

    scatter_kwargs = _default_scatter_kwargs(**scatter_kwargs)

    diff_data = data[diff_mask]
    diff_period = _plot_period_values(diff_data)
    subset_period = _plot_period_values(subset_data)
    ax.scatter(
        diff_period,
        np.asarray(diff_data["M_G"], dtype=float),
        color=SECONDARY_COLOR,
        label=f"Removed ($N$={len(diff_data)})",
        zorder=1,
        **scatter_kwargs,
    )
    ax.scatter(
        subset_period,
        np.asarray(subset_data["M_G"], dtype=float),
        color=PRIMARY_COLOR,
        label=f"Kept ($N$={len(subset_data)})",
        zorder=2,
        **scatter_kwargs,
    )

    ax.set_xscale("log")
    _set_descending_magnitude_yaxis(ax)
    ax.set_xlabel(r"$P$ [days]")
    ax.set_ylabel(r"$M_G$ [mag]")
    _set_title(ax, title)
    ax.legend()
    _apply_grid(ax)
    return ax


def plot_period_mean_g(data: Table):
    _, ax = _single_panel(_columnwidth_figsize(10 / 4))

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
            s=RRLYRAE_SCATTER_S,
            alpha=RRLYRAE_POINT_ALPHA,
            rasterized=True,
        )

    ax.set_xscale("log")
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

    fig, axes = _grid_1x2(figsize=_textwidth_figsize(4))

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
            s=RRLYRAE_SCATTER_S,
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
            s=RRLYRAE_SCATTER_S,
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


def plot_hr(source, ax=None, title=None, **scatter_kwargs):
    data = _as_table(source)
    if ax is None:
        _, ax = _single_panel(_columnwidth_figsize(7 / 2))

    scatter_kwargs = _default_scatter_kwargs(**scatter_kwargs)
    class_masks = _plot_class_masks(data)
    if class_masks:
        for i, (label, mask) in enumerate(class_masks):
            ax.scatter(
                data["bp_rp"][mask],
                np.asarray(data["M_G"], dtype=float)[mask],
                color=_rrlyrae_class_color(label, i),
                label=label,
                **scatter_kwargs,
            )
    else:
        ax.scatter(
            data["bp_rp"],
            np.asarray(data["M_G"], dtype=float),
            color=PRIMARY_COLOR,
            label="RR Lyrae",
            **scatter_kwargs,
        )

    ax.invert_yaxis()
    ax.set_xlabel(r"$G_\mathrm{BP} - G_\mathrm{RP}$ [mag]")
    ax.set_ylabel(r"$M_G$ [mag]")
    _set_title(ax, title)
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

    fig, axes = _grid_1x2(figsize=_textwidth_figsize(84 / 25))

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
        s=RRLYRAE_SCATTER_S,
        alpha=RRLYRAE_POINT_ALPHA,
        color=PRIMARY_COLOR,
        zorder=2,
        rasterized=True,
        label="Raw light curve",
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Time [days]")
    axes[0].set_ylabel("G [mag]")
    axes[0].legend(loc="best")
    _apply_grid(axes[0])

    axes[1].scatter(
        phase,
        mag,
        s=RRLYRAE_SCATTER_S,
        alpha=RRLYRAE_POINT_ALPHA,
        color=SECONDARY_COLOR,
        zorder=2,
        rasterized=True,
        label="Phase-folded light curve",
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Phase")
    axes[1].set_ylabel("G [mag]")
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
        figsize=(A4_USABLE_WIDTH_IN, A4_USABLE_HEIGHT_IN),
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
            ms=RRLYRAE_MARKER_MS,
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
            ms=RRLYRAE_MARKER_MS,
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
                ms=RRLYRAE_MARKER_MS,
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
                ms=RRLYRAE_MARKER_MS,
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

    _, ax = _single_panel(_columnwidth_figsize(9 / 4))
    ax.plot(
        Ks[valid],
        chi2r_train[valid],
        marker="o",
        ls="none",
        ms=RRLYRAE_MARKER_MS,
        alpha=ALPHA_DENSE,
        label=r"Training $\chi_r^2$",
    )
    ax.plot(
        Ks[valid],
        chi2r_cv[valid],
        marker="o",
        ls="none",
        ms=RRLYRAE_MARKER_MS,
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
        source_id=int(best_fit.source_id),
        classification=getattr(best_fit, "classification", None),
        period=float(best_fit.period),
    )


def plot_fourier_normalized_residual_histograms(
    train_norm_low: np.ndarray,
    cv_norm_low: np.ndarray,
    train_norm_best: np.ndarray,
    cv_norm_best: np.ndarray,
    low_K: int,
    best_K: int,
    *,
    source_id: int | None = None,
    classification: str | None = None,
    period: float | None = None,
):
    fig, axes = _grid_1x2(figsize=_textwidth_figsize(12 / 5))
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
        best_panel_title=rf"Best $K = {int(best_fit.K)}$",
        high_panel_title=rf"High $K = {int(high_fit.K)}$",
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
    show_panel_titles: bool = True,
    best_panel_title: str | None = None,
    high_panel_title: str | None = None,
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

    fig, axes = _grid_1x2(figsize=_textwidth_figsize(68 / 25))
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
            s=RRLYRAE_SCATTER_S,
            alpha=ALPHA_EXTRA_LIGHT,
            color=PRIMARY_COLOR,
            rasterized=True,
            label="Training",
        )
        ax.scatter(
            cv_phase,
            cv_mags,
            s=RRLYRAE_SCATTER_S,
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
        if show_panel_titles:
            if K == best_K:
                detail = best_panel_title if best_panel_title is not None else rf"Best $K = {K}$"
            else:
                detail = high_panel_title if high_panel_title is not None else rf"High $K = {K}$"
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

    _, ax = _single_panel(_textwidth_figsize(3))
    ax.errorbar(
        epochs[observed_mask],
        mags[observed_mask],
        yerr=mag_errs[observed_mask],
        fmt="o",
        ms=RRLYRAE_MARKER_MS,
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
        ms=MARKER_MS_STANDARD,
        color="k",
        lw=LW_NONE,
        label=rf"10-day prediction: $G={mag_pred:.3f}\pm{mag_pred_err:.3f}$",
    )
    ax.invert_yaxis()
    ax.set_xlim(float(np.min(epoch_grid)), float(np.max(epoch_grid)))
    ax.set_xlabel(time_label)
    ax.set_ylabel(r"$G$ [mag]")
    ax.legend(loc="upper right")
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
    all_values = [
        simple_mean_g[simple_valid],
        fourier_mean_g[fourier_valid],
        int_average_g[simple_valid],
        int_average_g[fourier_valid],
    ]
    if np.any(simple_valid):
        simple_err_safe = np.where(np.isfinite(simple_mean_g_err), simple_mean_g_err, 0.0)
        int_err_safe = np.where(np.isfinite(int_average_g_err), int_average_g_err, 0.0)
        all_values.extend(
            [
                (simple_mean_g - simple_err_safe)[simple_valid],
                (simple_mean_g + simple_err_safe)[simple_valid],
                (int_average_g - int_err_safe)[simple_valid],
                (int_average_g + int_err_safe)[simple_valid],
            ]
        )
    if np.any(fourier_valid):
        fourier_err_safe = np.where(np.isfinite(fourier_mean_g_err), fourier_mean_g_err, 0.0)
        int_err_safe = np.where(np.isfinite(int_average_g_err), int_average_g_err, 0.0)
        all_values.extend(
            [
                (fourier_mean_g - fourier_err_safe)[fourier_valid],
                (fourier_mean_g + fourier_err_safe)[fourier_valid],
                (int_average_g - int_err_safe)[fourier_valid],
                (int_average_g + int_err_safe)[fourier_valid],
            ]
        )
    all_values = np.concatenate(all_values)
    pad = 0.05 * (np.max(all_values) - np.min(all_values))
    lims = np.array([np.min(all_values) - pad, np.max(all_values) + pad])
    resid_max = 1.1 * np.nanmax(
        np.abs(
            np.concatenate(
                [
                    resid_simple[simple_valid] - simple_resid_err[simple_valid],
                    resid_simple[simple_valid] + simple_resid_err[simple_valid],
                    resid_fourier[fourier_valid] - fourier_resid_err[fourier_valid],
                    resid_fourier[fourier_valid] + fourier_resid_err[fourier_valid],
                ]
            )
        )
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=_textwidth_figsize(111 / 25),
        sharex="col",
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.0},
    )
    (ax_simple, ax_fourier), (ax_simple_resid, ax_fourier_resid) = axes

    simple_xerr = np.where(np.isfinite(int_average_g_err), int_average_g_err, 0.0)
    simple_yerr = np.where(np.isfinite(simple_mean_g_err), simple_mean_g_err, 0.0)
    if np.any(simple_valid):
        ax_simple.errorbar(
            int_average_g[simple_valid],
            simple_mean_g[simple_valid],
            xerr=simple_xerr[simple_valid],
            yerr=simple_yerr[simple_valid],
            fmt="none",
            ecolor=PRIMARY_COLOR,
            elinewidth=LW_FINE,
            alpha=ALPHA_DIM,
            zorder=1,
        )

    ax_simple.scatter(
        int_average_g[simple_valid],
        simple_mean_g[simple_valid],
        s=RRLYRAE_SCATTER_S,
        alpha=RRLYRAE_POINT_ALPHA,
        color=PRIMARY_COLOR,
        rasterized=True,
    )
    ax_simple.plot(lims, lims, color=NEUTRAL_COLOR, ls="--", lw=LW_STANDARD)
    ax_simple.set_xlim(lims[1], lims[0])
    ax_simple.set_ylim(lims)
    ax_simple.invert_yaxis()
    ax_simple.set_ylabel(r"$\langle G \rangle_{\rm epoch}$ [mag]")
    _apply_grid(ax_simple)

    if np.any(simple_valid):
        ax_simple_resid.errorbar(
            int_average_g[simple_valid],
            resid_simple[simple_valid],
            xerr=simple_xerr[simple_valid],
            yerr=simple_resid_err[simple_valid],
            fmt="none",
            ecolor=PRIMARY_COLOR,
            elinewidth=LW_FINE,
            alpha=ALPHA_DIM,
            zorder=1,
        )
    ax_simple_resid.scatter(
        int_average_g[simple_valid],
        resid_simple[simple_valid],
        s=RRLYRAE_SCATTER_S,
        alpha=RRLYRAE_POINT_ALPHA,
        color=PRIMARY_COLOR,
        rasterized=True,
    )
    ax_simple_resid.axhline(0.0, color=NEUTRAL_COLOR, ls="--", lw=LW_STANDARD)
    ax_simple_resid.set_xlim(lims[1], lims[0])
    ax_simple_resid.set_xlabel(r"Gaia $\mathtt{int\_average\_g}$ [mag]")
    ax_simple_resid.set_ylabel("Res.")
    ax_simple_resid.set_ylim(-resid_max, resid_max)
    _apply_grid(ax_simple_resid)

    fourier_xerr = np.where(np.isfinite(int_average_g_err), int_average_g_err, 0.0)
    fourier_yerr = np.where(np.isfinite(fourier_mean_g_err), fourier_mean_g_err, 0.0)
    if np.any(fourier_valid):
        ax_fourier.errorbar(
            int_average_g[fourier_valid],
            fourier_mean_g[fourier_valid],
            xerr=fourier_xerr[fourier_valid],
            yerr=fourier_yerr[fourier_valid],
            fmt="none",
            ecolor=SECONDARY_COLOR,
            elinewidth=LW_FINE,
            alpha=ALPHA_DIM,
            zorder=1,
        )
    ax_fourier.scatter(
        int_average_g[fourier_valid],
        fourier_mean_g[fourier_valid],
        s=RRLYRAE_SCATTER_S,
        alpha=RRLYRAE_POINT_ALPHA,
        color=SECONDARY_COLOR,
        rasterized=True,
    )
    ax_fourier.plot(lims, lims, color=NEUTRAL_COLOR, ls="--", lw=LW_STANDARD)
    ax_fourier.set_xlim(lims[1], lims[0])
    ax_fourier.set_ylim(lims)
    ax_fourier.invert_yaxis()
    ax_fourier.set_ylabel(r"$\langle G \rangle_{\rm Fourier}$ [mag]")
    _apply_grid(ax_fourier)

    if np.any(fourier_valid):
        ax_fourier_resid.errorbar(
            int_average_g[fourier_valid],
            resid_fourier[fourier_valid],
            xerr=fourier_xerr[fourier_valid],
            yerr=fourier_resid_err[fourier_valid],
            fmt="none",
            ecolor=SECONDARY_COLOR,
            elinewidth=LW_FINE,
            alpha=ALPHA_DIM,
            zorder=1,
        )
    ax_fourier_resid.scatter(
        int_average_g[fourier_valid],
        resid_fourier[fourier_valid],
        s=RRLYRAE_SCATTER_S,
        alpha=RRLYRAE_POINT_ALPHA,
        color=SECONDARY_COLOR,
        rasterized=True,
    )
    ax_fourier_resid.axhline(0.0, color=NEUTRAL_COLOR, ls="--", lw=LW_STANDARD)
    ax_fourier_resid.set_xlim(lims[1], lims[0])
    ax_fourier_resid.set_xlabel(r"Gaia $\mathtt{int\_average\_g}$ [mag]")
    ax_fourier_resid.set_ylabel("Res.")
    ax_fourier_resid.set_ylim(-resid_max, resid_max)
    _apply_grid(ax_fourier_resid)

    return axes


def plot_inlier_prob_map(source, ax=None, title=None):
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
        s=SCATTER_S_FINE,
        alpha=ALPHA_DENSE,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="Posterior inlier probability")
    ax.set_xscale("log")
    _set_descending_magnitude_yaxis(ax)
    ax.set_xlabel(r"$P$ [days]")
    ax.set_ylabel(r"$M_G$ [mag]")
    _set_title(ax, title)
    _apply_grid(ax)
    return ax


def plot_posterior(source, ax=None, title=None, pdf_fn=None, param_idx=0, label=None):
    if ax is None:
        _, ax = _single_panel(_columnwidth_figsize(5 / 2))

    n_burn = getattr(source, "n_burn", 0)
    samples = source.samples[n_burn:, param_idx]
    xlabel = label if label is not None else _labels(source)[param_idx]
    ax.hist(samples, bins=50, density=True, alpha=ALPHA_LIGHT, color=PRIMARY_COLOR, label="MCMC samples")

    if pdf_fn is not None:
        margin = 0.5 * (samples.max() - samples.min())
        grid = np.linspace(samples.min() - margin, samples.max() + margin, 500)
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
    _set_title(ax, title)
    ax.legend()
    _apply_grid(ax)
    return ax


def plot_trace(source, axes=None, title=None, labels=None):
    n_burn = getattr(source, "n_burn", 0)
    steps = np.arange(len(source.samples[n_burn:]))
    ndim = source.samples.shape[1]
    lbls = _labels(source, labels)

    if axes is None:
        trace_height_out_of_8 = 8 * (31 / 20) * (ndim + 1) / TEXTWIDTH_IN
        _, axes = _stacked_panels(
            ndim + 1,
            figsize=_textwidth_figsize(trace_height_out_of_8),
            height_ratios=[1] * (ndim + 1),
        )

    for i, lbl in enumerate(lbls):
        axes[i].plot(steps, source.samples[n_burn:, i], lw=LW_FINE, alpha=ALPHA_EMPHASIS, color=PRIMARY_COLOR)
        axes[i].set_ylabel(lbl)
        _apply_grid(axes[i])

    if title is not None:
        axes[0].set_title(title, fontsize=EMPHASIS_SIZE)

    axes[-1].plot(steps, source.log_probs[n_burn:], lw=LW_FINE, alpha=ALPHA_EMPHASIS, color=SECONDARY_COLOR)
    axes[-1].set_ylabel(r"$\ln P$")
    axes[-1].set_xlabel("Step")
    _apply_grid(axes[-1])
    return axes


def plot_corner(source, fig=None, title=None, labels=None):
    n_burn = getattr(source, "n_burn", 0)
    samples = source.samples[n_burn:]
    lbls = _labels(source, labels)

    fig = corner.corner(
        samples,
        labels=lbls,
        fig=fig,
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        color=PRIMARY_COLOR,
    )
    if title is not None:
        fig.suptitle(_escape_latex_text(title), fontsize=EMPHASIS_SIZE)
    return fig
