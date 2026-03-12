from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.figure import Figure

from .models.gaia import rrlyrae_representative_period
from .paths import FIGURES_DIR, ensure_output_dirs

TEXTWIDTH_IN = 7.59
COLUMNWIDTH_IN = 3.73
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8
ANNOTATION_SIZE = 9
EMPHASIS_SIZE = 10

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

ALPHA_SHADE = 0.10
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


def _set_title(ax: Any, title: str | None) -> None:
    if title is not None:
        ax.set_title(_escape_latex_text(title), fontsize=EMPHASIS_SIZE)


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


def plot_lomb_scargle_periodogram(data: Table):
    if len(data) == 0:
        raise ValueError("No light-curve rows available for plotting.")

    from ugdatalab.lightcurves import lomb_scargle_periodogram

    periods, power, best_period = lomb_scargle_periodogram(data)
    periods = np.asarray(periods, dtype=float)
    power = np.asarray(power, dtype=float)
    best_period = float(best_period)

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
    ax.set_xlabel(r"$P$ [days]")
    ax.set_ylabel("Lomb-Scargle power")
    source_id = int(data["source_id"][0])
    ax.set_title(
        _escape_latex_text(f"Lomb-Scargle periodogram for\nGaia DR3 {source_id} ({classification})"),
        fontsize=EMPHASIS_SIZE,
    )
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
    ax.invert_yaxis()
    ax.set_xlabel(r"$P$ [days]")
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
    ax.invert_yaxis()
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
    ax.set_xlabel(r"$P$ [days]")
    ax.set_ylabel(r"$\langle G \rangle$ [mag]")
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
    ax_fundamental.set_title(r"L-S periods vs. \texttt{vari\_rrlyrae} fundamental periods")
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
    ax_first_overtone.set_title(r"L-S periods vs. \texttt{vari\_rrlyrae} first-overtone periods (RRd only)")
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

    time_unit = getattr(data["g_transit_time"], "unit", None)
    mag_unit = getattr(data["g_transit_mag"], "unit", None)
    time_label = f"Time [{time_unit}]" if time_unit is not None else "Time"
    mag_label = f"G [{mag_unit}]" if mag_unit is not None else "G"

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
    axes[0].set_xlabel(time_label)
    axes[0].set_ylabel(mag_label)
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
    axes[1].set_ylabel(mag_label)
    axes[1].legend(loc="best")
    _apply_grid(axes[1])

    axes[0].autoscale_view()
    axes[1].autoscale_view()

    source_id = int(data["source_id"][0])
    fig.suptitle(
        _escape_latex_text(f"Lightcurves for Gaia DR3 {source_id} ({classification})"),
        fontsize=EMPHASIS_SIZE,
    )
    return axes


def plot_fourier_harmonic_fits(
    data: Table,
    target_id: int,
    K_values: list[int] | tuple[int, ...],
):
    data = _as_table(data)
    data = data[data["source_id"] == int(target_id)]
    if len(data) == 0:
        raise ValueError("No light-curve rows available for plotting.")

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
        figsize=(TEXTWIDTH_IN, 2.5 * len(K_values)),
        hspace=0.36,
    )
    axes = np.empty((len(K_values), 2), dtype=object)
    first_curve_ax = None
    for i in range(len(K_values)):
        inner_grid = outer_grid[i, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
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

        ax_curve.errorbar(
            phase,
            mag,
            yerr=mag_err,
            fmt="o",
            ms=RRLYRAE_MARKER_MS,
            elinewidth=LW_FINE,
            color=PRIMARY_COLOR,
            alpha=RRLYRAE_POINT_ALPHA,
            zorder=2,
        )
        ax_curve.plot(phase_grid, model_mag, color=SECONDARY_COLOR, lw=LW_MEDIUM, zorder=3)
        ax_curve.invert_yaxis()
        ax_curve.set_ylabel(r"$G$ [mag]")
        ax_curve.set_title(
            rf"$K={K}$, $\chi_\nu^2 = {fit.chi2_r:.2f}$",
            loc="left",
            y=1.05,
            pad=2.0,
        )
        ax_curve.tick_params(axis="x", labelbottom=False)
        _apply_grid(ax_curve)

        ax_resid.errorbar(
            phase,
            residuals,
            yerr=mag_err,
            fmt="o",
            ms=RRLYRAE_MARKER_MS,
            elinewidth=LW_FINE,
            color=PRIMARY_COLOR,
            alpha=RRLYRAE_POINT_ALPHA,
            zorder=2,
        )
        _zero_line(ax_resid)
        ax_resid.set_ylabel("Res.")
        ax_resid.set_xlabel("Phase")
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


def plot_rrlyrae_shape_comparison(panels: list[dict[str, Any]]):
    if len(panels) == 0:
        raise ValueError("panels must contain at least one prepared RR Lyrae panel.")

    rrc_panels = [panel for panel in panels if panel["rr_class"] == "RRc"]
    rrab_panels = [panel for panel in panels if panel["rr_class"] == "RRab"]
    if len(rrc_panels) == 0 or len(rrab_panels) == 0:
        raise ValueError("panels must include both RRc and RRab entries.")

    nrows = max(len(rrc_panels), len(rrab_panels))
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(TEXTWIDTH_IN, 2.35 * nrows),
        sharex=True,
        squeeze=False,
    )
    # fig.subplots_adjust(hspace=0.28, wspace=0.18)

    all_values = []
    for panel in panels:
        for key in ("mag_centered", "model_centered"):
            values = np.asarray(panel[key], dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite):
                all_values.append(finite)
    if len(all_values) == 0:
        raise ValueError("No finite centered magnitudes are available for plotting.")

    values = np.concatenate(all_values)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    pad = max(0.05 * (value_max - value_min), 0.05)
    y_limits = (value_min - pad, value_max + pad)

    panel_columns = (
        (rrc_panels, SECONDARY_COLOR, "RRc"),
        (rrab_panels, PRIMARY_COLOR, "RRab"),
    )
    ylabel = r"$G - \langle G \rangle_{\rm Fourier}$"

    for col, (panel_list, point_color, rr_class) in enumerate(panel_columns):
        for row in range(nrows):
            ax = axes[row, col]
            if row >= len(panel_list):
                ax.axis("off")
                continue

            panel = panel_list[row]
            phase_data = np.asarray(panel["phase_data"], dtype=float)
            mag_centered = np.asarray(panel["mag_centered"], dtype=float)
            phase_grid = np.asarray(panel["phase_grid"], dtype=float)
            model_centered = np.asarray(panel["model_centered"], dtype=float)
            source_id = int(panel["source_id"])
            best_K = int(panel["best_K"])
            period = float(panel["period"])

            ax.scatter(
                phase_data,
                mag_centered,
                s=RRLYRAE_SCATTER_S,
                alpha=RRLYRAE_POINT_ALPHA,
                color=point_color,
                rasterized=True,
                label=rr_class if row == 0 else "_nolegend_",
            )
            ax.plot(
                phase_grid,
                model_centered,
                color=NEUTRAL_COLOR,
                lw=LW_STANDARD,
                label="Fourier model" if row == 0 else "_nolegend_",
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(y_limits)
            ax.invert_yaxis()
            ax.set_ylabel(ylabel)
            ax.set_title(
                rf"Gaia DR3 {source_id}" + "\n" + rf"$K={best_K}$, $P={period:.4f}\,\mathrm{{d}}$"
            )
            if row == nrows - 1:
                ax.set_xlabel("Phase")
            _apply_grid(ax)

        axes[0, col].legend(loc="best")

    return axes


def plot_fourier_cross_validation(
    Ks: np.ndarray,
    chi2r_train: np.ndarray,
    chi2r_cv: np.ndarray,
    best_K: int,
    target_id: int,
    n_train: int,
    n_cv: int,
):
    Ks = np.asarray(Ks, dtype=int)
    chi2r_train = np.asarray(chi2r_train, dtype=float)
    chi2r_cv = np.asarray(chi2r_cv, dtype=float)
    valid = np.isfinite(chi2r_train) & np.isfinite(chi2r_cv)
    if not np.any(valid):
        raise ValueError("No finite cross-validation values are available for plotting.")

    best_mask = Ks == int(best_K)
    if not np.any(best_mask):
        raise ValueError("best_K must be present in Ks.")
    cv_best = float(chi2r_cv[best_mask][0])

    _, ax = _single_panel(_textwidth_figsize(16 / 3))
    ax.plot(Ks[valid], chi2r_train[valid], marker="o", ms=MARKER_MS_FINE, lw=LW_STANDARD, alpha=ALPHA_GUIDE, label=r"Training $\chi_r^2$")
    ax.plot(Ks[valid], chi2r_cv[valid], marker="o", ms=MARKER_MS_FINE, lw=LW_STANDARD, alpha=ALPHA_GUIDE, label=r"Cross-validation $\chi_r^2$")
    ax.axhline(1.0, color=NEUTRAL_COLOR, ls="--", lw=LW_GUIDE, alpha=ALPHA_STANDARD, label=r"$\chi_r^2 = 1$")
    ax.axvline(best_K, color=TERTIARY_COLOR, ls=":", lw=LW_LIGHT, alpha=ALPHA_GUIDE, label=rf"Best $K={best_K}$")
    ax.axhline(cv_best, color=TERTIARY_COLOR, ls=":", lw=LW_LIGHT, alpha=ALPHA_GUIDE, label=rf"Best CV $\chi_r^2={cv_best:.3f}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$K$ (number of Fourier harmonics)")
    ax.set_ylabel(r"$\chi_r^2$")
    ax.set_title(
        _escape_latex_text(
            f"Cross-validation for Gaia DR3 {target_id}\n20% CV, train={n_train} points, CV={n_cv} points"
        ),
        fontsize=EMPHASIS_SIZE,
    )
    ax.legend(loc="lower center")
    _apply_grid(ax)
    return ax


def plot_fourier_normalized_residual_histograms(
    train_norm_low: np.ndarray,
    cv_norm_low: np.ndarray,
    train_norm_best: np.ndarray,
    cv_norm_best: np.ndarray,
    low_K: int,
    best_K: int,
):
    fig, axes = _grid_1x2(figsize=_textwidth_figsize(12 / 5))
    bins = np.linspace(-5.0, 5.0, 31)
    x_gauss = np.linspace(-5.0, 5.0, 400)
    gaussian = np.exp(-0.5 * x_gauss**2) / np.sqrt(2.0 * np.pi)

    panels = (
        (axes[0], np.asarray(train_norm_low, dtype=float), np.asarray(cv_norm_low, dtype=float), rf"Low $K={low_K}$"),
        (axes[1], np.asarray(train_norm_best, dtype=float), np.asarray(cv_norm_best, dtype=float), rf"Best $K={best_K}$"),
    )

    for ax, train_norm, cv_norm, label in panels:
        train_norm = train_norm[np.isfinite(train_norm)]
        cv_norm = cv_norm[np.isfinite(cv_norm)]
        ax.hist(train_norm, bins=bins, histtype="step", density=True, color=PRIMARY_COLOR, lw=LW_MEDIUM, label="Training")
        ax.hist(cv_norm, bins=bins, histtype="step", density=True, color=SECONDARY_COLOR, lw=LW_MEDIUM, label="CV")
        ax.plot(x_gauss, gaussian, color=NEUTRAL_COLOR, ls="--", lw=LW_STANDARD, label=r"$\mathcal{N}(0,1)$")
        ax.axvline(0.0, color=NEUTRAL_COLOR, ls=":", lw=LW_STANDARD)
        ax.set_title(label)
        ax.set_xlabel(r"$(G - G_{\rm model})/\sigma$")
        _apply_grid(ax)

    axes[0].set_ylabel("Density")
    axes[1].set_ylabel("Density")
    axes[0].legend(loc="best")
    return axes


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
):
    train_phase = np.asarray(train_phase, dtype=float)
    train_mags = np.asarray(train_mags, dtype=float)
    cv_phase = np.asarray(cv_phase, dtype=float)
    cv_mags = np.asarray(cv_mags, dtype=float)
    phase_grid = np.asarray(phase_grid, dtype=float)
    model_mag_best = np.asarray(model_mag_best, dtype=float)
    model_mag_high = np.asarray(model_mag_high, dtype=float)

    fig, axes = _grid_1x2(figsize=_textwidth_figsize(68 / 25))
    all_mags = np.concatenate([train_mags[np.isfinite(train_mags)], cv_mags[np.isfinite(cv_mags)]])
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
            alpha=ALPHA_FAINT,
            color=PRIMARY_COLOR,
            rasterized=True,
            label="Training",
        )
        ax.scatter(
            cv_phase,
            cv_mags,
            s=RRLYRAE_SCATTER_S,
            alpha=ALPHA_LIGHT,
            color=SECONDARY_COLOR,
            rasterized=True,
            label="CV",
        )
        ax.plot(phase_grid, model_mag, color=NEUTRAL_COLOR, lw=LW_STANDARD, label="Model")
        ax.set_ylim(y_limits)
        ax.invert_yaxis()
        ax.set_xlabel("Phase")
        ax.set_title(rf"Phase-folded fit with $K={K}$")
        _apply_grid(ax)

    axes[0].set_ylabel(r"$G$ [mag]")
    axes[1].set_ylabel(r"$G$ [mag]")
    axes[0].legend(loc="best")
    return axes


def plot_fourier_extrapolation(
    fit,
    epoch_grid: np.ndarray,
    mag_grid: np.ndarray,
    epoch_pred: float,
    mag_pred: float,
    time_label: str,
):
    epochs = np.asarray(fit.epochs, dtype=float)
    mags = np.asarray(fit.mags, dtype=float)
    mag_errs = np.asarray(fit.mag_errs, dtype=float)
    epoch_grid = np.asarray(epoch_grid, dtype=float)
    mag_grid = np.asarray(mag_grid, dtype=float)
    epoch_last = float(np.max(epochs))

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
    ax.axvline(epoch_last, color=NEUTRAL_COLOR, ls=":", lw=LW_STANDARD, alpha=ALPHA_GUIDE)
    ax.plot(
        epoch_pred,
        mag_pred,
        marker="*",
        ms=MARKER_MS_STANDARD,
        color="k",
        lw=LW_NONE,
        label=rf"10-day prediction: $G={mag_pred:.3f}$",
    )
    ax.invert_yaxis()
    ax.set_xlim(float(np.min(epoch_grid)), float(np.max(epoch_grid)))
    ax.set_xlabel(time_label)
    ax.set_ylabel(r"$G$ [mag]")
    title = rf"Fourier extrapolation for Gaia DR3 {fit.source_id} ($K={fit.K}$, $P={fit.period:.4f}\,\mathrm{{d}}$)"
    ax.set_title(title, fontsize=EMPHASIS_SIZE)
    ax.legend(loc="upper left")
    _apply_grid(ax)
    return ax


def plot_mean_g_catalog_comparison(
    gaia_int_average_g: np.ndarray,
    simple_mean_g: np.ndarray,
    fourier_mean_g: np.ndarray,
    resid_simple: np.ndarray,
    resid_fourier: np.ndarray,
    best_K: int,
):
    gaia_int_average_g = np.asarray(gaia_int_average_g, dtype=float)
    simple_mean_g = np.asarray(simple_mean_g, dtype=float)
    fourier_mean_g = np.asarray(fourier_mean_g, dtype=float)
    resid_simple = np.asarray(resid_simple, dtype=float)
    resid_fourier = np.asarray(resid_fourier, dtype=float)

    simple_valid = np.isfinite(simple_mean_g) & np.isfinite(gaia_int_average_g) & np.isfinite(resid_simple)
    fourier_valid = np.isfinite(fourier_mean_g) & np.isfinite(gaia_int_average_g) & np.isfinite(resid_fourier)
    all_values = np.concatenate(
        [
            simple_mean_g[simple_valid],
            fourier_mean_g[fourier_valid],
            gaia_int_average_g[simple_valid],
            gaia_int_average_g[fourier_valid],
        ]
    )
    pad = 0.05 * (np.max(all_values) - np.min(all_values))
    lims = np.array([np.min(all_values) - pad, np.max(all_values) + pad])
    resid_max = 1.1 * np.nanmax(
        np.abs(np.concatenate([resid_simple[simple_valid], resid_fourier[fourier_valid]]))
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=_textwidth_figsize(111 / 25),
        sharex="col",
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.0},
    )
    (ax_simple, ax_fourier), (ax_simple_resid, ax_fourier_resid) = axes

    ax_simple.scatter(
        gaia_int_average_g[simple_valid],
        simple_mean_g[simple_valid],
        s=RRLYRAE_SCATTER_S,
        alpha=RRLYRAE_POINT_ALPHA,
        color=PRIMARY_COLOR,
        rasterized=True,
    )
    ax_simple.plot(lims, lims, color=NEUTRAL_COLOR, ls="--", lw=LW_STANDARD)
    ax_simple.set_xlim(lims)
    ax_simple.set_ylim(lims)
    ax_simple.set_ylabel(r"$\langle G \rangle_{\rm epoch}$ [mag]")
    ax_simple.set_title(r"$\langle G \rangle_{\rm epoch}$ vs Gaia $\mathtt{int\_average\_g}$")
    _apply_grid(ax_simple)

    ax_simple_resid.scatter(
        gaia_int_average_g[simple_valid],
        resid_simple[simple_valid],
        s=RRLYRAE_SCATTER_S,
        alpha=RRLYRAE_POINT_ALPHA,
        color=PRIMARY_COLOR,
        rasterized=True,
    )
    ax_simple_resid.axhline(0.0, color=NEUTRAL_COLOR, ls="--", lw=LW_STANDARD)
    ax_simple_resid.set_xlabel(r"Gaia $\mathtt{int\_average\_g}$ [mag]")
    ax_simple_resid.set_ylabel("Res.")
    ax_simple_resid.set_ylim(-resid_max, resid_max)
    _apply_grid(ax_simple_resid)

    ax_fourier.scatter(
        gaia_int_average_g[fourier_valid],
        fourier_mean_g[fourier_valid],
        s=RRLYRAE_SCATTER_S,
        alpha=RRLYRAE_POINT_ALPHA,
        color=SECONDARY_COLOR,
        rasterized=True,
    )
    ax_fourier.plot(lims, lims, color=NEUTRAL_COLOR, ls="--", lw=LW_STANDARD)
    ax_fourier.set_xlim(lims)
    ax_fourier.set_ylim(lims)
    ax_fourier.set_ylabel(r"$\langle G \rangle_{\rm Fourier}$ [mag]")
    ax_fourier.set_title(rf"$\langle G \rangle_{{\rm Fourier}}$ ($K={best_K}$) vs Gaia $\mathtt{{int\_average\_g}}$")
    _apply_grid(ax_fourier)

    ax_fourier_resid.scatter(
        gaia_int_average_g[fourier_valid],
        resid_fourier[fourier_valid],
        s=RRLYRAE_SCATTER_S,
        alpha=RRLYRAE_POINT_ALPHA,
        color=SECONDARY_COLOR,
        rasterized=True,
    )
    ax_fourier_resid.axhline(0.0, color=NEUTRAL_COLOR, ls="--", lw=LW_STANDARD)
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
    ax.invert_yaxis()
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
        ax.plot(grid, pdf_fn(grid), lw=LW_MEDIUM, color=SECONDARY_COLOR, label=rf"Analytic $p({xlabel}\mid x)$")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    _set_title(ax, title)
    ax.legend()
    _apply_grid(ax)
    return ax


def plot_trace(source, axes=None, title=None, labels=None):
    n_burn = getattr(source, "n_burn", 0)
    steps = np.arange(n_burn, len(source.samples))
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
