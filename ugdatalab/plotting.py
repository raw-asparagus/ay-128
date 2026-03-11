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

from .paths import FIGURES_DIR, ensure_output_dirs

TEXTWIDTH_IN = 7.59
COLUMNWIDTH_IN = 3.73
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8
ANNOTATION_SIZE = 9
EMPHASIS_SIZE = 10

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
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
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


def _as_lightcurve_table(source) -> Table:
    if isinstance(source, Table):
        return source
    if hasattr(source, "lightcurve_data") and getattr(source, "lightcurve_data") is not None:
        return getattr(source, "lightcurve_data")
    return _as_table(source)


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


def _grid_2x2(
    *,
    figsize: tuple[float, float],
    sharex: str | bool = "col",
    hspace: float = 0.38,
    wspace: float = 0.32,
):
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=sharex)
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    return fig, axes


def _zero_line(ax: Any) -> None:
    ax.axhline(0.0, color=NEUTRAL_COLOR, lw=0.8, ls="--")


def _unity_line(ax: Any, *, label: str | None = None) -> None:
    ax.axhline(1.0, color=NEUTRAL_COLOR, lw=0.8, ls="--", alpha=0.7, label=label)


def _reference_vline(
    ax: Any,
    x: float,
    *,
    label: str | None = None,
    color: str = NEUTRAL_COLOR,
    lw: float = 0.9,
    ls: str = "--",
    alpha: float = 0.8,
) -> None:
    ax.axvline(x, color=color, lw=lw, ls=ls, alpha=alpha, label=label)


def _set_title(ax: Any, title: str | None) -> None:
    if title is not None:
        ax.set_title(_escape_latex_text(title), fontsize=EMPHASIS_SIZE)


def _default_scatter_kwargs(**kwargs):
    kwargs.setdefault("s", 6)
    kwargs.setdefault("alpha", 0.7)
    kwargs.setdefault("rasterized", True)
    return kwargs


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
        alpha=0.10,
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
        alpha=0.10,
        label=r"$|b| < 30^\circ$ band",
    )

    ax.set_xlabel(r"Galactic longitude $l$")
    ax.set_ylabel(r"Galactic latitude $b$")
    _set_title(ax, title)
    ax.legend(loc="lower right")
    _apply_grid(ax)
    return ax


def plot_lomb_scargle_periodogram(
    source,
    source_id=None,
    *,
    periodogram=None,
    period=None,
    ax=None,
    title=None,
    **plot_kwargs,
):
    data = _as_lightcurve_table(source)
    if source_id is not None and "source_id" in data.colnames:
        data = data[np.asarray(data["source_id"], dtype=np.int64) == int(source_id)]

    if len(data) == 0:
        raise ValueError("No light-curve rows available for plotting.")

    if periodogram is None:
        from ugdatalab.lightcurves import lomb_scargle_periodogram

        periodogram = lomb_scargle_periodogram(data)

    periods, power, best_period = periodogram
    periods = np.asarray(periods, dtype=float)
    power = np.asarray(power, dtype=float)
    best_period = float(best_period)

    order = np.argsort(periods)
    periods = periods[order]
    power = power[order]

    if period is None:
        if "period_ls" in data.colnames:
            period_values = np.asarray(data["period_ls"], dtype=float)
            finite_periods = period_values[np.isfinite(period_values)]
            period = float(finite_periods[0]) if len(finite_periods) else best_period
        else:
            period = best_period
    else:
        period = float(period)

    if ax is None:
        _, ax = _single_panel(_columnwidth_figsize(35 / 16))

    plot_kwargs.setdefault("color", PRIMARY_COLOR)
    plot_kwargs.setdefault("lw", 1.1)
    ax.plot(periods, power, **plot_kwargs)
    ax.axvline(
        period,
        color=SECONDARY_COLOR,
        ls="--",
        lw=0.9,
        alpha=0.9,
        label=rf"$P={period:.4f}$ d",
    )
    ax.set_xlabel(r"$P$ [days]")
    ax.set_ylabel("Lomb-Scargle power")
    _set_title(ax, title)
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
    if "is_rrc" in data.colnames:
        rrc = np.asarray(data["is_rrc"], dtype=bool)
    elif "best_classification" in data.colnames:
        rrc = np.asarray(data["best_classification"]) == "RRc"
    elif "pf" in data.colnames and "p1_o" in data.colnames:
        pf = np.asarray(data["pf"], dtype=float)
        p1_o = np.asarray(data["p1_o"], dtype=float)
        rrc = ~np.isfinite(pf) & np.isfinite(p1_o)
    else:
        raise ValueError("Could not determine RR Lyrae class labels for plotting.")
    rrab = ~rrc
    if period_column is None:
        period_column = "period_ls" if use_periodogram else "period"
    if abs_mag_column is None:
        abs_mag_column = "M_G_ls" if use_periodogram else "M_G"
    if abs_mag_err_column is None:
        abs_mag_err_column = "sigma_M_ls" if use_periodogram else "sigma_M"

    if ax is None:
        _, ax = _single_panel(_columnwidth_figsize(5 / 2))

    scatter_kwargs = _default_scatter_kwargs(**scatter_kwargs)

    periods = np.asarray(data[period_column], dtype=float)
    m_g = np.asarray(data[abs_mag_column], dtype=float)
    sigma_m = np.asarray(data[abs_mag_err_column], dtype=float)
    ax.errorbar(
        periods,
        m_g,
        yerr=sigma_m,
        fmt="none",
        color=LIGHT_NEUTRAL_COLOR,
        alpha=0.6,
        zorder=1,
    )
    ax.scatter(periods[rrab], m_g[rrab], color=PRIMARY_COLOR, label="RRab", zorder=2, **scatter_kwargs)
    ax.scatter(periods[rrc], m_g[rrc], color=SECONDARY_COLOR, label="RRc", zorder=2, **scatter_kwargs)

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
    ax.scatter(
        diff_data["period"],
        np.asarray(diff_data["M_G"], dtype=float),
        color=SECONDARY_COLOR,
        label=f"Removed ($N$={len(diff_data)})",
        zorder=1,
        **scatter_kwargs,
    )
    ax.scatter(
        subset_data["period"],
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


def plot_period_mean_g(source, ax=None, title=None, **scatter_kwargs):
    data = _as_table(source)
    if ax is None:
        _, ax = _single_panel(_textwidth_figsize(22 / 5))

    classifications = np.asarray(data["best_classification"]).astype(str)
    periods = np.asarray(data["best_period"], dtype=float)
    mean_g = np.asarray(data["mean_apparent_g"], dtype=float)
    mean_g_err = np.asarray(data["mean_apparent_g_err"], dtype=float)
    finite = np.isfinite(periods) & np.isfinite(mean_g)

    ax.errorbar(
        periods[finite],
        mean_g[finite],
        yerr=mean_g_err[finite],
        fmt="none",
        color=NEUTRAL_COLOR,
        alpha=0.7,
        zorder=1,
    )

    scatter_kwargs.setdefault("s", 12)
    scatter_kwargs.setdefault("alpha", 0.7)
    scatter_kwargs.setdefault("rasterized", True)
    primary_colors = (
        PRIMARY_COLOR,
        SECONDARY_COLOR,
        TERTIARY_COLOR,
        QUATERNARY_COLOR,
        QUINARY_COLOR,
        SENARY_COLOR,
        SEPTENARY_COLOR,
        NEUTRAL_COLOR,
        LIGHT_NEUTRAL_COLOR,
        NONARY_COLOR,
    )

    for i, label in enumerate(np.unique(classifications[finite])):
        mask = finite & (classifications == label)
        ax.scatter(
            periods[mask],
            mean_g[mask],
            color=primary_colors[i % len(primary_colors)],
            label=label,
            zorder=2,
            **scatter_kwargs,
        )

    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel(r"$P$ [days]")
    ax.set_ylabel(r"$\langle G \rangle$ [mag]")
    if title is None:
        title = rf"L-S periods and $\langle G \rangle$ ({len(data)} RR Lyrae)"
    _set_title(ax, title)
    ax.legend()
    _apply_grid(ax)
    _tight_layout(ax.figure)
    return ax


def plot_vari_rrlyrae_period_comparison(source, axes=None):
    data = _as_table(source)

    classifications = np.asarray(data["best_classification"]).astype(str)
    fundamental_period = np.asarray(data["pf"], dtype=float)
    best_period = np.asarray(data["best_period"], dtype=float)
    first_overtone_period = np.asarray(data["p1_o"], dtype=float)
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

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=_textwidth_figsize(4))
        fig.subplots_adjust(wspace=0.28)
    else:
        fig = axes[0].figure
    ax_fundamental, ax_first_overtone = axes

    unique_classes = np.unique(classifications[finite_fundamental])
    rr_d_color = PRIMARY_COLOR
    for i, label in enumerate(unique_classes):
        color = COMPONENT_COLORS[i % len(COMPONENT_COLORS)]
        mask = finite_fundamental & (classifications == label)
        ax_fundamental.scatter(
            fundamental_period[mask],
            best_period[mask],
            s=14,
            color=color,
            label=label,
            zorder=2,
        )
        if label == "RRd":
            rr_d_color = color

    lo = float(np.nanmin(np.concatenate([fundamental_period[finite_fundamental], best_period[finite_fundamental]])))
    hi = float(np.nanmax(np.concatenate([fundamental_period[finite_fundamental], best_period[finite_fundamental]])))
    ax_fundamental.plot(
        [lo, hi],
        [lo, hi],
        color=NEUTRAL_COLOR,
        ls="--",
        lw=1.0,
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
            lw=1.2,
            label=rf"$P_{{\rm LS}}\approx({ratio:.3f})\,P_{{\rm F}}$",
            zorder=1,
        )

    ax_fundamental.set_xlabel(r"Catalog fundamental period $P_{\rm F}$ [days]")
    ax_fundamental.set_ylabel(r"L-S period $P_{\rm LS}$ [days]")
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
        ax_first_overtone.scatter(
            first_overtone_period[finite_rrd],
            best_period[finite_rrd],
            s=14,
            color=rr_d_color,
            zorder=2,
        )
        ax_first_overtone.plot(
            [lo_rrd, hi_rrd],
            [lo_rrd, hi_rrd],
            color=NEUTRAL_COLOR,
            ls="--",
            lw=1.0,
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

    ax_first_overtone.set_xlabel(r"Catalog first-overtone period $P_{1\rm O}$ [days]")
    ax_first_overtone.set_ylabel(r"L-S period $P_{\rm LS}$ [days]")
    ax_first_overtone.set_title(r"L-S periods vs. \texttt{vari\_rrlyrae} first-overtone periods (RRd only)")
    _apply_grid(ax_first_overtone)

    _tight_layout(fig)
    return axes


def plot_hr(source, ax=None, title=None, **scatter_kwargs):
    data = _as_table(source)
    rrc = np.asarray(data["is_rrc"], dtype=bool)
    rrab = ~rrc

    if ax is None:
        _, ax = _single_panel(_columnwidth_figsize(7 / 2))

    scatter_kwargs = _default_scatter_kwargs(**scatter_kwargs)
    ax.scatter(
        data["bp_rp"][rrab],
        np.asarray(data["M_G"], dtype=float)[rrab],
        color=PRIMARY_COLOR,
        label="RRab",
        **scatter_kwargs,
    )
    ax.scatter(
        data["bp_rp"][rrc],
        np.asarray(data["M_G"], dtype=float)[rrc],
        color=SECONDARY_COLOR,
        label="RRc",
        **scatter_kwargs,
    )

    ax.invert_yaxis()
    ax.set_xlabel(r"$G_\mathrm{BP} - G_\mathrm{RP}$ [mag]")
    ax.set_ylabel(r"$M_G$ [mag]")
    _set_title(ax, title)
    ax.legend()
    _apply_grid(ax)
    return ax


def plot_raw_phase_folded_lightcurve(
    source,
    source_id=None,
    period=None,
    axes=None,
    title=None,
    time_column="g_transit_time",
    mag_column="g_transit_mag",
    **scatter_kwargs,
):
    data = _as_lightcurve_table(source)
    if source_id is not None and "source_id" in data.colnames:
        data = data[np.asarray(data["source_id"], dtype=np.int64) == int(source_id)]

    if len(data) == 0:
        raise ValueError("No light-curve rows available for plotting.")

    if period is None:
        if "best_classification" in data.colnames and "pf" in data.colnames and "p1_o" in data.colnames:
            classification = np.asarray(data["best_classification"], dtype=str)
            pf = np.asarray(data["pf"], dtype=float)
            p1_o = np.asarray(data["p1_o"], dtype=float)
            period_values = np.where(classification == "RRab", pf, p1_o)
            unknown = ~np.isfinite(period_values)
            period_values[unknown] = np.where(np.isfinite(p1_o[unknown]), p1_o[unknown], pf[unknown])
        elif "pf" in data.colnames and "p1_o" in data.colnames:
            pf = np.asarray(data["pf"], dtype=float)
            p1_o = np.asarray(data["p1_o"], dtype=float)
            period_values = np.where(np.isfinite(p1_o), p1_o, pf)
        elif "pf" in data.colnames:
            period_values = np.asarray(data["pf"], dtype=float)
        elif "p1_o" in data.colnames:
            period_values = np.asarray(data["p1_o"], dtype=float)
        else:
            raise ValueError("Could not determine a period for phase folding.")

        finite_periods = period_values[np.isfinite(period_values)]
        if len(finite_periods) == 0:
            raise ValueError("Could not determine a finite period for phase folding.")
        period = float(finite_periods[0])
    else:
        period = float(period)

    epoch = np.asarray(data[time_column], dtype=float)
    mag = np.asarray(data[mag_column], dtype=float)
    phase = (epoch % period) / period

    if axes is None:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=_textwidth_figsize(84 / 25),
            sharey=True,
        )
    else:
        fig = axes[0].figure

    scatter_kwargs = _default_scatter_kwargs(**scatter_kwargs)
    scatter_kwargs.setdefault("s", 10)
    scatter_kwargs.setdefault("color", PRIMARY_COLOR)

    time_unit = getattr(data[time_column], "unit", None)
    mag_unit = getattr(data[mag_column], "unit", None)
    time_label = f"Time [{time_unit}]" if time_unit is not None else "Time"
    mag_label = f"G [{mag_unit}]" if mag_unit is not None else "G"

    axes[0].scatter(epoch, mag, **scatter_kwargs)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(time_label)
    axes[0].set_ylabel(mag_label)
    axes[0].set_title("Raw Light Curve")
    _apply_grid(axes[0])

    axes[1].scatter(phase, mag, **scatter_kwargs)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Phase")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[1].set_title("Phase-folded Light Curve")
    _apply_grid(axes[1])

    if title is not None:
        fig.suptitle(_escape_latex_text(title), fontsize=EMPHASIS_SIZE)
    return axes


def plot_inlier_prob_map(source, ax=None, title=None):
    data = _as_table(source, attr="all_data")
    if ax is None:
        _, ax = _single_panel(_columnwidth_figsize(5 / 2))

    sc = ax.scatter(
        data["period"],
        np.asarray(data["M_G"], dtype=float),
        c=np.asarray(data["inlier_prob"], dtype=float),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        s=6,
        alpha=0.75,
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
    ax.hist(samples, bins=50, density=True, alpha=0.6, color=PRIMARY_COLOR, label="MCMC samples")

    if pdf_fn is not None:
        margin = 0.5 * (samples.max() - samples.min())
        grid = np.linspace(samples.min() - margin, samples.max() + margin, 500)
        ax.plot(grid, pdf_fn(grid), lw=1.2, color=SECONDARY_COLOR, label=rf"Analytic $p({xlabel}\mid x)$")

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
        axes[i].plot(steps, source.samples[n_burn:, i], lw=0.6, alpha=0.9, color=PRIMARY_COLOR)
        axes[i].set_ylabel(lbl)
        _apply_grid(axes[i])

    if title is not None:
        axes[0].set_title(title, fontsize=EMPHASIS_SIZE)

    axes[-1].plot(steps, source.log_probs[n_burn:], lw=0.6, alpha=0.9, color=SECONDARY_COLOR)
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
