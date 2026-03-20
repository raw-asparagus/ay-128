from __future__ import annotations

import numpy as np
import corner
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import ugdatalab.plotting as ugplt
from ugdatalab.spectra import (
    _get_spectra,
    BAD_PIXEL_ERR,
    bitmask_summary,
    pseudo_continuum_normalize,
)

LABELS = ["teff", "logg", "fe_h", "mg_fe", "si_fe"]
LABEL_NAMES = [
    r"$T_\mathrm{eff}$ (K)",
    r"$\log g$",
    r"[Fe/H]",
    r"[Mg/Fe]",
    r"[Si/Fe]",
]

FIELD_DISPLAY = {
    "M15":          ("M15",              ugplt.PRIMARY_COLOR),
    "N6791":        ("NGC 6791",         ugplt.SECONDARY_COLOR),
    "K2_C4_168-21": (r"K2\_C4\_168-21", ugplt.TERTIARY_COLOR),
    "060+00":       ("060+00",           ugplt.QUATERNARY_COLOR),
}


def _extract(data: Table, labels: list[str]) -> np.ndarray:
    return np.column_stack([np.asarray(data[col].data, dtype=float) for col in labels])


def _global_range(data: Table, labels: list[str]) -> list[tuple[float, float]]:
    X = _extract(data, labels)
    return [(X[:, i].min(), X[:, i].max()) for i in range(len(labels))]


def plot_field_corners(
    data: Table,
    labels: list[str],
    label_names: list[str],
    field_display: dict[str, tuple[str, str]],
    shared_range: bool,
) -> list[Figure]:
    """One corner plot per field, optionally sharing axis ranges across fields."""
    ax_range = _global_range(data, labels) if shared_range else None
    figs = []

    for field, (display_name, _) in field_display.items():
        mask   = np.asarray(data["field"], dtype=str) == field
        subset = data[mask]
        X      = _extract(subset, labels)

        fig = plt.figure(figsize=ugplt._textwidth_figsize(8))
        corner.corner(
            X,
            labels=label_names,
            show_titles=True,
            title_fmt=".2f",
            bins=30,
            plot_datapoints=False,
            fill_contours=True,
            levels=(0.68, 0.95),
            color=ugplt.PRIMARY_COLOR,
            range=ax_range,
            fig=fig,
        )
        figs.append(fig)

    return figs


def plot_combined_corner(
    data: Table,
    labels: list[str],
    label_names: list[str],
    field_display: dict[str, tuple[str, str]],
) -> Figure:
    """Overlay corner plot with all fields in different colours."""
    fig = plt.figure(figsize=ugplt._textwidth_figsize(8))
    handles = []

    for field, (display_name, color) in field_display.items():
        mask   = np.asarray(data["field"], dtype=str) == field
        subset = data[mask]
        X      = _extract(subset, labels)

        fig = corner.corner(
            X,
            labels=label_names,
            bins=30,
            plot_datapoints=False,
            fill_contours=False,
            levels=(0.68, 0.95),
            color=color,
            fig=fig,
            hist_kwargs={"histtype": "step", "alpha": 0.6},
            contour_kwargs={"alpha": 0.5},
        )
        handles.append(Line2D([], [], color=color, lw=ugplt.LW_STANDARD,
                              label=rf"{display_name} ($N={len(subset)}$)"))

    fig.legend(handles=handles, loc="upper right", frameon=True,
               bbox_to_anchor=(0.98, 0.98), fontsize=ugplt.LEGEND_SIZE)
    return fig


def plot_spectrum(wavelength: np.ndarray, flux: np.ndarray) -> Axes:
    """Plot a single continuum-normalised (or raw) APOGEE spectrum."""
    fig, ax = plt.subplots(figsize=ugplt._textwidth_figsize(3))
    ax.plot(wavelength, flux,
            lw=ugplt.LW_FINE, color=ugplt.PRIMARY_COLOR, alpha=ugplt.ALPHA_STANDARD)
    ax.set_xlabel(r"Wavelength [Å]")
    ax.set_ylabel(r"$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    ugplt._apply_grid(ax)
    ugplt._tight_layout(fig)
    return ax


def plot_bitmask_spectrum(
    apogee_id: str,
    telescope: str,
    field: str,
) -> tuple[Axes, Axes]:
    """Two-panel plot: flux with good/bad pixels highlighted, and error panel."""
    spec       = _get_spectra(apogee_id, telescope, field)
    wavelength = np.asarray(spec["wavelength"][0], dtype=float)
    flux       = np.asarray(spec["flux"][0],       dtype=float)
    bad, flux_err, _ = bitmask_summary(apogee_id, telescope, field)

    fig, axes = plt.subplots(
        2, 1,
        figsize=ugplt._textwidth_figsize(5),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    ax_flux, ax_err = axes

    good = ~bad & np.isfinite(flux)
    ax_flux.plot(
        wavelength, np.where(good, flux, np.nan),
        lw=ugplt.LW_FINE, color=ugplt.PRIMARY_COLOR,
        alpha=ugplt.ALPHA_STANDARD, label="Good pixel",
    )
    bad_finite = bad & np.isfinite(flux)
    ax_flux.scatter(
        wavelength[bad_finite], flux[bad_finite],
        s=ugplt.MARKER_MS_BIG, color=ugplt.QUATERNARY_COLOR,
        alpha=1.0, marker="x", lw=ugplt.LW_FINE,
        label=r"Bad pixel (error $\to 10^{10}$)", zorder=3,
    )
    ax_flux.set_ylabel(r"$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    ax_flux.legend(fontsize=ugplt.LEGEND_SIZE)
    ax_flux.tick_params(bottom=False)
    ugplt._apply_grid(ax_flux)

    display_err = np.where(bad | ~np.isfinite(flux_err), np.nan, flux_err)
    ax_err.plot(
        wavelength, display_err,
        lw=ugplt.LW_FINE, color=ugplt.SECONDARY_COLOR,
    )
    ax_err.set_ylim(0, np.nanpercentile(display_err, 97))
    ax_err.set_ylabel(r"$\sigma_{F_\lambda}$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    ax_err.set_xlabel(r"Wavelength [Å]")
    ugplt._apply_grid(ax_err)

    ugplt._tight_layout(fig)
    fig.subplots_adjust(hspace=0)
    return ax_flux, ax_err


def plot_normalized_spectrum(
    apogee_id: str,
    telescope: str,
    field: str,
    poly_deg: int,
) -> tuple[Axes, Axes]:
    """Two-panel plot: raw flux + continuum overlay / normalised.

    Raw flux panel colours:
      grey  — non-continuum pixels
      C1    — continuum anchor pixels
      C0/C3/C2 — continuum fit for blue/green/red chips
    """
    result        = pseudo_continuum_normalize(apogee_id, telescope, field, poly_deg)
    wavelength    = result["wavelength"]
    flux          = result["flux"]
    flux_err      = result["flux_err"]
    norm_flux     = result["norm_flux"]
    cont_mask     = result["cont_mask"]
    chips         = result["chips"]
    chip_continua = result["chip_continua"]

    # chips stored as [blue, green, red]; map to user-specified colours
    _CHIP_COLORS  = ["C0", "C3", "C2"]
    _CHIP_LABELS  = ["Blue chip", "Green chip", "Red chip"]

    good     = np.isfinite(flux) & (flux_err < BAD_PIXEL_ERR / 2)
    non_cont = good & ~cont_mask
    anchor   = good & cont_mask

    fig, (ax_raw, ax_norm) = plt.subplots(2, 1, figsize=ugplt._textwidth_figsize(5), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # non-continuum raw flux
    ax_raw.plot(wavelength, np.where(non_cont, flux, np.nan),
                lw=ugplt.LW_FINE, color="gray",
                alpha=ugplt.ALPHA_MUTED)
    # continuum anchor pixels — larger and fully opaque to stand out
    ax_raw.scatter(wavelength[anchor], flux[anchor],
                   s=ugplt.MARKER_MS_SMALL, color=ugplt.SECONDARY_COLOR,
                   alpha=1.0, lw=0, label="Continuum pixels", zorder=2)
    # per-chip continuum fit — each chip's polynomial evaluated on [lo-ext, hi+ext]
    for (wave_eval, cont_eval), color, label in zip(chip_continua, _CHIP_COLORS, _CHIP_LABELS):
        if len(wave_eval) == 0:
            continue
        ax_raw.scatter(wave_eval, cont_eval,
                       s=ugplt.MARKER_MS_MICRO, color=color, lw=0,
                       alpha=1.0,
                       label=f"Continuum fit ({label})", zorder=3)
    ax_raw.set_ylabel(r"$F_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    ax_raw.legend(fontsize=ugplt.LEGEND_SIZE)
    ugplt._apply_grid(ax_raw)

    # normalised flux
    ax_norm.plot(wavelength, norm_flux,
                 lw=ugplt.LW_FINE, color=ugplt.PRIMARY_COLOR,
                 alpha=ugplt.ALPHA_STANDARD, label="Normalised flux")
    ax_norm.axhline(1.0, lw=ugplt.LW_GUIDE, ls="--",
                    color=ugplt.NEUTRAL_COLOR, alpha=ugplt.ALPHA_GUIDE)
    ax_norm.set_ylabel(r"$F_\lambda\,/\,F_\mathrm{cont}$")
    ax_norm.set_xlabel(r"Wavelength [Å]")
    ax_norm.legend(fontsize=ugplt.LEGEND_SIZE)
    ugplt._apply_grid(ax_norm)

    ugplt._tight_layout(fig)
    return ax_raw, ax_norm

