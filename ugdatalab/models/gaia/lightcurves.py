"""Gaia DR3 epoch-photometry I/O, periodograms, and Fourier mean magnitudes."""

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import numpy as np
import requests
from astropy import table
from astropy.io.votable import parse as parse_votable
from tqdm.auto import tqdm

from ugdatalab.utils.cache import cache_stable
from ugdatalab.utils.tables import _sanitize_table
from ugdatalab.models.gaia.constants import (
    ZP_ERR_G, ZP_G, _GAIA_DATALINK_URL, RRLYRAE_PERIOD_MIN,
    RRLYRAE_PERIOD_MAX, _EPOCH_SCHEMA,
)
from ugdatalab.methods.periodogram import lomb_scargle
from ugdatalab.methods.fourier import FourierFit, fourier_fit, build_design_matrix
from ugdatalab.methods.cross_validate import cross_validate


# Number of phase-grid points used when integrating Fourier fits to compute
# flux-space mean magnitudes; chosen for negligible quadrature error at the
# RR Lyrae cadence we work with.
_EPOCH_GRID_N = 1000

# Upper bound on the harmonic-order grid passed to ``cross_validate`` when
# selecting the best Fourier order per source.
_MAX_HARMONIC_K = 25


# ---------------------------------------------------------------------------
# Epoch photometry I/O
# ---------------------------------------------------------------------------

@cache_stable(module="ugdatalab.gaia")
def _get_epoch_photometry(source_id: int) -> table.Table:
    """Download Gaia DR3 epoch photometry for one source via direct HTTP."""
    resp = requests.post(
        _GAIA_DATALINK_URL,
        data={
            "RETRIEVAL_TYPE": "EPOCH_PHOTOMETRY",
            "ID": str(source_id),
            "RELEASE": "Gaia DR3",
            "DATA_STRUCTURE": "INDIVIDUAL",
            "FORMAT": "votable",
            "USE_ZIP_ALWAYS": "true",
        },
    )
    resp.raise_for_status()

    tables = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".xml"):
                with zf.open(name) as f:
                    vot = parse_votable(io.BytesIO(f.read()))
                    tables.extend(t.to_table() for t in vot.iter_tables())

    if not tables:
        raise KeyError(f"Could not find epoch photometry for source_id={source_id}.")

    for t in tables:
        for col in t.columns.values():
            col.unit = None

    data = table.vstack(tables)
    data["source_id"] = source_id
    return data


def _fetch_epoch_photometry(
    source_ids: Iterable[int], max_workers: int = 8,
) -> table.Table:
    """Download epoch photometry for many Gaia sources in parallel."""
    source_ids = tuple(source_ids)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_get_epoch_photometry, sid): sid for sid in source_ids}
        chunks = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Epoch photometry"):
            chunks.append(future.result())
    return table.vstack(chunks)


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

def _fetch_joined_epoch_photometry(catalog: table.Table) -> table.Table:
    """Fetch epoch photometry for a catalog, clean, and join on source_id."""
    epoch_data = _fetch_epoch_photometry(catalog["source_id"])
    _sanitize_table(epoch_data, _EPOCH_SCHEMA)
    # Drop epochs with any non-finite values
    mask = np.all(
        [np.isfinite(epoch_data[name]) for name in _EPOCH_SCHEMA[float]], axis=0,
    )
    joined = table.join(catalog, epoch_data[mask], keys="source_id")
    joined.sort(["source_id", "g_transit_time"])
    return joined


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

# ---------------------------------------------------------------------------
# Fourier mean magnitude (Gaia G-band specific)
# ---------------------------------------------------------------------------

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
    return mean_mag, float(np.sqrt(np.clip(mean_mag_var, 0.0, None)))


def _attach_fourier_mean_magnitudes(data: table.Table) -> None:
    """Fit a per-source Fourier model and attach ``fourier_mean_g_mag`` and ``fourier_mean_g_mag_err``."""
    grouped = data.group_by("source_id")
    means = np.empty(len(grouped.groups))
    errs = np.empty(len(grouped.groups))
    for i, g in enumerate(grouped.groups):
        period = float(g["period_ls"][0])
        cv_result = cross_validate(
            g["g_transit_time"], g["g_transit_mag"], g["g_transit_mag_err"],
            lambda x, y, ye, k, p=period: fourier_fit(x, y, ye, p, k),
            np.arange(1, _MAX_HARMONIC_K + 1, dtype=int),
        )
        best_k = int(cv_result.best_param)
        fit = fourier_fit(
            g["g_transit_time"], g["g_transit_mag"], g["g_transit_mag_err"],
            period, best_k,
        )
        means[i], errs[i] = _fourier_mean_mag_with_err(fit)

    _, inv = np.unique(data["source_id"], return_inverse=True)
    data["fourier_mean_g_mag"] = means[inv]
    data["fourier_mean_g_mag_err"] = errs[inv]
