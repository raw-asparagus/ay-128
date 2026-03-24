import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import numpy as np
import requests
from astropy import table
from astropy.io.votable import parse as parse_votable

from ugdatalab.models.cache import cache_stable
from ugdatalab.models.utils import _sanitize_table
from ugdatalab.models.gaia.constants import ZP_ERR_G, ZP_G
from ugdatalab.methods.periodogram import lomb_scargle
from ugdatalab.methods.fourier import FourierFit, fourier_fit, build_design_matrix
from ugdatalab.methods.cross_validate import holdout_validate

DEFAULT_PERIOD_MIN = 0.2
DEFAULT_PERIOD_MAX = 1.2

_GAIA_DATALINK_URL = "https://gea.esac.esa.int/data-server/data"


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
    source_ids: Iterable[int], max_workers: int = 4,
) -> table.Table:
    """Download epoch photometry for many Gaia sources in parallel."""
    from tqdm.auto import tqdm

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

_EPOCH_SCHEMA = {
    float: ["g_transit_time", "g_transit_mag", "g_transit_flux", "g_transit_flux_error"],
}


def _fetch_joined_epoch_photometry(catalog: table.Table) -> table.Table:
    """Fetch epoch photometry for a catalog, clean, and join on source_id."""
    epoch_data = _fetch_epoch_photometry(catalog["source_id"])
    _sanitize_table(epoch_data, _EPOCH_SCHEMA)
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

    # Per-source flux-space mean magnitude
    source_column = data["source_id"]
    source_ids = np.unique(source_column)
    lookup = {}
    for sid in source_ids:
        src_flux = flux[source_column == sid]
        src_flux_err = flux_err[source_column == sid]
        mean_flux = np.mean(src_flux)
        mean_flux_err = np.sqrt(np.sum(src_flux_err**2)) / len(src_flux_err)
        mean_mag = -2.5 * np.log10(mean_flux) + ZP_G
        mean_meas_err = (2.5 / np.log(10.0)) * (mean_flux_err / mean_flux)
        lookup[sid] = (mean_mag, np.sqrt(mean_meas_err**2 + ZP_ERR_G**2))

    data["mean_g_transit_mag"] = [lookup[sid][0] for sid in data["source_id"]]
    data["mean_g_transit_mag_err"] = [lookup[sid][1] for sid in data["source_id"]]


def _attach_periodogram_periods(data: table.Table) -> None:
    """Attach per-source Lomb-Scargle best periods as ``period_ls``."""
    source_column = data["source_id"]
    source_ids = np.unique(source_column)
    lookup = {}
    for sid in source_ids:
        subset = data[source_column == sid]
        result = lomb_scargle(
            subset["g_transit_time"],
            subset["g_transit_flux"],
            subset["g_transit_flux_error"],
            DEFAULT_PERIOD_MIN,
            DEFAULT_PERIOD_MAX,
        )
        lookup[sid] = result.best_period
    data["period_ls"] = [lookup[sid] for sid in data["source_id"]]

# ---------------------------------------------------------------------------
# Fourier mean magnitude (Gaia G-band specific)
# ---------------------------------------------------------------------------

def _fourier_mean_mag(fit: FourierFit) -> float:
    """Flux-space mean magnitude from a fitted Fourier model."""
    epoch_grid = np.linspace(0.0, fit.period, 1000, endpoint=False)
    fluepoch_grid = 10.0 ** (-0.4 * (fit.predict(epoch_grid) - ZP_G))
    return -2.5 * np.log10(np.mean(fluepoch_grid)) + ZP_G


def _fourier_mean_mag_err(fit: FourierFit) -> float:
    """Propagate coefficient covariance into the flux-space mean magnitude."""
    epoch_grid = np.linspace(0.0, fit.period, 1000, endpoint=False)
    omega = 2.0 * np.pi / fit.period
    X_grid = build_design_matrix(epoch_grid, omega, fit.k)
    mag_grid = X_grid @ fit.beta
    fluepoch_grid = 10.0 ** (-0.4 * (mag_grid - ZP_G))
    mean_flux = np.mean(fluepoch_grid)

    grad = np.mean(X_grid * fluepoch_grid[:, None], axis=0) / mean_flux
    mean_mag_var = grad @ fit.beta_cov @ grad
    return float(np.sqrt(np.clip(mean_mag_var, 0.0, None)))


def _attach_fourier_mean_magnitudes(data: table.Table) -> None:
    """Fit Fourier models per source and attach ``fourier_mean_g_mag`` and ``fourier_mean_g_mag_err``."""
    source_column = data["source_id"]
    source_ids = np.unique(source_column)
    lookup = {}
    for sid in source_ids:
        subset = data[source_column == sid]
        period = float(subset["period_ls"][0])

        cv_result = holdout_validate(
            subset["g_transit_time"], subset["g_transit_mag"], subset["g_transit_mag_err"],
            lambda x, y, ye, k, p=period: fourier_fit(x, y, ye, p, k),
            np.arange(1, 26, dtype=int),
        )
        best_k = int(cv_result.best_param)
        fit = fourier_fit(
            subset["g_transit_time"], subset["g_transit_mag"], subset["g_transit_mag_err"],
            period, best_k,
        )
        lookup[sid] = (_fourier_mean_mag(fit), _fourier_mean_mag_err(fit))

    data["fourier_mean_g_mag"] = [lookup[sid][0] for sid in data["source_id"]]
    data["fourier_mean_g_mag_err"] = [lookup[sid][1] for sid in data["source_id"]]
