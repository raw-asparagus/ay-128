from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from astroquery.gaia import Gaia
from astropy import table
from astropy.timeseries import LombScargle

from ugdatalab.models.cache import _cache_stable
from ugdatalab.models.gaia import ZP_ERR_G, ZP_G

DEFAULT_PERIOD_MIN = 0.2
DEFAULT_PERIOD_MAX = 1.2
EPOCH_DATA_RELEASE = "Gaia DR3"
EPOCH_FILE_FORMAT = "xml"


def _as_float_array(column) -> np.ndarray:
    values = np.ma.asarray(column, dtype=float)
    return np.asarray(np.ma.filled(values, np.nan), dtype=float)


# Data loading and photometric preprocessing


def _epoch_payload(datalink: dict, source_id: int):
    key = f"EPOCH_PHOTOMETRY-{EPOCH_DATA_RELEASE} {source_id}.{EPOCH_FILE_FORMAT}"
    if key in datalink:
        return datalink[key]

    source_id_str = str(source_id)
    for dl_key, payload in datalink.items():
        if source_id_str in dl_key:
            return payload
    raise KeyError(f"Could not find epoch photometry for source_id={source_id}.")


def _empty_epoch_table() -> table.Table:
    return table.Table({"source_id": np.asarray([], dtype=np.int64)})


def _empty_joined_table(catalog: table.Table, epoch_data: table.Table) -> table.Table:
    res = table.Table()
    for name in catalog.colnames:
        res[name] = np.asarray(catalog[name])[:0]
    for name in epoch_data.colnames:
        if name not in res.colnames:
            res[name] = np.asarray(epoch_data[name])[:0]
    return res


@_cache_stable(module="ugdatalab.gaia")
def get_epoch_photometry(source_id: int) -> table.Table:
    """Download Gaia epoch photometry for one source."""
    datalink = Gaia.load_data(
        ids=[source_id],
        data_release=EPOCH_DATA_RELEASE,
        retrieval_type="EPOCH_PHOTOMETRY",
        data_structure="INDIVIDUAL",
        linking_parameter="SOURCE_ID",
    )
    payload = _epoch_payload(datalink, source_id)
    data = table.vstack([chunk.to_table() for chunk in payload])
    data["source_id"] = source_id
    return data


def fetch_epoch_photometry(source_ids: Iterable[int]) -> table.Table:
    """Download epoch photometry for many Gaia sources and stack the results."""
    source_ids = tuple(source_id for source_id in source_ids)
    chunks = []
    for source_id in source_ids:
        chunk = get_epoch_photometry(source_id)
        if len(chunk):
            chunks.append(chunk)

    if not chunks:
        return _empty_epoch_table()
    if len(chunks) == 1:
        return chunks[0].copy()
    return table.vstack(chunks)


def clean_epoch_photometry(data: table.Table) -> table.Table:
    """Drop rows with missing Gaia G epoch time, flux, or magnitude values."""
    if len(data) == 0:
        return data.copy()

    mask = (
        np.isfinite(_as_float_array(data["g_transit_time"]))
        & np.isfinite(_as_float_array(data["g_transit_mag"]))
        & np.isfinite(_as_float_array(data["g_transit_flux"]))
        & np.isfinite(_as_float_array(data["g_transit_flux_error"]))
    )
    return data[mask]


def join_catalog_with_epoch_photometry(catalog: table.Table, epoch_data: table.Table) -> table.Table:
    """Join a source catalog to epoch photometry on `source_id` and clean the result."""
    if len(epoch_data) == 0 or "source_id" not in epoch_data.colnames:
        return _empty_joined_table(catalog, epoch_data)
    return clean_epoch_photometry(table.join(catalog, epoch_data, keys="source_id"))


def fetch_joined_epoch_photometry(catalog: table.Table) -> table.Table:
    """Fetch epoch photometry for a catalog and return the cleaned joined table."""
    epoch_data = fetch_epoch_photometry(catalog["source_id"])
    return join_catalog_with_epoch_photometry(catalog, epoch_data)


def _add_g_transit_mag_error(data: table.Table) -> table.Table:
    """Attach `g_transit_mag_err` computed from flux errors and the G zero point."""
    flux = _as_float_array(data["g_transit_flux"])
    flux_err = _as_float_array(data["g_transit_flux_error"])
    meas_err = (2.5 / np.log(10.0)) * np.abs(flux_err / flux)
    data["g_transit_mag_err"] = np.sqrt(meas_err**2 + ZP_ERR_G ** 2)
    return data


def _get_mean_mags(epoch_table: table.Table) -> tuple[float, float]:
    flux = _as_float_array(epoch_table["g_transit_flux"])
    flux_err = _as_float_array(epoch_table["g_transit_flux_error"])

    mean_flux = np.mean(flux)
    mean_flux_err = np.sqrt(np.sum(flux_err**2)) / len(flux_err)
    mean_g_mag = -2.5 * np.log10(mean_flux) + ZP_G
    mean_meas_err = (2.5 / np.log(10.0)) * (mean_flux_err / mean_flux)
    mean_g_mag_err = np.sqrt(mean_meas_err**2 + ZP_ERR_G ** 2)
    return mean_g_mag, mean_g_mag_err


def attach_flux_mean_magnitudes(data: table.Table) -> table.Table:
    """Attach per-epoch magnitude errors and repeated per-source flux means."""
    _add_g_transit_mag_error(data)
    source_column = np.asarray(data["source_id"], dtype=np.int64)
    source_ids = np.unique(source_column)

    lookup = {}
    for source_id in np.asarray(source_ids, dtype=np.int64):
        lookup[source_id] = _get_mean_mags(data[source_column == source_id])

    data["mean_g_transit_mag"] = [lookup[source_id][0] for source_id in data["source_id"]]
    data["mean_g_transit_mag_err"] = [lookup[source_id][1] for source_id in data["source_id"]]
    return data


# Lomb-Scargle period finding


def attach_periodogram_periods(data: table.Table) -> table.Table:
    """Attach repeated per-source Lomb-Scargle periods to a joined epoch table."""
    source_ids, periods = estimate_periods_from_epoch_photometry(data)
    lookup = {source_id: float(period) for source_id, period in zip(source_ids, periods)}
    data["period_ls"] = [lookup[source_id] for source_id in data["source_id"]]
    return data


def lomb_scargle_periodogram(target: table.Table) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute a Lomb-Scargle periodogram for one source's epoch photometry."""
    epoch = _as_float_array(target["g_transit_time"])
    values = _as_float_array(target["g_transit_flux"])
    value_err = _as_float_array(target["g_transit_flux_error"])

    freqs, power = LombScargle(epoch, values, value_err).autopower(
        minimum_frequency=1.0 / DEFAULT_PERIOD_MAX,
        maximum_frequency=1.0 / DEFAULT_PERIOD_MIN,
    )

    periods = 1.0 / np.asarray(freqs, dtype=float)
    power = np.asarray(power, dtype=float)
    order = np.argsort(power)[::-1]
    periods = periods[order]
    power = power[order]
    max_power = np.max(power)
    near_max = np.where(power >= 0.98 * max_power)[0]
    if len(near_max) == 0:
        best_period = periods[int(np.argmax(power))]
    else:
        best_period = periods[int(near_max[np.argmax(periods[near_max])])]
    return periods, power, float(best_period)


def estimate_periods_from_epoch_photometry(data: table.Table) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the best Lomb-Scargle period for each source in a joined table."""
    source_column = np.asarray(data["source_id"], dtype=np.int64)
    source_ids = np.unique(source_column)
    periods = np.empty(len(source_ids), dtype=float)
    for i, source_id in enumerate(source_ids):
        _, _, periods[i] = lomb_scargle_periodogram(data[source_column == source_id])
    return source_ids, periods


# Fourier-series light-curve modeling


def phase_fold(epoch: np.ndarray, period: float) -> np.ndarray:
    """Map times to phase in [0, 1)."""
    epoch = np.asarray(epoch, dtype=float)
    return (epoch % period) / period


def build_fourier_matrix(epoch: Iterable[float], omega: float, k: int) -> np.ndarray:
    """Build the design matrix X for a Fourier series with known angular frequency."""
    epochs = np.asarray(epoch, dtype=float)
    period = 2.0 * np.pi / omega
    epochs_mod = epochs % period

    X = np.ones((len(epochs_mod), 2 * k + 1), dtype=float)
    for k in range(1, k + 1):
        X[:, 2 * k - 1] = np.cos(k * omega * epochs_mod)
        X[:, 2 * k] = np.sin(k * omega * epochs_mod)
    return X


def _fourier_predict(epoch_eval: Iterable[float], period: float, k: int, beta: np.ndarray) -> np.ndarray:
    omega = 2.0 * np.pi / period
    return build_fourier_matrix(epoch_eval, omega, k) @ np.asarray(beta, dtype=float)


@dataclass(frozen=True)
class FourierFit:
    source_id: int
    period: float
    K: int
    epochs: np.ndarray
    mags: np.ndarray
    mag_errs: np.ndarray
    beta: np.ndarray
    chi2_r: float

    def predict(self, epoch_eval: Iterable[float]) -> np.ndarray:
        return _fourier_predict(epoch_eval, self.period, self.K, self.beta)


def fourier_fit(target: table.Table, period: float, k: int) -> FourierFit:
    """Fit a weighted Fourier series to one light curve with a fixed period."""
    source_id = int(np.asarray(target["source_id"], dtype=np.int64)[0])
    epochs = _as_float_array(target["g_transit_time"])
    mags = _as_float_array(target["g_transit_mag"])
    mag_errs = _as_float_array(target["g_transit_mag_err"])

    if len(epochs) <= 2 * k + 1:
        raise ValueError("Not enough epochs for the requested number of Fourier harmonics.")

    omega = 2.0 * np.pi / period
    X = build_fourier_matrix(epochs, omega, k)
    weights = 1.0 / mag_errs
    beta, _, _, _ = np.linalg.lstsq(X * weights[:, None], mags * weights, rcond=None)

    resid = mags - X @ beta
    nu = len(epochs) - (2 * k + 1)
    chi2_r = float(np.sum((resid / mag_errs) ** 2) / nu)
    period = float(period)

    return FourierFit(
        source_id=source_id,
        period=period,
        K=k,
        epochs=epochs,
        mags=mags,
        mag_errs=mag_errs,
        beta=beta,
        chi2_r=chi2_r,
    )


def cross_validate_harmonics(target: table.Table, period: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Cross-validate the harmonic order of a fixed-period Fourier model."""
    Ks = np.arange(1, 26, dtype=int)
    epochs = _as_float_array(target["g_transit_time"])

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(epochs))
    n_cv = max(1, int(round(0.2 * len(epochs))))
    cv_idx = idx[:n_cv]
    train_idx = idx[n_cv:]

    chi2r_train = np.full(len(Ks), np.nan, dtype=float)
    chi2r_cv = np.full(len(Ks), np.nan, dtype=float)

    for i, K in enumerate(Ks):
        if len(train_idx) <= 2 * K + 1:
            continue
        fit = fourier_fit(target[train_idx], period, K)
        chi2r_train[i] = fit.chi2_r

        cv_epochs = _as_float_array(target[cv_idx]["g_transit_time"])
        cv_mags = _as_float_array(target[cv_idx]["g_transit_mag"])
        cv_mag_errs = _as_float_array(target[cv_idx]["g_transit_mag_err"])
        resid_cv = cv_mags - fit.predict(cv_epochs)
        chi2r_cv[i] = float(np.sum((resid_cv / cv_mag_errs) ** 2) / len(cv_idx))

    best_k = Ks[int(np.nanargmin(chi2r_cv))]
    return (
        Ks,
        chi2r_train,
        chi2r_cv,
        best_k,
        np.asarray(train_idx, dtype=int),
        np.asarray(cv_idx, dtype=int),
    )


def predict_future_magnitude(fit: FourierFit) -> tuple[float, float]:
    """Predict the magnitude a fixed number of days after the last observed epoch."""
    epoch_last = float(np.max(fit.epochs))
    epoch_pred = epoch_last + 10.0
    mag_pred = float(fit.predict([epoch_pred])[0])
    return epoch_pred, mag_pred


def fourier_mean_magnitude(fit: FourierFit) -> float:
    """Compute the flux-space mean magnitude implied by a fitted Fourier model."""
    epoch_grid = np.linspace(0.0, fit.period, 1000, endpoint=False)
    flux_grid = 10.0 ** (-0.4 * (fit.predict(epoch_grid) - ZP_G))
    return -2.5 * np.log10(np.mean(flux_grid)) + ZP_G
