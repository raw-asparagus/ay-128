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
_EPOCH_FLOAT_COLUMNS = (
    "g_transit_time",
    "g_transit_mag",
    "g_transit_flux",
    "g_transit_flux_error",
)
_MIN_PERIOD = np.finfo(float).tiny


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
def _get_epoch_photometry(source_id: int) -> table.Table:
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


def _fetch_epoch_photometry(source_ids: Iterable[int]) -> table.Table:
    """Download epoch photometry for many Gaia sources and stack the results."""
    source_ids = tuple(source_id for source_id in source_ids)
    chunks = []
    for source_id in source_ids:
        chunk = _get_epoch_photometry(source_id)
        if len(chunk):
            chunks.append(chunk)

    if not chunks:
        return _empty_epoch_table()
    if len(chunks) == 1:
        return chunks[0].copy()
    return table.vstack(chunks)


def _clean_epoch_photometry(data: table.Table) -> table.Table:
    """Drop rows with missing Gaia G epoch time, flux, or magnitude values."""
    if len(data) == 0:
        out = data.copy()
        for name in _EPOCH_FLOAT_COLUMNS:
            if name in out.colnames:
                out[name] = _as_float_array(out[name])
        return out

    mask = (
        np.isfinite(_as_float_array(data["g_transit_time"]))
        & np.isfinite(_as_float_array(data["g_transit_mag"]))
        & np.isfinite(_as_float_array(data["g_transit_flux"]))
        & np.isfinite(_as_float_array(data["g_transit_flux_error"]))
    )
    out = data[mask].copy()
    for name in _EPOCH_FLOAT_COLUMNS:
        out[name] = _as_float_array(out[name])
    return out


def _join_catalog_with_epoch_photometry(catalog: table.Table, epoch_data: table.Table) -> table.Table:
    """Join a source catalog to epoch photometry on `source_id` and clean the result."""
    epoch_data = _clean_epoch_photometry(epoch_data)
    if len(epoch_data) == 0 or "source_id" not in epoch_data.colnames:
        return _empty_joined_table(catalog, epoch_data)
    return table.join(catalog, epoch_data, keys="source_id")


def _fetch_joined_epoch_photometry(catalog: table.Table) -> table.Table:
    """Fetch epoch photometry for a catalog and return the cleaned joined table."""
    epoch_data = _fetch_epoch_photometry(catalog["source_id"])
    return _join_catalog_with_epoch_photometry(catalog, epoch_data)


def _add_g_transit_mag_error(data: table.Table) -> table.Table:
    """Attach `g_transit_mag_err` computed from flux errors and the G zero point."""
    flux = data["g_transit_flux"]
    flux_err = data["g_transit_flux_error"]
    meas_err = (2.5 / np.log(10.0)) * np.abs(flux_err / flux)
    data["g_transit_mag_err"] = np.sqrt(meas_err**2 + ZP_ERR_G ** 2)
    return data


def _get_mean_mags(epoch_table: table.Table) -> tuple[float, float]:
    flux = epoch_table["g_transit_flux"]
    flux_err = epoch_table["g_transit_flux_error"]

    mean_flux = np.mean(flux)
    mean_flux_err = np.sqrt(np.sum(flux_err**2)) / len(flux_err)
    mean_g_mag = -2.5 * np.log10(mean_flux) + ZP_G
    mean_meas_err = (2.5 / np.log(10.0)) * (mean_flux_err / mean_flux)
    mean_g_mag_err = np.sqrt(mean_meas_err**2 + ZP_ERR_G ** 2)
    return mean_g_mag, mean_g_mag_err


def attach_flux_mean_magnitudes(data: table.Table) -> table.Table:
    """Attach per-epoch magnitude errors and repeated per-source flux means."""
    _add_g_transit_mag_error(data)
    source_column = data["source_id"]
    source_ids = np.unique(source_column)

    lookup = {}
    for source_id in source_ids:
        lookup[source_id] = _get_mean_mags(data[source_column == source_id])

    data["mean_g_transit_mag"] = [lookup[source_id][0] for source_id in data["source_id"]]
    data["mean_g_transit_mag_err"] = [lookup[source_id][1] for source_id in data["source_id"]]
    return data


# Lomb-Scargle period finding


def attach_periodogram_periods(data: table.Table) -> table.Table:
    """Attach repeated per-source Lomb-Scargle periods to a joined epoch table."""
    source_ids, periods = _estimate_periods_from_epoch_photometry(data)
    lookup = {source_id: period for source_id, period in zip(source_ids, periods)}
    data["period_ls"] = [lookup[source_id] for source_id in data["source_id"]]
    return data


def _lomb_scargle_spectrum(target: table.Table) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Lomb-Scargle frequency spectrum for one light curve."""
    epoch = target["g_transit_time"]
    values = target["g_transit_flux"]
    value_err = target["g_transit_flux_error"]

    freqs, power = LombScargle(epoch, values, value_err).autopower(
        minimum_frequency=1.0 / DEFAULT_PERIOD_MAX,
        maximum_frequency=1.0 / DEFAULT_PERIOD_MIN,
    )
    return np.asarray(freqs, dtype=float), np.asarray(power, dtype=float)


def lomb_scargle_periodogram(target: table.Table) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute a Lomb-Scargle periodogram for one source's epoch photometry."""
    freqs, power = _lomb_scargle_spectrum(target)
    periods = 1.0 / freqs
    order = np.argsort(power)[::-1]
    periods = periods[order]
    power = power[order]
    max_power = np.max(power)
    near_max = np.where(power >= 0.98 * max_power)[0]
    if len(near_max) == 0:
        best_period = periods[int(np.argmax(power))]
    else:
        best_period = periods[int(near_max[np.argmax(periods[near_max])])]
    return periods, power, best_period


def _estimate_periods_from_epoch_photometry(data: table.Table) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the best Lomb-Scargle period for each source in a joined table."""
    source_column = data["source_id"]
    source_ids = np.unique(source_column)
    periods = np.empty(len(source_ids), dtype=float)
    for i, source_id in enumerate(source_ids):
        _, _, periods[i] = lomb_scargle_periodogram(data[source_column == source_id])
    return source_ids, periods


# Fourier-series light-curve modeling


def phase_fold(epochs: np.ndarray, period: float) -> np.ndarray:
    """Map times to phase in [0, 1)."""
    return (epochs % period) / period


def _build_fourier_matrix(epochs: Iterable[float], omega: float, k: int) -> np.ndarray:
    """Build the design matrix X for a Fourier series with known angular frequency."""
    epochs = np.asarray(epochs, dtype=float)
    period = 2.0 * np.pi / omega
    epochs_mod = epochs % period

    X = np.ones((len(epochs_mod), 2 * k + 1), dtype=float)
    for k in range(1, k + 1):
        X[:, 2 * k - 1] = np.cos(k * omega * epochs_mod)
        X[:, 2 * k] = np.sin(k * omega * epochs_mod)
    return X


def _fourier_predict(epoch_eval: Iterable[float], period: float, k: int, beta: np.ndarray) -> np.ndarray:
    omega = 2.0 * np.pi / period
    return _build_fourier_matrix(epoch_eval, omega, k) @ beta


@dataclass(frozen=True)
class FourierFit:
    source_id: int
    period: float
    K: int
    epochs: np.ndarray
    mags: np.ndarray
    mag_errs: np.ndarray
    beta: np.ndarray
    beta_cov: np.ndarray | None
    chi2_r: float
    classification: str | None = None

    def predict(self, epoch_eval: Iterable[float]) -> np.ndarray:
        return _fourier_predict(epoch_eval, self.period, self.K, self.beta)

    def predict_std(self, epoch_eval: Iterable[float]) -> np.ndarray:
        epoch_eval = np.atleast_1d(np.asarray(epoch_eval, dtype=float))

        omega = 2.0 * np.pi / self.period
        X_eval = _build_fourier_matrix(epoch_eval, omega, self.K)
        pred_var = np.einsum("ij,jk,ik->i", X_eval, self.beta_cov, X_eval)
        return np.sqrt(np.clip(pred_var, 0.0, None))


@dataclass(frozen=True)
class HarmonicCrossValidationResult:
    source_id: int
    period: float
    Ks: np.ndarray
    chi2r_train: np.ndarray
    chi2r_cv: np.ndarray
    best_K: int
    train_idx: np.ndarray
    cv_idx: np.ndarray
    classification: str


def fourier_fit(target: table.Table, period: float, k: int) -> FourierFit:
    """Fit a weighted Fourier series to one light curve with a fixed period."""
    source_id = target["source_id"][0]
    classification = target["best_classification"][0]
    epochs = target["g_transit_time"]
    mags = target["g_transit_mag"]
    mag_errs = target["g_transit_mag_err"]

    if len(epochs) <= 2 * k + 1:
        raise ValueError("Not enough epochs for the requested number of Fourier harmonics.")

    omega = 2.0 * np.pi / period
    X = _build_fourier_matrix(epochs, omega, k)
    weights = 1.0 / mag_errs
    beta, _, _, _ = np.linalg.lstsq(X * weights[:, None], mags * weights, rcond=None)

    resid = mags - X @ beta
    nu = len(epochs) - (2 * k + 1)
    chi2_r = float(np.sum((resid / mag_errs) ** 2) / nu)
    inv_var = 1.0 / np.square(mag_errs)
    normal_matrix = X.T @ (X * inv_var[:, None])
    try:
        beta_cov = np.linalg.inv(normal_matrix)
    except np.linalg.LinAlgError:
        beta_cov = np.linalg.pinv(normal_matrix)
    beta_cov = beta_cov * max(chi2_r, 1.0)

    return FourierFit(
        source_id=source_id,
        period=period,
        K=k,
        epochs=epochs,
        mags=mags,
        mag_errs=mag_errs,
        beta=beta,
        beta_cov=beta_cov,
        chi2_r=chi2_r,
        classification=classification,
    )


def cross_validate_harmonics(target: table.Table) -> HarmonicCrossValidationResult:
    """Cross-validate the harmonic order using the source's Lomb-Scargle period."""
    source_id = target["source_id"][0]
    classification = target["best_classification"][0]
    Ks = np.arange(1, 26, dtype=int)
    period = target["period_ls"][0]
    epochs = target["g_transit_time"]

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

        cv_epochs = target[cv_idx]["g_transit_time"]
        cv_mags = target[cv_idx]["g_transit_mag"]
        cv_mag_errs = target[cv_idx]["g_transit_mag_err"]
        resid_cv = cv_mags - fit.predict(cv_epochs)
        chi2r_cv[i] = float(np.sum((resid_cv / cv_mag_errs) ** 2) / len(cv_idx))

    best_k = Ks[int(np.nanargmin(chi2r_cv))]
    return HarmonicCrossValidationResult(
        source_id=source_id,
        period=period,
        Ks=Ks,
        chi2r_train=chi2r_train,
        chi2r_cv=chi2r_cv,
        best_K=best_k,
        train_idx=train_idx,
        cv_idx=cv_idx,
        classification=classification,
    )

def predict_future_magnitude(fit: FourierFit) -> tuple[float, float, float]:
    """Predict the magnitude and its uncertainty a fixed time after the last epoch."""
    epoch_last = float(np.max(fit.epochs))
    epoch_pred = epoch_last + 10.0
    mag_pred = float(fit.predict([epoch_pred])[0])
    mag_pred_err = float(fit.predict_std([epoch_pred])[0])
    return epoch_pred, mag_pred, mag_pred_err


def _fourier_mean_magnitude_from_beta(period: float, k: int, beta: np.ndarray) -> float:
    epoch_grid = np.linspace(0.0, period, 1000, endpoint=False)
    flux_grid = 10.0 ** (-0.4 * (_fourier_predict(epoch_grid, period, k, beta) - ZP_G))
    return -2.5 * np.log10(np.mean(flux_grid)) + ZP_G


def fourier_mean_magnitude(fit: FourierFit) -> float:
    """Compute the flux-space mean magnitude implied by a fitted Fourier model."""
    return _fourier_mean_magnitude_from_beta(fit.period, fit.K, fit.beta)


def fourier_mean_magnitude_error(fit: FourierFit) -> float:
    """Propagate fitted-coefficient covariance into the flux-space mean magnitude."""
    epoch_grid = np.linspace(0.0, fit.period, 1000, endpoint=False)
    omega = 2.0 * np.pi / fit.period
    X_grid = _build_fourier_matrix(epoch_grid, omega, fit.K)
    mag_grid = X_grid @ fit.beta
    flux_grid = 10.0 ** (-0.4 * (mag_grid - ZP_G))
    mean_flux = np.mean(flux_grid)

    grad = np.mean(X_grid * flux_grid[:, None], axis=0) / mean_flux
    mean_mag_var = grad @ fit.beta_cov @ grad
    return float(np.sqrt(np.clip(mean_mag_var, 0.0, None)))
