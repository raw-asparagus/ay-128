from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from astropy import table

from ugdatalab.models.gaia import ZP_G


def _as_float_array(column) -> np.ndarray:
    data = np.asarray(column)
    if hasattr(data, "filled"):
        data = data.filled(np.nan)
    return np.asarray(data, dtype=float)


def phase_fold(epoch: np.ndarray, period: float) -> np.ndarray:
    """Map times to phase in [0, 1)."""
    epoch = np.asarray(epoch, dtype=float)
    return (epoch % float(period)) / float(period)


def build_fourier_matrix(
    epoch: Iterable[float],
    omega: float,
    k_harmonics: int | None = None,
    *,
    K: int | None = None,
) -> np.ndarray:
    """Return the Fourier design matrix for an angular frequency and harmonic order."""
    if k_harmonics is None:
        if K is None:
            raise TypeError("Provide `k_harmonics` (or the alias `K`).")
        k_harmonics = int(K)
    elif K is not None and int(K) != int(k_harmonics):
        raise ValueError("`k_harmonics` and `K` disagree.")
    if k_harmonics < 0:
        raise ValueError("k_harmonics must be non-negative.")

    epochs = np.asarray(epoch, dtype=float)
    period = 2.0 * np.pi / float(omega)
    epochs_mod = epochs % period

    X = np.ones((len(epochs_mod), 2 * k_harmonics + 1), dtype=float)
    for k in range(1, k_harmonics + 1):
        X[:, 2 * k - 1] = np.cos(k * omega * epochs_mod)
        X[:, 2 * k] = np.sin(k * omega * epochs_mod)
    return X


@dataclass(frozen=True)
class FourierFitResult:
    period: float
    k_harmonics: int
    epochs: np.ndarray
    mags: np.ndarray
    mag_errs: np.ndarray
    beta: np.ndarray
    chi2_r: float

    @property
    def omega(self) -> float:
        return 2.0 * np.pi / self.period

    def predict(self, epoch_eval: Iterable[float]) -> np.ndarray:
        return build_fourier_matrix(epoch_eval, self.omega, self.k_harmonics) @ self.beta


@dataclass(frozen=True)
class FourierCrossValidationResult:
    k_values: np.ndarray
    chi2r_train: np.ndarray
    chi2r_cv: np.ndarray
    best_k: int
    cv_fraction: float
    train_idx: np.ndarray
    cv_idx: np.ndarray

    @property
    def K_range(self) -> np.ndarray:
        return self.k_values

    @property
    def best_K(self) -> int:
        return self.best_k


FourierCVResult = FourierCrossValidationResult


def fourier_fit(target: table.Table, period: float, k_harmonics: int) -> FourierFitResult:
    """Weighted least-squares Fourier series fit to Gaia epoch photometry."""
    epochs = _as_float_array(target["g_transit_time"])
    mags = _as_float_array(target["g_transit_mag"])
    mag_errs = _as_float_array(target["g_transit_mag_err"])
    valid = np.isfinite(epochs) & np.isfinite(mags) & np.isfinite(mag_errs) & (mag_errs > 0)

    epochs = epochs[valid]
    mags = mags[valid]
    mag_errs = mag_errs[valid]

    if len(epochs) <= 2 * k_harmonics + 1:
        raise ValueError("Not enough epochs for the requested number of Fourier harmonics.")

    omega = 2.0 * np.pi / float(period)
    X = build_fourier_matrix(epochs, omega, k_harmonics)
    weights = 1.0 / mag_errs
    beta, _, _, _ = np.linalg.lstsq(X * weights[:, None], mags * weights, rcond=None)

    resid = mags - X @ beta
    nu = len(epochs) - (2 * k_harmonics + 1)
    chi2_r = float(np.sum((resid / mag_errs) ** 2) / nu)

    return FourierFitResult(
        period=float(period),
        k_harmonics=int(k_harmonics),
        epochs=epochs,
        mags=mags,
        mag_errs=mag_errs,
        beta=beta,
        chi2_r=chi2_r,
    )


def cross_validate_harmonics(
    target: table.Table,
    period: float,
    k_values: Iterable[int] | None = None,
    cv_fraction: float = 0.2,
    seed: int = 42,
    selection_tolerance: float = 0.05,
) -> FourierCrossValidationResult:
    """Cross-validate Fourier harmonic order using held-out epoch photometry."""
    if not (0.0 < cv_fraction < 1.0):
        raise ValueError("cv_fraction must lie strictly between 0 and 1.")

    if k_values is None:
        k_values = range(1, 26)

    k_values = np.asarray(list(k_values), dtype=int)
    epochs = _as_float_array(target["g_transit_time"])
    mags = _as_float_array(target["g_transit_mag"])
    mag_errs = _as_float_array(target["g_transit_mag_err"])
    valid = np.isfinite(epochs) & np.isfinite(mags) & np.isfinite(mag_errs) & (mag_errs > 0)
    clean = target[valid]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(clean))
    n_cv = max(1, int(round(cv_fraction * len(clean))))
    cv_idx = idx[:n_cv]
    train_idx = idx[n_cv:]
    if len(train_idx) == 0:
        raise ValueError("Cross-validation split left no training data.")

    chi2r_train = np.full(len(k_values), np.nan, dtype=float)
    chi2r_cv = np.full(len(k_values), np.nan, dtype=float)

    for i, k_harmonics in enumerate(k_values):
        if len(train_idx) <= 2 * k_harmonics + 1:
            continue
        fit = fourier_fit(clean[train_idx], period=period, k_harmonics=int(k_harmonics))
        chi2r_train[i] = fit.chi2_r

        cv_epochs = _as_float_array(clean[cv_idx]["g_transit_time"])
        cv_mags = _as_float_array(clean[cv_idx]["g_transit_mag"])
        cv_mag_errs = _as_float_array(clean[cv_idx]["g_transit_mag_err"])
        resid_cv = cv_mags - fit.predict(cv_epochs)
        chi2r_cv[i] = float(np.sum((resid_cv / cv_mag_errs) ** 2) / len(cv_idx))

    if not np.isfinite(chi2r_cv).any():
        raise ValueError("No valid harmonic orders for cross-validation.")

    best_idx = int(np.nanargmin(chi2r_cv))
    best_score = float(chi2r_cv[best_idx])
    tolerance_mask = np.isfinite(chi2r_cv) & (chi2r_cv <= best_score * (1.0 + float(selection_tolerance)))
    best_k = int(np.min(k_values[tolerance_mask]))
    return FourierCrossValidationResult(
        k_values=k_values,
        chi2r_train=chi2r_train,
        chi2r_cv=chi2r_cv,
        best_k=best_k,
        cv_fraction=float(cv_fraction),
        train_idx=np.asarray(train_idx, dtype=int),
        cv_idx=np.asarray(cv_idx, dtype=int),
    )


def predict_future_magnitude(
    fit: FourierFitResult,
    days_after_last: float = 10.0,
) -> tuple[float, float]:
    """Predict the magnitude a fixed number of days after the last observed epoch."""
    epoch_last = float(np.max(fit.epochs))
    epoch_pred = epoch_last + float(days_after_last)
    mag_pred = float(fit.predict([epoch_pred])[0])
    return epoch_pred, mag_pred


def fourier_mean_magnitude(
    fit: FourierFitResult,
    n_phase_samples: int = 1000,
    zero_point_g: float = ZP_G,
) -> float:
    """Flux-space mean magnitude implied by a fitted Fourier model over one period."""
    epoch_grid = np.linspace(0.0, fit.period, int(n_phase_samples), endpoint=False)
    flux_grid = 10.0 ** (-0.4 * (fit.predict(epoch_grid) - float(zero_point_g)))
    return float(-2.5 * np.log10(np.mean(flux_grid)) + float(zero_point_g))


def estimate_fourier_mean_magnitudes(
    data: table.Table,
    periods: np.ndarray,
    k_harmonics: int | None = None,
    source_ids: Iterable[int] | None = None,
    zero_point_g: float = ZP_G,
    *,
    K: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Fourier-model mean magnitudes for many stars in a joined epoch table."""
    if k_harmonics is None:
        if K is None:
            raise TypeError("Provide `k_harmonics` (or the alias `K`).")
        k_harmonics = int(K)
    if source_ids is None:
        source_ids = np.unique(np.asarray(data["source_id"], dtype=np.int64))

    source_ids = np.asarray(list(source_ids), dtype=np.int64)
    periods = np.asarray(periods, dtype=float)
    if len(periods) != len(source_ids):
        raise ValueError("periods must align one-to-one with source_ids.")

    mean_mags = np.empty(len(source_ids), dtype=float)
    source_column = np.asarray(data["source_id"], dtype=np.int64)

    for i, source_id in enumerate(source_ids):
        star = data[source_column == source_id]
        fit = fourier_fit(star, period=float(periods[i]), k_harmonics=int(k_harmonics))
        mean_mags[i] = fourier_mean_magnitude(fit, zero_point_g=zero_point_g)

    return source_ids, mean_mags
