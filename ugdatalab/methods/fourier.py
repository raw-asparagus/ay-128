from dataclasses import dataclass

import numpy as np

from ugdatalab.methods.base import Fit


def phase_fold(x: np.ndarray, period: float) -> np.ndarray:
    """Map values to phase in [0, 1)."""
    return (np.asarray(x, dtype=float) % period) / period


def _build_design_matrix(x: np.ndarray, omega: float, k: int) -> np.ndarray:
    """Build the Fourier design matrix for angular frequency *omega* and *k* harmonics."""
    x = np.asarray(x, dtype=float)
    period = 2.0 * np.pi / omega
    x_mod = x % period

    X = np.ones((len(x_mod), 2 * k + 1), dtype=float)
    for j in range(1, k + 1):
        X[:, 2 * j - 1] = np.cos(j * omega * x_mod)
        X[:, 2 * j] = np.sin(j * omega * x_mod)
    return X


@dataclass(frozen=True)
class FourierFit(Fit):
    """Result of a weighted least-squares Fourier fit.

    Attributes
    ----------
    period : float
        Fixed period used for the fit.
    k : int
        Number of harmonics.
    x, y, y_err : ndarray
        Input independent variable, dependent variable, and uncertainties.
    beta : ndarray
        Fitted coefficients (length 2k+1).
    beta_cov : ndarray or None
        Covariance matrix of *beta*.
    chi2_r : float
        Reduced chi-squared of the fit.
    """
    period: float
    k: int
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray
    beta: np.ndarray
    beta_cov: np.ndarray | None
    chi2_r: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict values at the given positions."""
        omega = 2.0 * np.pi / self.period
        X = _build_design_matrix(x, omega, self.k)
        return X @ self.beta

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        """Predict the standard deviation of the prediction at the given positions."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        omega = 2.0 * np.pi / self.period
        X = _build_design_matrix(x, omega, self.k)
        pred_var = np.einsum("ij,jk,ik->i", X, self.beta_cov, X)
        return np.sqrt(np.clip(pred_var, 0.0, None))


def fourier_fit(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    period: float,
    k: int,
) -> FourierFit:
    """Fit a weighted Fourier series with *k* harmonics at a fixed *period*.

    Parameters
    ----------
    x, y, y_err : array-like
        Independent variable, dependent variable, and uncertainties.
    period : float
        Known period.
    k : int
        Number of Fourier harmonics.

    Returns
    -------
    FourierFit
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)

    if len(x) <= 2 * k + 1:
        raise ValueError(
            "Not enough data points for the requested number of harmonics."
        )

    omega = 2.0 * np.pi / period
    X = _build_design_matrix(x, omega, k)
    weights = 1.0 / y_err
    beta, _, _, _ = np.linalg.lstsq(
        X * weights[:, None], y * weights, rcond=None,
    )

    resid = y - X @ beta
    nu = len(x) - (2 * k + 1)
    chi2_r = float(np.sum((resid / y_err) ** 2) / nu)

    inv_var = 1.0 / np.square(y_err)
    normal_matrix = X.T @ (X * inv_var[:, None])
    try:
        beta_cov = np.linalg.inv(normal_matrix)
    except np.linalg.LinAlgError:
        beta_cov = np.linalg.pinv(normal_matrix)
    beta_cov = beta_cov * max(chi2_r, 1.0)

    return FourierFit(
        period=period,
        k=k,
        x=x,
        y=y,
        y_err=y_err,
        beta=beta,
        beta_cov=beta_cov,
        chi2_r=chi2_r,
    )
