"""Weighted least-squares Fourier fitting at a known period."""

from dataclasses import dataclass

import numpy as np

from ugdatalab.methods.base import DataFit


# ---------------------------------------------------------------------------
# Phase and design-matrix helpers
# ---------------------------------------------------------------------------


def phase_fold(x: np.ndarray, period: float) -> np.ndarray:
    """Map values to phase in [0, 1).

    Parameters
    ----------
    x : array-like
        Times (or other periodic coordinate) in the same units as *period*.
    period : float
        Folding period.

    Returns
    -------
    ndarray
        Phase values ``(x mod period) / period``, same shape as *x*.
    """
    return (np.asarray(x, dtype=float) % period) / period


def build_design_matrix(x: np.ndarray, omega: float, k: int) -> np.ndarray:
    """Build the Fourier design matrix for angular frequency *omega* and *k* harmonics.

    Parameters
    ----------
    x : array-like
        Independent-variable positions.
    omega : float
        Angular frequency ``2*pi/period``.
    k : int
        Number of harmonics (the matrix has ``2*k + 1`` columns: a constant
        term plus cos/sin pairs for each harmonic).

    Returns
    -------
    ndarray, shape (len(x), 2*k + 1)
        Design matrix with columns ``[1, cos(omega*x), sin(omega*x),
        cos(2*omega*x), sin(2*omega*x), ...]``.
    """
    x = np.asarray(x, dtype=float)
    period = 2.0 * np.pi / omega
    x_mod = x % period

    X = np.ones((len(x_mod), 2 * k + 1), dtype=float)
    for j in range(1, k + 1):
        X[:, 2 * j - 1] = np.cos(j * omega * x_mod)
        X[:, 2 * j] = np.sin(j * omega * x_mod)
    return X


# ---------------------------------------------------------------------------
# Fit result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FourierFit(DataFit):
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
        Fitted coefficients (length ``2*k + 1``).
    beta_cov : ndarray or None
        Covariance matrix of ``beta``.
    """
    period: float
    k: int
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray
    beta: np.ndarray
    beta_cov: np.ndarray | None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted Fourier series at *x*.

        Parameters
        ----------
        x : array-like
            Positions at which to evaluate the model.

        Returns
        -------
        ndarray
            Predicted y values, same shape as *x*.
        """
        omega = 2.0 * np.pi / self.period
        X = build_design_matrix(x, omega, self.k)
        return X @ self.beta

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        """Compute the std of the mean prediction at *x*.

        Returns ``sqrt(diag(X @ beta_cov @ X.T))`` where ``X`` is the
        Fourier design matrix at *x*. Reflects coefficient uncertainty
        only, not data scatter.

        Parameters
        ----------
        x : array-like
            Positions at which to evaluate the predictive std.

        Returns
        -------
        ndarray
            Standard deviation of the mean prediction.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        omega = 2.0 * np.pi / self.period
        X = build_design_matrix(x, omega, self.k)
        pred_var = np.einsum("ij,jk,ik->i", X, self.beta_cov, X)
        return np.sqrt(np.clip(pred_var, 0.0, None))

    @property
    def n_params(self) -> int:
        """Return the number of free coefficients (``2*k + 1``)."""
        return 2 * self.k + 1


# ---------------------------------------------------------------------------
# Fit driver
# ---------------------------------------------------------------------------


def fourier_fit(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    period: float,
    k: int,
) -> FourierFit:
    """Fit a weighted Fourier series with *k* harmonics at a fixed *period*.

    The coefficient covariance is rescaled by ``max(chi2_r, 1)``.

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

    Raises
    ------
    ValueError
        If ``len(x) <= 2 * k + 1``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)

    if len(x) <= 2 * k + 1:
        raise ValueError(
            "Not enough data points for the requested number of harmonics."
        )

    omega = 2.0 * np.pi / period
    X = build_design_matrix(x, omega, k)
    weights = 1.0 / y_err
    beta, _, _, _ = np.linalg.lstsq(
        X * weights[:, None], y * weights, rcond=None,
    )

    inv_var = 1.0 / np.square(y_err)
    normal_matrix = X.T @ (X * inv_var[:, None])
    try:
        beta_cov = np.linalg.inv(normal_matrix)
    except np.linalg.LinAlgError:
        beta_cov = np.linalg.pinv(normal_matrix)

    # Inflate covariance by chi²_r when the fit is worse than expected under
    # the stated y_err — a standard rescaling so beta_cov reflects observed
    # scatter rather than overconfident formal errors.
    resid = y - X @ beta
    nu = len(x) - (2 * k + 1)
    chi2_r = float(np.sum((resid / y_err) ** 2) / nu)
    beta_cov = beta_cov * max(chi2_r, 1.0)

    return FourierFit(
        period=period, k=k, x=x, y=y, y_err=y_err,
        beta=beta, beta_cov=beta_cov,
    )
