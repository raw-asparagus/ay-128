"""Concrete ``InlierComponent`` and ``OutlierComponent`` implementations."""

from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ugdatalab.methods.bayesian.base import (
    InlierComponent,
    OutlierComponent,
    Parameter,
)


# ---------------------------------------------------------------------------
# Linear inlier
# ---------------------------------------------------------------------------

# Prior-scale multipliers (data-driven weakly informative widths).
# 2.5σ Normal covers ~99% of the data range under the slope/intercept.
_PRIOR_SCALE_SLOPE_INTERCEPT = 2.5
# log10(σ_s) prior width of 2.0 lets intrinsic scatter span ~2 decades.
_PRIOR_LOG10_SIG_WIDTH = 2.0
# Floor on residual std before taking log10, to avoid log(0) when the
# OLS solution fits the data exactly.
_STD_FLOOR = 1e-300


@dataclass
class LinearInlier(InlierComponent):
    """Linear inlier ``y = a x + b`` with intrinsic scatter and per-point x-error.

    Variance::

        V_i = y_err_i**2 + sigma_s**2 + (a * x_err_i)**2

    Attributes
    ----------
    x_err : ndarray
        Per-point uncertainties on ``x``. Pass ``np.zeros_like(x)`` for
        the no-x-error special case.
    """
    x_err: np.ndarray

    def __post_init__(self):
        """Coerce x_err to a float ndarray."""
        self.x_err = np.asarray(self.x_err, dtype=float)

    @property
    def parameters(self) -> list[Parameter]:
        """Return ``(name, label)`` for slope, intercept, intrinsic scatter."""
        return [
            Parameter("a", r"$a$"),
            Parameter("b", r"$b$"),
            Parameter("sigma_s", r"$\sigma_s$"),
        ]

    def predict(self, x, theta):
        """Return ``a * x + b`` for ``theta = (a, b, ...)``."""
        a, b, *_ = theta
        return a * x + b

    def total_variance(self, x, y_err, theta):
        """Return ``y_err**2 + sigma_s**2 + (a * x_err)**2``."""
        a, _, sigma_s = theta
        return y_err**2 + sigma_s**2 + (a * self.x_err)**2

    def build_pymc(self, x, y, y_err):
        """Add linear-inlier priors to the active PyMC model and return tensors."""
        x_t = pt.as_tensor_variable(x)
        y_err_t = pt.as_tensor_variable(y_err)
        guess = self._initial_guess(x, y)
        scales = self._prior_scales(x, y)

        a = pm.Normal("a", mu=guess[0], sigma=scales[0])
        b = pm.Normal("b", mu=guess[1], sigma=scales[1])
        log10_sig = pm.Normal("log10_sig", mu=guess[2], sigma=scales[2])
        # Publish the physical scatter so MCMCResult.samples is in σ-space,
        # not log10(σ)-space. ``log10_sig`` remains the sampled internal RV;
        # users never see it.
        sigma_s = pm.Deterministic("sigma_s", 10.0**log10_sig)
        mu_in = a * x_t + b
        x_err_t = pt.as_tensor_variable(self.x_err)
        var_in = y_err_t**2 + sigma_s**2 + (a * x_err_t)**2
        return mu_in, var_in

    def _initial_guess(self, x, y):
        """Return an OLS-based starting point ``(a, b, log10(std(resid)))``."""
        A = np.column_stack([x, np.ones_like(x)])
        a0, b0 = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - (a0 * x + b0)
        sig_resid = max(np.std(resid), _STD_FLOOR)
        return np.array([a0, b0, np.log10(sig_resid)])

    def _prior_scales(self, x, y):
        """Return data-driven weakly informative prior widths for ``(a, b, log10_sig)``."""
        sy = float(np.std(y))
        sx = float(np.std(x))
        return np.array([
            _PRIOR_SCALE_SLOPE_INTERCEPT * sy / sx,
            _PRIOR_SCALE_SLOPE_INTERCEPT * sy,
            _PRIOR_LOG10_SIG_WIDTH,
        ])


# ---------------------------------------------------------------------------
# Gaussian outlier
# ---------------------------------------------------------------------------

# Outlier background and prior on the inlier fraction.
# Both are deliberately wide (weakly informative).
_OUTLIER_WIDTH_STDS = 3.0          # background σ as a multiple of std(y)
_LOGIT_F_PRIOR_MU = 0.0            # logit(0.5) — neutral on inlier fraction
_LOGIT_F_PRIOR_SIGMA = 3.0         # wide → essentially uniform on f


@dataclass
class GaussianOutlier(OutlierComponent):
    """Broad-Gaussian outlier component with logit-Normal mixture weight.

    Adds a ``N(mu_bg, y_err**2 + sig_bg**2)`` background mixed with the
    inlier component via a sigmoid-transformed Normal weight ``f``,
    where ``mu_bg = median(y)`` and ``sig_bg = 3 * std(y)``.
    """

    @staticmethod
    def _background_stats(y: np.ndarray) -> tuple[float, float]:
        """Return ``(mu_bg, sig_bg)`` for the broad-Gaussian background."""
        return float(np.median(y)), float(_OUTLIER_WIDTH_STDS * np.std(y))

    def build_pymc_mixture(self, y, y_err, mu_in, var_in):
        """Add outlier RVs and a logaddexp mixture potential to the active model."""
        y_t = pt.as_tensor_variable(y)
        y_err_t = pt.as_tensor_variable(y_err)
        mu_bg, sig_bg = self._background_stats(y)

        logit_f = pm.Normal("logit_f", mu=_LOGIT_F_PRIOR_MU, sigma=_LOGIT_F_PRIOR_SIGMA)
        f = pm.math.sigmoid(logit_f)
        pm.Deterministic("f", f)
        var_out = y_err_t**2 + sig_bg**2

        ll_in = pt.log(f) - 0.5 * (pt.log(2 * np.pi * var_in) + (y_t - mu_in)**2 / var_in)
        ll_out = pt.log(1 - f) - 0.5 * (pt.log(2 * np.pi * var_out) + (y_t - mu_bg)**2 / var_out)
        pm.Potential("mixture_loglike", pt.sum(pt.logaddexp(ll_in, ll_out)))

    def inlier_probs(self, y, y_err, mu_in_samples, var_in_samples, f_samples):
        """Return per-point posterior inlier probabilities.

        For each draw ``s`` and point ``i``::

            r_si = f_s * N_in / (f_s * N_in + (1 - f_s) * N_out)

        Returns ``mean_s(r_si)``.
        """
        mu_bg, sig_bg = self._background_stats(y)
        var_out = y_err**2 + sig_bg**2

        f = f_samples[:, None]
        ll_in = np.log(f) - 0.5 * (
            np.log(2 * np.pi * var_in_samples) + (y - mu_in_samples)**2 / var_in_samples
        )
        ll_out = np.log(1 - f) - 0.5 * (
            np.log(2 * np.pi * var_out) + (y - mu_bg)**2 / var_out
        )
        responsibilities = np.exp(ll_in - np.logaddexp(ll_in, ll_out))
        return responsibilities.mean(axis=0)
