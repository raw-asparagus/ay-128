"""Concrete inlier and outlier component implementations.

:class:`LinearInlier` is a :class:`RegressionInlierComponent` for
1-D linear regression with intrinsic scatter and per-point x-error.
:class:`MultivariateLinearInlier` is its k-feature analogue without
x-error. :class:`GaussianOutlier` is a shape-agnostic
:class:`OutlierComponent` that pairs with any regression inlier.
"""

from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ugdatalab.methods.bayesian.base import (
    OutlierComponent,
    Parameter,
    RegressionInlierComponent,
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
class LinearInlier(RegressionInlierComponent):
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

    def predict_at(self, x, theta):
        """Return ``a * x + b`` for ``theta = (a, b, ...)``."""
        a, b, *_ = theta
        return a * x + b

    def total_variance_at(self, x, y_err, theta):
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
        # Publish the physical scatter so Posterior.samples is in σ-space,
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
# Multivariate linear inlier
# ---------------------------------------------------------------------------


@dataclass
class MultivariateLinearInlier(RegressionInlierComponent):
    """Linear inlier ``y = X @ a + b`` with intrinsic scatter, k-feature ``x``.

    Variance::

        V_i = y_err_i**2 + sigma_s**2

    No per-feature x-uncertainty term — :class:`LinearInlier` already
    covers the univariate ``(a * x_err)**2`` case; multidimensional
    x-error would require per-feature uncertainties and is not yet a
    use case.

    Attributes
    ----------
    n_features : int
        Number of features ``k``. Determines the slope-vector length
        in :attr:`parameters` and the prior structure in
        :meth:`build_pymc`. Must be set at construction because
        ``parameters`` cannot inspect the data.
    """
    n_features: int

    def __post_init__(self):
        """Validate n_features."""
        if self.n_features < 1:
            raise ValueError(
                f"n_features must be >= 1, got {self.n_features}."
            )

    @property
    def parameters(self) -> list[Parameter]:
        """Return ``a_0…a_{k-1}``, ``b``, ``sigma_s``."""
        slope_params = [
            Parameter(f"a_{i}", rf"$a_{{{i}}}$") for i in range(self.n_features)
        ]
        return slope_params + [
            Parameter("b", r"$b$"),
            Parameter("sigma_s", r"$\sigma_s$"),
        ]

    def predict_at(self, x, theta):
        """Return ``X @ a + b`` for ``theta = (a_0,…,a_{k-1}, b, sigma_s)``."""
        a = np.asarray(theta[:self.n_features], dtype=float)
        b = float(theta[self.n_features])
        return np.asarray(x, dtype=float) @ a + b

    def total_variance_at(self, x, y_err, theta):
        """Return ``y_err**2 + sigma_s**2`` (no x-uncertainty term)."""
        sigma_s = float(theta[self.n_features + 1])
        return y_err**2 + sigma_s**2

    def build_pymc(self, x, y, y_err):
        """Add multivariate-linear-inlier priors to the active PyMC model and return tensors."""
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.n_features:
            raise ValueError(
                f"x must have shape (n, {self.n_features}); got {x_arr.shape}."
            )
        x_t = pt.as_tensor_variable(x_arr)
        y_err_t = pt.as_tensor_variable(y_err)
        guess_a, guess_b, guess_log10_sig = self._initial_guess(x_arr, y)
        scale_a, scale_b, scale_log10_sig = self._prior_scales(x_arr, y)

        a = pm.Normal("a", mu=guess_a, sigma=scale_a, shape=self.n_features)
        b = pm.Normal("b", mu=guess_b, sigma=scale_b)
        log10_sig = pm.Normal("log10_sig", mu=guess_log10_sig, sigma=scale_log10_sig)
        # Publish physical scatter so Posterior.samples is in σ-space, not log10(σ).
        sigma_s = pm.Deterministic("sigma_s", 10.0**log10_sig)
        # ``a`` is published per-component as ``a_0…a_{k-1}`` so trace
        # extraction lines up with :attr:`parameters`.
        for i in range(self.n_features):
            pm.Deterministic(f"a_{i}", a[i])
        mu_in = pt.dot(x_t, a) + b
        var_in = y_err_t**2 + sigma_s**2
        return mu_in, var_in

    def _initial_guess(self, x: np.ndarray, y: np.ndarray):
        """Return an OLS-based starting point ``(a, b, log10(std(resid)))``."""
        design = np.column_stack([x, np.ones(x.shape[0])])
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        a0, b0 = coeffs[:self.n_features], coeffs[self.n_features]
        resid = y - (x @ a0 + b0)
        sig_resid = max(np.std(resid), _STD_FLOOR)
        return a0, float(b0), float(np.log10(sig_resid))

    def _prior_scales(self, x: np.ndarray, y: np.ndarray):
        """Return data-driven weakly informative prior widths."""
        sy = float(np.std(y))
        sx_per_feature = np.std(x, axis=0)
        sx_per_feature = np.where(sx_per_feature > 0, sx_per_feature, 1.0)
        return (
            _PRIOR_SCALE_SLOPE_INTERCEPT * sy / sx_per_feature,
            _PRIOR_SCALE_SLOPE_INTERCEPT * sy,
            _PRIOR_LOG10_SIG_WIDTH,
        )


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
