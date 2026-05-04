"""Concrete Bayesian likelihood implementations (linear, etc.)."""

from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ugdatalab.methods.bayesian.base import GaussianLikelihood


# Prior-scale multipliers (data-driven weakly informative widths).
# 2.5σ Normal covers ~99% of the data range under the slope/intercept.
_PRIOR_SCALE_SLOPE_INTERCEPT = 2.5
# log10(σ_s) prior width of 2.0 lets intrinsic scatter span ~2 decades.
_PRIOR_LOG10_SIG_WIDTH = 2.0
# Numerical floor to avoid log(0) when residuals are perfectly fit.
_LOG_FLOOR = 1e-300


@dataclass
class LinearGaussianLikelihood(GaussianLikelihood):
    """Linear model with Gaussian noise and intrinsic scatter.

    Model::

        y = a * x + b
        V_i = sigma_yi**2 + sigma_s**2 + (a * sigma_xi)**2

    Posterior parameters exposed in ``MCMCResult.samples`` / ``theta``:
    ``a`` (slope), ``b`` (intercept), ``sigma_s`` (intrinsic scatter).

    Attributes
    ----------
    x, y, y_err : ndarray
        Independent variable, dependent variable, and uncertainties.
    x_err : ndarray, optional
        Uncertainties on ``x``; defaults to zeros.
    """
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray
    x_err: np.ndarray = None

    def __post_init__(self):
        """Coerce x, y, y_err, and x_err to float ndarrays, defaulting x_err to zeros."""
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.y_err = np.asarray(self.y_err, dtype=float)
        self.x_err = np.asarray(self.x_err, dtype=float) if self.x_err is not None else np.zeros_like(self.x)

    @property
    def param_labels(self) -> list[str]:
        """Return LaTeX labels for slope, intercept, intrinsic scatter."""
        return [r"$a$", r"$b$", r"$\sigma_s$"]

    @property
    def physical_param_names(self) -> list[str]:
        """Return physical parameter names to extract from the trace."""
        return ["a", "b", "sigma_s"]

    def predict(self, x, theta):
        """Return ``a * x + b`` for ``theta = (a, b, ...)``."""
        a, b, *_ = theta
        return a * np.asarray(x, dtype=float) + b

    def total_variance(self, theta):
        """Return ``y_err**2 + sigma_s**2 + (a * x_err)**2``."""
        a, _, sigma_s = theta
        return self.y_err**2 + sigma_s**2 + (a * self.x_err)**2

    def _initial_guess(self):
        """Return an OLS-based starting point ``(a, b, log10(std(resid)))``."""
        A = np.column_stack([self.x, np.ones_like(self.x)])
        a0, b0 = np.linalg.lstsq(A, self.y, rcond=None)[0]
        resid = self.y - (a0 * self.x + b0)
        sig_resid = max(np.std(resid), _LOG_FLOOR)
        return np.array([a0, b0, np.log10(sig_resid)])

    def _prior_scales(self):
        """Return data-driven weakly informative prior widths for ``(a, b, log10_sig)``."""
        sy = float(np.std(self.y))
        sx = float(np.std(self.x))
        return np.array([
            _PRIOR_SCALE_SLOPE_INTERCEPT * sy / sx,
            _PRIOR_SCALE_SLOPE_INTERCEPT * sy,
            _PRIOR_LOG10_SIG_WIDTH,
        ])

    def _pymc_inlier_model(self, model):
        """Add linear inlier priors to *model* and return ``(mu_in, var_in)``."""
        x = pt.as_tensor_variable(self.x)
        y_err = pt.as_tensor_variable(self.y_err)
        guess = self._initial_guess()
        scales = self._prior_scales()

        a = pm.Normal("a", mu=guess[0], sigma=scales[0])
        b = pm.Normal("b", mu=guess[1], sigma=scales[1])
        log10_sig = pm.Normal("log10_sig", mu=guess[2], sigma=scales[2])
        # Publish the physical scatter so MCMCResult.samples is in σ-space,
        # not log10(σ)-space. ``log10_sig`` remains the sampled internal RV;
        # users never see it.
        sigma_s = pm.Deterministic("sigma_s", 10.0**log10_sig)
        mu_in = a * x + b
        x_err = pt.as_tensor_variable(self.x_err)
        var_in = y_err**2 + sigma_s**2 + (a * x_err)**2
        return mu_in, var_in

    def build_pymc(self):
        """Return a PyMC model with ``y ~ N(a*x + b, sqrt(var_in))``."""
        with pm.Model() as model:
            mu_in, var_in = self._pymc_inlier_model(model)
            pm.Normal("obs", mu=mu_in, sigma=pt.sqrt(var_in), observed=self.y)
        return model
