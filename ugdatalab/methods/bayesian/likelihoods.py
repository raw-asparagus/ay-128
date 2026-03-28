from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ugdatalab.methods.bayesian.base import GaussianLikelihood


@dataclass
class LinearGaussianLikelihood(GaussianLikelihood):
    """Linear model with Gaussian noise and intrinsic scatter.

    Model: y = a*x + b
    Noise: V_i = sigma_i^2 + sigma_s^2

    Parameters: a (slope), b (intercept), log10(sigma_s) (intrinsic scatter).

    The log-likelihood is:
        log L = -0.5 * sum_i [ log(2*pi*V_i) + (y_i - a*x_i - b)^2 / V_i ]
    """
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.y_err = np.asarray(self.y_err, dtype=float)

    @property
    def param_labels(self) -> list[str]:
        return [r"$a$", r"$b$", r"$\log_{10}\,\sigma_s$"]

    def _predict(self, x, theta):
        a, b, *_ = theta
        return a * np.asarray(x, dtype=float) + b

    def _inlier_variance(self, theta):
        _, _, log10_sig = theta
        return self.y_err**2 + (10.0**log10_sig)**2

    def _initial_guess(self):
        A = np.column_stack([self.x, np.ones_like(self.x)])
        a0, b0 = np.linalg.lstsq(A, self.y, rcond=None)[0]
        resid = self.y - (a0 * self.x + b0)
        sig_resid = max(np.std(resid), 1e-300)
        return np.array([a0, b0, np.log10(sig_resid)])

    def _prior_scales(self):
        """Data-driven weakly informative prior widths.

        The log10(sigma_s) width of 2.0 allows the intrinsic scatter to
        span approximately two orders of magnitude.
        """
        sy = float(np.std(self.y))
        sx = float(np.std(self.x))
        return np.array([2.5 * sy / sx, 2.5 * sy, 2.0])

    def _pymc_inlier_model(self, model):
        """Add linear inlier priors and return (mu_in, var_in)."""
        x = pt.as_tensor_variable(self.x)
        y_err = pt.as_tensor_variable(self.y_err)
        guess = self._initial_guess()
        scales = self._prior_scales()

        a = pm.Normal("a", mu=guess[0], sigma=scales[0])
        b = pm.Normal("b", mu=guess[1], sigma=scales[1])
        log10_sig = pm.Normal("log10_sig", mu=guess[2], sigma=scales[2])
        mu_in = a * x + b
        var_in = y_err**2 + (10.0**log10_sig)**2
        return mu_in, var_in

    def build_pymc(self):
        """Build a native PyMC model for NUTS sampling.

        Model: y ~ N(a*x + b, sigma_obs^2 + sigma_s^2)
        """
        with pm.Model() as model:
            mu_in, var_in = self._pymc_inlier_model(model)
            pm.Normal("obs", mu=mu_in, sigma=pt.sqrt(var_in), observed=self.y)
        return model
