from abc import ABC, abstractmethod

import numpy as np


class Fit(ABC):
    """Base class for fitted models."""
    chi2_r: float

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray: ...


class Likelihood(ABC):
    """Base class for likelihood objects.

    Subclasses hold data (x, y, y_err) and expose methods consumed by engines:
    - build_pymc()          — for plain NUTS parameter estimation
    - build_pymc_mixture()  — for mixture contamination with outlier rejection
    - inlier_probs()        — per-point inlier probabilities from mixture posterior
    """
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray

    @abstractmethod
    def inlier_probs(self, trace, model_var_names: list[str]) -> np.ndarray:
        """Compute per-point inlier probabilities from posterior samples."""
        ...

    @abstractmethod
    def build_pymc(self):
        """Build a PyMC model for NUTS sampling."""
        ...

    @abstractmethod
    def build_pymc_mixture(self):
        """Build a PyMC mixture model with an outlier component for NUTS sampling."""
        ...


class GaussianLikelihood(Likelihood):
    """Intermediate base for likelihoods with Gaussian noise and a Gaussian
    outlier background."""

    @property
    def mu_bg(self) -> float:
        """Background mean, fixed to median(y)."""
        return float(np.median(self.y))

    @property
    def sig_bg(self) -> float:
        """Background width, fixed to 3 * MAD * 1.4826."""
        return float(3.0 * np.median(np.abs(self.y - self.mu_bg)) * 1.4826)

    @abstractmethod
    def _predict(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Predict y values at *x* for parameter vector *theta*."""
        ...

    @abstractmethod
    def _inlier_variance(self, theta: np.ndarray) -> np.ndarray:
        """Per-point total variance for the inlier component given *theta*."""
        ...

    @abstractmethod
    def _pymc_inlier_model(self, model):
        """Add inlier priors/variables to a pm.Model. Returns (mu_in, var_in)."""
        ...

    def inlier_probs(self, trace, model_var_names) -> np.ndarray:
        """Compute per-point inlier probabilities from posterior samples.

        For each draw s and point i:
            r_si = f_s * N_in / (f_s * N_in + (1-f_s) * N_out)
        Returns mean_s(r_si).
        """
        posterior = trace.posterior
        param_samples = np.column_stack([
            posterior[name].values.flatten() for name in model_var_names
        ])
        f_samples = posterior["f"].values.flatten()

        var_out = self.y_err**2 + self.sig_bg**2
        n_samples = len(f_samples)
        responsibilities = np.empty((n_samples, len(self.x)), dtype=float)

        for s in range(n_samples):
            theta_s = param_samples[s]
            y_pred = self._predict(self.x, theta_s)
            var_in = self._inlier_variance(theta_s)
            f = f_samples[s]

            ll_in = np.log(f) - 0.5 * (np.log(2 * np.pi * var_in) + (self.y - y_pred)**2 / var_in)
            ll_out = np.log(1 - f) - 0.5 * (np.log(2 * np.pi * var_out) + (self.y - self.mu_bg)**2 / var_out)
            responsibilities[s] = np.exp(ll_in - np.logaddexp(ll_in, ll_out))

        return responsibilities.mean(axis=0)

    def build_pymc_mixture(self):
        """Build a PyMC two-component Gaussian mixture model for NUTS sampling.

        Inlier component defined by _pymc_inlier_model().
        Outlier: N(mu_bg, sigma_obs^2 + sigma_bg^2).
        """
        import pymc as pm
        import pytensor.tensor as pt

        y = pt.as_tensor_variable(self.y)
        y_err = pt.as_tensor_variable(self.y_err)

        with pm.Model() as model:
            mu_in, var_in = self._pymc_inlier_model(model)
            f = pm.Beta("f", alpha=2, beta=1)
            var_out = y_err**2 + self.sig_bg**2

            ll_in = pt.log(f) - 0.5 * (pt.log(2 * np.pi * var_in) + (y - mu_in)**2 / var_in)
            ll_out = pt.log(1 - f) - 0.5 * (pt.log(2 * np.pi * var_out) + (y - self.mu_bg)**2 / var_out)
            pm.Potential("mixture_loglike", pt.sum(pt.logaddexp(ll_in, ll_out)))

        return model
