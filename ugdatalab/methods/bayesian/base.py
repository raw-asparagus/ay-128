"""Base classes for Bayesian likelihoods consumable by the NUTS sampler.

``Likelihood`` is the minimal contract. ``MixtureLikelihood`` adds a
two-component outlier-mixture contract. ``GaussianLikelihood`` provides
a shared inlier+outlier Gaussian-mixture implementation.
"""

from abc import ABC, abstractmethod

import numpy as np
import pymc as pm
import pytensor.tensor as pt


# GaussianLikelihood mixture model — outlier background and prior on
# the inlier fraction. Both are deliberately wide (weakly informative).
_OUTLIER_WIDTH_STDS = 3.0          # background σ as a multiple of std(y)
_LOGIT_F_PRIOR_MU = 0.0            # logit(0.5) — neutral on inlier fraction
_LOGIT_F_PRIOR_SIGMA = 3.0         # wide → essentially uniform on f


# ---------------------------------------------------------------------------
# Likelihood ABCs
# ---------------------------------------------------------------------------


class Likelihood(ABC):
    """Base class for likelihood objects consumable by ``nuts_sample``.

    Subclasses hold the data ``(x, y, y_err)`` and implement
    ``build_pymc``, ``predict``, ``param_labels``, and
    ``physical_param_names``; ``total_variance`` defaults to
    ``y_err ** 2`` and may be overridden.
    """
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray

    @property
    @abstractmethod
    def param_labels(self) -> list[str]:
        """Return LaTeX-formatted parameter labels in physical units."""
        ...

    @property
    @abstractmethod
    def physical_param_names(self) -> list[str]:
        """Return trace variable names to extract into ``samples``/``theta``.

        Order matches ``param_labels`` and defines the *theta* component
        indexing consumed by ``predict`` and ``total_variance``.
        """
        ...

    @abstractmethod
    def build_pymc(self):
        """Return a PyMC model ready for NUTS sampling."""
        ...

    @abstractmethod
    def predict(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return the noiseless model prediction at *x* for *theta*."""
        ...

    def total_variance(self, theta: np.ndarray) -> np.ndarray:
        """Return per-point predictive variance at ``self.x`` for *theta*.

        Default implementation returns ``y_err ** 2``. Subclasses with
        intrinsic scatter or x-uncertainty override.
        """
        return self.y_err ** 2


class MixtureLikelihood(Likelihood):
    """Likelihood with a two-component (inlier + outlier) mixture contract.

    Subclasses build a PyMC mixture model and report per-point posterior
    inlier probabilities; consumed by ``mixture_contamination``.
    """

    @abstractmethod
    def build_pymc_mixture(self):
        """Return a PyMC inlier+outlier mixture model for NUTS sampling."""
        ...

    @abstractmethod
    def inlier_probs(self, trace, model_var_names: list[str]) -> np.ndarray:
        """Return per-point posterior inlier probabilities from a trace."""
        ...


# ---------------------------------------------------------------------------
# Gaussian inlier+outlier mixture base
# ---------------------------------------------------------------------------


class GaussianLikelihood(MixtureLikelihood):
    """Intermediate base for likelihoods with Gaussian noise and Gaussian outliers."""

    @property
    def mu_bg(self) -> float:
        """Return the background mean, fixed to ``median(y)``."""
        return float(np.median(self.y))

    @property
    def sig_bg(self) -> float:
        """Return the background width, fixed to ``_OUTLIER_WIDTH_STDS * std(y)``."""
        return float(_OUTLIER_WIDTH_STDS * np.std(self.y))

    @abstractmethod
    def _pymc_inlier_model(self, model):
        """Add inlier priors/variables to *model* and return ``(mu_in, var_in)``."""
        ...

    def inlier_probs(self, trace, model_var_names) -> np.ndarray:
        """Return per-point posterior inlier probabilities.

        For each draw ``s`` and point ``i``::

            r_si = f_s * N_in / (f_s * N_in + (1 - f_s) * N_out)

        Returns ``mean_s(r_si)``.

        Parameters
        ----------
        trace : arviz.InferenceData
            Posterior trace from ``pm.sample``.
        model_var_names : list of str
            Names of physical inlier parameters to read from
            ``trace.posterior``.

        Returns
        -------
        ndarray, shape (len(self.x),)
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
            y_pred = self.predict(self.x, theta_s)
            var_in = self.total_variance(theta_s)
            f = f_samples[s]

            ll_in = np.log(f) - 0.5 * (np.log(2 * np.pi * var_in) + (self.y - y_pred)**2 / var_in)
            ll_out = np.log(1 - f) - 0.5 * (np.log(2 * np.pi * var_out) + (self.y - self.mu_bg)**2 / var_out)
            responsibilities[s] = np.exp(ll_in - np.logaddexp(ll_in, ll_out))

        return responsibilities.mean(axis=0)

    def build_pymc_mixture(self):
        """Return a PyMC two-component Gaussian mixture model.

        Inlier component is supplied by ``_pymc_inlier_model``; the
        outlier component is ``N(mu_bg, y_err**2 + sig_bg**2)`` mixed
        via a logit-Normal weight ``f``.
        """
        y = pt.as_tensor_variable(self.y)
        y_err = pt.as_tensor_variable(self.y_err)

        with pm.Model() as model:
            mu_in, var_in = self._pymc_inlier_model(model)
            logit_f = pm.Normal("logit_f", mu=_LOGIT_F_PRIOR_MU, sigma=_LOGIT_F_PRIOR_SIGMA)
            f = pm.math.sigmoid(logit_f)
            pm.Deterministic("f", f)
            var_out = y_err**2 + self.sig_bg**2

            ll_in = pt.log(f) - 0.5 * (pt.log(2 * np.pi * var_in) + (y - mu_in)**2 / var_in)
            ll_out = pt.log(1 - f) - 0.5 * (pt.log(2 * np.pi * var_out) + (y - self.mu_bg)**2 / var_out)
            pm.Potential("mixture_loglike", pt.sum(pt.logaddexp(ll_in, ll_out)))

        return model
