"""Concrete Bayesian likelihoods built by composing inlier + outlier components."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ugdatalab.methods.bayesian.base import (
    InlierComponent,
    Likelihood,
    MixtureLikelihood,
    OutlierComponent,
    Parameter,
)
from ugdatalab.methods.bayesian.components import GaussianOutlier, LinearInlier


class ComposedLikelihood(Likelihood):
    """Likelihood assembled from an :class:`InlierComponent`.

    The inlier component supplies the model form: predictions, prior,
    parameter metadata, and per-point variance. Use
    :class:`ComposedMixtureLikelihood` when an outlier mixture is
    required.

    Parameters
    ----------
    x, y, y_err : array-like
        Observations and per-point uncertainties on ``y``.
    inlier : InlierComponent
        Strategy describing the inlier (signal) part of the model.
    """

    def __init__(self, x, y, y_err, inlier: InlierComponent):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.y_err = np.asarray(y_err, dtype=float)
        self.inlier = inlier

    @property
    def parameters(self) -> list[Parameter]:
        """Delegate to the inlier component."""
        return self.inlier.parameters

    def predict(self, x, theta):
        """Delegate to the inlier component."""
        return self.inlier.predict(x, theta)

    def total_variance(self, theta):
        """Delegate to the inlier component, supplying ``self.x`` and ``self.y_err``."""
        return self.inlier.total_variance(self.x, self.y_err, theta)

    def build_pymc(self):
        """Return a PyMC model with ``y ~ N(mu_in, sqrt(var_in))``."""
        with pm.Model() as model:
            mu_in, var_in = self.inlier.build_pymc(self.x, self.y, self.y_err)
            pm.Normal("obs", mu=mu_in, sigma=pt.sqrt(var_in), observed=self.y)
        return model


class ComposedMixtureLikelihood(ComposedLikelihood, MixtureLikelihood):
    """Composed likelihood with an outlier mixture branch.

    Adds an :class:`OutlierComponent` to a :class:`ComposedLikelihood`
    and implements the :class:`MixtureLikelihood` contract by routing
    ``build_pymc_mixture`` and ``inlier_probs`` through both
    components.

    Parameters
    ----------
    x, y, y_err : array-like
        Observations and per-point uncertainties on ``y``.
    inlier : InlierComponent
        Strategy describing the inlier (signal) part of the model.
    outlier : OutlierComponent
        Strategy supplying the outlier branch of the mixture.
    """

    def __init__(
        self,
        x,
        y,
        y_err,
        inlier: InlierComponent,
        outlier: OutlierComponent,
    ):
        super().__init__(x, y, y_err, inlier)
        self.outlier = outlier

    def build_pymc_mixture(self):
        """Return a PyMC inlier+outlier mixture model."""
        with pm.Model() as model:
            mu_in, var_in = self.inlier.build_pymc(self.x, self.y, self.y_err)
            self.outlier.build_pymc_mixture(self.y, self.y_err, mu_in, var_in)
        return model

    def inlier_probs(self, samples, f_samples) -> np.ndarray:
        """Return per-point posterior inlier probabilities.

        Evaluates the inlier prediction and variance at every posterior
        draw, then delegates to the outlier component to combine those
        with the mixture weight ``f``.
        """
        mu_in_samples = np.array([
            self.inlier.predict(self.x, s) for s in samples
        ])
        var_in_samples = np.array([
            self.inlier.total_variance(self.x, self.y_err, s) for s in samples
        ])
        return self.outlier.inlier_probs(
            self.y, self.y_err, mu_in_samples, var_in_samples, f_samples,
        )


class LinearGaussianLikelihood(ComposedMixtureLikelihood):
    """Linear model with Gaussian noise, intrinsic scatter, and a Gaussian outlier mixture.

    Convenience class that wires :class:`LinearInlier` and
    :class:`GaussianOutlier` into a :class:`ComposedMixtureLikelihood`
    with the constructor surface used throughout the labs.

    Model::

        y = a * x + b
        V_i = sigma_yi**2 + sigma_s**2 + (a * sigma_xi)**2

    Posterior parameters exposed in ``MCMCResult.samples``: ``a``
    (slope), ``b`` (intercept), ``sigma_s`` (intrinsic scatter).

    Parameters
    ----------
    x, y, y_err : array-like
        Independent variable, dependent variable, and uncertainties on ``y``.
    x_err : array-like, optional
        Uncertainties on ``x``. Defaults to zero, matching the
        no-x-error special case of the variance expression.
    """

    def __init__(self, x, y, y_err, x_err=None):
        x = np.asarray(x, dtype=float)
        x_err = np.zeros_like(x) if x_err is None else x_err
        super().__init__(
            x=x,
            y=y,
            y_err=y_err,
            inlier=LinearInlier(x_err=x_err),
            outlier=GaussianOutlier(),
        )
