"""Concrete Bayesian likelihoods built by composing inlier + outlier components.

Three composers — one regression non-mixture, one regression mixture,
one grid non-mixture:

- :class:`ComposedRegressionLikelihood` /
  :class:`ComposedMixtureRegressionLikelihood` for
  ``y_i = f(x_i; theta) + eps`` (1-D or k-D regression).
- :class:`ComposedGridLikelihood` for ``y = f(theta)`` at fixed
  coordinates.

Grid + outlier mixture is not provided — there is no current use case
and the speculative composer was dropped.

:class:`LinearGaussianLikelihood` is a convenience wrapper that wires
:class:`LinearInlier` + :class:`GaussianOutlier` into the regression
mixture composer with the constructor surface used throughout the labs.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ugdatalab.methods.bayesian.base import (
    GridInlierComponent,
    GridLikelihood,
    MixtureLikelihood,
    OutlierComponent,
    Parameter,
    RegressionInlierComponent,
    RegressionLikelihood,
)
from ugdatalab.methods.bayesian.components import GaussianOutlier, LinearInlier


# ---------------------------------------------------------------------------
# Regression composers
# ---------------------------------------------------------------------------


class ComposedRegressionLikelihood(RegressionLikelihood):
    """Regression likelihood assembled from a :class:`RegressionInlierComponent`.

    The inlier component supplies the model form: predictions, prior,
    parameter metadata, and per-point variance. Use
    :class:`ComposedMixtureRegressionLikelihood` when an outlier
    mixture is required.

    Parameters
    ----------
    x, y, y_err : array-like
        Independent variable(s), dependent variable, and per-point
        uncertainties on ``y``. ``x`` may be 1-D ``(n,)`` or k-D
        ``(n, k)`` — the inlier component decides what it accepts.
    inlier : RegressionInlierComponent
        Strategy describing the inlier (signal) part of the model.
    """

    def __init__(self, x, y, y_err, inlier: RegressionInlierComponent):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.y_err = np.asarray(y_err, dtype=float)
        self.inlier = inlier

    @property
    def parameters(self) -> list[Parameter]:
        """Delegate to the inlier component."""
        return self.inlier.parameters

    def predict_at(self, x, theta):
        """Delegate to the inlier component."""
        return self.inlier.predict_at(x, theta)

    def total_variance_at(self, theta):
        """Delegate to the inlier component, supplying ``self.x`` and ``self.y_err``."""
        return self.inlier.total_variance_at(self.x, self.y_err, theta)

    def build_pymc(self):
        """Return a PyMC model with ``y ~ N(mu_in, sqrt(var_in))``."""
        with pm.Model() as model:
            mu_in, var_in = self.inlier.build_pymc(self.x, self.y, self.y_err)
            pm.Normal("obs", mu=mu_in, sigma=pt.sqrt(var_in), observed=self.y)
        return model


class ComposedMixtureRegressionLikelihood(
    ComposedRegressionLikelihood, MixtureLikelihood
):
    """Regression likelihood with an outlier mixture branch.

    Adds an :class:`OutlierComponent` to a
    :class:`ComposedRegressionLikelihood` and implements the
    :class:`MixtureLikelihood` contract by routing ``build_pymc_mixture``
    and ``inlier_probs`` through both components.

    Parameters
    ----------
    x, y, y_err : array-like
        Independent variable(s), dependent variable, and per-point
        uncertainties on ``y``.
    inlier : RegressionInlierComponent
        Strategy describing the inlier (signal) part of the model.
    outlier : OutlierComponent
        Strategy supplying the outlier branch of the mixture.
    """

    def __init__(
        self,
        x,
        y,
        y_err,
        inlier: RegressionInlierComponent,
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
        """Return per-point posterior inlier probabilities."""
        mu_in_samples = np.array([
            self.inlier.predict_at(self.x, s) for s in samples
        ])
        var_in_samples = np.array([
            self.inlier.total_variance_at(self.x, self.y_err, s) for s in samples
        ])
        return self.outlier.inlier_probs(
            self.y, self.y_err, mu_in_samples, var_in_samples, f_samples,
        )


# ---------------------------------------------------------------------------
# Grid composer
# ---------------------------------------------------------------------------


class ComposedGridLikelihood(GridLikelihood):
    """Grid likelihood assembled from a :class:`GridInlierComponent`.

    The inlier component supplies the model form: predictions, prior,
    parameter metadata, and per-entry variance. No mixture composer
    is provided for grid likelihoods — there is no current use case.

    Parameters
    ----------
    coords : array-like
        Coordinate labels for the entries of ``y``. Used by plotters,
        not by the model itself.
    y, y_err : array-like
        Observations and per-entry uncertainties, shape matching
        ``coords``.
    inlier : GridInlierComponent
        Strategy describing the inlier (signal) part of the model.
    """

    def __init__(self, coords, y, y_err, inlier: GridInlierComponent):
        self.coords = np.asarray(coords)
        self.y = np.asarray(y, dtype=float)
        self.y_err = np.asarray(y_err, dtype=float)
        self.inlier = inlier

    @property
    def parameters(self) -> list[Parameter]:
        """Delegate to the inlier component."""
        return self.inlier.parameters

    def predict_at(self, theta):
        """Delegate to the inlier component."""
        return self.inlier.predict_at(theta)

    def total_variance_at(self, theta):
        """Delegate to the inlier component, supplying ``self.y_err``."""
        return self.inlier.total_variance_at(self.y_err, theta)

    def build_pymc(self):
        """Return a PyMC model with ``y ~ N(mu_in, sqrt(var_in))``."""
        with pm.Model() as model:
            mu_in, var_in = self.inlier.build_pymc(self.y, self.y_err)
            pm.Normal("obs", mu=mu_in, sigma=pt.sqrt(var_in), observed=self.y)
        return model


# ---------------------------------------------------------------------------
# Convenience: linear regression with Gaussian outlier mixture
# ---------------------------------------------------------------------------


class LinearGaussianLikelihood(ComposedMixtureRegressionLikelihood):
    """Linear model with Gaussian noise, intrinsic scatter, and a Gaussian outlier mixture.

    Convenience class that wires :class:`LinearInlier` and
    :class:`GaussianOutlier` into a
    :class:`ComposedMixtureRegressionLikelihood` with the constructor
    surface used throughout the labs.

    Model::

        y = a * x + b
        V_i = sigma_yi**2 + sigma_s**2 + (a * sigma_xi)**2

    Posterior parameters exposed in ``Posterior.samples``: ``a``
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
