"""Abstract contracts for Bayesian likelihoods and their building blocks.

``Likelihood`` is the minimal contract consumed by ``NUTSSampler.sample``.
``MixtureLikelihood`` extends it with the inlier+outlier mixture path
consumed by ``NUTSSampler.sample_mixture``. ``InlierComponent`` and
``OutlierComponent`` are strategies that ``ComposedLikelihood`` /
``ComposedMixtureLikelihood`` glue together via composition; this
replaces the prior inheritance chain with two independent axes (model
form, outlier treatment) that can vary independently.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Parameter metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameter:
    """Metadata for one fitted parameter.

    Single source of truth for the ``(name, label)`` pair so that
    posterior-trace extraction, plotting labels, and ``samples``
    column ordering can never drift apart.

    Attributes
    ----------
    name : str
        Trace variable name in the PyMC model; used to extract draws
        from the posterior.
    label : str
        LaTeX-formatted parameter label in physical units; used for
        plot axes, corner plots, and tables.
    """
    name: str
    label: str


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


class Likelihood(ABC):
    """Minimal contract for an MCMC-sampleable likelihood.

    Subclasses hold ``(x, y, y_err)`` and implement ``build_pymc``,
    ``predict``, and ``parameters``. ``total_variance`` defaults to
    ``y_err ** 2`` and may be overridden when the model adds
    intrinsic scatter or x-uncertainty.
    """
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray

    @property
    @abstractmethod
    def parameters(self) -> list[Parameter]:
        """Return the fitted parameters in trace/sample-column order.

        The order defines the *theta* component indexing consumed by
        ``predict`` and ``total_variance``.
        """
        ...

    @property
    def param_labels(self) -> list[str]:
        """LaTeX-formatted parameter labels, derived from :attr:`parameters`."""
        return [p.label for p in self.parameters]

    @property
    def physical_param_names(self) -> list[str]:
        """Trace variable names, derived from :attr:`parameters`."""
        return [p.name for p in self.parameters]

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
    """Likelihood that supports an inlier+outlier mixture sampling path.

    Subclasses add ``build_pymc_mixture`` and ``inlier_probs``. A
    ``Likelihood`` that is *not* a ``MixtureLikelihood`` cannot be
    passed to :meth:`NUTSSampler.sample_mixture` — the distinction is
    expressed at the type level rather than as a runtime guard.
    """

    @abstractmethod
    def build_pymc_mixture(self):
        """Return a PyMC inlier+outlier mixture model ready for NUTS sampling."""
        ...

    @abstractmethod
    def inlier_probs(
        self, samples: np.ndarray, f_samples: np.ndarray
    ) -> np.ndarray:
        """Return per-point posterior inlier probabilities, shape ``(len(x),)``.

        Parameters
        ----------
        samples : ndarray, shape ``(n_draws, n_params)``
            Inlier-parameter posterior draws.
        f_samples : ndarray, shape ``(n_draws,)``
            Posterior draws of the inlier-fraction RV.
        """
        ...


# ---------------------------------------------------------------------------
# Composable strategies
# ---------------------------------------------------------------------------


class InlierComponent(ABC):
    """Strategy describing the inlier (signal) portion of a likelihood.

    An ``InlierComponent`` is a stateless (or near-stateless) strategy
    that knows how to (1) add inlier random variables to a PyMC model
    and produce the per-point ``(mu_in, var_in)`` tensors, (2) evaluate
    the deterministic prediction in NumPy, and (3) report the
    parameter metadata that flows into ``MCMCResult``.
    """

    @property
    @abstractmethod
    def parameters(self) -> list[Parameter]:
        """Return the inlier parameters in trace/sample-column order."""
        ...

    @property
    def param_labels(self) -> list[str]:
        """LaTeX-formatted parameter labels, derived from :attr:`parameters`."""
        return [p.label for p in self.parameters]

    @property
    def physical_param_names(self) -> list[str]:
        """Trace variable names, derived from :attr:`parameters`."""
        return [p.name for p in self.parameters]

    @abstractmethod
    def build_pymc(self, x: np.ndarray, y: np.ndarray, y_err: np.ndarray):
        """Add inlier RVs to the active PyMC model, return ``(mu_in, var_in)``.

        Must be called inside an active ``pm.Model()`` context.
        """
        ...

    @abstractmethod
    def predict(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return the noiseless inlier prediction at *x* for *theta*."""
        ...

    @abstractmethod
    def total_variance(
        self, x: np.ndarray, y_err: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Return per-point inlier variance at *x* for *theta*."""
        ...


class OutlierComponent(ABC):
    """Strategy describing the outlier (background) portion of a mixture.

    An ``OutlierComponent`` knows how to (1) extend an active PyMC
    model with the outlier mixture potential given the inlier
    ``(mu_in, var_in)`` tensors, and (2) compute per-point posterior
    inlier probabilities from precomputed per-sample inlier prediction
    arrays. The component never holds a back-reference to the
    likelihood — the ``ComposedMixtureLikelihood`` evaluates the
    inlier model and passes the resulting numerics in.
    """

    @abstractmethod
    def build_pymc_mixture(
        self,
        y: np.ndarray,
        y_err: np.ndarray,
        mu_in,
        var_in,
    ) -> None:
        """Add outlier RVs and the mixture log-likelihood to the active model.

        Must be called inside an active ``pm.Model()`` context. The
        ``mu_in`` / ``var_in`` arguments are PyTensor expressions
        produced by :meth:`InlierComponent.build_pymc`.
        """
        ...

    @abstractmethod
    def inlier_probs(
        self,
        y: np.ndarray,
        y_err: np.ndarray,
        mu_in_samples: np.ndarray,
        var_in_samples: np.ndarray,
        f_samples: np.ndarray,
    ) -> np.ndarray:
        """Return per-point posterior inlier probabilities, shape ``(len(y),)``.

        Parameters
        ----------
        y, y_err : ndarray, shape ``(n_points,)``
            Observations and per-point uncertainties on ``y``.
        mu_in_samples : ndarray, shape ``(n_draws, n_points)``
            Inlier mean prediction evaluated at each posterior draw.
        var_in_samples : ndarray, shape ``(n_draws, n_points)``
            Inlier per-point variance evaluated at each posterior draw.
        f_samples : ndarray, shape ``(n_draws,)``
            Posterior draws of the inlier-fraction RV.
        """
        ...
