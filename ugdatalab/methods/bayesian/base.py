"""Abstract contracts for Bayesian likelihoods, components, and the posterior.

Three orthogonal axes structure the framework:

1. **Data layout** — sibling ABCs of :class:`Likelihood`. Today
   :class:`RegressionLikelihood` (``y_i = f(x_i; theta) + eps``, ``x``
   either 1-D or k-D) and :class:`GridLikelihood` (``y = f(theta)`` on
   fixed coordinates). Add a sibling for any new shape; the sampler
   never inspects which shape it has.

2. **Model form** — :class:`RegressionInlierComponent` and
   :class:`GridInlierComponent` strategies. The composers in
   :mod:`ugdatalab.methods.bayesian.likelihoods` glue these to the data.

3. **Outlier treatment** — :class:`OutlierComponent` strategies. By the
   time an outlier runs it sees only ``(mu_in, var_in)`` per
   observation, so a single outlier strategy works for any data shape.

:class:`MixtureLikelihood` is a :class:`Likelihood` subtype that adds
an outlier branch. Concrete mixture likelihoods inherit both their
data-shape ABC and :class:`MixtureLikelihood`;
``NUTSSampler.sample_mixture`` accepts any :class:`MixtureLikelihood`
regardless of underlying shape.

:class:`Posterior` is the single result container: a sample cloud plus
a reference to the bound likelihood, with shape-agnostic statistical
summaries and shape-dispatched evaluators (``predict``,
``total_variance``, ``chi2_r``) that delegate to the likelihood.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Parameter metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameter:
    """Metadata for one fitted parameter.

    Single source of truth for the ``(name, label)`` pair so that
    posterior-trace extraction, plotting labels, and ``samples`` column
    ordering can never drift apart.

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
# Parameter-metadata mixin
# ---------------------------------------------------------------------------


class HasParameters(ABC):
    """Mixin contract for any object that exposes a list of :class:`Parameter`.

    Defines the single abstract :attr:`parameters` and derives
    :attr:`param_labels` / :attr:`physical_param_names` from it. Inherited
    by both :class:`Likelihood` (for trace extraction and plotting) and
    the inlier component ABCs (for the parameters they contribute to a
    composed model).
    """

    @property
    @abstractmethod
    def parameters(self) -> list[Parameter]:
        """Return the parameters in trace/sample-column order."""
        ...

    @property
    def param_labels(self) -> list[str]:
        """LaTeX-formatted parameter labels, derived from :attr:`parameters`."""
        return [p.label for p in self.parameters]

    @property
    def physical_param_names(self) -> list[str]:
        """Trace variable names, derived from :attr:`parameters`."""
        return [p.name for p in self.parameters]


# ---------------------------------------------------------------------------
# Likelihood — minimal contract
# ---------------------------------------------------------------------------


class Likelihood(HasParameters):
    """Minimal contract for an MCMC-sampleable likelihood.

    Two abstractions only: :attr:`parameters` (for trace extraction
    and plotting labels) and :meth:`build_pymc` (the PyMC model).
    Data-layout concepts (``x``, ``coords``, ``y``, ``y_err``,
    ``predict_at``, ``total_variance_at``, ``chi2_r_at``) live on the
    shape-specific subclasses.

    Concrete user-facing likelihoods subclass one of those shape ABCs,
    never this base directly — unless a non-standard data layout
    (hierarchical, state-space, etc.) is needed, in which case a new
    sibling shape ABC is the right extension point.
    """

    @abstractmethod
    def build_pymc(self):
        """Return a PyMC model ready for NUTS sampling."""
        ...


class RegressionLikelihood(Likelihood):
    """Regression-shaped likelihood: ``y_i = f(x_i; theta) + eps``.

    ``x`` may be 1-D ``(n,)`` for univariate regression or k-D
    ``(n, k)`` for multivariate regression; subclasses decide what
    shape they accept. The framework treats ``x`` as opaque
    per-observation input.

    Attributes
    ----------
    x : ndarray, shape ``(n,)`` or ``(n, k)``
        Independent-variable values.
    y : ndarray, shape ``(n,)``
        Dependent-variable observations.
    y_err : ndarray, shape ``(n,)``
        Per-point observational uncertainty on ``y``.
    """
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray

    @abstractmethod
    def predict_at(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return the noiseless model prediction at *x* for *theta*.

        ``x`` may differ from ``self.x`` — this is a parameterized
        evaluation suitable for prediction grids, posterior-predictive
        bands, and extrapolation.
        """
        ...

    def total_variance_at(self, theta: np.ndarray) -> np.ndarray:
        """Return per-point predictive variance at ``self.x`` for *theta*.

        Default returns ``y_err ** 2``. Subclasses with intrinsic
        scatter or x-uncertainty override.
        """
        return self.y_err ** 2

    def chi2_r_at(self, theta: np.ndarray) -> float:
        """Return reduced chi-squared at the bound data for *theta*.

        Weighted sum of squared residuals at ``self.x`` divided by
        ``max(len(y) - len(theta), 1)``.
        """
        resid = self.y - self.predict_at(self.x, theta)
        nu = max(self.y.size - len(theta), 1)
        return float(np.sum(resid ** 2 / self.total_variance_at(theta)) / nu)


class GridLikelihood(Likelihood):
    """Fixed-grid likelihood: ``y = f(theta)`` at locked coordinates.

    The model has no notion of "evaluate at a new x" — output shape is
    fixed at construction. ``coords`` is a coordinate label kept for
    plotting and bookkeeping; the model itself does not consume it.

    Attributes
    ----------
    coords : ndarray
        Coordinate labels for the entries of ``y`` (wavelength, pixel
        index, time, ...). Used by plotters, not by the model.
    y : ndarray
        Observations laid out on the fixed grid.
    y_err : ndarray
        Per-entry observational uncertainty, shape matching ``y``.
    """
    coords: np.ndarray
    y: np.ndarray
    y_err: np.ndarray

    @abstractmethod
    def predict_at(self, theta: np.ndarray) -> np.ndarray:
        """Return the noiseless model prediction for *theta*.

        Output shape matches ``y``. There is no ``x`` argument because
        a grid model's output grid is fixed.
        """
        ...

    def total_variance_at(self, theta: np.ndarray) -> np.ndarray:
        """Return per-entry predictive variance for *theta*.

        Default returns ``y_err ** 2``. Subclasses with intrinsic
        scatter override.
        """
        return self.y_err ** 2

    def chi2_r_at(self, theta: np.ndarray) -> float:
        """Return reduced chi-squared on the fixed grid for *theta*.

        Weighted sum of squared residuals divided by
        ``max(y.size - len(theta), 1)``.
        """
        resid = self.y - self.predict_at(theta)
        nu = max(self.y.size - len(theta), 1)
        return float(np.sum(resid ** 2 / self.total_variance_at(theta)) / nu)


# ---------------------------------------------------------------------------
# Mixture likelihood — Likelihood subtype with an outlier branch
# ---------------------------------------------------------------------------


class MixtureLikelihood(Likelihood):
    """Likelihood with an inlier+outlier mixture branch.

    Concrete mixture likelihoods inherit their data-shape ABC (e.g.
    :class:`RegressionLikelihood`) *and* this class.
    ``NUTSSampler.sample_mixture`` accepts any :class:`MixtureLikelihood`
    regardless of underlying shape.
    """

    @abstractmethod
    def build_pymc_mixture(self):
        """Return a PyMC inlier+outlier mixture model ready for NUTS sampling."""
        ...

    @abstractmethod
    def inlier_probs(
        self, samples: np.ndarray, f_samples: np.ndarray
    ) -> np.ndarray:
        """Return per-observation posterior inlier probabilities.

        Parameters
        ----------
        samples : ndarray, shape ``(n_draws, n_params)``
            Inlier-parameter posterior draws.
        f_samples : ndarray, shape ``(n_draws,)``
            Posterior draws of the inlier-fraction RV.
        """
        ...


# ---------------------------------------------------------------------------
# Strategy components
# ---------------------------------------------------------------------------


class RegressionInlierComponent(HasParameters):
    """Strategy describing the inlier portion of a regression likelihood.

    Stateless or near-stateless: data flows in through :meth:`build_pymc`
    and the ``*_at`` evaluators rather than being held on the component.
    A single component instance can be reused across datasets; the
    composer owns data binding.
    """

    @abstractmethod
    def build_pymc(self, x: np.ndarray, y: np.ndarray, y_err: np.ndarray):
        """Add inlier RVs to the active PyMC model, return ``(mu_in, var_in)``.

        Must be called inside an active ``pm.Model()`` context. The
        returned tensors are PyTensor expressions of shape ``y.shape``.
        """
        ...

    @abstractmethod
    def predict_at(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return the noiseless inlier prediction at *x* for *theta*."""
        ...

    @abstractmethod
    def total_variance_at(
        self, x: np.ndarray, y_err: np.ndarray, theta: np.ndarray,
    ) -> np.ndarray:
        """Return per-point inlier variance at *x* for *theta*."""
        ...


class GridInlierComponent(HasParameters):
    """Strategy describing the inlier portion of a grid likelihood.

    The grid model has no ``x`` input — its output grid is locked by
    training or instrument geometry — so :meth:`build_pymc` and the
    ``*_at`` evaluators take only ``y``, ``y_err``, and ``theta``.
    """

    @abstractmethod
    def build_pymc(self, y: np.ndarray, y_err: np.ndarray):
        """Add inlier RVs to the active PyMC model, return ``(mu_in, var_in)``.

        Must be called inside an active ``pm.Model()`` context. The
        returned tensors are PyTensor expressions of shape ``y.shape``.
        """
        ...

    @abstractmethod
    def predict_at(self, theta: np.ndarray) -> np.ndarray:
        """Return the noiseless inlier prediction for *theta*.

        Output shape matches the data ``y``.
        """
        ...

    @abstractmethod
    def total_variance_at(
        self, y_err: np.ndarray, theta: np.ndarray,
    ) -> np.ndarray:
        """Return per-entry inlier variance for *theta*."""
        ...


class OutlierComponent(ABC):
    """Strategy describing the outlier (background) portion of a mixture.

    Data-shape-agnostic: operates on per-observation ``(mu_in, var_in)``
    tensors produced by the inlier, so a single outlier strategy works
    for both :class:`RegressionInlierComponent` and
    :class:`GridInlierComponent` siblings.
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
        produced by an inlier component's ``build_pymc``.
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
        """Return per-observation posterior inlier probabilities.

        Parameters
        ----------
        y, y_err : ndarray
            Observations and per-entry uncertainties.
        mu_in_samples : ndarray, shape ``(n_draws, *y.shape)``
            Inlier mean prediction evaluated at each posterior draw.
        var_in_samples : ndarray, shape ``(n_draws, *y.shape)``
            Inlier per-entry variance evaluated at each posterior draw.
        f_samples : ndarray, shape ``(n_draws,)``
            Posterior draws of the inlier-fraction RV.
        """
        ...


# ---------------------------------------------------------------------------
# Posterior — single result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Posterior:
    """MCMC posterior over the parameters of a :class:`Likelihood`.

    Shape-agnostic at the statistical level (``samples``, ``log_probs``,
    ``theta``); shape-aware via dispatch through
    :meth:`Likelihood.predict_at` for predictive evaluators
    (``predict``, ``total_variance``, ``chi2_r``). A regression
    posterior takes ``predict(x_new)``; a grid posterior takes
    ``predict()``.

    Attributes
    ----------
    samples : ndarray, shape ``(n_samples, n_params)``
        Posterior samples with all chains flattened, parameter order
        matches ``likelihood.parameters``.
    log_probs : ndarray, shape ``(n_samples,)``
        Log-posterior at each sample.
    likelihood : RegressionLikelihood or GridLikelihood
        Bound likelihood — sole source of ``predict_at``,
        ``total_variance_at``, ``chi2_r_at``, and parameter metadata.
        Typed as the union of the two shape ABCs so the predictive
        evaluators are statically resolvable; extend the union when a
        new shape ABC is added.
    f_samples : ndarray, optional
        Posterior draws of the inlier-fraction RV ``f``. ``None``
        unless the likelihood was a :class:`MixtureLikelihood` sampled
        via :meth:`NUTSSampler.sample_mixture`.
    inlier_prob : ndarray, optional
        Per-observation posterior inlier probability, averaged over
        MCMC draws. ``None`` unless this is a mixture result.
    """
    samples: np.ndarray
    log_probs: np.ndarray
    likelihood: RegressionLikelihood | GridLikelihood
    f_samples: Optional[np.ndarray] = None
    inlier_prob: Optional[np.ndarray] = None

    @property
    def labels(self) -> list[str]:
        """LaTeX-formatted parameter labels from the likelihood."""
        return self.likelihood.param_labels

    @property
    def theta(self) -> np.ndarray:
        """Per-parameter posterior median (marginal, not a joint draw)."""
        return np.median(self.samples, axis=0)

    @property
    def n_params(self) -> int:
        """Number of fitted inlier parameters (``samples.shape[1]``).

        For mixture fits this *excludes* the inlier-fraction RV ``f``,
        which is reported separately in :attr:`f_samples`.
        """
        return self.samples.shape[1]

    def predict(self, *args) -> np.ndarray:
        """Predict at the posterior median.

        Forwards ``*args`` to :meth:`Likelihood.predict_at`. For a
        regression likelihood pass ``x``; for a grid likelihood pass
        nothing.
        """
        return self.likelihood.predict_at(*args, self.theta)

    def total_variance(self) -> np.ndarray:
        """Per-observation predictive variance at the posterior median."""
        return self.likelihood.total_variance_at(self.theta)

    @property
    def chi2_r(self) -> float:
        """Reduced chi-squared at the posterior median."""
        return self.likelihood.chi2_r_at(self.theta)
