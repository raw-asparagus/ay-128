"""PyMC NUTS engine, ``NUTSSampler`` class, and ``MCMCResult`` container."""

from dataclasses import dataclass

import numpy as np
import pymc as pm

from ugdatalab.methods.base import Fit
from ugdatalab.methods.bayesian.base import Likelihood, MixtureLikelihood


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCMCResult(Fit):
    """Result of PyMC NUTS parameter estimation.

    Holds the fitted likelihood by composition; ``predict``,
    ``total_variance``, ``x``, ``y``, and ``y_err`` delegate to it.
    The posterior median ``theta`` is derived from ``samples`` on
    demand.

    All ``Fit``-protocol summaries (``predict``, ``total_variance``,
    ``chi2_r``) are evaluated at the **per-parameter posterior median**.
    This is a deliberate point estimate, not a posterior draw — the
    median of marginals is not guaranteed to lie at the joint mode and
    will differ from the MAP for correlated parameters. Use
    :meth:`predict_at` / :meth:`total_variance_at` to evaluate at any
    other ``theta`` (e.g., a specific sample, the joint mean, or an
    external best-fit).

    Attributes
    ----------
    samples : ndarray, shape (n_samples, n_params)
        Posterior samples with all chains flattened.
    log_probs : ndarray, shape (n_samples,)
        Log-posterior probability at each sample.
    likelihood : Likelihood
        The likelihood the fit was performed against; the sole source
        of ``x``, ``y``, ``y_err``, ``predict``, ``total_variance``,
        and ``param_labels``.
    """
    samples: np.ndarray
    log_probs: np.ndarray
    likelihood: Likelihood

    # ---- pass-throughs to the likelihood ------------------------------------

    @property
    def x(self) -> np.ndarray:
        """Return ``self.likelihood.x``."""
        return self.likelihood.x

    @property
    def y(self) -> np.ndarray:
        """Return ``self.likelihood.y``."""
        return self.likelihood.y

    @property
    def y_err(self) -> np.ndarray:
        """Return ``self.likelihood.y_err``."""
        return self.likelihood.y_err

    @property
    def labels(self) -> list[str]:
        """Return LaTeX-formatted parameter labels from the likelihood."""
        return self.likelihood.param_labels

    @property
    def theta(self) -> np.ndarray:
        """Return the per-parameter posterior median (marginal, not a joint draw)."""
        return np.median(self.samples, axis=0)

    # ---- model evaluators ---------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the noiseless model prediction at *x* at the posterior median.

        Conforms to the :class:`Fit` protocol. To evaluate at any other
        parameter vector, use :meth:`predict_at`.
        """
        return self.likelihood.predict(x, self.theta)

    def predict_at(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return the noiseless model prediction at *x* for an explicit *theta*."""
        return self.likelihood.predict(x, theta)

    def total_variance(self) -> np.ndarray:
        """Return per-point predictive variance at ``self.x`` at the posterior median.

        Conforms to the :class:`Fit` protocol. To evaluate at any other
        parameter vector, use :meth:`total_variance_at`.
        """
        return self.likelihood.total_variance(self.theta)

    def total_variance_at(self, theta: np.ndarray) -> np.ndarray:
        """Return per-point predictive variance at ``self.x`` for an explicit *theta*."""
        return self.likelihood.total_variance(theta)

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        """Return the std of the mean prediction at *x* across posterior samples.

        Captures parameter-spread uncertainty only; observational and
        intrinsic scatter are not included.

        Parameters
        ----------
        x : array-like
            Positions at which to evaluate the prediction.

        Returns
        -------
        ndarray
            Std of ``predict(x, theta)`` across rows of ``self.samples``.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        preds = np.array([self.predict(x, s) for s in self.samples])
        return np.std(preds, axis=0)

    @property
    def n_params(self) -> int:
        """Return the number of fitted parameters."""
        return self.samples.shape[1]


@dataclass(frozen=True)
class MixtureResult(MCMCResult):
    """Result of a NUTS fit to an inlier+outlier mixture likelihood.

    Extends :class:`MCMCResult` with the per-point posterior inlier
    probability; inherits ``predict``, ``total_variance``, ``chi2_r``,
    ``predict_std``, ``theta``, and the ``x`` / ``y`` / ``y_err`` /
    ``labels`` pass-throughs.

    Attributes
    ----------
    f_samples : ndarray, shape (n_samples,)
        Posterior draws of the inlier-fraction RV ``f``.
    inlier_prob : ndarray, shape (len(x),)
        Per-data-point posterior inlier probability, averaged over
        MCMC draws and computed via the outlier component attached to
        ``likelihood``.
    """
    f_samples: np.ndarray
    inlier_prob: np.ndarray


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


@dataclass
class NUTSSampler:
    """PyMC NUTS sampler bound to a chain configuration.

    Encapsulates ``draws / tune / random_seed`` and routes each call
    through ``sample`` (single-component) or ``sample_mixture``
    (inlier+outlier). The two paths share posterior extraction so that
    :class:`MCMCResult` and :class:`MixtureResult` are populated
    consistently.

    Parameters
    ----------
    n_steps : int
        Total NUTS draws kept per chain after tuning. Default ``2000`` —
        comfortably yields ESS ≳ 1000 on the 2–3 parameter linear/Gaussian
        models used across the labs; raise if posterior summaries are still
        noisy or if ``arviz.summary`` reports ESS below a few hundred.
    n_burn : int
        Tuning / adaptation steps discarded before sampling. Default
        ``1000`` — enough for NUTS to adapt the mass matrix and step size
        on the well-behaved likelihoods here; raise if you see divergences
        or persistent r-hat warnings on a harder posterior.
    seed : int
        Random seed for chain initialization and draws. Default ``42`` —
        an arbitrary fixed value chosen for reproducibility; override to
        check sensitivity of results to the random seed.
    """
    n_steps: int = 2000
    n_burn: int = 1000
    seed: int = 42

    def sample(self, likelihood: Likelihood) -> MCMCResult:
        """Sample ``likelihood.build_pymc()`` and return an :class:`MCMCResult`."""
        trace = self._run(likelihood.build_pymc())
        samples, log_probs = self._extract(trace, likelihood.physical_param_names)
        return MCMCResult(
            samples=samples,
            log_probs=log_probs,
            likelihood=likelihood,
        )

    def sample_mixture(self, likelihood: MixtureLikelihood) -> MixtureResult:
        """Sample ``likelihood.build_pymc_mixture()`` and return a :class:`MixtureResult`.

        The static type ``MixtureLikelihood`` enforces that the
        likelihood exposes the mixture sampling path; no runtime
        capability check is needed.
        """
        trace = self._run(likelihood.build_pymc_mixture())
        samples, log_probs = self._extract(trace, likelihood.physical_param_names)
        f_samples = trace.posterior["f"].values.flatten()
        inlier_prob = likelihood.inlier_probs(samples, f_samples)

        return MixtureResult(
            samples=samples,
            log_probs=log_probs,
            likelihood=likelihood,
            f_samples=f_samples,
            inlier_prob=inlier_prob,
        )

    # ---- internals ---------------------------------------------------------

    def _run(self, model):
        """Run NUTS on *model* with this sampler's chain configuration."""
        with model:
            return pm.sample(
                draws=self.n_steps,
                tune=self.n_burn,
                random_seed=self.seed,
                progressbar=False,
            )

    def _extract(self, trace, names: list[str]):
        """Return ``(samples, log_probs)`` for *names*."""
        posterior = trace.posterior
        samples = np.column_stack([
            posterior[name].values.flatten() for name in names
        ])
        log_probs = trace.sample_stats["lp"].values.flatten()
        return samples, log_probs
