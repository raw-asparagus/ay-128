"""PyMC NUTS engine and ``MCMCResult`` container."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pymc as pm

from ugdatalab.methods.base import DataFit


@dataclass(frozen=True)
class MCMCResult(DataFit):
    """Result of PyMC NUTS parameter estimation.

    Carries the data the fit was performed on plus bound model
    evaluators, exposing ``predict`` / ``total_variance`` / ``chi2_r``.

    Attributes
    ----------
    theta : ndarray
        Per-parameter posterior median (marginal, not a joint draw).
    samples : ndarray, shape (n_samples, n_params)
        Posterior samples with all chains flattened.
    log_probs : ndarray, shape (n_samples,)
        Log-posterior probability at each sample.
    labels : list of str
        LaTeX-formatted parameter labels.
    x, y, y_err : ndarray
        Data the fit was performed on.
    _predict_fn, _variance_fn : Callable
        Bound model evaluators; accessed via ``predict`` / ``total_variance``.
    """
    theta: np.ndarray
    samples: np.ndarray
    log_probs: np.ndarray
    labels: list
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray
    _predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    _variance_fn: Callable[[np.ndarray], np.ndarray]

    def predict(self, x: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        """Return the noiseless model prediction at *x*.

        Uses the posterior median when *theta* is omitted.

        Parameters
        ----------
        x : ndarray
            Positions at which to evaluate the model.
        theta : ndarray, optional
            Parameter vector; defaults to ``self.theta``.

        Returns
        -------
        ndarray
        """
        return self._predict_fn(x, self.theta if theta is None else theta)

    def total_variance(self, theta: np.ndarray | None = None) -> np.ndarray:
        """Return per-point predictive variance at ``self.x``.

        Uses the posterior median when *theta* is omitted.

        Parameters
        ----------
        theta : ndarray, optional
            Parameter vector; defaults to ``self.theta``.

        Returns
        -------
        ndarray
        """
        return self._variance_fn(self.theta if theta is None else theta)

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
        return len(self.theta)


def nuts_sample(
    likelihood,
    n_steps: int = 2000,
    n_burn: int = 1000,
    seed: int = 42,
) -> MCMCResult:
    """Run PyMC NUTS on a likelihood's model and return an ``MCMCResult``.

    Parameters
    ----------
    likelihood : Likelihood
        Object satisfying the ``Likelihood`` ABC.
    n_steps : int
        Total NUTS draws after tuning.
    n_burn : int
        Tuning steps, discarded.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    MCMCResult
    """
    model = likelihood.build_pymc()
    with model:
        trace = pm.sample(
            draws=n_steps,
            tune=n_burn,
            random_seed=seed,
            progressbar=False,
        )

    var_names = likelihood.physical_param_names
    posterior = trace.posterior
    theta_median = np.array([float(posterior[name].median()) for name in var_names])
    samples = np.column_stack([
        posterior[name].values.flatten() for name in var_names
    ])
    log_probs = trace.sample_stats["lp"].values.flatten()

    return MCMCResult(
        theta=theta_median,
        samples=samples,
        log_probs=log_probs,
        labels=likelihood.param_labels,
        x=likelihood.x,
        y=likelihood.y,
        y_err=likelihood.y_err,
        _predict_fn=likelihood.predict,
        _variance_fn=likelihood.total_variance,
    )
