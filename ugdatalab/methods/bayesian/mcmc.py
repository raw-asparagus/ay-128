from dataclasses import dataclass

import numpy as np
import pymc as pm

from ugdatalab.methods.base import Fit


@dataclass(frozen=True)
class MCMCResult(Fit):
    """Result of PyMC NUTS parameter estimation.

    Attributes
    ----------
    theta : ndarray
        Posterior median of the model parameters.
    samples : ndarray
        Posterior samples, shape (n_samples, n_params).
    log_probs : ndarray
        Log-posterior probability at each sample.
    labels : list[str]
        LaTeX-formatted parameter labels for plotting.
    chi2_r : float
        Reduced chi-squared evaluated at the posterior median.
    """
    theta: np.ndarray
    samples: np.ndarray
    log_probs: np.ndarray
    labels: list
    chi2_r: float
    _likelihood: object = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict y values at *x* using the posterior median parameters."""
        return self._likelihood._predict(x, self.theta)

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        """Prediction uncertainty at *x* from posterior spread."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        preds = np.array([self._likelihood._predict(x, s) for s in self.samples])
        return np.std(preds, axis=0)


def nuts_sample(
    likelihood,
    n_steps: int = 2000,
    n_burn: int = 1000,
    seed: int = 42,
) -> MCMCResult:
    """Run PyMC NUTS on a likelihood's model.

    Parameters
    ----------
    likelihood : Likelihood
        An object satisfying the Likelihood ABC.
    n_steps : int
        Total NUTS draws (after tuning).
    n_burn : int
        Tuning steps (discarded).
    seed : int
        Random seed for reproducibility.
    """
    model = likelihood.build_pymc()
    with model:
        trace = pm.sample(
            draws=n_steps,
            tune=n_burn,
            random_seed=seed,
            progressbar=False,
        )

    var_names = [v.name for v in model.free_RVs]
    posterior = trace.posterior
    theta_median = np.array([float(posterior[name].median()) for name in var_names])
    samples = np.column_stack([
        posterior[name].values.flatten() for name in var_names
    ])
    log_probs = trace.sample_stats["lp"].values.flatten()

    y_pred = likelihood._predict(likelihood.x, theta_median)
    good = (np.isfinite(likelihood.y_err) & (likelihood.y_err < 1e5)
            & np.isfinite(likelihood.y) & np.isfinite(y_pred))
    resid = (likelihood.y - y_pred)[good]
    sigma = likelihood.y_err[good]
    nu = int(good.sum()) - len(var_names)
    chi2_r = float(np.sum((resid / sigma) ** 2) / max(nu, 1))

    return MCMCResult(
        theta=theta_median,
        samples=samples,
        log_probs=log_probs,
        labels=likelihood.param_labels,
        chi2_r=chi2_r,
        _likelihood=likelihood,
    )
