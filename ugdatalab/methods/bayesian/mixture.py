from dataclasses import dataclass

import numpy as np
import pymc as pm


@dataclass(frozen=True)
class MixtureResult:
    """Result of mixture contamination model fitting.

    Attributes
    ----------
    inlier_prob : ndarray
        Per-data-point posterior inlier probability, averaged over MCMC draws.
    theta : ndarray
        Posterior median of the inlier model parameters.
    samples : ndarray
        Posterior samples, shape (n_samples, n_params).
    log_probs : ndarray
        Log-posterior probability at each sample.
    labels : list[str]
        LaTeX-formatted parameter labels for plotting.
    """
    inlier_prob: np.ndarray
    theta: np.ndarray
    samples: np.ndarray
    log_probs: np.ndarray
    labels: list


def mixture_contamination(
    likelihood,
    n_steps: int = 2000,
    n_burn: int = 1000,
    seed: int = 42,
) -> MixtureResult:
    """Run a mixture contamination model via PyMC NUTS.

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
    model = likelihood.build_pymc_mixture()
    with model:
        trace = pm.sample(
            draws=n_steps,
            tune=n_burn,
            random_seed=seed,
            progressbar=False,
        )

    model_var_names = [v.name for v in model.free_RVs if v.name != "f"]
    posterior = trace.posterior
    theta_median = np.array([float(posterior[name].median()) for name in model_var_names])
    samples = np.column_stack([
        posterior[name].values.flatten() for name in model_var_names
    ])
    log_probs = trace.sample_stats["lp"].values.flatten()
    probs = likelihood.inlier_probs(trace, model_var_names)

    return MixtureResult(
        inlier_prob=probs,
        theta=theta_median,
        samples=samples,
        log_probs=log_probs,
        labels=likelihood.param_labels,
    )
