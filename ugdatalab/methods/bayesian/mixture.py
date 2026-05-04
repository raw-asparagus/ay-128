"""Mixture-contamination engine: NUTS over an inlier+outlier mixture model."""

from dataclasses import dataclass

import numpy as np
import pymc as pm

from ugdatalab.methods.bayesian.base import MixtureLikelihood


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
    likelihood: MixtureLikelihood,
    n_steps: int = 2000,
    n_burn: int = 1000,
    seed: int = 42,
) -> MixtureResult:
    """Run an inlier+outlier mixture model via PyMC NUTS.

    Parameters
    ----------
    likelihood : MixtureLikelihood
        Object satisfying the ``MixtureLikelihood`` ABC.
    n_steps : int
        Total NUTS draws kept per chain after tuning. Default ``2000`` —
        matches :func:`~ugdatalab.methods.bayesian.mcmc.nuts_sample`; gives
        ESS ≳ 1000 on the inlier/outlier mixtures used in the labs. Raise
        if the per-row ``inlier_prob`` summaries appear noisy.
    n_burn : int
        Tuning / adaptation steps discarded before sampling. Default
        ``1000`` — sufficient for NUTS to adapt on the mixture posteriors
        here; raise if divergences or r-hat warnings appear.
    seed : int
        Random seed for chain initialization and draws. Default ``42`` —
        an arbitrary fixed value for reproducibility; override to check
        sensitivity of the inlier-probability assignments to randomness.

    Returns
    -------
    MixtureResult
    """
    model = likelihood.build_pymc_mixture()
    with model:
        trace = pm.sample(
            draws=n_steps,
            tune=n_burn,
            random_seed=seed,
            progressbar=False,
        )

    model_var_names = likelihood.physical_param_names
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
