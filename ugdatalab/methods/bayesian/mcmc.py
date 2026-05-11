"""PyMC NUTS engine.

The :class:`NUTSSampler` runs PyMC's NUTS on any :class:`Likelihood`
(or :class:`MixtureLikelihood` for mixture fits) and packages the draws
into a :class:`Posterior`. Both result types live in
:mod:`ugdatalab.methods.bayesian.base` alongside the likelihood ABCs;
this module is intentionally narrow — just the sampler engine.
"""

from dataclasses import dataclass

import numpy as np
import pymc as pm

from ugdatalab.methods.bayesian.base import (
    Likelihood,
    MixtureLikelihood,
    Posterior,
)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


@dataclass
class NUTSSampler:
    """PyMC NUTS sampler bound to a chain configuration.

    Encapsulates ``draws / tune / random_seed`` and routes each call
    through :meth:`sample` (single-component) or
    :meth:`sample_mixture` (inlier+outlier). Both paths return a
    :class:`Posterior`; data-shape polymorphism is owned by the bound
    likelihood and the posterior delegates to it via ``predict_at``.

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

    def sample(self, likelihood: Likelihood) -> Posterior:
        """Sample ``likelihood.build_pymc()`` and return a :class:`Posterior`."""
        trace = self._run(likelihood.build_pymc())
        samples, log_probs = self._extract(trace, likelihood.physical_param_names)
        return Posterior(
            samples=samples,
            log_probs=log_probs,
            likelihood=likelihood,
        )

    def sample_mixture(self, likelihood: MixtureLikelihood) -> Posterior:
        """Sample ``likelihood.build_pymc_mixture()`` and return a :class:`Posterior`.

        The static type :class:`MixtureLikelihood` enforces that the
        likelihood exposes the mixture sampling path; no runtime
        capability check is needed.
        """
        trace = self._run(likelihood.build_pymc_mixture())
        samples, log_probs = self._extract(trace, likelihood.physical_param_names)
        f_samples = trace.posterior["f"].values.flatten()
        inlier_prob = likelihood.inlier_probs(samples, f_samples)
        return Posterior(
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
