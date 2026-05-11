"""Bayesian methods: composable likelihoods, NUTS sampler, and Posterior result.

Public API:

- **Contracts** — :class:`Likelihood` (minimal),
  :class:`RegressionLikelihood`, :class:`GridLikelihood`,
  :class:`MixtureLikelihood`, :class:`RegressionInlierComponent`,
  :class:`GridInlierComponent`, :class:`OutlierComponent`,
  :class:`Parameter`.
- **Concrete strategies** — :class:`LinearInlier`,
  :class:`MultivariateLinearInlier`, :class:`GaussianOutlier`.
- **Concrete likelihoods** — composers (``Composed*Likelihood``) and
  the user-facing :class:`LinearGaussianLikelihood`.
- **Sampler & result** — :class:`NUTSSampler`, :class:`Posterior`.
"""

from ugdatalab.methods.bayesian.base import (
    GridInlierComponent,
    GridLikelihood,
    Likelihood,
    MixtureLikelihood,
    OutlierComponent,
    Parameter,
    Posterior,
    RegressionInlierComponent,
    RegressionLikelihood,
)
from ugdatalab.methods.bayesian.components import (
    GaussianOutlier,
    LinearInlier,
    MultivariateLinearInlier,
)
from ugdatalab.methods.bayesian.likelihoods import (
    ComposedGridLikelihood,
    ComposedMixtureRegressionLikelihood,
    ComposedRegressionLikelihood,
    LinearGaussianLikelihood,
)
from ugdatalab.methods.bayesian.mcmc import NUTSSampler

__all__ = [
    # contracts
    "GridInlierComponent",
    "GridLikelihood",
    "Likelihood",
    "MixtureLikelihood",
    "OutlierComponent",
    "Parameter",
    "RegressionInlierComponent",
    "RegressionLikelihood",
    # concrete strategies
    "GaussianOutlier",
    "LinearInlier",
    "MultivariateLinearInlier",
    # concrete likelihoods
    "ComposedGridLikelihood",
    "ComposedMixtureRegressionLikelihood",
    "ComposedRegressionLikelihood",
    "LinearGaussianLikelihood",
    # sampler & result
    "NUTSSampler",
    "Posterior",
]
