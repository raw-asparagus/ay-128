# ugdatalab

A set of reusable data analysis tools for astronomical data. These are mostly
wrappers around existing packages, but they are collated together to allow for
easy use and organization into data pipelines and visaulizations.

## Package structure

```
ugdatalab/
    plotting.py                  — Matplotlib style guide (rcParams, constants, style dicts)
    methods/
        base.py                  — Fit ABC
        periodogram.py           — Lomb-Scargle periodogram
        fourier.py               — Fourier series fitting
        cross_validate.py        — Holdout and k-fold validation
        bayesian/
            base.py              — Likelihood and GaussianLikelihood ABCs
            likelihoods.py       — LinearGaussianLikelihood
            mcmc.py              — PyMC NUTS sampling engine
            mixture.py           — Mixture contamination model engine
    models/
        cache.py                 — Joblib caching decorator
        utils.py                 — Table sanitization helpers
        gaia/
            constants.py         — Gaia zero-point data
            gaia.py              — GaiaData, GaiaSample, filter classes
            wise.py              — WISEData, WISESample
            lightcurves.py       — Epoch photometry I/O and derived columns
```

## Usage

Submodules are imported on demand to avoid pulling heavy dependencies
(PyMC, astroquery) at package import time.

```python
# Plotting (applies rcParams on import)
import ugdatalab.plotting as ugplt

fig = ugplt.textwidth_figure(4)
axes = ugplt.subpanels(fig, 2, height_ratios=(3, 1))
axes[0].plot(x, y, **ugplt.FIT_STYLE)
axes[1].axhline(0, **ugplt.GUIDE_STYLE)

# Periodogram
from ugdatalab.methods.periodogram import lomb_scargle
result = lomb_scargle(times, values, errors, period_min=0.2, period_max=1.2)

# Fourier fitting
from ugdatalab.methods.fourier import fourier_fit, phase_fold
fit = fourier_fit(x, y, y_err, period=0.567, k=3)

# Cross-validation (holdout by default; set n_folds=k for k-fold)
from ugdatalab.methods.cross_validate import cross_validate
cv = cross_validate(x, y, y_err, fit_fn, param_values)

# Bayesian parameter estimation (PyMC NUTS)
from ugdatalab.methods.bayesian.likelihoods import LinearGaussianLikelihood
from ugdatalab.methods.bayesian.mcmc import nuts_sample
lk = LinearGaussianLikelihood(x, y, y_err)
result = nuts_sample(lk)

# Bayesian outlier rejection (mixture contamination)
from ugdatalab.methods.bayesian.mixture import mixture_contamination
result = mixture_contamination(lk)

# Gaia data models
from ugdatalab.models.gaia import GaiaData, GaiaSample, LindegrenC1
```

## Architecture

### Methods

Generic, reusable analysis routines. 

- **`Fit`** -- ABC for fitted models bound to `(x, y, y_err)`, with `predict(x)`, `predict_std(x)`, and a derived `chi2_r` property.
- **Bayesian framework (`methods/bayesian/`)** -- three orthogonal axes:
  - *Data layout*: `RegressionLikelihood` (`y_i = f(x_i; θ)`) and `GridLikelihood` (`y = f(θ)` at fixed coords) as sibling ABCs of `Likelihood`.
  - *Inlier model*: `RegressionInlierComponent` / `GridInlierComponent` strategies (e.g. `LinearInlier`).
  - *Outlier treatment*: shared `OutlierComponent` strategies (e.g. `GaussianOutlier`); `MixtureLikelihood` is the `Likelihood` subtype that adds the outlier branch.
- **Bayesian sampler** -- `NUTSSampler.sample(likelihood)` for any `Likelihood`, `NUTSSampler.sample_mixture(likelihood)` for any `MixtureLikelihood`. Returns a `Posterior`. See `FRAMEWORK.md` for the full architecture.
- **Signal detection** -- `lomb_scargle` (Lomb-Scargle periodogram).
- **Fitting** -- `fourier_fit` (weighted least-squares Fourier series).
- **Model selection** -- `cross_validate` (holdout when `n_folds=1`, k-fold when `n_folds>1`).

### Models

Data access and caching for specific surveys.

- **Gaia pipeline** — `GaiaData` → `GaiaSample` → `Local`, `StrictG`, `StrictBPRP`, `LindegrenC1`, `LindegrenC2`, `StrictReddening` for progressive query, quality filtering, and caching.
- **`WISEData` / `WISESample`** — Gaia–AllWISE cross-match with photometric quality cuts.

### Plotting

Publication-quality matplotlib defaults applied on `import ugdatalab.plotting`.
Style dictionaries (`GRID_STYLE`, `GUIDE_STYLE`, `FIT_STYLE`, etc.) unpack
as `**kwargs` into matplotlib calls.
