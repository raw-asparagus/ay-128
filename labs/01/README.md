# Lab 1 Workflow Guide

`ugdatalab` now covers the reusable logic across essentially the full RR Lyrae and dust workflow in Lab 1. The quickest path is to treat the package as the library layer and use the existing notebooks in this directory as the lab-specific orchestration layer.

## What Is Already Implemented

- `ugdatalab.models.gaia`
  Provides `GaiaData` for raw cached Gaia query results and `GaiaQuality` for the quality-filtered sample with `mu`, `M_G`, `sigma_M`, `period`, and RRab/RRc class flags.
- `ugdatalab.queries`
  Provides the Lab 1 ADQL builders for the top-100 RR Lyrae sample, the bright RRab/RRc comparison sample, the local low-dust calibration sample, and the full Gaia cross-match.
- `ugdatalab.lightcurves`
  Fetches and joins Gaia epoch photometry, computes flux-space mean magnitudes, adds epoch magnitude uncertainties, and estimates periods with Lomb-Scargle.
- `ugdatalab.fourier`
  Builds Fourier design matrices, fits harmonic light-curve models, cross-validates the harmonic order, extrapolates future magnitudes, and computes Fourier mean magnitudes.
- `ugdatalab.models`
  Provides `Local`, `StrictGBPRP`, `Cut1`, and `Cut2` for the main quality and locality cuts.
- `ugdatalab.plotting`
  Provides the reusable Gaia/deoutlier and MCMC plotting helpers used by the Lab 1 notebooks.
- `ugdatalab.models.deoutlier.MixtureContaminationModel`
  Performs class-specific outlier rejection in period-luminosity space and adds posterior inlier probabilities.
- `ugdatalab.relations`
  Prepares period-luminosity and period-color datasets and fits them with either Metropolis-Hastings or NUTS.
- `ugdatalab.dust`
  Loads saved posterior samples, computes class-specific intrinsic colors, `E_bprp`, `A_G_calc`, empirical-vs-catalog extinction residuals, and optional SFD comparisons.
- Plot helpers
  The package already includes Mollweide sky maps, period-luminosity plots, HR diagrams, kept-vs-removed comparisons, and inlier-probability plots.

## Notebook Map

- `01.ipynb`
  Uses the query, epoch-photometry, Lomb-Scargle, and Fourier helpers for Lab 1 parts 1-10.
- `part-2.ipynb`
  Uses `GaiaQuality`, the quality-cut classes, `MixtureContaminationModel`, `prepare_relation_data`, `fit_relation_mh`, and `fit_relation_nuts` to build the cleaned calibration sample and fit the class-specific period-luminosity and period-color relations.
- `part-3.ipynb`
  Continues from the fitted period-color posteriors and computes class-specific intrinsic colors, `E_bprp`, `A_G_calc`, the comparison to Gaia `g_absorption`, and the full-sky reddening map.
- `flat_pc_rrab.npy` and `flat_pc_rrc.npy`
  Store posterior samples reused by `part-3.ipynb` for the dust/reddening stage.

## Recommended Order

1. Install the package in the project environment with `pip install -e .[dev]`.
2. Run `01.ipynb` for the light-curve, Lomb-Scargle, and Fourier sections (parts 1-10).
3. Run `part-2.ipynb` to generate or validate the RR Lyrae calibration sample and fitted PL/PC relations.
4. Confirm the saved period-color posterior samples exist as `flat_pc_rrab.npy` and `flat_pc_rrc.npy`.
5. Run `part-3.ipynb` to compute the empirical extinction quantities and produce the reddening-map figures.
6. Use the notebook outputs, not the raw package API alone, as the direct basis for the lab writeup.

## Package-To-Analysis Mapping

- Sample construction:
  `GaiaQuality(query)` followed by `Local`, `StrictGBPRP`, `Cut1`, and `Cut2`.
- Light-curve download and period finding:
  `build_rrlyrae_top_n_query(...)`, `get_epoch_photometry(...)`, `join_catalog_with_epoch_photometry(...)`, and `lomb_scargle_periodogram(...)`.
- Fourier modeling:
  `fourier_fit(...)`, `cross_validate_harmonics(...)`, `predict_future_magnitude(...)`, and `estimate_fourier_mean_magnitudes(...)`.
- Outlier rejection:
  `MixtureContaminationModel(rrlyrae)`.
- Period-luminosity calibration:
  `prepare_relation_data(..., "pl")` and `fit_relation_mh(...)` or `fit_relation_nuts(...)`.
- Period-color calibration:
  `prepare_relation_data(..., "pc")` and `fit_relation_nuts(...)`.
- Dust stage:
  `load_relation_posteriors(...)`, `compute_period_color_extinction(...)`, and `empirical_vs_catalog_extinction(...)`.

## Important Constraints

- Live Gaia access is required for `GaiaData(...)`, `GaiaQuality(...)`, and `get_gaia(...)`.
- Query results are cached locally; reruns should reuse `.joblib-cache` when the same ADQL string is used.
- In restricted environments, you may need to set writable `MPLCONFIGDIR` and `XDG_CACHE_HOME` values before using Matplotlib, `corner`, or `pymc`.
- The exact one-to-one mapping to every question in the lab PDF was inferred mainly from the existing notebooks, especially the numbered sections in `part-3.ipynb`.

## Notebook Cache Pattern

For notebook-defined Gaia helpers, import `_cache_stable` from the package and give the helper its own namespace:

```python
from ugdatalab import _cache_stable, get_gaia

@_cache_stable(module="lab02.epoch_photometry")
def get_epoch_photometry(source_id):
    ...
```
