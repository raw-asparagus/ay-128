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
- `ugdatalab.deoutlier.MixtureContaminationModel`
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
- `02-01.ipynb`
  Builds the cleaned Gaia calibration sample and exports it as `rrlyrae_calibration_sample.npz`.
- `02-02.ipynb`
  Fits the class-specific Gaia $G$-band period-luminosity relations from `rrlyrae_calibration_sample.npz` and exports `rrlyrae_optical_pl_comparison_data.npz`.
- `03.ipynb`
  Loads `rrlyrae_calibration_sample.npz`, caches the Gaia+WISE join in `rrlyrae_gaia_wise_query_data.npz`, compares against `rrlyrae_optical_pl_comparison_data.npz`, and exports `rrlyrae_infrared_pl_comparison_data.npz`.
- `04.ipynb`
  Consumes `rrlyrae_optical_pl_comparison_data.npz` and `rrlyrae_infrared_pl_comparison_data.npz` for the optical-literature comparison and the Gaia $G$ versus WISE $W2$ discussion.
- `05.ipynb`
  Fits the class-specific Gaia period-color relations with native PyMC NUTS and writes notebook-local handoff archives `rrlyrae_optical_pc_comparison_data.npz` and `rrlyrae_rrab_rrc_full_catalog.npz` for `06.ipynb`.
- `06.ipynb`
  Loads `rrlyrae_optical_pc_comparison_data.npz` and `rrlyrae_rrab_rrc_full_catalog.npz`, computes `E_bprp` and empirical `A_G`, and compares that result to Gaia DR3 `g_absorption`.
- `part-3.ipynb`
  Legacy broader dust-stage notebook that combines the empirical extinction comparison with the full-sky reddening map.
- `rrlyrae_calibration_sample.npz`
  Stores the cleaned Gaia calibration sample exported by `02-01.ipynb`.
- `rrlyrae_optical_pl_comparison_data.npz`
  Stores the summarized RRab and RRc Gaia $G$-band relation values exported by `02-02.ipynb`.
- `rrlyrae_infrared_pl_comparison_data.npz`
  Stores the summarized RRab and RRc WISE $W2$ relation values exported by `03.ipynb`.
- `rrlyrae_optical_pc_comparison_data.npz` and `rrlyrae_rrab_rrc_full_catalog.npz`
  Store the local dust/reddening handoff data used by `06.ipynb`.

## Recommended Order

1. Install the package in the project environment with `pip install -e .[dev]`.
2. Run `01.ipynb` for the light-curve, Lomb-Scargle, and Fourier sections (parts 1-10).
3. Run `02-01.ipynb` to generate `rrlyrae_calibration_sample.npz`.
4. Run `02-02.ipynb` to generate `rrlyrae_optical_pl_comparison_data.npz`.
5. Run `03.ipynb` to generate the WISE comparison and `rrlyrae_infrared_pl_comparison_data.npz`.
6. Run `04.ipynb` after `02-02.ipynb` and `03.ipynb` to compare the derived Gaia $G$ and WISE $W2$ relations against the literature.
7. Run `05.ipynb` to generate `rrlyrae_optical_pc_comparison_data.npz` and `rrlyrae_rrab_rrc_full_catalog.npz` for the reddening/extinction stage.
8. Run `06.ipynb` to load those local archives, compute the empirical extinction quantities, and compare them to Gaia DR3 `g_absorption`.
9. Use `part-3.ipynb` only if you also want the older full-sky reddening-map workflow.
10. Use the notebook outputs, not the raw package API alone, as the direct basis for the lab writeup.

## Package-To-Analysis Mapping

- Sample construction:
  `GaiaQuality(query)` followed by `Local`, `StrictGBPRP`, `Cut1`, and `Cut2`.
- Light-curve download and period finding:
  `build_rrlyrae_top_n_query(...)`, `get_epoch_photometry(...)`, `join_catalog_with_epoch_photometry(...)`, and `lomb_scargle_periodogram(...)`.
- Fourier modeling:
  `fourier_fit(...)`, `cross_validate_harmonics(...)`, `predict_future_magnitude(...)`, and `fourier_mean_magnitude(...)`.
- Outlier rejection:
  `MixtureContaminationModel(rrlyrae)`.
- Period-luminosity calibration:
  `prepare_relation_data(..., "pl")` and `fit_relation_mh(...)` or `fit_relation_nuts(...)`.
- Period-color calibration:
  `prepare_relation_data(..., "pc")` and `fit_relation_nuts(...)`.
- Dust stage:
  `load_or_create_rrab_rrc_full_catalog(...)`, `load_optical_pc_comparison_data(...)`, `compute_period_color_extinction(...)`, and `empirical_vs_catalog_extinction(...)`.

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
