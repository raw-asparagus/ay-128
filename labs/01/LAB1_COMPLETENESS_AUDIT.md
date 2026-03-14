# Lab 1 Completeness Audit

This note records the PDF-to-notebook coverage check for `course_materials_sp2026/labs/lab_1/AY128_Lab1_rrlyrae_dust_spring2026.pdf` after the notebook markdown/display refactor. The goal was to make each required handout item explicit in the notebook trail without changing the numerical analysis pipeline or the saved science results.

## Coverage Matrix

| PDF item | Notebook | Status | Notes |
| --- | --- | --- | --- |
| 1 | `labs/01/01-01.ipynb` | Complete | ADQL query, Astroquery submission, and first-row display are explicit. |
| 2 | `labs/01/01-01.ipynb` | Complete | Raw and phase-folded Gaia light curves are fetched and plotted. |
| 3 | `labs/01/01-01.ipynb` | Complete | Lomb-Scargle period estimation, periodogram display, and flux-space mean magnitude calculation are explicit. |
| 4 | `labs/01/01-01.ipynb` | Complete | Notebook compares recovered periods to Gaia catalog values and discusses RRd mode-identification limits. |
| 5 | `labs/01/01-01.ipynb` | Complete | Fourier-series design matrix is derived explicitly in markdown. |
| 6 | `labs/01/01-01.ipynb` | Complete | Model-versus-data figure is present and interpreted. |
| 7 | `labs/01/01-01.ipynb` | Complete | Train/CV `\chi_r^2` scan is plotted and discussed. |
| 8 | `labs/01/01-01.ipynb` | Complete | Ten-day extrapolation plot and interpretation are explicit. |
| 9 | `labs/01/01-01.ipynb` | Complete | Fourier-based mean magnitude comparison and residual discussion are explicit. |
| 10 | `labs/01/01-02.ipynb` | Complete | RRab/RRc light-curve morphology comparison and astrophysical interpretation are explicit. |
| 11 | `labs/01/01-02.ipynb` | Complete | Netzel-style non-strict periodicity discussion is explicit and scoped correctly. |
| 12 | `labs/01/02-01.ipynb` | Complete | Low-dust local Gaia selection is explicit, including why low extinction is required. |
| 13 | `labs/01/02-01.ipynb` | Complete after refactor | Galactic-coordinate low-dust sky plot is now shown explicitly. |
| 14 | `labs/01/02-01.ipynb` | Complete after refactor | Raw pre-C1/C2 period-versus-absolute-magnitude plot with errors is now shown explicitly. |
| 15 | `labs/01/02-01.ipynb` | Complete | C1/C2 quality cuts, scatter reduction, and astrophysical interpretation are explicit. |
| 16 | `labs/01/02-01.ipynb` | Complete | Final cleaned sample plot and outlier-rejection discussion are explicit. |
| 17 | `labs/01/02-pre02.ipynb` | Complete | One-dimensional MH sampler benchmark, acceptance tuning, histogram, and trace diagnostics are explicit. |
| 18(i) | `labs/01/02-02.ipynb` | Complete after refactor | Custom MH fit, priors, likelihood, corner plot, and explicit 50-line posterior-over-data view are surfaced. |
| 18(ii) | `labs/01/02-02.ipynb` | Complete | PyMC NUTS + `Potential` workflow and interpretation are explicit. |
| 18(iii) | `labs/01/02-02.ipynb` | Complete after refactor | Separate method-colored corner plots remain primary; a supplementary checkpoint-style overlay now satisfies the single-comparison-corner requirement. |
| 19 | `labs/01/03.ipynb` | Complete | Gaia-to-WISE cross-match logic through best-neighbour tables is explicit. |
| 20 | `labs/01/03.ipynb` | Complete after refactor | WISE `W2` PL calibration is carried out with the same Bayesian structure as the optical fit, including explicit 50-line posterior-over-data views. |
| 21 | `labs/01/03.ipynb` | Complete | Notebook explicitly answers which band is steeper and why. |
| 22 | `labs/01/04.ipynb` | Complete after refactor | Literature comparison is explicit for Gaia `G` versus optical `V` context and `W2` versus infrared literature, with an adopted numerical visual-band benchmark shown in the table. |
| 23 | `labs/01/05.ipynb` | Complete after refactor | Period-color model, color-uncertainty propagation, and subclass-specific interpretation are explicit. |
| 24 | `labs/01/05.ipynb` | Complete after refactor | Full `vari_rrlyrae` + `gaia_source` ADQL provenance is explicit, and the RRab/RRc-only downstream scope is now surfaced in a table. |
| 25 | `labs/01/06.ipynb` | Complete after refactor | Intrinsic color, color excess, and empirical `A_G` construction are explicit, with the RRab/RRc-only downstream scope called out directly. |
| 26 | `labs/01/06.ipynb` | Complete | Comparison to Gaia DR3 `g_absorption` is explicit, with sanity cuts and interpretation. |
| 27 | `labs/01/07.ipynb` | Complete after refactor | Cleaned Aitoff reddening map, cut rationale, retained-count logic, and non-uniform tracer discussion are explicit. |
| 28 | `labs/01/08.ipynb` | Complete | SFD sampling at the same sightlines, map comparison, and matched-sightline diagnostics are explicit. |
| 29 | `labs/01/08.ipynb` | Complete | Notebook explicitly explains why the RR Lyrae map should not exactly match SFD. |

## Refactor-Specific Notes

- `02-01.ipynb` was the main completeness gap before the refactor. Items 13 and 14 were described in prose but not shown as explicit notebook outputs.
- `02-02.ipynb` intentionally keeps the separate sampler-colored corner plots as the cleaner diagnostic view. A compact overlay was added only so the notebook also matches the checkpoint wording in the handout.
- `05.ipynb` now makes item 24 auditable in the notebook itself instead of leaving that logic implicit in `ugdatalab/dust.py`.
- `07.ipynb` now explicitly reconciles the notebook's `E(G_BP-G_RP)` map with the handout wording that refers to coloring by `A_G`.

## Non-Blocking Future Improvements

These would improve pedagogy or reproducibility without changing the current analysis results:

- Re-execute the edited notebooks and save fresh outputs once the text changes are final, so the stored notebook outputs match the new explanatory flow exactly.
- Add explicit handout-item callouts in major markdown headers, for example `## Item 15: Gaia DR2 Quality Cuts`, to make grading cross-reference faster.
- Add multi-chain diagnostics (`\hat{R}`, ESS, divergence summaries) to the MCMC notebooks if the goal shifts from pedagogical demonstration toward research-grade posterior validation.
- Add a short appendix notebook or markdown note summarizing the main limitations of direct absolute-magnitude-space fitting versus parallax-space or hierarchical alternatives.
