# Missing Numerical Claims: Report vs Notebooks

All numerical values cited in `report/main.tex` that are **not** computed and displayed in any notebook cell.

## Notebook 01 (01-lightcurves.ipynb)

| Report claim | Tex line | Status |
|---|---|---|
| 89/100 periods agree within 1% | 41, 320 | **MISSING** |
| 11 outliers match p1_o | 320 | **MISSING** |
| P1/P0 ≈ 0.745 | 320 | **MISSING** — computed in plotter but not displayed |
| Fourier mean RMS = 0.0147 mag | 364, 379 | **MISSING** |
| Epoch mean RMS = 0.0282 mag | 364, 379 | **MISSING** |
| Optimal K = 5 for example star | 343 | Partial — used but not printed |

## Notebook 02 (02-calibration-sample.ipynb)

| Report claim | Tex line | Status |
|---|---|---|
| 1050 initial sample size | 275 | **MISSING** |
| 1050 → 972 after C1/C2 | 420 | **MISSING** |
| 972 → 884 after mixture model | 437 | **MISSING** |
| 1006 pass C1 alone | 420 | **MISSING** |
| 973 pass C2 alone | 420 | **MISSING** |
| 78 removed by C1/C2 | 422 | **MISSING** |
| 7.4% rejection rate | 420 | **MISSING** |
| LMC rejection 3/24 = 12.5% | 422 | **MISSING** |
| SMC rejection 19/24 = 79.2% | 422 | **MISSING** |
| Basal rejection 56/945 = 5.9% | 422 | **MISSING** |
| C1/C2 RMS improvement 57% | 439 | **MISSING** |
| RMS 1.229 → 0.259 (79% reduction) | 439 | **MISSING** |
| 48 stars >1 mag → 1 | 439 | **MISSING** |
| 560 RRab / 316 RRc / 8 RRd | 518, 536 | **MISSING** |

## Notebook 03b (03b-optical-sampler-comparison.ipynb)

| Report claim | Tex line | Status |
|---|---|---|
| MH acceptance rate 0.549 | 501 | Partial — scan shown, final not printed |
| MH 25,000 steps / 5,000 burn-in | 501 | **MISSING** — in code but not printed |
| NUTS 1,500 tune / 3,000 draw | 503–505 | **DISCREPANCY** — code uses 1,000/2,000 |
| G RRab: a = −2.2316 ± 0.1283 | 520 | **MISSING** |
| G RRab: b = +0.6531 ± 0.0085 | 521 | **MISSING** |
| G RRab: σ = 0.1762 ± 0.0077 | 522 | **MISSING** |
| G RRc: a = −2.2169 +0.18/−0.17 | 538 | **MISSING** |
| G RRc: b = +0.5757 ± 0.0108 | 539 | **MISSING** |
| G RRc: σ = 0.1669 ± 0.0093 | 540 | **MISSING** |

## Notebook 03c (03c-optical-ir-comparison.ipynb)

| Report claim | Tex line | Status |
|---|---|---|
| 271,779 full join rows | 551 | **MISSING** |
| 875 matched to calibration | 551 | **MISSING** |
| 839 after quality cuts | 551 | **MISSING** |
| 527 RRab + 304 RRc + 8 RRd | 551 | **MISSING** |
| W2 RRab: a = −2.9845 ± 0.1189 | 559 | **MISSING** |
| W2 RRc: a = −3.2460 ± 0.1555 | 560 | **MISSING** |
| W2 σ_int ≈ 0.14–0.15 | 570 | **MISSING** |
| Δa_RRab = −0.75 ± 0.17 | 566 | **MISSING** |
| Δa_RRc = −1.03 ± 0.24 | 566 | **MISSING** |

## Notebook 03d (03d-period-color.ipynb)

| Report claim | Tex line | Status |
|---|---|---|
| PC RRab: a = +0.2483 ± 0.0417 | 617 | **MISSING** |
| PC RRab: b = +0.7095 ± 0.0103 | 618 | **MISSING** |
| PC RRab: σ = 0.0502 ± 0.0025 | 619 | **MISSING** |
| PC RRc: a = +0.4766 ± 0.0548 | 627 | **MISSING** |
| PC RRc: b = +0.6817 ± 0.0267 | 628 | **MISSING** |
| PC RRc: σ = 0.0508 ± 0.0027 | 629 | **MISSING** |

## Notebook 04 (04-reddening-sfd-maps.ipynb)

| Report claim | Tex line | Status |
|---|---|---|
| 269,772 full catalog | 279 | **MISSING** |
| Cut: 19,607 removed (7.7%) | 671 | **MISSING** |
| Cut: 1,474 removed (0.6%) | 672 | **MISSING** |
| Cut: 55,304 removed (28.4%) | 673 | **MISSING** |
| Cut: 8,325 removed (3.3%) | 674 | **MISSING** |
| Cut: 38,582 removed (15.2%) | 675 | **MISSING** |
| 130,877 final (48.5%) | 678 | **MISSING** |
| 88,028 RRab + 42,849 RRc | 695 | **MISSING** |
| 109,325 RRab with finite g_absorption | 661 | **MISSING** |
| Median offset 0.114 mag | 656 | **MISSING** |
| Pearson R² = 0.459 overall | 737 | **MISSING** |
| All latitude-bin R² (Table 2) | 727–737 | **MISSING** |
| Similar-scale slope = 1.11 | 746 | Partial — computed but not printed |
| Similar-scale N = 127,893 | 759 | **MISSING** |
| Similar-scale R² = 0.878 | 763 | **MISSING** |
| Large-SFD N = 74 | 760 | **MISSING** |

## Discrepancy

The report claims NUTS used **1,500 tune / 3,000 draw** steps (lines 503–505), but the notebook code uses `nuts_sample` defaults of **1,000 tune / 2,000 draw**.
