# Framework & Workflow

This document describes the architectural conventions for building data analysis
pipelines with `ugdatalab`. It covers how data is acquired, filtered, analyzed,
visualized, and collated into a report.

## Architecture Overview

The project separates concerns into three layers:

```
ugdatalab/                              labs/NN/
┌──────────────────────────────┐   ┌────────────────────────────┐
│  models/   — data acquisition│   │  notebooks  — analysis     │
│  methods/  — analysis engines│──▶│  plotters.py — figures      │
│  plotters/ — generic plots   │   │  report/     — write-up    │
│  plotting  — style constants │   └────────────────────────────┘
└──────────────────────────────┘
```

**`ugdatalab`** provides reusable, survey-agnostic analysis engines and
survey-specific data access. **Each lab** wires those building blocks into
a specific scientific investigation, with its own plotters and report.

## Layer 1 — Data Acquisition (`models/`)

Each astronomical survey gets its own subpackage under `models/`. A survey
module is responsible for:

1. **Querying** the archive (TAP, HTTP, file I/O).
2. **Caching** raw results so re-runs are fast (`@_cache_stable` via joblib).
3. **Sanitizing** column types (`_sanitize_table` with a schema dict).
4. **Attaching derived columns** (distance modulus, absolute magnitude, errors).
5. **Defining quality filters** as subclasses of the base data class.

### Data class pattern

```python
@dataclass
class SurveyData:
    """Fetches and caches raw query results."""
    query: str
    data: Table = field(init=False)

    def __post_init__(self):
        data = _get_cached(self.query)
        _sanitize_table(data, SCHEMA)
        _attach_derived_columns(data)
        self.data = data
```

### Filter-as-subclass pattern

Each quality cut is a single class that takes a parent data object and
applies one filter to produce a new `.data` table:

```python
class StrictSNR(SurveyData):
    def __init__(self, source: SurveyData):
        self.data = source.data[source.data["snr"] > 5]
```

Filters are composed in notebooks by chaining constructors:

```python
raw    = SurveyData(query)
local  = LocalCut(raw)
strict = StrictSNR(local)
clean  = QualityCut(strict)
```

This makes the filtering chain readable, auditable, and self-documenting:
each step is one line, one class, one row in the sample-size table. Filters
are survey-specific and defined alongside their survey module — they are not
shared across surveys.

### Caching

All expensive queries (TAP, HTTP downloads) use `@_cache_stable` backed by
joblib. The cache lives in `.joblib-cache/` and persists across sessions.
Re-running a notebook after the first fetch is fast.

## Layer 2 — Analysis Engines (`methods/`)

Methods are survey-agnostic. They operate on numpy arrays (`x`, `y`, `y_err`)
and return frozen dataclass results.

### Contracts

Two ABC ladders define the contracts:

- **`Fit`** — every fitted model exposes `predict(x) -> ndarray`. Use this
  for trained-model artifacts that aren't bound to specific data (e.g.
  `CannonModel`).
- **`DataFit(Fit)`** — adds `x, y, y_err`, abstract `n_params`, and a
  derived `chi2_r` property (computed from `predict` + `total_variance`).
  Use for fits bound to the data they were produced from
  (`MCMCResult`, `FourierFit`).
- **`Likelihood`** — Bayesian models expose `param_labels`,
  `physical_param_names`, `build_pymc()`, `predict(x, theta)`, and
  `total_variance(theta)`.
- **`MixtureLikelihood(Likelihood)`** — additionally exposes
  `build_pymc_mixture()` and `inlier_probs()`. Required by
  `mixture_contamination`.

Engines consume these contracts without knowing the concrete implementation:

```
Likelihood (ABC)
    └── GaussianLikelihood (ABC — Gaussian noise + outlier background)
            └── LinearGaussianLikelihood (concrete — y = ax + b + σ_s)
            └── PolynomialGaussianLikelihood (concrete — future lab)
```

### Engine functions

Engines are stateless functions that take a `Likelihood` and return a
frozen result:

| Engine | Input | Output | Purpose |
|---|---|---|---|
| `nuts_sample` | `Likelihood` | `MCMCResult` | NUTS parameter estimation |
| `mixture_contamination` | `Likelihood` | `MixtureResult` | Outlier rejection via mixture model |
| `lomb_scargle` | `x, y, y_err` | `PeriodogramResult` | Period detection |
| `fourier_fit` | `x, y, y_err, period, k` | `FourierFit` | Weighted Fourier series fit |
| `cross_validate` | `x, y, y_err, fit_fn, params, *, n_folds=1, cv_fraction=0.2` | `ValidationResult` | Holdout (n_folds=1) or k-fold model selection |

### Result dataclasses

All results are frozen dataclasses carrying everything downstream code needs:

- **`MCMCResult`**: `theta`, `samples`, `log_probs`, `labels`, `chi2_r`,
  `x`, `y`, `y_err`, plus `predict(x, theta=None)`, `total_variance(theta=None)`,
  and `predict_std(x)`. Self-contained — no Likelihood back-reference.
- **`MixtureResult`**: `inlier_prob`, `theta`, `samples`, `log_probs`,
  `labels`.
- **`FourierFit`**: `beta`, `beta_cov`, `period`, `k`, `x`, `y`, `y_err`,
  plus `predict()`, `predict_std()`, and inherited `chi2_r` (property).
- **`PeriodogramResult`**: `periods`, `power`, `best_period`, `best_power`,
  `fap`.
- **`ValidationResult`** (and subclasses): `param_values`, `chi2r_train`,
  `chi2r_cv`, `best_param`.

Results are immutable and self-contained. Plotters and notebooks consume
them without needing to know how they were produced.

### Adding a new likelihood

To support a new model (e.g., polynomial), subclass `GaussianLikelihood`
and implement three methods:

```python
class PolynomialGaussianLikelihood(GaussianLikelihood):
    def predict(self, x, theta) -> ndarray: ...
    def total_variance(self, theta) -> ndarray: ...
    def _pymc_inlier_model(self, model): ...
```

This plugs directly into `nuts_sample` and `mixture_contamination` with
no changes to the engines.

### Default arguments

Package code and business code follow different rules for default parameter
values.

**Package code (`ugdatalab/`)** — defaults are allowed and encouraged for
reusable APIs, since callers shouldn't need to know implementation details
like optimal concurrency or polynomial degree. Every default must be
documented: what the value is, why it was chosen, and when to override.
Defaults encoding scientific choices (e.g., `bad_bits`, `degree`,
`period_min`) require especially clear documentation.

**Business code (`labs/NN/`)** — no default parameter values unless
absolutely necessary for functional reuse. Every argument should be
explicit at the call site so the notebook is self-documenting. Reading a
plotter call should tell you exactly what is being plotted without chasing
defaults in the function signature.

```python
# Package code — defaults are fine, documented in docstring
def _normalize_spectrum(flux, error, wavelength, continuum_mask, degree=4):
    """..., degree: Chebyshev polynomial order (default 4; reduce to 2-3
    if residuals show overfitting)."""

# Business code — required data first, optional overlays after
def plot_kiel_diagram(fitted_labels, isochrone_tracks=None):
    ...

# In notebook:
plotters.plot_kiel_diagram(fitted_cv, [iso_solar, iso_poor])
```

Module-level structural constants (`_FIGURES_DIR`, `savefig`) are not
function defaults and are fine in both layers.

## Layer 3 — Visualization

Visualization has two tiers: **generic plotters** in `ugdatalab/plotters/`
and **lab-specific plotters** in each `labs/NN/plotters.py`.

### Generic plotters (`ugdatalab/plotters/`)

Reusable across labs. Consume result dataclasses directly:

- `plot_trace(result)` — trace plot of MCMC parameters + log-probability.
- `plot_corner(result)` — corner plot with 16/50/84 quantiles.
- `plot_posterior(result, param_idx=)` — 1D posterior histogram.
- `predict_posterior(result)` — compute posterior predictive summaries.
- `plot_posterior_predictive(result)` — data + median + credible bands.

### Lab-specific plotters (`labs/NN/plotters.py`)

Each lab defines its own plotter module. Every function follows the same
pattern:

```python
def plot_something(ugdatalab_result, ...):
    fig, ax = textwidth_figure(height)       # figure helpers from plotting.py
    # ... plot using style constants & dicts
    savefig(fig, "fig_something.pdf")         # → report/figures/
    return ax                                 # → notebook inline display
```

Conventions:

- **Imports** style constants and dicts from `ugdatalab.plotting` — never
  hardcodes `lw=0.6` or `alpha=0.75`.
- **Uses figure helpers** (`textwidth_figure`, `columnwidth_figure`,
  `subpanels`) — never raw `plt.figure(figsize=...)`.
- **Accepts ugdatalab objects** as arguments (data classes, result
  dataclasses).
- **Single save point**: `savefig(fig, name)` writes to `report/figures/`.
- **Returns axes** for notebook inline display.
- **No rcParams mutations** inside functions — all rcParams are set once by
  importing `ugdatalab.plotting`.

### Style system (`plotting.py`)

Importing `ugdatalab.plotting` activates all rcParams (LaTeX, ticks, grid,
DPI). The module exports:

- **Page dimensions**: `TEXTWIDTH_IN`, `COLUMNWIDTH_IN`.
- **Named constants**: `LW_*`, `MS_*`, `SS_*`, `ALPHA_*`, `NEUTRAL_COLOR`.
- **Style dicts**: `GRID_STYLE`, `GUIDE_STYLE`, `FIT_STYLE`, `MODEL_STYLE`,
  `ERRORBAR_STYLE`, `FILL_STYLE`, `SCATTER_STYLE` — unpack as `**kwargs`.
- **Figure helpers**: `textwidth_figure`, `columnwidth_figure`,
  `landscapewidth_figure`, `corner_figure`, `subpanels`.
- **Reference lines**: `zero_line(ax)`, `unity_line(ax)`.

Full details in `PLOTTING_STYLE.md`.

## Notebook Workflow

Each lab is a directory (`labs/NN/`) containing notebooks, a plotter
module, data files, and a report subdirectory:

```
labs/NN/
    01-data-acquisition.ipynb
    02-analysis.ipynb
    ...
    plotters.py
    intermediate_data.npz
    report/
        main.tex
        figures/
            fig_something.pdf
```

### Notebook structure

Each notebook follows the same pattern:

1. **Single import cell** at the top — `ugdatalab` classes, `plotters`,
   standard libraries.
2. **Data acquisition or loading** — construct model objects or load `.npz`
   from a prior notebook.
3. **Analysis** — call engine functions, inspect results.
4. **Visualization** — call plotter functions (which save PDFs and display
   inline).
5. **Persistence** — save intermediate products as `.npz` for downstream
   notebooks.

### Inter-notebook data flow

Notebooks pass intermediate products via `.npz` files within the lab
directory. Each `.npz` is produced by exactly one notebook and consumed
by one or more downstream notebooks:

```
NB 01 (acquire)
    ↓
NB 02 (filter) → sample.npz
    ↓
NB 03 (fit) → results.npz
    ↓
NB 04 (validate)
```

A summary notebook at the end re-derives all computed values and serves as
the single source of truth for numerical claims in the report.

### Reproducibility

- All random operations use explicit seeds (`seed=` parameter on engines,
  `np.random.default_rng(seed)` for manual sampling).
- All data queries are cached — re-running notebooks after the first fetch
  is deterministic.
- Source IDs come from data queries, not hardcoded values. Paths use
  `Path(__file__).parent` or equivalent.

## Report Collation

Each lab produces a LaTeX report in `labs/NN/report/`:

```
report/
    main.tex          — full write-up
    figures/
        fig_*.pdf     — one PDF per plotter function
```

### Figure–notebook traceability

Every figure in the report traces to a specific plotter function call in a
specific notebook. The chain is:

```
notebook cell  →  plotters.plot_X(result)  →  savefig(fig, "fig_X.pdf")
                                                     ↓
report/main.tex  →  \includegraphics{figures/fig_X.pdf}
```

### Numerical claims

Every number cited in the report (sample sizes, fit parameters, chi-squared
values) has a corresponding notebook cell that computes it. The summary
notebook is authoritative — if the report and notebook disagree, update the
report to match the notebook.

## Putting It Together

A typical analysis pipeline, from raw data to report figure:

```python
# 1. Data acquisition (models/)
sample = GaiaSample(query)
clean  = StrictSNR(Local(sample))

# 2. Analysis (methods/)
lk     = LinearGaussianLikelihood(x, y, y_err)
result = nuts_sample(lk)

# 3. Visualization (plotters/)
from ugdatalab.plotters.bayesian import plot_corner, plot_posterior_predictive
plot_corner(result)                           # generic
plot_posterior_predictive(result)              # generic

from plotters import plot_custom_figure
plot_custom_figure(result, clean)             # lab-specific → report/figures/
```

Each layer knows only about the layer above it: plotters consume result
objects, engines consume likelihood objects, models produce data tables.
Nothing in `ugdatalab` knows about any specific lab.

## New Lab Checklist

When starting `labs/NN/`, follow these steps:

### 1. Create the directory structure

```
labs/NN/
    report/
        figures/
```

### 2. Create `plotters.py` with the standard scaffold

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ugdatalab.plotting import (
    # page dimensions
    TEXTWIDTH_IN,
    # line weights
    LW_NONE, LW_FINE, LW_LIGHT, LW_STANDARD, LW_MEDIUM,
    # marker sizes (for ax.plot ms= and ax.scatter s=)
    MS_MICRO, MS_FINE, MS_STANDARD, MS_MEDIUM, MS_LARGE,
    SS_MICRO, SS_FINE, SS_STANDARD,
    # transparency
    ALPHA_EXTRA_LIGHT, ALPHA_FAINT, ALPHA_LIGHT, ALPHA_STANDARD,
    # colors
    NEUTRAL_COLOR,
    # style dicts
    GUIDE_STYLE, FIT_STYLE, MODEL_STYLE, FILL_STYLE,
    # figure helpers
    textwidth_figure, columnwidth_figure, subpanels, zero_line,
)

_FIGURES_DIR = Path(__file__).parent / "report" / "figures"


def savefig(fig, name):
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES_DIR / name)
```

Import additional constants and helpers as needed. Delete unused imports
before finalizing.

### 3. Create the first notebook

Number notebooks sequentially: `01-descriptive-name.ipynb`. The first
cell should be a single import block:

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy import table

# ugdatalab — models
from ugdatalab.models.sdss import SDSSData  # or whatever survey

# ugdatalab — methods
from ugdatalab import fourier_fit, lomb_scargle, cross_validate

# lab plotters
import plotters
```

### 4. Add survey module to `ugdatalab` if needed

If the lab uses a new survey, create `ugdatalab/models/survey_name/`
following the existing `models/gaia/` pattern:

- `__init__.py` — re-export public classes.
- `constants.py` — zero-points, filter curves, fixed values.
- `survey.py` — base data class with `@_cache_stable` queries,
  `_sanitize_table`, derived columns, and filter subclasses.
- Update `ugdatalab/__init__.py` to re-export the new public classes.

### 5. Add new likelihoods to `ugdatalab` if needed

If the lab requires a model beyond `LinearGaussianLikelihood`, add a new
concrete class in `ugdatalab/methods/bayesian/likelihoods.py` (or a new
file if substantially different). Subclass `GaussianLikelihood` and
implement `predict`, `total_variance`, and `_pymc_inlier_model`.

### 6. Build notebooks in pipeline order

Each notebook should either produce a `.npz` file consumed by later
notebooks, or be a terminal analysis step. Keep the dependency graph
linear or tree-shaped — avoid cycles.

### 7. Write the summary notebook last

The final notebook re-derives every computed value cited in the report.
Display results as `pd.DataFrame` tables (for computed values) or
`astropy.table.Table` (for catalog data). This notebook is the single
source of truth.

### 8. Set up the report

Create `report/main.tex`. Include every figure from `report/figures/`
via `\includegraphics`. Ensure every numerical claim in the text traces
to a cell in the summary notebook.
