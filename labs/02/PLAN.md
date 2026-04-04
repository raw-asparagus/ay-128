# Plan: Lab 02 — Modeling Stellar Spectra

## Context

Lab 02 uses APOGEE DR17 spectra of red giant stars to build a data-driven spectral model ("The Cannon" per Ness et al. 2015). The pipeline: download spectra + labels from SDSS, normalize, train a per-pixel 2nd-order polynomial model, cross-validate by fitting labels on held-out stars, fit a mystery spectrum via MCMC, and compare with a neural network approach. This requires a new APOGEE survey module in `ugdatalab/models/`, a new Cannon engine in `ugdatalab/methods/`, and 6 lab notebooks with ~14 plotter functions.

Implementation will proceed in phases — the user will guide each step.

---

## 1. New `ugdatalab` Modules

### 1A. `ugdatalab/models/apogee/` — APOGEE survey module

Following the `models/gaia/` pattern: query, cache, sanitize, derive, filter.

**`constants.py`**
```python
APOGEE_BAD_PIXMASK_BITS = [0, 1, 2, 3, 4, 5, 6, 7, 12]
LABEL_NAMES = ["TEFF", "LOGG", "FE_H", "MG_FE", "SI_FE"]
LABEL_LATEX = [
    r"$T_{\rm eff}$", r"$\log g$", r"$[\mathrm{Fe/H}]$",
    r"$[\mathrm{Mg/Fe}]$", r"$[\mathrm{Si/Fe}]$",
]
N_LABELS = 5
FILLER_VALUE = -9999.0  # ASPCAP NULL sentinel

APOGEE_DR17_URL = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars"

# Catalog column type mapping
_APOGEE_SCHEMA = {
    str: ["apogee_id", "field", "telescope"],
    float: ["teff", "logg", "fe_h", "mg_fe", "si_fe", "snr", "vhelio_avg"],
    int: ["nvisits", "starflag", "aspcapflag"],
}
```

**`apogee.py`** — Catalog data classes
- `_get_apogee_allstar(fields: tuple[str, ...])` — `@_cache_stable`. SQL query via `astroquery.sdss.SDSS.query_sql(data_release=17)` joining `apogeeStar` and `aspcapStar`. Query includes: `apogee_id`, `field`, `telescope`, `snr`, `nvisits`, `vhelio_avg`, `teff`, `logg`, `fe_h`, `mg_fe`, `si_fe`, `starflag`, `aspcapflag`. Returns `astropy.table.Table`. Uses `tuple(fields)` for hashable cache key.
- `APOGEEData` — `@dataclass` with `fields: list[str]`. `__post_init__` calls `_get_apogee_allstar(tuple(self.fields))`, sanitizes via `_sanitize_table(_APOGEE_SCHEMA)`.
- `APOGEETrainingSet(APOGEEData)` — filter subclass:
  - All 5 labels finite and != -9999
  - SNR > 50
  - logg <= 4 and Teff <= 5700 (giants only)
  - [Fe/H] >= -1

**`spectra.py`** — Spectrum I/O and normalization

Chip boundaries are specified in the apStar FITS header (no manual detection needed):
- Blue: `BMIN`–`BMAX` (overlap-excluded: `BOVERMIN`–`BOVERMAX`)
- Green: `GMIN`–`GMAX` (overlap-excluded: `GOVERMIN`–`GOVERMAX`)
- Red: `RMIN`–`RMAX` (overlap-excluded: `ROVERMIN`–`ROVERMAX`)
Verified example: B=200-3332, G=3537-6139, R=6294-8391. Pixels outside these ranges (inter-chip gaps) have NaN flux.

Functions:
- `_reconstruct_wavelength(header) -> ndarray` — builds the wavelength array from the FITS header's log-linear WCS: `10**(CRVAL1 + CDELT1 * arange(NAXIS1))`, shape (8575,). Produces the same array as `continuum_wavelengths.npz["wavelengths"]`. Needed when processing individual FITS files independently of the continuum file (e.g., the mystery spectrum, or displaying a raw spectrum before normalization).
- `__get_apstar_spectra(apogee_id: str, telescope: str, field: str) -> dict` — `@_cache_stable`. HTTP GET to `{APOGEE_DR17_URL}/{telescope}/{field}/apStar-dr17-{apogee_id}.fits`. Opens via `astropy.io.fits.open(io.BytesIO(response.content))`. Returns dict with:
  - `flux`: coadded spectrum (HDU 1, row 0), shape (8575,)
  - `error`: HDU 2 row 0, shape (8575,)
  - `bitmask`: HDU 3 row 0, shape (8575,) int16
  - `wavelength`: from `_reconstruct_wavelength(header)`
- `_fetch_apstar_batch(catalog, max_workers=4) -> list[dict]` — parallel download via `ThreadPoolExecutor` and `as_completed`, same pattern as `_fetch_epoch_photometry` in `gaia/lightcurves.py`.
- `_apply_bitmask(flux, error, bitmask, bad_bits=APOGEE_BAD_PIXMASK_BITS) -> (flux, error)` — `mask = any(bitmask & (1 << bit) for bit in bad_bits)`, sets `error[mask] = np.inf`. Also sets `error[np.isnan(flux)] = np.inf`.
- `_identify_chips(header) -> list[slice]` — reads chip boundaries from FITS header keywords `BMIN/BMAX`, `GMIN/GMAX`, `RMIN/RMAX`. Returns 3 slices for blue/green/red chips.
- `_normalize_spectrum(flux, error, wavelength, continuum_mask, degree=4) -> (flux_norm, error_norm, continuum_fit)`:
  - Fits a separate Chebyshev polynomial for each of the 3 chips:
    1. Get chip slices from `_identify_chips(header)`
    2. **For each chip**: select pixels within the chip where `continuum_mask == True` AND `np.isfinite(error)` AND `error < np.inf`
    3. If fewer than `degree + 1` valid continuum pixels in a chip, set entire chip to `error = np.inf` (skip)
    4. Fit weighted Chebyshev polynomial to the chip's continuum pixels: `numpy.polynomial.chebyshev.Chebyshev.fit(wavelength[valid], flux[valid], deg=degree, w=1/error[valid]**2)` — `Chebyshev.fit` automatically maps the wavelength domain to [-1, 1] for numerical stability
    5. Evaluate the chip's polynomial at all pixels in that chip
    6. Divide flux and error by the chip's continuum (`flux_norm = flux / continuum`, `error_norm = error / continuum`)
  - Each chip gets its own independent polynomial — no polynomial spans across chip gaps
  - Continuum fit uncertainty is neglected (standard practice: many continuum pixels make the fit uncertainty small relative to per-pixel noise)
  - Returns: normalized flux, normalized error, full continuum fit array
- `_fetch_normalized_spectra(catalog, continuum_path, degree=4, max_workers=4) -> (flux, error, wavelength, continuum_mask, apogee_ids)`:
  - Loads `continuum_path` (an .npz file with keys `"wavelengths"` and `"continuum"`)
  - Downloads all spectra via `_fetch_apstar_batch`
  - Applies bitmask and normalizes each
  - Returns stacked arrays: flux (N, 8575), error (N, 8575), wavelength (8575,), continuum_mask (8575,) bool, apogee_ids (N,) str

**`__init__.py`** — re-exports `APOGEEData`, `APOGEETrainingSet`

### `ugdatalab/models/isochrones.py` — MIST isochrone access (survey-agnostic)

Not APOGEE-specific — lives directly in `models/`.

- `_get_mist_isochrone(age_gyr: float, feh: float) -> pd.DataFrame` — `@_cache_stable`. Uses `isochrones` package (tested, works with local grid data, no web API dependency):
  ```python
  from isochrones import get_ichrone
  mist = get_ichrone('mist')
  return mist.isochrone(age=np.log10(age_gyr * 1e9), feh=feh)
  ```
  Returns DataFrame with columns: `logTeff`, `Teff`, `logg`, `initial_mass`, `phase`, etc.

**Update `ugdatalab/__init__.py`** — add APOGEE imports (lazy, like Gaia).

### 1B. `ugdatalab/methods/cannon.py` — The Cannon engine

**Why not a Likelihood subclass**: The Cannon training step inverts the usual direction. A `Likelihood` models `y = f(x; theta)` for one dataset with MCMC over theta. The Cannon trains 8575 independent per-pixel models across all stars simultaneously (design matrix is stars x polynomial terms), then inverts to fit labels per star. It's a standalone engine like `fourier_fit` or `lomb_scargle`.

**`Fit` ABC compatibility**: The `Fit` ABC defines `predict(self, x: ndarray) -> ndarray` — model input → model output. Both Fourier and Cannon satisfy this naturally:
- `FourierFit.predict(x)`: time/phase array → magnitude predictions
- `CannonModel.predict(labels)`: label vector (5,) → spectrum (8575,)
Both map "what defines the prediction" to "the predicted values." No ABC changes needed.

**Design matrix**: 2nd-order polynomial in 5 labels = 21 terms:
```
[1, l1, l2, l3, l4, l5, l1², l1·l2, l1·l3, l1·l4, l1·l5, l2², l2·l3, l2·l4, l2·l5, l3², l3·l4, l3·l5, l4², l4·l5, l5²]
```

Labels centered (subtract mean) and scaled (divide by std) BEFORE building the design matrix. Statistics computed from training set only.

**Comparison with Ness et al. 2015**: The paper centers labels but does NOT scale by std. Our plan adds scaling (following the lab manual's recommendation to "rescale labels to order unity") for better numerical conditioning — the model is mathematically equivalent, scaling is absorbed into the theta coefficients. The paper uses 3 labels (Teff, logg, [Fe/H]) → 10 coefficients; we use 5 labels → 21 coefficients per the lab manual.

**Dataclass**:
```python
@dataclass(frozen=True)
class CannonModel(Fit):
    theta: np.ndarray        # (n_pixels, 21) per-pixel coefficients
    scatter: np.ndarray      # (n_pixels,) per-pixel intrinsic scatter s²_lambda
    label_names: list        # e.g. ["TEFF", "LOGG", ...]
    label_means: np.ndarray  # (5,) centering constants (from training set)
    label_stds: np.ndarray   # (5,) scaling constants (from training set)
    wavelength: np.ndarray   # (n_pixels,)
    chi2_r: float            # mean reduced chi2 across training set

    def predict(self, labels: np.ndarray) -> np.ndarray:
        """Predict spectrum for a single star. labels shape (5,) → (n_pixels,)."""
        scaled = (labels - self.label_means) / self.label_stds
        dv = _build_cannon_design_vector(scaled)  # (21,)
        return self.theta @ dv                     # (n_pixels,)

    def predict_batch(self, labels: np.ndarray) -> np.ndarray:
        """Predict spectra for N stars. labels shape (N, 5) → (N, n_pixels)."""
        scaled = (labels - self.label_means) / self.label_stds
        X = _build_cannon_design_matrix(scaled)    # (N, 21)
        return X @ self.theta.T                    # (N, n_pixels)

    def gradient(self, label_idx: int) -> np.ndarray:
        """Gradient spectrum df/dl_i, shape (n_pixels,).
        Computed analytically from theta and polynomial structure.
        Returns derivative w.r.t. UNSCALED label (divides by label_std)."""
```

**Functions**:
```python
def _build_cannon_design_matrix(labels: np.ndarray) -> np.ndarray:
    """Build design matrix. labels shape (N, 5) centered/scaled → (N, 21)."""
    N, L = labels.shape  # L=5
    terms = [np.ones(N)]
    for i in range(L):
        terms.append(labels[:, i])
    for i in range(L):
        for j in range(i, L):
            terms.append(labels[:, i] * labels[:, j])
    return np.column_stack(terms)  # (N, 21)

def _build_cannon_design_vector(labels: np.ndarray) -> np.ndarray:
    """Single-star design vector. labels shape (5,) centered/scaled → (21,)."""
    return _build_cannon_design_matrix(labels.reshape(1, -1)).ravel()

def _train_pixel(flux_pixel, error_pixel, design_matrix) -> tuple:
    """Train one pixel via profile likelihood.

    For fixed s², theta is WLS: theta_opt = (X^T W X)^{-1} X^T W f
    where W = diag(1/(sigma² + s²)).

    Optimize over log(s²) in [-20, 2] via scipy.optimize.minimize_scalar,
    solving WLS analytically at each evaluation.

    Guard: if fewer than 25 stars have finite error, return (zeros, inf).

    Returns (theta_vec shape (21,), s2 float).
    """

def train_cannon(flux, error, labels, wavelength, label_names=None) -> CannonModel:
    """Train The Cannon.

    1. Compute label_means, label_stds from training set.
    2. Center/scale labels.
    3. Build design matrix X (N_train, 21).
    4. Loop over 8575 pixels calling _train_pixel.
    5. Compute training chi2_r.
    6. Return frozen CannonModel.

    ~30-60s for 8575 pixels with ~2000 training stars.
    """

def fit_star_labels(model, flux, error, x0=None) -> np.ndarray:
    """Fit labels for a single star (inverse problem).

    Minimizes chi2(labels) = sum_lambda [(f - predict(labels))² / (sigma² + s²)]

    Uses scipy.optimize.least_squares(method='trf') with:
    - Residual: r_lambda = (f - predict(labels)) / sqrt(sigma² + s²)
    - Analytical Jacobian: dr/dl_j = -(1/sqrt(var)) * d(predict)/dl_j
      where d(predict)/dl_j = theta @ d(design_vector)/dl_j
      and d(design_vector)/dl_j has known closed form from polynomial
    - Bounds: Teff in [3500, 6500], logg in [-1, 5], [Fe/H] in [-2, 1],
      [Mg/Fe] in [-1, 1], [Si/Fe] in [-1, 1]
    - x0 default: training set mean labels

    Returns fitted labels in ORIGINAL (unscaled) space, shape (5,).
    """

def fit_labels_batch(model, flux, error) -> np.ndarray:
    """Fit labels for N stars. Returns (N, 5)."""
```

### 1C. `ugdatalab/methods/cannon_likelihood.py` — PyMC wrapper for Problem 12

Wraps a trained `CannonModel` as a `Likelihood` for `nuts_sample`.

**Key implementation detail**: `nuts_sample` calls `likelihood._predict(likelihood.x, theta_median)` at line 80 to compute `chi2_r`. So `CannonLabelLikelihood` must implement `_predict(x, theta)` where `theta` is a 5-element label vector (the fitted parameters) and `x` is ignored (or is wavelength). The return is the predicted spectrum.

```python
@dataclass
class CannonLabelLikelihood(Likelihood):
    """Likelihood for fitting stellar labels given a trained Cannon model."""
    x: np.ndarray        # wavelength — present to fulfill ABC, used as x in MCMCResult.predict()
    y: np.ndarray        # observed normalized flux (n_pixels,)
    y_err: np.ndarray    # observed error (n_pixels,)
    model: CannonModel   # trained model (not a Likelihood param, added field)

    @property
    def param_labels(self) -> list[str]:
        return list(LABEL_LATEX)

    def _predict(self, x, theta):
        """Predict spectrum given label vector theta (5,).
        x is ignored (wavelength is baked into the model)."""
        return self.model.predict(theta)

    def _inlier_variance(self, theta):
        """Total variance per pixel: measurement + intrinsic scatter."""
        return self.y_err**2 + self.model.scatter

    def build_pymc(self):
        """Build PyMC model with 5 label parameters.

        Priors: Normal on each label, centered on training means,
        width = 2 × training stds (weakly informative).

        Likelihood: product over good pixels (finite error) of
            N(f_obs | predict(labels), sigma² + s²)

        Prediction in PyTensor:
            1. Center/scale: l_scaled = (l - mean) / std
            2. Build design vector (21,) via pt.concatenate of constant,
               linear, and cross terms
            3. flux_pred = pt.dot(theta_matrix, design_vec)
        """

    def _build_cannon_design_vector_pytensor(self, labels_scaled):
        """Build (21,) design vector from (5,) scaled labels in PyTensor.
        Helper for build_pymc()."""
        terms = [pt.ones(1)]
        for i in range(5):
            terms.append(labels_scaled[i:i+1])
        for i in range(5):
            for j in range(i, 5):
                terms.append(labels_scaled[i:i+1] * labels_scaled[j:j+1])
        return pt.concatenate(terms)

    def build_pymc_mixture(self):
        raise NotImplementedError("Mixture model not applicable for Cannon label fitting")

    def inlier_probs(self, trace, model_var_names):
        raise NotImplementedError("Inlier probabilities not applicable for Cannon label fitting")
```

This plugs into `nuts_sample` → `MCMCResult` → `plot_corner`/`plot_trace`.

Note: `plot_posterior_predictive` and `predict_posterior` from `ugdatalab.plotters.bayesian` will NOT work with this likelihood because they assume a 1D x-grid and call `_inlier_variance(s)` for each posterior sample, which doesn't match the Cannon semantics. For Problem 12, we only use `plot_corner` and `plot_trace`.

---

## 2. Notebook Breakdown

### `labs/02/01-training-set.ipynb` — Problems 1-4

| Step | Problem | Action |
|------|---------|--------|
| **Import cell** | — | ugdatalab models, plotters, numpy, matplotlib, astropy |
| Query catalog | P1 | `APOGEEData(fields=["M15", "N6791", "K2_C4_168-21", "060+00"])` |
| Apply cuts | P1 | `APOGEETrainingSet(source)` — display before/after counts |
| Display sample | P1 | Show `data[:10]` as astropy Table |
| **Log g calculation** | P1 | Markdown derivation: `g = GM/R²` in CGS. Three stages: MS (R=1 R☉ → log g ≈ 4.44), pre-He flash (R=100 R☉ → log g ≈ 0.44), core He burning (R=15 R☉ → log g ≈ 2.08). Discussion: the logg ≤ 4 cut selects giants. |
| **Corner plot** | P1 | `plotters.plot_label_corner(labels, label_names)` — Checkpoint #1 |
| Download spectra | P2-3 | `_fetch_normalized_spectra(catalog, continuum_path)` |
| **Raw spectrum plot** | P2 | `plotters.plot_example_spectrum(wavelength, flux_raw)` — before normalization |
| **Units discussion** | P2 | Markdown: flux in 10⁻¹⁷ erg/s/cm²/Å (spectral flux density). Barycentric frame: observed wavelengths corrected for Earth's orbital motion so that absorption features appear at rest wavelengths. Different for each visit because Earth's velocity component changes. |
| **Bitmask discussion** | P3 | Markdown: bits 0-7 flag bad detector pixels, cosmic rays, saturation, persistence, sky subtraction failures. Bit 12 flags unvisited pixels. Setting error=inf makes flagged pixels contribute zero weight to any chi-squared fit. |
| **Pseudo-continuum explanation** | P4 | Markdown: pseudo-continuum normalization divides by a smooth estimate of the underlying continuum level. "Pseudo" because we use empirically identified continuum pixels (where flux doesn't depend on spectral labels) rather than a physical stellar atmosphere model for the true continuum. It removes the broadband SED shape so that spectral features can be modeled as fractional deviations. |
| **Normalization plot** | P4 | `plotters.plot_normalization_diagnostic(...)` for star 2M21235315+1244123 — 3-panel: raw, continuum fit, normalized — **Checkpoint #1** |
| **Robustness check** | P4 | `plotters.plot_similar_stars_comparison(...)` — find ~5 stars with similar Teff/logg/[Fe/H], overlay their normalized spectra |
| **Save** | — | `training_spectra.npz` |

**`training_spectra.npz` fields**:
- `wavelength`: (8575,) float64
- `flux`: (N, 8575) float64 — normalized
- `error`: (N, 8575) float64 — normalized
- `labels`: (N, 5) float64 — [Teff, logg, [Fe/H], [Mg/Fe], [Si/Fe]]
- `label_names`: (5,) str — ["TEFF", "LOGG", "FE_H", "MG_FE", "SI_FE"]
- `apogee_ids`: (N,) str
- `continuum_mask`: (8575,) bool
- `starflag`: (N,) int — for P10 outlier investigation
- `aspcapflag`: (N,) int — for P10 outlier investigation

### `labs/02/02-cannon-training.ipynb` — Problems 5-8

| Step | Problem | Action |
|------|---------|--------|
| **Import cell** | — | load `training_spectra.npz`, import `train_cannon` from `ugdatalab.methods.cannon` |
| **Train/CV split** | P5 | 50/50 random split with `seed=42`, display train/CV sizes |
| **Design matrix derivation** | P6a | Markdown: the spectral model at pixel λ is `f_λ = X_n · θ_λ` where `X_n` is the design vector for star n (21 terms: 1 constant + 5 linear + 15 quadratic). Display X structure: print first 5 rows, column labels `["1", "Teff", "logg", ..., "Teff²", "Teff·logg", ...]`. Confirm shape: `(N_train, 21)`. |
| **Parameter count** | P6a | Code cell: 21 coefficients + 1 scatter per pixel × 8575 pixels = **188,650 total free parameters** |
| **Log-likelihood derivation** | P6b | Markdown with LaTeX: `ln L = -½ Σ_n Σ_λ [ln(2π(σ²_nλ + s²_λ)) + (f_nλ - X_n·θ_λ)²/(σ²_nλ + s²_λ)]`. Explain: s² makes this nonlinear — can't solve all parameters simultaneously via matrix algebra. Profile likelihood trick: for fixed s², θ is WLS; optimize 1D over log(s²). |
| **Train Cannon** | P6c | `model = train_cannon(flux_train, error_train, labels_train, wavelength)` |
| **Prediction demonstration** | P6d | Code cell: `predicted = model.predict(labels_train[0])` — show it returns (8575,) spectrum |
| **Training validation** | P7 | Find star 2M03533659+2512012 in training set. `plotters.plot_training_prediction(wavelength, flux_obs, error_obs, flux_pred)` — window 16000-16100 Å. **Checkpoint #2** |
| **Gradient spectra** | P8a | `plotters.plot_gradient_spectra(model)` — 5 panels. Mark known Mg I (15740, 15748, 15765 Å) and Si I (15888, 16060, 16094 Å) lines on the [Mg/Fe] and [Si/Fe] gradient panels. |
| **Scatter spectrum** | P8b | `plotters.plot_scatter_spectrum(model)` — s² vs wavelength. Markdown: regions of high scatter correspond to strong absorption lines where the 2nd-order polynomial is insufficient. |
| **Save** | — | `cannon_model.npz` |

**`cannon_model.npz` fields**:
- `theta`: (8575, 21) float64
- `scatter`: (8575,) float64
- `label_means`: (5,) float64
- `label_stds`: (5,) float64
- `wavelength`: (8575,) float64
- `label_names`: (5,) str
- `chi2_r`: float scalar
- `train_idx`: (N_train,) int
- `cv_idx`: (N_cv,) int

### `labs/02/03-cross-validation.ipynb` — Problems 9-11

| Step | Problem | Action |
|------|---------|--------|
| **Import cell** | — | load `training_spectra.npz` + `cannon_model.npz`, import `fit_labels_batch`, plotters |
| **Fit CV labels** | P9 | `fitted_labels = fit_labels_batch(model, flux_cv, error_cv)` |
| **1-to-1 plots** | P9 | `plotters.plot_label_recovery(true, fitted, label_names)` — 5 panels with 1:1 line + residual sub-panels. **Checkpoint #2** |
| **Bias/scatter table** | P9 | Per-label: mean offset (bias) and std (scatter) as DataFrame. Expected: Teff ~30 K, [Fe/H] ~0.02 dex. |
| **Outlier investigation** | P10 | Identify worst-fit stars (largest chi² or largest label residual). Code cells to: (a) check if fitted labels ≈ initial guess (optimizer stuck), (b) `plotters.plot_outlier_spectra(...)` — overlay observed and model for worst 3-5 stars, (c) check `starflag`/`aspcapflag` columns for ASPCAP warning bits, (d) check continuum normalization quality for outliers. Markdown discussion of findings. |
| **MIST isochrones** | P11 | `from ugdatalab.models.isochrones import _get_mist_isochrone`; get [Fe/H]=0 and [Fe/H]=-1 isochrones |
| **Kiel diagram** | P11 | `plotters.plot_kiel_diagram(fitted_labels, isochrone_tracks)` — logg vs Teff with inverted axes (hot→cold left→right, low-g at top), scatter colored by [Fe/H], two MIST isochrone lines overlaid. Markdown: identify RGB, red clump, and [Fe/H] trend. **Checkpoint #3** |
| **Save** | — | `cv_results.npz` |

**`cv_results.npz` fields**:
- `fitted_labels`: (N_cv, 5) float64
- `true_labels`: (N_cv, 5) float64
- `apogee_ids_cv`: (N_cv,) str

### `labs/02/04-mcmc-and-synthesis.ipynb` — Problems 12-15

| Step | Problem | Action |
|------|---------|--------|
| **Import cell** | — | load `cannon_model.npz`, astropy.io.fits, `nuts_sample`, bayesian plotters |
| **Read mystery spectrum** | P12 | Read `mystery_spec_wiped.fits` (HDU 1 = flux (8575,), HDU 2 = error, HDU 3 = bitmask). Apply bitmask + normalize using same continuum pipeline. Note: this is a 1D spectrum (not multi-row like training data). |
| **Prior statement** | P12 | Markdown: "We place weakly informative Normal priors on each label centered on training-set means: Teff ~ N(μ_T, 2σ_T), logg ~ N(μ_g, 2σ_g), [Fe/H] ~ N(μ_F, 2σ_F), [Mg/Fe] ~ N(μ_Mg, 2σ_Mg), [Si/Fe] ~ N(μ_Si, 2σ_Si), where μ and σ are the training-set mean and std." Print actual values from model. |
| **MCMC fit** | P12 | `lk = CannonLabelLikelihood(wavelength, flux, error, model)` → `result = nuts_sample(lk)` |
| **Corner plot** | P12 | `plot_corner(result)` from `ugdatalab.plotters.bayesian` — **Checkpoint #3** |
| **Trace plot** | P12 | `plot_trace(result)` from `ugdatalab.plotters.bayesian` |
| **Posterior summary** | P12 | Print median ± 68% CI for each label as DataFrame |
| **Uncertainty discussion** | P12 | Markdown: compare formal MCMC uncertainties to (a) CV scatter from P9, (b) typical ASPCAP errors (~100 K for Teff, ~0.1 dex for logg). If MCMC uncertainties are much smaller than CV scatter, discuss: the model treats pixels as independent but they share correlated systematics (continuum, line blending); formal errors underestimate true uncertainty. |
| **Metallicity sequence** | P13 | Generate synthetic spectra at fixed Teff=4800 K, logg=2.5, [Mg/Fe]=0, [Si/Fe]=0, varying [Fe/H] from -1 to +0.5 in 0.25 dex steps. `plotters.plot_metallicity_sequence(model, ...)` — 16000-16200 Å, color-coded with vertical offsets. |
| **RGB evolution** | P14 | Get MIST isochrone at [Fe/H]=0. Extract (Teff, logg) track along RGB from logg=3.5 to 0.5. Generate synthetic spectra at each point. `plotters.plot_rgb_evolution(model, ...)` — 16000-16200 Å. Markdown: compare composition effects (P13) to evolutionary effects (P14). Key question: how distinguish a cool low-g star from a warmer, higher-g, more metal-rich star? Answer: the pattern of line strengths differs (Mg/Si lines vs Fe lines respond differently to temperature vs abundance). |
| **Binary discussion** | P15 | Markdown: unresolved binary → composite spectrum → Cannon fits intermediate labels. Bias: Teff biased high (secondary adds blue flux), logg may be biased, abundances diluted. Reference: El-Badry et al. (2018). Correction: fit two-component model, or flag binaries via RV scatter. |

### `labs/02/05-neural-network.ipynb` — Problem 16

| Step | Problem | Action |
|------|---------|--------|
| **Import cell** | — | load `training_spectra.npz`, `cannon_model.npz` (for same train/CV split), import torch |
| **Data preparation** | P16 | Normalize labels to order unity (subtract mean, divide by std). Prepare PyTorch DataLoader for train/validation. |
| **Network definition** | P16 | PyTorch MLP: `Linear(8575, 512) → ReLU → Linear(512, 256) → ReLU → Linear(256, 5)`. Experiment with architecture. |
| **Training loop** | P16 | MSE loss on normalized labels. Adam optimizer. Track train and validation loss per epoch. Early stopping on validation loss. |
| **Loss curves** | P16 | `plotters.plot_nn_loss(train_losses, val_losses)` |
| **CV evaluation** | P16 | Apply to CV set. Unnormalize predicted labels. `plotters.plot_nn_label_recovery(true, predicted, label_names)` |
| **Comparison table** | P16 | Cannon vs NN: bias and scatter per label as DataFrame. Markdown: discuss advantages (NN: nonlinear, potentially better; Cannon: interpretable gradients, explicit uncertainty model) and disadvantages. |
| **Save** | — | `nn_results.npz` {nn_fitted_labels, true_labels} |

### `labs/02/06-numerical-results.ipynb` — Summary

Re-derives all computed values as DataFrames. Single source of truth for report:
- Training set size and cuts table
- Log g calculations
- Cannon model: total parameters, training chi2_r
- CV bias/scatter per label
- MCMC posterior medians and CIs for mystery star
- NN vs Cannon comparison table

---

## 3. Lab Plotter Functions (`labs/02/plotters.py`)

~14 functions, each following the standard `plot_X(ugdatalab_objects) → axes` pattern with `savefig(fig, name)`:

| Function | Problem | Figure | Notes |
|----------|---------|--------|-------|
| `plot_label_corner` | P1 | `fig_label_corner.pdf` | 5D corner of Teff/logg/[Fe/H]/[Mg/Fe]/[Si/Fe] using `corner` |
| `plot_example_spectrum` | P2 | `fig_example_spectrum.pdf` | Full wavelength range, raw flux vs lambda |
| `plot_normalization_diagnostic` | P4 | `fig_normalization.pdf` | 3-panel: raw, continuum fit, normalized |
| `plot_similar_stars_comparison` | P4 | `fig_similar_stars.pdf` | Overlay normalized spectra of similar-label stars |
| `plot_training_prediction` | P7 | `fig_training_prediction.pdf` | Observed vs model, 16000-16100 Å window |
| `plot_gradient_spectra` | P8 | `fig_gradient_spectra.pdf` | 5-panel df/dl, Mg/Si line markers on relevant panels |
| `plot_scatter_spectrum` | P8 | `fig_scatter_spectrum.pdf` | s² vs wavelength, identify high-scatter regions |
| `plot_label_recovery` | P9 | `fig_label_recovery.pdf` | 5×(main+residual) with 1:1 line, bias/scatter annotations |
| `plot_outlier_spectra` | P10 | `fig_outlier_spectra.pdf` | Worst-fit spectra with model overlaid |
| `plot_kiel_diagram` | P11 | `fig_kiel_diagram.pdf` | logg vs Teff (both inverted), scatter colored by [Fe/H], + 2 MIST isochrones |
| `plot_metallicity_sequence` | P13 | `fig_metallicity_sequence.pdf` | Synthetic spectra varying [Fe/H], 16000-16200 Å |
| `plot_rgb_evolution` | P14 | `fig_rgb_evolution.pdf` | Synthetic spectra along RGB, 16000-16200 Å |
| `plot_nn_loss` | P16 | `fig_nn_loss.pdf` | Training and validation loss vs epoch |
| `plot_nn_label_recovery` | P16 | `fig_nn_label_recovery.pdf` | Same format as `plot_label_recovery` for NN predictions |

Generic plotters from `ugdatalab.plotters.bayesian` handle P12 corner/trace directly — no new code.

---

## 4. Data Flow

```
course_materials_sp2026/labs/lab_2/
    continuum_wavelengths.npz (provided — keys: "wavelengths", "continuum")
    mystery_spec_wiped.fits   (provided — 1D: flux/error/bitmask each (8575,))

labs/02/
    NB 01 (acquire + normalize)
        → training_spectra.npz  {wavelength, flux, error, labels, label_names,
                                  apogee_ids, continuum_mask, starflag, aspcapflag}

    NB 02 (train Cannon)
        → cannon_model.npz  {theta, scatter, label_means, label_stds,
                              wavelength, label_names, chi2_r, train_idx, cv_idx}

    NB 03 (cross-validate + Kiel)
        → cv_results.npz  {fitted_labels, true_labels, apogee_ids_cv}

    NB 04 (MCMC + synthesis)
        → [terminal analysis, no .npz]

    NB 05 (neural network)
        → nn_results.npz  {nn_fitted_labels, true_labels}

    NB 06 (numerical results)
        → [summary tables, single source of truth]
```

Provided files are referenced by path from `course_materials_sp2026/` — no copying needed.

---

## 5. Reusable Components

**Used directly (no changes)**:
- `_cache_stable` — for APOGEE catalog and spectrum downloads
- `_sanitize_table` — for catalog column types
- `nuts_sample` / `MCMCResult` — for Problem 12 MCMC
- `plot_corner`, `plot_trace` — for Problem 12 visualization
- All `ugdatalab.plotting` constants, figure factories, style dicts
- `Fit` ABC — parent class for `CannonModel`
- `Likelihood` ABC — parent class for `CannonLabelLikelihood`

**Genuinely new**:
- `ugdatalab/models/apogee/` (4 files) — survey module
- `ugdatalab/models/isochrones.py` — MIST isochrone access (survey-agnostic)
- `ugdatalab/methods/cannon.py` — Cannon engine
- `ugdatalab/methods/cannon_likelihood.py` — PyMC wrapper
- `labs/02/plotters.py` (~14 functions)
- 6 notebooks + report

---

## 6. Key Technical Decisions

### Continuum normalization

- **Degree**: Start with degree 4 Chebyshev per chip for all spectra (Ness et al. 2015 use degree 2, but with a different continuum pixel selection procedure; our provided continuum pixel list may benefit from a higher degree). The continuum shape is instrumental (blaze function), smooth and similar across stars. Degree 4 captures the broad shape without fitting absorption features. Can tune down to degree 2-3 if residuals show overfitting.
- **Library**: `numpy.polynomial.chebyshev.Chebyshev.fit()` — handles domain mapping automatically.
- **Weights**: `1/error²`, with `error = inf` pixels getting zero weight (automatically handled by the fitting routine).
- **Guards**: If fewer than `degree + 1` valid continuum pixels in a chip, skip that chip (set error = inf). Also ensure NaN flux pixels get `error = inf` before fitting.
- **Error propagation**: Continuum fit uncertainty neglected (standard: many continuum pixels → small fractional uncertainty).
- **Validation**: Compare normalized spectra of stars with similar labels — they should be nearly identical. Wiggles or trends indicate degree is wrong.

### APOGEE chip boundaries

Verified from test data: the 8575-pixel wavelength grid is uniform in log-lambda with no wavelength gaps. Chip boundaries are specified in the apStar FITS header via `BMIN/BMAX`, `GMIN/GMAX`, `RMIN/RMAX` keywords (e.g., B=200-3332, G=3537-6139, R=6294-8391). Inter-chip gap pixels have NaN flux. `_identify_chips(header)` reads these keywords directly — no heuristic detection needed.

### Continuum file

The provided `continuum_wavelengths.npz` contains:
- `"wavelengths"`: float64 (8575,) — matches `10**(4.179 + 6e-6 * arange(8575))`
- `"continuum"`: bool (8575,) — 529 True pixels identifying continuum regions

### APOGEE data access (verified working)

- **Catalog**: `astroquery.sdss.SDSS.query_sql(data_release=17)` — tested, returns correct columns. NULL sentinel is `-9999.0`.
- **Spectra**: HTTP GET to `{APOGEE_DR17_URL}/apo25m/{field}/apStar-dr17-{apogee_id}.fits` — tested, 200 OK. All 4 fields use telescope `apo25m`.
- **FITS structure**: HDU 1 shape (N_visits+1, 8575) for training spectra (row 0 = coadd). Mystery spectrum is 1D (8575,).
- **Parallel download**: `ThreadPoolExecutor(max_workers=4)`.
- **Expected yield**: ~2335 raw → ~1855-1886 after quality cuts (exact count depends on NULL handling).

### MIST isochrones (verified working)

Uses `isochrones` package with local MIST grid data — no web API dependency:
```python
from isochrones import get_ichrone
mist = get_ichrone('mist')
iso = mist.isochrone(age=np.log10(6e9), feh=0.0)  # returns DataFrame
```
Verified: returns DataFrame with `Teff`, `logTeff`, `logg`, `initial_mass`, etc. Both [Fe/H]=0 and [Fe/H]=-1 work. RGB region (3500 < Teff < 5700, logg < 4) has 350+ points.

### Mystery spectrum

Verified structure: HDUs 1-3 are 1D (8575,) for flux/error/bitmask. Same wavelength grid via header `CRVAL1=4.179, CDELT1=6e-6`. All metadata fields zeroed. Same bitmask + normalization pipeline applies.

### Per-pixel training optimization

- **Profile likelihood**: For fixed s², theta is WLS. Outer 1D optimization over log(s²) in bounds [-20, 2] via `scipy.optimize.minimize_scalar`.
- **Degenerate pixels**: If fewer than 25 stars have finite error at a pixel, set theta=0 and scatter=inf (skip pixel).
- **Performance**: ~30-60s for 8575 pixels with ~2000 training stars. No parallelization needed.

### `Fit` ABC compatibility

The `Fit` ABC's `predict(self, x: ndarray) -> ndarray` maps model input → model output. `CannonModel.predict(labels)` maps (5,) → (8575,), satisfying the contract naturally — labels are the model input, the spectrum is the output. Same pattern as `FourierFit.predict(phases)` mapping phases → magnitudes.

### Comparison with Ness et al. 2015 (arXiv:1501.07604)

Our implementation follows the paper's core algorithm with lab-manual-driven adaptations:

| Aspect | Paper | Our plan | Rationale |
|--------|-------|----------|-----------|
| Labels | 3 (Teff, logg, [Fe/H]) | 5 (+[Mg/Fe], +[Si/Fe]) | Lab manual requirement |
| Coefficients/pixel | 10 | 21 (= 1+5+15) | 2nd-order with 5 labels |
| Label preprocessing | Center by mean only | Center AND scale by std | Lab manual: "rescale labels to order unity" — better numerical conditioning |
| Continuum degree | 2nd-order Chebyshev per chip | Degree 4 Chebyshev per chip (tunable) | Different continuum pixel set; start higher, can reduce |
| s² training optimizer | `scipy.optimize.curve_fit` | `scipy.optimize.minimize_scalar` on log(s²) | Equivalent profile likelihood; 1D optimizer is cleaner |
| Label inference | `scipy.optimize.curve_fit` | `scipy.optimize.least_squares(method='trf')` | Lab manual recommends 'trf'; supports bounds |
| Analytical Jacobians | Not mentioned | Yes, from polynomial structure | Improvement: faster convergence |
| Cross-validation | Leave-one-out (N=542) | 50/50 split | Lab manual Problem 5 specification |
| Bad pixel handling | Inverse variance downweighting | Same: error=inf → weight≈0 | Identical approach |

The mathematical model is identical: per-pixel log-likelihood `ln L = -½ Σ_n [ln(σ²_n + s²) + (f_n - θ·ℓ_n)²/(σ²_n + s²)]` with WLS for theta at fixed s² (Eq. 4-5 of paper). The adaptations are all motivated by the lab manual's specific requirements and do not change the underlying algorithm.

### `CannonLabelLikelihood` and `nuts_sample`

`nuts_sample` line 80 calls `likelihood._predict(likelihood.x, theta_median)` where `theta_median` is the posterior median of the 5 labels. `CannonLabelLikelihood._predict(x, theta)` calls `self.model.predict(theta)` (ignoring `x` = wavelength). This produces the correct spectrum for chi2_r computation.

`MCMCResult.predict(x)` will also call `_predict(x, self.theta)` → `model.predict(labels)`, returning the posterior-median spectrum. `MCMCResult.predict_std(x)` iterates over all posterior samples. Both work correctly.

`plot_posterior_predictive` and `predict_posterior` will NOT work (they assume 1D x-grid and call methods we haven't implemented). Only `plot_corner` and `plot_trace` are used for Problem 12.

---

## 7. Directory Structure

```
ugdatalab/
    __init__.py                          # UPDATE: add APOGEE exports
    models/
        isochrones.py                    # NEW — survey-agnostic MIST access
        apogee/
            __init__.py                  # NEW
            constants.py                 # NEW
            apogee.py                    # NEW
            spectra.py                   # NEW
    methods/
        cannon.py                        # NEW
        cannon_likelihood.py             # NEW

labs/02/
    01-training-set.ipynb                # NEW — Problems 1-4
    02-cannon-training.ipynb             # NEW — Problems 5-8
    03-cross-validation.ipynb            # NEW — Problems 9-11
    04-mcmc-and-synthesis.ipynb          # NEW — Problems 12-15
    05-neural-network.ipynb              # NEW — Problem 16
    06-numerical-results.ipynb           # NEW — Summary
    plotters.py                          # NEW
    report/
        main.tex                         # NEW
        figures/                         # NEW (populated by plotters)
```

---

## 8. Implementation Phases

The user will guide implementation step by step. Natural phase boundaries:

| Phase | Scope | Files | Depends on |
|-------|-------|-------|------------|
| **A** | APOGEE survey module + isochrones | `models/apogee/{constants,apogee,spectra,__init__}.py`, `models/isochrones.py`, `ugdatalab/__init__.py` | existing cache/utils |
| **B** | Plotters scaffold + NB 01 | `labs/02/plotters.py`, `labs/02/01-training-set.ipynb` | Phase A |
| **C** | Cannon engine | `methods/cannon.py` | independent |
| **D** | NB 02 (training) | `labs/02/02-cannon-training.ipynb` + plotter functions | Phases B, C |
| **E** | NB 03 (cross-validation + Kiel) | `labs/02/03-cross-validation.ipynb` + plotter functions | Phase D |
| **F** | Cannon likelihood + NB 04 | `methods/cannon_likelihood.py`, `labs/02/04-mcmc-and-synthesis.ipynb` | Phase E |
| **G** | NB 05 (neural network) | `labs/02/05-neural-network.ipynb` + plotter functions | Phase D |
| **H** | NB 06 (summary) + report | `labs/02/06-numerical-results.ipynb`, `labs/02/report/main.tex` | Phases E-G |

Phases A and C can proceed in parallel. Phases F and G can proceed in parallel after Phase E.

---

## 9. Verification

Each phase should be verified before proceeding:

- **Phase A**: `import ugdatalab` succeeds; `APOGEEData(fields=["M15"]).data` returns table with correct columns; `APOGEETrainingSet(source).data` applies all cuts and reduces sample; single spectrum download + bitmask + normalization works; `_get_mist_isochrone(6.0, 0.0)` returns DataFrame with `Teff`/`logg` columns
- **Phase B**: NB 01 runs top-to-bottom; `training_spectra.npz` produced with correct shapes; corner plot and normalization figures generated
- **Phase C**: `CannonModel` dataclass created; `_build_cannon_design_matrix` returns (N, 21); `_train_pixel` converges for test pixel; `train_cannon` completes in <2 min; `model.predict(training_labels[0])` matches observed spectrum; `fit_star_labels` recovers training labels
- **Phase D**: NB 02 runs; training prediction matches at 16000-16100 Å; gradient spectra show features at known Mg/Si lines
- **Phase E**: `fit_labels_batch` recovers CV labels; scatter comparable to expectations (~30 K Teff, ~0.02 dex [Fe/H]); Kiel diagram shows RGB structure with [Fe/H] gradient
- **Phase F**: `CannonLabelLikelihood` + `nuts_sample` runs; `plot_corner` shows reasonable posterior; uncertainties discussed
- **Phase G**: NN trains to convergence; performance comparable to Cannon
- **Phase H**: All figures in `report/figures/`; all numerical claims traceable to notebook cells
