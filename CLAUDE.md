# Project: ay-128 (ugdatalab)

## Code style

### Docstrings

- **Private functions / methods** (names prefixed with `_`): a single-line docstring describing what the function does.
- **Public functions / methods / classes**: PEP 257 format — one-line summary on the first line, a blank line, optional extended description, then `Args` / `Returns` / `Raises` sections (Google or NumPy style — match whatever the surrounding module already uses).
- **Module docstrings**: same shape — one-line summary, blank line, optional details.
- Docstrings must **describe** what the symbol does (behavior, inputs, outputs, side effects). They must **not justify** the symbol's existence at the module, class, function, or parameter level (no "we need this because…", no rationale for design choices, no historical context). Justification belongs in commit messages or PR descriptions, not docstrings. **Exception:** justification IS allowed inside *default-value descriptions* — per audit rule 19, every default must explain what the value is, why it was chosen, and when a caller should override it.
- Keep docstrings accurate: no references to removed parameters, renamed types, or stale behavior.

## Audit: "audit ugdatalab"

When the user says "audit ugdatalab", perform the following checks on all `.py` files in `ugdatalab/` (recursively), including `ugdatalab/methods/`, `ugdatalab/models/`, and `ugdatalab/plotting.py`.

### Checklist

1. **Dead code** — functions/classes defined but never called or imported
2. **Stale imports** — importing things that don't exist or aren't used
3. **Lazy imports** — no function-level imports; all imports at top of file
4. **Return type consistency** — in-place mutation functions must return `None`; functions that produce new data must return it
5. **Defensive guards** — no unnecessary guards (e.g., `if len(x) == 0` when input is guaranteed non-empty)
6. **Hardcoded values** — no unexplained magic numbers; data-driven defaults where possible
7. **Cross-references** — all import paths resolve; no references to deleted/renamed modules
8. **ABC conformance** — every concrete class implements all abstract methods from its base
9. **Naming** — private building blocks prefixed `_`; public methods match what engines consume; functions used only within the same file must be private; internal decorators and helpers must not be exported from `__init__.py`
10. **Docstring accuracy** — no references to removed parameters, wrong types, or stale descriptions
11. **Export consistency** — `__init__.py` files export exactly what's used externally; no private symbols (`_`-prefixed) exported; no internal utilities (e.g., caching decorators) in public API
12. **Duplicate code** — flag identical logic that could be shared
13. **Convention consistency** — schemas use `type: [colnames]` format; prior scales are data-driven; style dicts used for plot styling
14. **Redundant rcParams** — no rcParams that match matplotlib defaults
15. **Redundant type casting** — no `np.asarray(col, dtype=...)` on columns already sanitized by `_sanitize_table`; no casting of return values from libraries that already return numpy arrays (e.g., `LombScargle.autopower`)

16. **Circular imports** — no import cycles between modules; verify with `import ugdatalab`
17. **Signature consistency** — engines use consistent parameter names (e.g., `n_steps`, `n_burn`, `seed`); result dataclasses carry `labels`, `samples`, `trace` where applicable
18. **Numerical stability** — no unguarded `log(0)`, `1/0`, or `10**large_number`; magnitude conversions handle edge cases
19. **Default argument documentation** — package-level functions, factories, and classes may use default parameter values, but every default must be documented: what the value is, why it was chosen, and when a caller should override it. Defaults encoding scientific choices (e.g., `bad_bits`, `degree`, `period_min`) require especially clear documentation since a user may reasonably want a different value

### Output format

For each file: list issues or say "clean". End with `import ugdatalab` verification.

## Audit: "audit lab NN"

When the user says "audit lab NN" (e.g., "audit lab 01"), perform the following checks on all files in `labs/NN/` — notebooks (`.ipynb`), plotter scripts (`.py`), and the report directory (`report/`).

### Reuse & imports

1. **ugdatalab reuse** — no reimplementation of logic that exists in ugdatalab (e.g., hand-rolling chi-squared when `DataFit.chi2_r` / `_mean_chi2` exists, manual periodograms when `lomb_scargle` exists, manual distance modulus when `_add_gaia_photometry_columns` does it)
2. **No raw astroquery/requests** — all Gaia queries go through `GaiaData`/`GaiaSample`/`WISEData` or the cached `_get_gaia`; no uncached ad-hoc calls
3. **Plotting style imports** — plotters import constants/helpers from `ugdatalab.plotting`, not hardcoded equivalents (no bare `lw=0.6` when `LW_FINE` exists, no `alpha=0.75` when `ALPHA_STANDARD` exists)
4. **Figure creation** — all figures use `textwidth_figure`, `columnwidth_figure`, `landscapewidth_figure`, or `corner_figure`; no raw `plt.figure(figsize=(...))` with ad-hoc dimensions
5. **No duplicate helpers** — if a plotting function exists in `ugdatalab.plotters.bayesian` (e.g., `plot_trace`, `plot_corner`, `plot_posterior`), the lab plotter delegates rather than reimplements

### Plotter style consistency

6. **Style dict usage** — error bars use `**ERRORBAR_STYLE`, fills use `**FILL_STYLE`, scatter uses `**SCATTER_STYLE`, fits use `**FIT_STYLE`, guides use `**GUIDE_STYLE`; no partial re-specification of what the dicts already provide
7. **zorder discipline** — guides/models at zorder 1, error bars at 2, data at 3, emphasized overlays at 4+
8. **Axis conventions** — magnitudes have inverted y-axis; periods use log scale; residual panels share x-axis with main panel via `subpanels(..., sharex=True)`
9. **Savefig pattern** — all figures saved through a single `_savefig(fig, name)` helper that writes to `report/figures/`; no scattered `fig.savefig(...)` calls with inconsistent paths
10. **No inline style overrides** — no `plt.rcParams` mutations inside plotter functions; all rcParams set once by importing `ugdatalab.plotting`

### Default arguments

11. **No unnecessary defaults in business code** — lab plotter functions, notebook helper functions, and other single-use/single-project code must not have default parameter values unless absolutely necessary for functional reuse; every argument should be explicit at the call site so the notebook is self-documenting (e.g., `plot_kiel_diagram(fitted_labels, isochrone_tracks, feh_values)` not `plot_kiel_diagram(fitted_labels, isochrone_tracks, feh_values=[-1, 0])`)
12. **Structural constants are not defaults** — module-level constants like `_FIGURES_DIR` and the `savefig(fig, name)` helper are fine; the rule targets function signatures, not file-level configuration

### Notebook hygiene

13. **Imports at top** — each notebook has a single import cell at the top; no mid-notebook imports
14. **No dead cells** — no commented-out code blocks, no cells that produce no output and serve no setup purpose
15. **Reproducibility** — all random operations use explicit seeds; no bare `np.random.rand()` or unseeded `pm.sample()`
16. **No hardcoded source IDs or paths** — source IDs come from data queries, not magic numbers; paths use `Path(__file__).parent` or equivalent

### Report fidelity

17. **Figure coverage** — every PDF in `report/figures/` is `\includegraphics`'d in the `.tex`; no orphan figures
18. **Figure–notebook traceability** — every figure file can be traced to a specific plotter function call in a specific notebook
19. **No stale figures** — figure files are not older than the notebook cells that generate them (check mtimes or re-run)
20. **Numerical claims** — every number cited in the report text (sample sizes, best-fit parameters, chi-squared values) has a corresponding notebook cell that computes it

### Output format

For each file: list issues or say "clean". End with a figure coverage summary (figures found vs figures referenced in `.tex`).

## Audit: "audit lab NN completeness"

When the user says "audit lab NN completeness" (e.g., "audit lab 01 completeness"), perform the following checks once all notebooks and the report are finalized. Read the lab manual PDF from `course_materials_sp2026/labs/lab_N/` as the reference checklist.

### Notebook result completeness

1. **All computed values displayed** — every significant computed quantity (best periods, fit parameters, chi-squared, sample sizes after cuts, acceptance rates) is displayed in the notebook as either a `pd.DataFrame` (for computed/derived values) or an `astropy.table.Table` (for catalog data); no results exist only as intermediate variables
2. **All plots rendered** — every plotter function called in the notebooks produces a visible output in the notebook; no figure is saved without also being displayed
3. **All filtering stages shown** — each data quality cut (parallax, SNR, Lindegren C1/C2, outlier rejection) shows before/after sample sizes
4. **MCMC diagnostics present** — every NUTS/MH run shows: trace plots, convergence summary (Rhat, ESS or acceptance rate), and posterior summary (median + credible intervals)
5. **Cross-validation results shown** — every model selection step shows the training/CV metric vs parameter grid, with best parameter identified

### Report–notebook equivalence

6. **Report is superset** — every figure, table, and computed value in the notebooks appears in the report; the report adds only prose, derivations, and context
7. **No report-only results** — the report does not contain figures or numerical results that cannot be reproduced from the notebooks
8. **Section–notebook mapping** — each report section maps to one or more notebooks; no section draws from undocumented analysis
9. **Figure ordering** — figures appear in the report in the same logical order as the notebook progression
10. **Caption accuracy** — figure captions describe what is actually plotted (correct axes, correct sample, correct model)
11. **Numerical claim coverage** — extract every computed numerical value from the report (sample sizes, fit parameters, RMS values, R² correlations, rejection rates, percentages, quality cut counts) and verify each one appears in a notebook cell output; present a table of report claim vs notebook cell for every value
12. **Notebook as single source of truth** — for every numerical claim where the report and notebook both cover the same quantity, the notebook value (computed from data) is authoritative; if the report value disagrees, update the report to match the notebook; common sources of drift include stale query caches producing different sample sizes, MCMC sampling variability in fit parameters, and report text citing tuning/draw counts that don't match code defaults

### Report vs lab manual completeness

11. **All required tasks addressed** — every task/question in the lab manual PDF has a corresponding notebook section and report section
12. **All required plots produced** — every figure the lab manual asks for is present in the report
13. **All required quantities computed** — every numerical result the lab manual asks for appears in both a notebook cell and the report text
14. **Method justification** — where the lab manual asks "why did you choose X", the report provides the justification (not just the result)

### Output format

Three sections: (1) notebook completeness issues, (2) report–notebook equivalence issues, (3) lab manual coverage — listing each manual task and whether it is addressed. End with an overall completeness summary.
