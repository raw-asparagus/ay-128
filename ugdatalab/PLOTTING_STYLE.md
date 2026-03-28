# Plotting Style Guide

All plots use constants and helpers from `ugdatalab.plotting`. Importing the module activates rcParams globally.

## Figure Creation

- **Single panel**: `fig, ax = textwidth_figure(h)` or `fig, ax = columnwidth_figure(h)`
- **Grid of independent panels**: `fig, subfigs = textwidth_figure(h, subfigures=(nrows, ncols))`, then `subpanels(subfig, ...)` on each subfigure
- **Flat multi-panel** (shared axes): `fig, ax = textwidth_figure(h)` then `ax.remove()` then `axes = subpanels(fig, nrows, ncols, ...)`
- **Nested panels** (e.g. curve + residual per cell): use `subfigures` for the outer grid, `subpanels(sf, 2, height_ratios=(3, 1))` for each inner pair
- Height parameter scales as fraction: `textwidth_figure` divides by 16, `columnwidth_figure` divides by 7.5
- Never call `plt.figure()` or `fig.add_subplot(111)` directly — use the helpers

## Colors

- Use matplotlib default cycle `"C0"` through `"C9"` — no custom color constants
- RR Lyrae subclasses: `{"RRab": "C0", "RRc": "C1", "RRd": "C2"}`
- Neutral reference elements (guide lines, zero lines, model overlays on data): `NEUTRAL_COLOR` (`"C7"`)
- When multiple datasets overlap, assign sequential `"C0"`, `"C1"`, etc.

## Transparency

Every plot element must have an explicit `alpha=` set. Use these constants:

| Constant | Value | Use for |
|---|---|---|
| `ALPHA_EXTRA_LIGHT` | 0.2 | Shaded fill bands, subtle background regions |
| `ALPHA_FAINT` | 0.4 | Error bars, secondary data points, CV histograms |
| `ALPHA_LIGHT` | 0.6 | Guide/reference lines, axis bands |
| `ALPHA_STANDARD` | 0.75 | **Default for all primary plot elements** — data scatter, model lines, curves, markers |
| `ALPHA_FULL` | 1.0 | Only when full opacity is required |

**Rule**: If a plot call does not already have `alpha=` set (either directly or via a style dict), add `alpha=ALPHA_STANDARD`.

## Line Weights

| Constant | Value | Use for |
|---|---|---|
| `LW_NONE` | 0.0 | Fill regions, scatter markers with no edge |
| `LW_FINE` | 0.6 | Error bar lines, `elinewidth`, thin outlines |
| `LW_LIGHT` | 1.0 | Reference/guide lines, axis lines |
| `LW_STANDARD` | 1.5 | Primary data lines, model fits, curves |
| `LW_MEDIUM` | 2.0 | Emphasized model overlays |
| `LW_THICK` | 2.5 | Rarely used — heavy emphasis only |

## Marker Sizes

For `ax.plot(..., ms=)`:

| Constant | Value | Use for |
|---|---|---|
| `MS_MICRO` | 2.0 | Dense scatter plots, many-point datasets |
| `MS_FINE` | 3.0 | Standard scatter, error bar markers |
| `MS_STANDARD` | 6.0 | Moderate-density plots |
| `MS_MEDIUM` | 9.0 | Emphasized single points |
| `MS_LARGE` | 14.0 | Special markers (predictions, highlights) |

For `ax.scatter(..., s=)` use `SS_*` equivalents (`MS_*`²).

## Style Dictionaries

Unpack with `**` for consistent styling:

| Dict | Purpose |
|---|---|
| `**GRID_STYLE` | Grid lines (set via rcParams, rarely needed explicitly) |
| `**GUIDE_STYLE` | Reference lines (zero line, unity line, 1:1 diagonal) |
| `**FIT_STYLE` | Fitted model solid lines |
| `**MODEL_STYLE` | Model dashed lines (e.g. 1:1 comparison, PL ratio) |
| `**ERRORBAR_STYLE` | Standard error bar formatting |
| `**FILL_STYLE` | Shaded uncertainty bands |
| `**SCATTER_STYLE` | Standard scatter point formatting |

## Reference Lines

- `zero_line(ax)` — horizontal y=0 reference using `GUIDE_STYLE`
- `unity_line(ax)` — horizontal y=1 reference using `GUIDE_STYLE`
- For 1:1 diagonal comparisons: `ax.plot([lo, hi], [lo, hi], **MODEL_STYLE)`

## Ordering (zorder)

- Reference/model lines: `zorder=1`
- Error bars: `zorder=2`
- Data scatter points: `zorder=3`
- Model curves overlaid on data: `zorder=3` or higher

Data in front of reference lines.

## Axis Conventions

- **Magnitudes**: always inverted y-axis (brighter = up). After plotting: `ax.set_ylim(max, min)` or `ax.invert_yaxis()`
- **Periods**: log scale with `ax.set_xscale("log")` and tick formatter `FuncFormatter(lambda x, _: f"{x:g}")`
- **Square panels** (1:1 comparisons): `ax.set_aspect("equal", adjustable="box")`. Set identical x/y limits.
- **Residual panels**: share x-axis with main panel via `subpanels(..., sharex=True)`. Use `zero_line(ax)` as reference.

## Text and Labels

- LaTeX rendering is on by default (`text.usetex: True`)
- Use `set_title(..., loc="left", fontsize="small")` for per-panel annotations
- Multi-line titles: split with `"\n"` between string literals
- Chi-squared: `$\chi_r^2$` (reduced, subscript r)
- Avoid `\texttt{}` for column names in labels when possible

## Saving

- Every plotter function saves automatically via `_savefig(fig, "fig_name.pdf")`
- `savefig.bbox: "tight"` is set in rcParams — no manual `tight_layout()` needed
- Do not call `fig.tight_layout()` on figures with `set_aspect("equal")` — it produces warnings

## No Type Casting

Data from `ugdatalab` is sanitized upstream. Do not use `np.asarray(..., dtype=float)`, `float()`, or `int()` on table columns or result object attributes.

## rcParams (set automatically on import)

- LaTeX serif font (Computer Modern Roman)
- Grid on by default, dotted, faint
- Ticks: inward, all four sides
- DPI: 300 for both display and save
- `savefig.bbox: "tight"`
