"""Reusable MCMC diagnostic plots: trace, corner, posterior predictive.

All functions accept result objects with .samples, .labels, and .log_probs
(MCMCResult, MixtureResult, MHResult, or any compatible object)
and use ugdatalab.plotting constants with matplotlib "C0"–"C9" colors.
"""

import corner
import numpy as np

from ugdatalab.plotting import (
    LW_NONE,
    LW_FINE,
    LW_STANDARD,
    LW_MEDIUM,
    SS_MICRO,
    ALPHA_EXTRA_LIGHT,
    ALPHA_FAINT,
    ALPHA_LIGHT,
    ALPHA_STANDARD,
    NEUTRAL_COLOR,
    FILL_STYLE,
    SCATTER_STYLE,
    ERRORBAR_STYLE,
    FIT_STYLE,
    textwidth_figure,
    columnwidth_figure,
    corner_figure,
    subpanels,
)

_TRACE_PANEL_HEIGHT_PTS = 28
_TRACE_STYLE = dict(lw=LW_FINE, alpha=ALPHA_STANDARD, color="C0")

_CORNER_QUANTILES = [0.16, 0.5, 0.84]

_HIST_STYLE = dict(bins=50, density=True, alpha=ALPHA_LIGHT, color="C0")

_PP_N_DRAWS = 400
_PP_GRID_N = 300
_PP_SEED = 42


# ---------------------------------------------------------------------------
# Trace plot
# ---------------------------------------------------------------------------

def plot_trace(result):
    """Trace plot of MCMC samples and log-probability vs step number.

    Parameters
    ----------
    result : MCMCResult, MHResult, or similar
        Must have .samples (n_steps, n_params), .labels, and .log_probs.
    """
    samples = result.samples
    labels = result.labels
    log_probs = result.log_probs
    ndim = samples.shape[1]
    steps = np.arange(len(samples))

    fig, ax = textwidth_figure(_TRACE_PANEL_HEIGHT_PTS * (ndim + 1))
    ax.remove()
    axes = subpanels(fig, ndim + 1, height_ratios=[1] * (ndim + 1))

    for i, lbl in enumerate(labels):
        axes[i].plot(steps, samples[:, i], **_TRACE_STYLE)
        axes[i].set_ylabel(lbl)

    axes[-1].plot(steps, log_probs, **_TRACE_STYLE)
    axes[-1].set_ylabel(r"$\ln P$")
    axes[-1].set_xlabel("Step")
    return axes


# ---------------------------------------------------------------------------
# Corner plot
# ---------------------------------------------------------------------------

def plot_corner(result):
    """Corner plot of posterior samples.

    Parameters
    ----------
    result : MCMCResult or similar
        Must have .samples and .labels.
    """
    fig = corner.corner(
        result.samples,
        labels=result.labels,
        show_titles=True,
        title_fmt=".3f",
        quantiles=_CORNER_QUANTILES,
        color="C0",
        fig=corner_figure(),
    )
    return fig


# ---------------------------------------------------------------------------
# Posterior histogram (single parameter)
# ---------------------------------------------------------------------------

def plot_posterior(result, *, param_idx=0, pdf_fn=None):
    """Histogram of one parameter's posterior samples with optional analytic PDF.

    Parameters
    ----------
    result : MCMCResult or similar
        Must have .samples and .labels.
    param_idx : int
        Which parameter column to plot.
    pdf_fn : callable or None
        Analytic PDF to overlay.
    """
    post_burn = result.samples[:, param_idx]
    xlabel = result.labels[param_idx]

    fig, ax = columnwidth_figure(4)

    ax.hist(post_burn, **_HIST_STYLE, label="MCMC samples")

    if pdf_fn is not None:
        span = post_burn.max() - post_burn.min()
        grid = np.linspace(post_burn.min() - 0.5 * span,
                           post_burn.max() + 0.5 * span, _PP_GRID_N)
        legend_param = xlabel
        if legend_param.startswith("$") and legend_param.endswith("$"):
            legend_param = legend_param[1:-1]
        ax.plot(grid, pdf_fn(grid), lw=LW_MEDIUM, color="C1",
                label=rf"Analytic $p({legend_param}\mid x)$")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Posterior predictive check (median + credible bands)
# ---------------------------------------------------------------------------

def plot_posterior_predictive(result, *, color="C0", data_label="Data", ax=None):
    """Posterior predictive: data + median line + 68%/95% credible bands.

    Assumes a linear model y = a*x + b with intrinsic scatter sigma_s,
    where each row of result.samples is [a, b, log10(sigma_s)].
    Data (x, y, y_err) is read from result._likelihood.

    Parameters
    ----------
    result : MCMCResult
        Must have .samples and ._likelihood (with .x, .y, .y_err).
    color : str
    data_label : str
    ax : Axes or None
    """
    likelihood = result._likelihood
    x, y, y_err = likelihood.x, likelihood.y, likelihood.y_err
    samples = result.samples

    if ax is None:
        fig, ax = textwidth_figure(8)

    x_grid = np.linspace(np.min(x), np.max(x), _PP_GRID_N)
    order = np.argsort(x)
    sigma_grid = np.interp(x_grid, x[order], y_err[order])

    rng = np.random.default_rng(_PP_SEED)
    step = max(len(samples) // _PP_N_DRAWS, 1)
    pool = samples[::step]
    if len(pool) > _PP_N_DRAWS:
        pool = pool[rng.choice(len(pool), size=_PP_N_DRAWS, replace=False)]

    mean_draws = np.empty((len(pool), len(x_grid)))
    pred_draws = np.empty_like(mean_draws)
    for i, (a, b, log10_sig) in enumerate(pool):
        mu = a * x_grid + b
        sigma_pred = np.sqrt(sigma_grid**2 + (10.0**log10_sig)**2)
        mean_draws[i] = mu
        pred_draws[i] = rng.normal(mu, sigma_pred)

    median_line = np.median(mean_draws, axis=0)
    q16 = np.quantile(pred_draws, 0.16, axis=0)
    q84 = np.quantile(pred_draws, 0.84, axis=0)
    q025 = np.quantile(pred_draws, 0.025, axis=0)
    q975 = np.quantile(pred_draws, 0.975, axis=0)

    ax.errorbar(x, y, yerr=y_err, fmt="none", color=NEUTRAL_COLOR,
                alpha=ALPHA_LIGHT, lw=LW_FINE, zorder=1)
    ax.scatter(x, y, s=SS_MICRO, color=color, alpha=ALPHA_FAINT,
               rasterized=True, zorder=2, label=data_label)
    ax.fill_between(x_grid, q025, q975, color=color, **FILL_STYLE,
                    zorder=3, label=r"95\% predictive")
    ax.fill_between(x_grid, q16, q84, color=color, alpha=ALPHA_FAINT,
                    lw=LW_NONE, zorder=4, label=r"68\% predictive")
    ax.plot(x_grid, median_line, color=color, **FIT_STYLE,
            zorder=5, label="Posterior median")

    ax.autoscale_view()
    y0, y1 = ax.get_ylim()
    ax.set_ylim(max(y0, y1), min(y0, y1))
    ax.legend(loc="best")
    return ax
