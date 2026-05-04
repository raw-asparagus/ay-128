"""Trace, corner, posterior, and posterior-predictive plotters for MCMC results."""

import corner
import numpy as np

from ugdatalab.plotting import (
    LW_FINE,
    LW_MEDIUM,
    SS_MICRO,
    ALPHA_FAINT,
    ALPHA_LIGHT,
    ALPHA_STANDARD,
    NEUTRAL_COLOR,
    FILL_STYLE,
    FIT_STYLE,
    textwidth_figure,
    columnwidth_figure,
    corner_figure,
    subpanels,
)

_TRACE_PANEL_HEIGHT_PTS = 3
_TRACE_STYLE = dict(lw=LW_FINE, alpha=ALPHA_STANDARD, color="C0")

# Corner-plot quantile annotations: median + central 68% credible interval.
_CORNER_QUANTILES = [0.16, 0.5, 0.84]

_HIST_STYLE = dict(bins=50, density=True, alpha=ALPHA_LIGHT, color="C0")

# Posterior-predictive grid extends 5% beyond the data range on each side.
_PP_X_MARGIN_FRAC = 0.05

# Quantile pairs used to build the 68% / 95% predictive bands.
_PP_Q_LOW_68, _PP_Q_HIGH_68 = 0.16, 0.84
_PP_Q_LOW_95, _PP_Q_HIGH_95 = 0.025, 0.975

# Width factor applied around the posterior to extend the analytic-PDF
# overlay grid in ``plot_posterior``.
_POST_PDF_GRID_PAD_FRAC = 0.5

# ---------------------------------------------------------------------------
# Trace plot
# ---------------------------------------------------------------------------

def plot_trace(result):
    """Plot MCMC sample traces and log-probability versus step number.

    Parameters
    ----------
    result : MCMCResult, MHResult, or similar
        Object exposing ``samples`` (shape ``(n_steps, n_params)``),
        ``labels``, and ``log_probs``.

    Returns
    -------
    ndarray of matplotlib.axes.Axes
        One axes per parameter plus a final axes for the log-probability.
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

def plot_corner(result, *, figsize="textwidth"):
    """Draw a corner plot of posterior samples.

    Parameters
    ----------
    result : MCMCResult or similar
        Object exposing ``samples`` and ``labels``.
    figsize : {"textwidth", "corner"}, optional
        ``"textwidth"`` uses the full page width (better for five or more
        parameters); ``"corner"`` uses the 0.7x text width square.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if figsize == "textwidth":
        base, ax = textwidth_figure(16)
        ax.remove()
    else:
        base = corner_figure()
    fig = corner.corner(
        result.samples,
        labels=result.labels,
        show_titles=True,
        title_fmt=".3f",
        quantiles=_CORNER_QUANTILES,
        color="C0",
        fig=base,
    )
    return fig


# ---------------------------------------------------------------------------
# Posterior histogram (single parameter)
# ---------------------------------------------------------------------------

def plot_posterior(result, *, param_idx=0, pdf_fn=None):
    """Plot the marginal posterior histogram for a single parameter.

    Parameters
    ----------
    result : MCMCResult or similar
        Object exposing ``samples`` and ``labels``.
    param_idx : int, optional
        Index of the parameter column to plot.
    pdf_fn : callable, optional
        Analytic PDF ``f(x) -> density`` to overlay on the histogram.

    Returns
    -------
    matplotlib.axes.Axes
    """
    post_burn = result.samples[:, param_idx]
    xlabel = result.labels[param_idx]

    fig, ax = columnwidth_figure(4)

    ax.hist(post_burn, **_HIST_STYLE, label="MCMC samples")

    if pdf_fn is not None:
        span = post_burn.max() - post_burn.min()
        grid = np.linspace(post_burn.min() - _POST_PDF_GRID_PAD_FRAC * span,
                           post_burn.max() + _POST_PDF_GRID_PAD_FRAC * span, _PP_GRID_N)
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

_PP_N_DRAWS = 300
_PP_GRID_N = 1000
_PP_SEED = 42


def predict_posterior(result, n_grid=_PP_GRID_N, n_draws=_PP_N_DRAWS, seed=_PP_SEED):
    """Compute posterior-predictive summary curves from an MCMC result.

    Parameters
    ----------
    result : MCMCResult
        Object exposing ``x``, ``samples``, ``predict``, and
        ``total_variance``.
    n_grid : int, optional
        Number of grid points spanning the (margin-extended) data range.
    n_draws : int, optional
        Number of posterior draws used to build the predictive bands.
    seed : int, optional
        Seed for the random number generator used to sample predictions.

    Returns
    -------
    dict
        Mapping with keys ``x_grid``, ``median``, ``q16``, ``q84``,
        ``q025``, ``q975``; each value is an array of length ``n_grid``.
    """
    x = result.x
    samples = result.samples

    margin = _PP_X_MARGIN_FRAC * (np.max(x) - np.min(x))
    x_grid = np.linspace(np.min(x) - margin, np.max(x) + margin, n_grid)
    order = np.argsort(x)

    rng = np.random.default_rng(seed)
    step = max(len(samples) // n_draws, 1)
    pool = samples[::step]
    if len(pool) > n_draws:
        pool = pool[rng.choice(len(pool), size=n_draws, replace=False)]

    mean_draws = np.empty((len(pool), n_grid))
    pred_draws = np.empty_like(mean_draws)
    for i, theta in enumerate(pool):
        mu = result.predict(x_grid, theta)
        sigma_pred = np.sqrt(np.interp(x_grid, x[order], result.total_variance(theta)[order]))
        mean_draws[i] = mu
        pred_draws[i] = rng.normal(mu, sigma_pred)

    return {
        "x_grid": x_grid,
        "median": np.median(mean_draws, axis=0),
        "q16": np.quantile(pred_draws, _PP_Q_LOW_68, axis=0),
        "q84": np.quantile(pred_draws, _PP_Q_HIGH_68, axis=0),
        "q025": np.quantile(pred_draws, _PP_Q_LOW_95, axis=0),
        "q975": np.quantile(pred_draws, _PP_Q_HIGH_95, axis=0),
    }


def plot_posterior_predictive(result, *, color="C0", data_label="Data", ax=None):
    """Plot data with the posterior-predictive median and 68%/95% bands.

    Parameters
    ----------
    result : MCMCResult
        Object exposing ``x``, ``y``, ``y_err``, ``samples``, ``predict``,
        and ``total_variance``.
    color : str, optional
        Color used for the data points and predictive curves.
    data_label : str, optional
        Legend label for the data scatter.
    ax : matplotlib.axes.Axes, optional
        Target axes; a new text-width figure is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    x, y, y_err = result.x, result.y, result.y_err
    pp = predict_posterior(result)
    x_grid = pp["x_grid"]

    if ax is None:
        fig, ax = textwidth_figure(8)

    ax.errorbar(x, y, yerr=y_err, fmt="none", color=NEUTRAL_COLOR,
                alpha=ALPHA_LIGHT, lw=LW_FINE, zorder=1)
    ax.scatter(x, y, s=SS_MICRO, color=color, alpha=ALPHA_FAINT,
               rasterized=True, zorder=2, label=data_label)
    ax.fill_between(x_grid, pp["q025"], pp["q975"], color=color, **FILL_STYLE,
                    zorder=3, label=r"95\% predictive")
    ax.fill_between(x_grid, pp["q16"], pp["q84"], color=color,
                    **{**FILL_STYLE, "alpha": ALPHA_FAINT},
                    zorder=4, label=r"68\% predictive")
    ax.plot(x_grid, pp["median"], color=color, **FIT_STYLE,
            zorder=5, label="Posterior median")

    ax.autoscale_view()
    y0, y1 = ax.get_ylim()
    ax.set_ylim(max(y0, y1), min(y0, y1))
    ax.legend(loc="best")
    return ax
