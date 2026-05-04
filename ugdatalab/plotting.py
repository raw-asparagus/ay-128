"""Project-wide matplotlib styling and figure-sizing helpers.

Defines page dimensions, font sizes, line weights, marker sizes, alphas,
neutral colors, and reusable style dictionaries (``GRID_STYLE``,
``ERRORBAR_STYLE``, ...). Importing this module also installs the
``rcParams`` used by every plot in the package.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page dimensions
# ---------------------------------------------------------------------------

TEXTWIDTH_IN = 7.59
COLUMNWIDTH_IN = 3.73
A4_WIDTH_IN = 8.27
A4_HEIGHT_IN = 11.69
A4_MARGIN_IN = 0.75
A4_USABLE_WIDTH_IN = A4_WIDTH_IN - 2.0 * A4_MARGIN_IN
A4_USABLE_HEIGHT_IN = A4_HEIGHT_IN - 2.0 * A4_MARGIN_IN

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------

LABEL_SIZE = 10
TICK_SIZE = 10
LEGEND_SIZE = 9
EMPHASIS_SIZE = 12

# ---------------------------------------------------------------------------
# Line weights
# ---------------------------------------------------------------------------

LW_NONE = 0.0
LW_FINE = 0.6
LW_LIGHT = 1.0
LW_STANDARD = 1.5
LW_MEDIUM = 2.0
LW_THICK = 2.5

# ---------------------------------------------------------------------------
# Marker sizes (markersize= for ax.plot, diameter in points)
# ---------------------------------------------------------------------------

MS_MICRO = 2.0
MS_FINE = 3.0
MS_STANDARD = 6.0
MS_MEDIUM = 9.0
MS_LARGE = 14.0

# Scatter equivalents (s= for ax.scatter, area in points^2 = ms^2)
SS_MICRO = MS_MICRO**2
SS_FINE = MS_FINE**2
SS_STANDARD = MS_STANDARD**2
SS_MEDIUM = MS_MEDIUM**2
SS_LARGE = MS_LARGE**2

# ---------------------------------------------------------------------------
# Transparency
# ---------------------------------------------------------------------------

ALPHA_EXTRA_LIGHT = 0.2
ALPHA_FAINT = 0.4
ALPHA_LIGHT = 0.6
ALPHA_STANDARD = 0.75
ALPHA_FULL = 1.0

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

NEUTRAL_COLOR = "C7"

# ---------------------------------------------------------------------------
# Style dictionaries (unpack as **kwargs)
# ---------------------------------------------------------------------------

GRID_STYLE = dict(lw=LW_FINE, ls=":", alpha=ALPHA_FAINT, color=NEUTRAL_COLOR)
GUIDE_STYLE = dict(lw=LW_LIGHT, ls="-.", alpha=ALPHA_LIGHT, color=NEUTRAL_COLOR)
FIT_STYLE = dict(lw=LW_STANDARD, ls="-", alpha=ALPHA_STANDARD)
MODEL_STYLE = dict(lw=LW_STANDARD, ls="--", alpha=ALPHA_STANDARD)
ERRORBAR_STYLE = dict(capsize=2, lw=LW_FINE, elinewidth=LW_FINE, markersize=MS_FINE)
FILL_STYLE = dict(alpha=ALPHA_EXTRA_LIGHT, lw=LW_NONE)
SCATTER_STYLE = dict(s=SS_FINE, alpha=ALPHA_STANDARD, lw=LW_NONE)

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

mpl.rcParams.update(
    {
        # LaTeX rendering
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
        "axes.unicode_minus": False,
        # Font sizes
        "legend.fontsize": LEGEND_SIZE,
        # Axes
        "axes.linewidth": LW_LIGHT,
        "axes.grid": True,
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": LW_FINE,
        "ytick.major.width": LW_FINE,
        # Grid
        "grid.linewidth": GRID_STYLE["lw"],
        "grid.alpha": GRID_STYLE["alpha"],
        "grid.linestyle": GRID_STYLE["ls"],
        "grid.color": GRID_STYLE["color"],
        # Figure
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_figure(width, height_ratio, height, subfigures):
    """Build a figure of the given width and proportional height."""
    fig = plt.figure(figsize=(width, height / height_ratio * width))
    if subfigures is not None:
        return fig, fig.subfigures(*subfigures)
    return fig, fig.add_subplot(111)


def textwidth_figure(height_out_of_16, subfigures=None):
    """Create a figure spanning the full text width.

    Parameters
    ----------
    height_out_of_16 : float
        Figure height expressed as units out of 16 of the text width.
    subfigures : tuple of int, optional
        ``(nrows, ncols)`` to partition the figure into subfigures.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_or_subfigs : Axes or ndarray of SubFigure
        A single ``Axes`` if ``subfigures`` is ``None``, otherwise the
        array of subfigures.
    """
    return _make_figure(TEXTWIDTH_IN, 16, height_out_of_16, subfigures)


def columnwidth_figure(height_out_of_7_5, subfigures=None):
    """Create a figure spanning a single column width.

    Parameters
    ----------
    height_out_of_7_5 : float
        Figure height expressed as units out of 7.5 of the column width.
    subfigures : tuple of int, optional
        ``(nrows, ncols)`` to partition the figure into subfigures.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_or_subfigs : Axes or ndarray of SubFigure
        A single ``Axes`` if ``subfigures`` is ``None``, otherwise the
        array of subfigures.
    """
    return _make_figure(COLUMNWIDTH_IN, 7.5, height_out_of_7_5, subfigures)


def landscapewidth_figure(height_out_of_10, subfigures=None):
    """Create a figure spanning the A4 usable height (landscape width).

    Parameters
    ----------
    height_out_of_10 : float
        Figure height expressed as units out of 10 of the landscape width.
    subfigures : tuple of int, optional
        ``(nrows, ncols)`` to partition the figure into subfigures.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_or_subfigs : Axes or ndarray of SubFigure
        A single ``Axes`` if ``subfigures`` is ``None``, otherwise the
        array of subfigures.
    """
    return _make_figure(A4_USABLE_HEIGHT_IN, 10, height_out_of_10, subfigures)


def corner_figure():
    """Create a square figure sized at 0.7x the text width.

    Returns
    -------
    matplotlib.figure.Figure
    """
    side = 0.7 * TEXTWIDTH_IN
    return plt.figure(figsize=(side, side))


def subpanels(
    parent,
    nrows: int,
    ncols: int = 1,
    height_ratios: list[int] | tuple[int, ...] | None = None,
    width_ratios: list[int] | tuple[int, ...] | None = None,
    hspace: float = 0.0,
    wspace: float = 0.0,
    sharex: bool = True,
    sharey: bool = False,
):
    """Create a grid of axes inside a parent figure or subfigure.

    Parameters
    ----------
    parent : Figure or SubFigure
        The container in which to lay out the axes.
    nrows : int
        Number of rows of axes.
    ncols : int, optional
        Number of columns of axes.
    height_ratios, width_ratios : sequence of int, optional
        Relative sizes of rows and columns.
    hspace, wspace : float, optional
        Spacing between axes in fractions of axis height/width.
    sharex, sharey : bool, optional
        Whether axes share x- or y-limits.

    Returns
    -------
    Axes or ndarray of Axes

    Examples
    --------
    >>> fig = plt.figure(figsize=(7, 5))
    >>> axes = subpanels(fig, 2, height_ratios=(3, 1))
    """
    gridspec_kw = {"hspace": hspace, "wspace": wspace}
    if height_ratios is not None:
        gridspec_kw["height_ratios"] = list(height_ratios)
    if width_ratios is not None:
        gridspec_kw["width_ratios"] = list(width_ratios)

    return parent.subplots(
        nrows, ncols,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=gridspec_kw,
    )


def zero_line(ax) -> None:
    """Draw a horizontal reference line at ``y = 0``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    """
    ax.axhline(0.0, **GUIDE_STYLE)


def unity_line(ax, label: str | None = None) -> None:
    """Draw a horizontal reference line at ``y = 1``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    label : str, optional
        Legend label for the line.
    """
    ax.axhline(1.0, **GUIDE_STYLE, label=label)
