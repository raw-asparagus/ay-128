"""Lab 03 — Galaxy image classification plotters.

All functions follow the ugdatalab convention: no defaults on data
arguments, style constants from ``ugdatalab.plotting``, single
``savefig(fig, name)`` call, return axes.
"""

from pathlib import Path

import numpy as np
from matplotlib.ticker import ScalarFormatter


def _decimal_log_yaxis(ax):
    """Format the y-axis tick labels as plain decimals instead of scientific notation."""
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)

from ugdatalab.plotting import (
    LW_LIGHT,
    LW_STANDARD,
    LW_MEDIUM,
    LW_NONE,
    SS_MICRO,
    ALPHA_EXTRA_LIGHT,
    ALPHA_FAINT,
    ALPHA_LIGHT,
    ALPHA_STANDARD,
    ALPHA_FULL,
    NEUTRAL_COLOR,
    GUIDE_STYLE,
    LABEL_SIZE,
    LEGEND_SIZE,
    columnwidth_figure,
    landscapewidth_figure,
    textwidth_figure,
    subpanels,
)
from ugdatalab.models.galaxy_zoo.constants import LABEL_COLUMNS

_FIGURES_DIR = Path(__file__).parent / "report" / "figures"

# --- Layout ---
# Columns in the random-image montage
_RANDOM_IMAGES_NCOLS = 7
# Columns in the label-distribution histogram grid
_LABEL_DIST_NCOLS = 7
# Per-row figure height scale (inches)
_LABEL_DIST_FIGURE_SCALE = 2.2
# Columns in the prototype-image grid
_PROTOTYPE_IMAGES_NCOLS = 7
# Max columns in the pairwise-label scatter grid
_PAIRWISE_LABEL_MAX_COLS = 4

# --- Histogram bin counts ---
# Bin count for per-label distribution histograms
_LABEL_DIST_HIST_BINS = 50
# Bin count for train/val split-distribution histograms
_SPLIT_DIST_NBINS = 30
# 2D histogram bins for pairwise density panels
_PAIRWISE_2D_HIST_BINS = 60
# Bin count for pixel-statistic histograms
_PIXEL_STAT_HIST_BINS = 50

# --- Corner / contour ---
# 1σ and 2σ enclosed-mass contour levels
_CORNER_CONTOUR_LEVELS = (0.393, 0.865)

# --- MCMC effective-sample-size thresholds (mirrors labs/02) ---
# ESS at or above which posteriors are "well-mixed"
_NEFF_HIGH_THRESHOLD = 1000
# ESS below which posteriors are flagged as poorly-mixed
_NEFF_LOW_THRESHOLD = 200

# --- Label-axis padding ---
# Symmetric pad on the [0, 1] label-probability axis
_LABEL_AXIS_PAD = 0.05

# --- Grid layout: blank-panel insertions for label-grouped subplots ---
# Insert a blank panel AFTER plotting the label at each index below.
# 10 -> after Class5.2 (just-noticeable bulge), so Class5.3 (obvious bulge)
#       starts in the cell after the blank.
# 15 -> after Class7.1 (completely round), so Class7.2 (in-between) starts
#       in the cell after the blank.
_LABEL_GRID_SKIPS = (10, 15)


def _panel_for_label(label_index, skips=_LABEL_GRID_SKIPS):
    """Map label index to panel index, accounting for trailing-blank insertions."""
    return label_index + sum(1 for s in skips if label_index > s)


def savefig(fig, name):
    """Write *fig* to ``report/figures/<name>``, creating the directory if needed."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES_DIR / name)


# ---------------------------------------------------------------------------
# 1. Random sample images (Task 5)
# ---------------------------------------------------------------------------

def plot_random_images(images, galaxy_ids, n_samples, seed):
    """Grid of random galaxy images from the training set.

    Parameters
    ----------
    images : ndarray, shape (N, H, W, 3)
        Image array in [0, 1].
    galaxy_ids : ndarray, shape (N,)
    n_samples : int
        Number of images to display.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=n_samples, replace=False)

    ncols = _RANDOM_IMAGES_NCOLS
    nrows = (n_samples + ncols - 1) // ncols

    fig, _ = textwidth_figure(2 * nrows)
    _.remove()
    axes = subpanels(fig, nrows, ncols, hspace=0.42, wspace=-0.30, sharex=False)

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < n_samples:
            ax.imshow(images[idx[i]])
            ax.set_title(f"ID {galaxy_ids[idx[i]]}",
                         fontsize=LEGEND_SIZE - 2, loc="left")
            ax.axis("off")
        else:
            ax.set_visible(False)

    savefig(fig, "fig_example_galaxy.pdf")
    return axes


# ---------------------------------------------------------------------------
# 2. Label distributions (Task 6)
# ---------------------------------------------------------------------------

def plot_label_distributions(labels, label_names, label_descriptive):
    """Normalized histograms of all 37 classification labels, grouped by class.

    One row per GZ2 decision-tree class (Class1..Class11); columns are the
    answers within that class. This makes the hierarchical structure of the
    decision tree visually apparent: top-level classes are densely populated
    across [0, 1], while conditional sub-classes concentrate near zero because
    they are only asked along specific branches.

    Parameters
    ----------
    labels : ndarray, shape (N, 37)
    label_names : list of str
        Column names (e.g. "Class1.1").
    label_descriptive : dict
        Mapping column name -> descriptive name.
    """
    n_labels = labels.shape[1]
    ncols = _LABEL_DIST_NCOLS
    n_panels_needed = n_labels + len(_LABEL_GRID_SKIPS)
    nrows = (n_panels_needed + ncols - 1) // ncols

    fig, _ = textwidth_figure(_LABEL_DIST_FIGURE_SCALE * nrows)
    _.remove()
    axes = subpanels(fig, nrows, ncols, hspace=0.56, wspace=0.28, sharex=True)

    # Map each label index to its panel position (accounting for blank-cell skips)
    label_to_panel = [_panel_for_label(i) for i in range(n_labels)]
    panel_to_label = {p: i for i, p in enumerate(label_to_panel)}

    # Last row in each column that carries a visible histogram, for xtick gating
    last_row_for_col = [-1] * ncols
    for p in label_to_panel:
        r, c = divmod(p, ncols)
        if r > last_row_for_col[c]:
            last_row_for_col[c] = r

    for p in range(nrows * ncols):
        row, col = divmod(p, ncols)
        ax = axes[row, col]
        if p in panel_to_label:
            i = panel_to_label[p]
            class_id = int(label_names[i].split("Class")[1].split(".")[0])
            color = f"C{(class_id - 1) % 10}"
            ax.hist(labels[:, i], bins=_LABEL_DIST_HIST_BINS, density=True, color=color,
                    alpha=ALPHA_STANDARD, lw=LW_NONE)
            desc = label_descriptive.get(label_names[i], label_names[i])
            ax.set_title(f"{label_names[i]}:\n{desc}",
                         fontsize=LEGEND_SIZE - 2, loc="left")
            ax.set_xlim(-_LABEL_AXIS_PAD, 1.0 + _LABEL_AXIS_PAD)
            ax.tick_params(labelsize=LEGEND_SIZE - 2)
            ax.tick_params(labelbottom=(row == last_row_for_col[col]))
        else:
            ax.set_visible(False)

    fig.supxlabel("Label probability", fontsize=LABEL_SIZE, y=0.04)
    fig.supylabel("Density", fontsize=LABEL_SIZE, x=0.06)

    savefig(fig, "fig_label_distributions.pdf")
    return axes


# ---------------------------------------------------------------------------
# 3. Prototype images (Task 7)
# ---------------------------------------------------------------------------

def plot_prototype_images(images, galaxy_ids, labels, label_names,
                          label_descriptive):
    """Image with highest value for each of 37 labels.

    Parameters
    ----------
    images : ndarray, shape (N, H, W, 3)
    galaxy_ids : ndarray, shape (N,)
    labels : ndarray, shape (N, 37)
    label_names : list of str
    label_descriptive : dict
    """
    n_labels = labels.shape[1]
    ncols = _PROTOTYPE_IMAGES_NCOLS
    n_panels_needed = n_labels + len(_LABEL_GRID_SKIPS)
    nrows = (n_panels_needed + ncols - 1) // ncols

    # Portrait textwidth figure; aim for near-square panels with title
    # padding by sizing the figure height as 16 * (nrows / ncols) * scale.
    fig, _ = textwidth_figure(16 * (nrows / ncols) * 1.10)
    _.remove()
    axes = subpanels(fig, nrows, ncols, hspace=0.60, wspace=-0.10, sharex=False)

    panel_to_label = {_panel_for_label(i): i for i in range(n_labels)}

    for p in range(nrows * ncols):
        row, col = divmod(p, ncols)
        ax = axes[row, col]
        if p in panel_to_label:
            i = panel_to_label[p]
            best = np.argmax(labels[:, i])
            ax.imshow(images[best])
            desc = label_descriptive.get(label_names[i], label_names[i])
            ax.set_title(f"{label_names[i]}: {labels[best, i]:.2f}\n"
                         f"{desc}\n"
                         f"ID {galaxy_ids[best]}",
                         fontsize=LEGEND_SIZE - 3, loc="left")
            ax.axis("off")
        else:
            ax.set_visible(False)

    savefig(fig, "fig_prototype_images.pdf")
    return axes


# ---------------------------------------------------------------------------
# 4. Correlation matrix (Task 8)
# ---------------------------------------------------------------------------

def plot_correlation_matrix(corr_matrix, label_descriptive_list):
    """37x37 heatmap of label correlations.

    Parameters
    ----------
    corr_matrix : ndarray, shape (37, 37)
    label_descriptive_list : list of str
        Descriptive names in column order.
    """
    fig, ax = textwidth_figure(8)

    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1,
                   aspect="equal", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\rho_{ij}$")

    ax.set_xticks(range(len(label_descriptive_list)))
    ax.set_yticks(range(len(label_descriptive_list)))
    ax.set_xticklabels(label_descriptive_list, rotation=90,
                       fontsize=LEGEND_SIZE - 2)
    ax.set_yticklabels(label_descriptive_list, fontsize=LEGEND_SIZE - 2)

    savefig(fig, "fig_correlation_matrix.pdf")
    return ax


# ---------------------------------------------------------------------------
# 5. Image comparison before/after resizing (Task 10)
# ---------------------------------------------------------------------------

def plot_image_comparison(
    images_original,
    images_cache,
    images_cropped,
    images_rotated,
    galaxy_ids,
):
    """Show the four stages of the preprocessing + augmentation pipeline.

    Parameters
    ----------
    images_original : list of ndarray, shape (H, W, 3)
        Raw images (e.g. 424×424).
    images_cache : list of ndarray, shape (136, 136, 3)
        Pre-cropped + resized images (the on-disk cache, rotation-safe buffer).
    images_cropped : list of ndarray, shape (96, 96, 3)
        Cache center-cropped to the model input size, no rotation
        (the angle-0 input the CNN actually sees).
    images_rotated : list of ndarray, shape (96, 96, 3)
        Cache rotated then center-cropped (the augmented input the CNN
        sees at non-zero rotation angles; verifies that the
        rotation-safe buffer eliminates corner artifacts).
    galaxy_ids : array-like
        Galaxy IDs for titles.
    """
    n = len(images_original)

    fig, _ = textwidth_figure(2.5 * n)
    _.remove()
    axes = subpanels(fig, n, 4, hspace=0.3, wspace=0.1, sharex=False)
    if n == 1:
        axes = axes[np.newaxis, :]

    column_titles = [
        lambda i: f"Original — {galaxy_ids[i]}",
        lambda i: f"Cache ({images_cache[i].shape[0]}×{images_cache[i].shape[1]})",
        lambda i: f"Model input ({images_cropped[i].shape[0]}×{images_cropped[i].shape[1]})",
        lambda i: f"Rotated + crop ({images_rotated[i].shape[0]}×{images_rotated[i].shape[1]})",
    ]
    column_images = [images_original, images_cache, images_cropped, images_rotated]

    for i in range(n):
        for j, (col, title) in enumerate(zip(column_images, column_titles)):
            axes[i, j].imshow(col[i])
            axes[i, j].set_title(title(i), fontsize=LEGEND_SIZE)
            axes[i, j].axis("off")

    savefig(fig, "fig_image_comparison.pdf")
    return axes


# ---------------------------------------------------------------------------
# 6. Train/val split distribution comparison (Task 12)
# ---------------------------------------------------------------------------

def plot_split_distributions(train_labels, val_labels, label_names,
                             label_descriptive):
    """Overlaid histograms of train vs validation for each label.

    Parameters
    ----------
    train_labels : ndarray, shape (N_train, 37)
    val_labels : ndarray, shape (N_val, 37)
    label_names : list of str
    label_descriptive : dict
    """
    n_labels = train_labels.shape[1]
    ncols = _LABEL_DIST_NCOLS
    n_panels_needed = n_labels + len(_LABEL_GRID_SKIPS)
    nrows = (n_panels_needed + ncols - 1) // ncols

    fig, _ = textwidth_figure(_LABEL_DIST_FIGURE_SCALE * nrows)
    _.remove()
    axes = subpanels(fig, nrows, ncols, hspace=0.56, wspace=0.28, sharex=True)

    # Map each label index to its panel position (accounting for blank-cell skips)
    label_to_panel = [_panel_for_label(i) for i in range(n_labels)]
    panel_to_label = {p: i for i, p in enumerate(label_to_panel)}

    # Last row in each column that carries a visible histogram, for xtick gating
    last_row_for_col = [-1] * ncols
    for p in label_to_panel:
        r, c = divmod(p, ncols)
        if r > last_row_for_col[c]:
            last_row_for_col[c] = r

    bins = np.linspace(0, 1, _SPLIT_DIST_NBINS)
    occupied_panels = set(panel_to_label.keys())
    unused_panels = [p for p in range(nrows * ncols) if p not in occupied_panels]
    # Bottom-right empty cell hosts the legend (no histogram overlay)
    legend_panel = unused_panels[-1] if unused_panels else None

    for p in range(nrows * ncols):
        row, col = divmod(p, ncols)
        ax = axes[row, col]
        if p in panel_to_label:
            i = panel_to_label[p]
            ax.hist(train_labels[:, i], bins=bins, density=True,
                    histtype="step", color="C0", lw=LW_STANDARD,
                    ls="-", alpha=ALPHA_LIGHT)
            ax.hist(val_labels[:, i], bins=bins, density=True,
                    histtype="step", color="C1", lw=LW_STANDARD,
                    ls="--", alpha=ALPHA_LIGHT)
            desc = label_descriptive.get(label_names[i], label_names[i])
            ax.set_title(f"{label_names[i]}:\n{desc}",
                         fontsize=LEGEND_SIZE - 2, loc="left")
            ax.set_xlim(-_LABEL_AXIS_PAD, 1.0 + _LABEL_AXIS_PAD)
            ax.tick_params(labelsize=LEGEND_SIZE - 2)
            ax.tick_params(labelbottom=(row == last_row_for_col[col]))
        elif p == legend_panel:
            # Keep this axes alive but hide ticks/spines so the legend sits cleanly
            ax.axis("off")
            from matplotlib.lines import Line2D
            handles = [
                Line2D([], [], color="C0", lw=LW_STANDARD, ls="-",
                       alpha=ALPHA_LIGHT, label="Train"),
                Line2D([], [], color="C1", lw=LW_STANDARD, ls="--",
                       alpha=ALPHA_LIGHT, label="Val"),
            ]
            ax.legend(handles=handles, loc="center",
                      fontsize=LEGEND_SIZE - 1, frameon=False)
        else:
            ax.set_visible(False)

    fig.supxlabel("Label probability", fontsize=LABEL_SIZE, y=0.04)
    fig.supylabel("Density", fontsize=LABEL_SIZE, x=0.06)

    savefig(fig, "fig_split_distributions.pdf")
    return axes


# ---------------------------------------------------------------------------
# 7. Loss curves (Tasks 15, 16)
# ---------------------------------------------------------------------------

def plot_loss_curves(train_losses, val_losses, model_name):
    """Training and validation RMSE vs epoch.

    Parameters
    ----------
    train_losses, val_losses : array-like, shape (n_epochs,)
    model_name : str
        Used in title and filename.
    """
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = textwidth_figure(3)

    ax.plot(epochs, train_losses, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
            color="C0", label="Training", zorder=3)
    ax.plot(epochs, val_losses, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
            color="C1", label="Validation", zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_yscale("log")
    _decimal_log_yaxis(ax)
    ax.set_title(model_name, loc="left", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    savefig(fig, f"fig_loss_{safe_name}.pdf")
    return ax


# ---------------------------------------------------------------------------
# 8. Loss curves with learning rate panel (Tasks 20, 21)
# ---------------------------------------------------------------------------

def plot_loss_with_lr(train_losses, val_losses, learning_rates, model_name):
    """Two-panel: loss curves (top) and learning rate (bottom).

    Parameters
    ----------
    train_losses, val_losses : array-like, shape (n_epochs,)
    learning_rates : array-like, shape (n_epochs,)
    model_name : str
    """
    epochs = np.arange(1, len(train_losses) + 1)

    # Auto-pick figure width: 50-epoch (or shorter) runs render at
    # columnwidth; longer runs (typically 100 epochs) use textwidth.
    if len(epochs) <= 50:
        fig, _ = columnwidth_figure(7.5)
    else:
        fig, _ = textwidth_figure(5)
    _.remove()
    ax_loss, ax_lr = subpanels(fig, 2, height_ratios=(3, 1), sharex=True)

    ax_loss.plot(epochs, train_losses, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
                 color="C0", label="Training", zorder=3)
    ax_loss.plot(epochs, val_losses, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
                 color="C1", label="Validation", zorder=3)
    ax_loss.set_ylabel("RMSE")
    ax_loss.set_yscale("log")
    _decimal_log_yaxis(ax_loss)
    ax_loss.set_title(model_name, loc="left", fontsize=LABEL_SIZE)
    ax_loss.legend(fontsize=LEGEND_SIZE)

    ax_lr.plot(epochs, learning_rates, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
               color="C2", zorder=3)
    ax_lr.set_ylabel("LR")
    ax_lr.set_yscale("log")
    ax_lr.set_xlabel("Epoch")

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    savefig(fig, f"fig_loss_lr_{safe_name}.pdf")
    return ax_loss, ax_lr


def plot_loss_with_lr_pair(runs, filename):
    """Two-column loss + LR layout: paired short-run ablations side-by-side.

    Renders two ``plot_loss_with_lr``-style panels in a single textwidth
    figure. Each column gets a loss panel on top and an LR panel below,
    sharing the same Epoch axis. Designed to merge short paired
    experiments (e.g. scheduler-only vs.\ augmentation-only) into one
    compact figure rather than two separate full-width ones.

    Parameters
    ----------
    runs : sequence of dict
        Two dicts each with keys ``train_losses``, ``val_losses``,
        ``learning_rates`` (1D arrays of equal length), and ``title``.
    filename : str
        Output PDF filename (without the ``.pdf`` suffix, written to
        ``report/figures/``).
    """
    assert len(runs) == 2, "plot_loss_with_lr_pair expects exactly two runs"

    fig, _ = textwidth_figure(4.0)
    _.remove()
    axes = subpanels(
        fig, nrows=2, ncols=2, height_ratios=(3, 1),
        hspace=0.05, wspace=0.32, sharex=False,
    )

    for col, run in enumerate(runs):
        ax_loss = axes[0, col]
        ax_lr = axes[1, col]
        epochs = np.arange(1, len(run["train_losses"]) + 1)

        ax_loss.plot(epochs, run["train_losses"], lw=LW_STANDARD,
                     alpha=ALPHA_STANDARD, color="C0", label="Training",
                     zorder=3)
        ax_loss.plot(epochs, run["val_losses"], lw=LW_STANDARD,
                     alpha=ALPHA_STANDARD, color="C1", label="Validation",
                     zorder=3)
        ax_loss.set_yscale("log")
        _decimal_log_yaxis(ax_loss)
        ax_loss.set_title(run["title"], loc="left", fontsize=LABEL_SIZE)
        ax_loss.tick_params(labelbottom=False)
        if col == 0:
            ax_loss.set_ylabel("RMSE")
            ax_loss.legend(fontsize=LEGEND_SIZE - 1, loc="upper right")

        ax_lr.plot(epochs, run["learning_rates"], lw=LW_STANDARD,
                   alpha=ALPHA_STANDARD, color="C2", zorder=3)
        ax_lr.set_yscale("log")
        ax_lr.set_xlabel("Epoch")
        if col == 0:
            ax_lr.set_ylabel("LR")

    savefig(fig, f"{filename}.pdf")
    return axes


# ---------------------------------------------------------------------------
# 9. Model comparison (Task 23)
# ---------------------------------------------------------------------------

def plot_custom_vs_resnet_with_delta(custom_run, resnet_run, filename):
    """Overlaid train+val loss curves for two models with a delta panel.

    Renders a 2-panel figure:
      * top panel (tall): training (dashed) and validation (solid) RMSE
        for the two models, colour-coded by model.
      * bottom panel (short): per-model overfitting gap, val - train,
        plotted as solid lines in the model colours.

    No plot title --- the model names appear in the legend.

    Parameters
    ----------
    custom_run, resnet_run : dict
        Each with keys ``name`` (str), ``train_losses`` (1D), and
        ``val_losses`` (1D).
    filename : str
        Output PDF filename (without ``.pdf``); written to ``report/figures/``.
    """
    fig, _ = textwidth_figure(5.0)
    _.remove()
    ax_top, ax_delta = subpanels(
        fig, nrows=2, height_ratios=(3, 1), hspace=0.05, sharex=True,
    )

    for run, color in [(custom_run, "C0"), (resnet_run, "C1")]:
        train = np.asarray(run["train_losses"])
        val = np.asarray(run["val_losses"])
        epochs = np.arange(1, len(train) + 1)
        ax_top.plot(epochs, train, color=color, ls="--", lw=LW_STANDARD,
                    alpha=ALPHA_STANDARD,
                    label=f"{run['name']} (training)", zorder=3)
        ax_top.plot(epochs, val, color=color, ls="-", lw=LW_STANDARD,
                    alpha=ALPHA_STANDARD,
                    label=f"{run['name']} (validation)", zorder=3)

    # Delta panel: validation RMSE difference between the two models
    custom_val = np.asarray(custom_run["val_losses"])
    resnet_val = np.asarray(resnet_run["val_losses"])
    n_delta = min(len(custom_val), len(resnet_val))
    delta_epochs = np.arange(1, n_delta + 1)
    ax_delta.plot(delta_epochs, custom_val[:n_delta] - resnet_val[:n_delta],
                  color="black", ls="-", lw=LW_STANDARD,
                  alpha=ALPHA_STANDARD, zorder=3)

    ax_top.set_ylabel("RMSE")
    ax_top.set_yscale("log")
    _decimal_log_yaxis(ax_top)
    # Auto-scale the y-axis using all curves EXCEPT the ResNet training
    # curve (which dives to ~0.018 at convergence and would dominate the
    # axis). The ResNet training segment below the resulting floor is
    # clipped from view but the other three curves remain in full.
    relevant = np.concatenate([
        np.asarray(custom_run["train_losses"]),
        np.asarray(custom_run["val_losses"]),
        np.asarray(resnet_run["val_losses"]),
    ])
    ax_top.set_ylim(relevant.min() * 0.95, relevant.max() * 1.05)
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(fontsize=LEGEND_SIZE - 1, loc="upper right", ncol=2)

    ax_delta.axhline(0.0, ls=":", color=NEUTRAL_COLOR, lw=LW_LIGHT,
                     alpha=ALPHA_LIGHT, zorder=2)
    ax_delta.set_xlabel("Epoch")
    ax_delta.set_ylabel(r"$\Delta$ RMSE$_{\mathrm{val}}$")

    savefig(fig, f"{filename}.pdf")
    return ax_top, ax_delta


def plot_model_comparison(names, val_losses_list):
    """Overlaid validation loss curves for multiple models.

    Parameters
    ----------
    names : list of str
        Model names for legend.
    val_losses_list : list of array-like
        Validation RMSE per epoch for each model.
    """
    fig, ax = textwidth_figure(3)

    for i, (name, val_losses) in enumerate(zip(names, val_losses_list)):
        epochs = np.arange(1, len(val_losses) + 1)
        ax.plot(epochs, val_losses, lw=LW_STANDARD, alpha=ALPHA_LIGHT,
                color=f"C{i}", label=name, zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.set_yscale("log")
    _decimal_log_yaxis(ax)
    ax.legend(fontsize=LEGEND_SIZE)

    savefig(fig, "fig_model_comparison.pdf")
    return ax


# ---------------------------------------------------------------------------
# 10. True vs predicted scatter (Task 24)
# ---------------------------------------------------------------------------

def plot_label_scatter(true_labels, pred_labels, label_descriptive_list):
    """Grid of scatter plots comparing true vs predicted for each label.

    Parameters
    ----------
    true_labels : ndarray, shape (N, 37)
    pred_labels : ndarray, shape (N, 37)
    label_descriptive_list : list of str
        Descriptive names in column order.
    """
    n_labels = true_labels.shape[1]
    ncols = 6
    nrows = (n_labels + ncols - 1) // ncols

    fig, _ = textwidth_figure(2.5 * nrows)
    _.remove()
    axes = subpanels(fig, nrows, ncols, hspace=0.65, wspace=0.35, sharex=False)

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < n_labels:
            t = true_labels[:, i]
            p = pred_labels[:, i]
            resid = p - t
            bias = np.mean(resid)
            scatter_val = np.std(resid)

            ax.scatter(t, p, s=1.0, color="C0", alpha=ALPHA_EXTRA_LIGHT,
                       zorder=3, rasterized=True)
            ax.plot([0, 1], [0, 1], color=NEUTRAL_COLOR, lw=LW_MEDIUM,
                    alpha=ALPHA_STANDARD, zorder=1)
            ax.set_xlim(-_LABEL_AXIS_PAD, 1.0 + _LABEL_AXIS_PAD)
            ax.set_ylim(-_LABEL_AXIS_PAD, 1.0 + _LABEL_AXIS_PAD)
            ax.set_title(
                f"{LABEL_COLUMNS[i]}: {label_descriptive_list[i]}\n"
                rf"bias$={bias:+.3f}$, $\sigma={scatter_val:.3f}$",
                fontsize=LEGEND_SIZE - 2, loc="left",
            )
            ax.tick_params(labelsize=LEGEND_SIZE - 1)
        else:
            ax.set_visible(False)

    fig.supxlabel("True", fontsize=LABEL_SIZE)
    fig.supylabel("Predicted", fontsize=LABEL_SIZE)

    savefig(fig, "fig_label_scatter.pdf")
    return axes


# ---------------------------------------------------------------------------
# 11. Top-5 extreme images for a label (Task 25)
# ---------------------------------------------------------------------------

def plot_top5_images(images, true_labels, pred_labels, label_idx,
                     label_name, galaxy_ids):
    """Top 5 images by actual and predicted probability for one label.

    Two rows: top row = actual highest, bottom row = predicted highest.

    Parameters
    ----------
    images : ndarray, shape (N, H, W, 3)
    true_labels : ndarray, shape (N,)
        True label values for this label index.
    pred_labels : ndarray, shape (N,)
        Predicted label values for this label index.
    label_idx : int
        Label column index (for filename).
    label_name : str
        Descriptive label name for title.
    galaxy_ids : ndarray, shape (N,)
    """
    fig, _ = textwidth_figure(4)
    _.remove()
    axes = subpanels(fig, 2, 5, hspace=0.2, wspace=0.1, sharex=False)

    top5_true = np.argsort(true_labels)[-5:][::-1]
    top5_pred = np.argsort(pred_labels)[-5:][::-1]

    for j in range(5):
        idx_t = top5_true[j]
        axes[0, j].imshow(images[idx_t])
        axes[0, j].set_title(f"{true_labels[idx_t]:.2f}",
                             fontsize=LEGEND_SIZE - 1)
        axes[0, j].axis("off")

        idx_p = top5_pred[j]
        axes[1, j].imshow(images[idx_p])
        axes[1, j].set_title(f"{pred_labels[idx_p]:.2f}",
                             fontsize=LEGEND_SIZE - 1)
        axes[1, j].axis("off")

    axes[0, 0].set_ylabel("Actual", fontsize=LABEL_SIZE)
    axes[1, 0].set_ylabel("Predicted", fontsize=LABEL_SIZE)
    fig.suptitle(label_name, fontsize=LABEL_SIZE)

    safe_name = label_name.lower().replace(" ", "_").replace(":", "").replace("/", "_")
    savefig(fig, f"fig_top5_{safe_name}.pdf")
    return axes


# ---------------------------------------------------------------------------
# Branch color scheme for the GZ2 decision tree
# ---------------------------------------------------------------------------

# Each branch maps to a matplotlib color used for both the tree diagram
# and the heatmap tick labels.
_BRANCH_COLORS = {
    "root":     "C7",   # Q1 top-level (gray)
    "smooth":   "C3",   # Q7 roundness (red)
    "edge-on":  "C1",   # Q2 edge-on, Q9 bulge shape (orange)
    "face-on":  "C2",   # Q3 bar (green)
    "spiral":   "C0",   # Q4 spiral, Q10 tightness, Q11 arm count (blue)
    "bulge":    "C4",   # Q5 bulge prominence (purple)
    "odd":      "C5",   # Q6 odd, Q8 odd features (brown)
}

# Reordering of labels by tree branch, with the branch tag for each label.
_BRANCH_ORDER = [
    ("Class1.1",  "root"),
    ("Class1.2",  "root"),
    ("Class1.3",  "root"),
    ("Class7.1",  "smooth"),
    ("Class7.2",  "smooth"),
    ("Class7.3",  "smooth"),
    ("Class2.1",  "edge-on"),
    ("Class2.2",  "edge-on"),
    ("Class9.1",  "edge-on"),
    ("Class9.2",  "edge-on"),
    ("Class9.3",  "edge-on"),
    ("Class3.1",  "face-on"),
    ("Class3.2",  "face-on"),
    ("Class4.1",  "spiral"),
    ("Class4.2",  "spiral"),
    ("Class5.1",  "bulge"),
    ("Class5.2",  "bulge"),
    ("Class5.3",  "bulge"),
    ("Class5.4",  "bulge"),
    ("Class10.1", "spiral"),
    ("Class10.2", "spiral"),
    ("Class10.3", "spiral"),
    ("Class11.1", "spiral"),
    ("Class11.2", "spiral"),
    ("Class11.3", "spiral"),
    ("Class11.4", "spiral"),
    ("Class11.5", "spiral"),
    ("Class11.6", "spiral"),
    ("Class6.1",  "odd"),
    ("Class6.2",  "odd"),
    ("Class8.1",  "odd"),
    ("Class8.2",  "odd"),
    ("Class8.3",  "odd"),
    ("Class8.4",  "odd"),
    ("Class8.5",  "odd"),
    ("Class8.6",  "odd"),
    ("Class8.7",  "odd"),
]


# ---------------------------------------------------------------------------
# 12. Hierarchical label tree heatmap (Diagnostic 1)
# ---------------------------------------------------------------------------

def plot_hierarchical_correlation(corr_matrix, label_descriptive_list,
                                  label_tree, label_columns):
    """Correlation heatmap with labels reordered and colored by branch.

    Parameters
    ----------
    corr_matrix : ndarray, shape (37, 37)
    label_descriptive_list : list of str
    label_tree : dict
        Parent -> children mapping from constants.
    label_columns : list of str
        Label column names in original order.
    """
    col_to_idx = {col: i for i, col in enumerate(label_columns)}
    reorder = [col_to_idx[col] for col, _ in _BRANCH_ORDER]
    branch_tags = [tag for _, tag in _BRANCH_ORDER]

    reordered = corr_matrix[np.ix_(reorder, reorder)]
    reordered_names = [label_descriptive_list[i] for i in reorder]
    tick_colors = [_BRANCH_COLORS[tag] for tag in branch_tags]

    # Branch boundary positions (cumulative group sizes)
    branch_sizes = [3, 3, 5, 2, 2, 4, 9, 2, 7]
    boundaries = np.cumsum(branch_sizes)[:-1]

    fig, ax = textwidth_figure(8)
    im = ax.imshow(reordered, cmap="RdBu_r", vmin=-1, vmax=1,
                   aspect="equal", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\rho_{ij}$")

    # Draw branch separators
    for b in boundaries:
        ax.axhline(b - 0.5, color="k", lw=LW_LIGHT, alpha=ALPHA_STANDARD)
        ax.axvline(b - 0.5, color="k", lw=LW_LIGHT, alpha=ALPHA_STANDARD)

    ax.set_xticks(range(len(reordered_names)))
    ax.set_yticks(range(len(reordered_names)))
    ax.set_xticklabels(reordered_names, rotation=90, fontsize=LEGEND_SIZE - 2)
    ax.set_yticklabels(reordered_names, fontsize=LEGEND_SIZE - 2)

    # Color tick labels by branch
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), tick_colors):
        tick.set_color(color)

    savefig(fig, "fig_hierarchical_correlation.pdf")
    return ax


# ---------------------------------------------------------------------------
# 13. t-SNE / UMAP of label space (Diagnostic 2)
# ---------------------------------------------------------------------------

def plot_label_tsne(embedding, top_label, colorbar_label):
    """2D scatter of label-space embedding colored by a continuous label.

    Parameters
    ----------
    embedding : ndarray, shape (N, 2)
        2D coordinates from t-SNE or UMAP.
    top_label : ndarray, shape (N,)
        Continuous value for coloring (e.g., Smooth fraction).
    colorbar_label : str
        Colorbar label text.
    """
    fig, ax = textwidth_figure(7)

    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=top_label, cmap="coolwarm", s=SS_MICRO,
                    alpha=ALPHA_FAINT, lw=LW_NONE, rasterized=True,
                    vmin=0, vmax=1, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(colorbar_label)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_aspect("equal")

    savefig(fig, "fig_label_tsne.pdf")
    return ax


# ---------------------------------------------------------------------------
# 14. Label-conditional image montages (Diagnostic 3)
# ---------------------------------------------------------------------------

def plot_label_conditional_montage(images, label_a, label_b, name_a, name_b,
                                   threshold, seed):
    """2x2 grid of 3x3 image montages for high/low combinations of two labels.

    Parameters
    ----------
    images : ndarray, shape (N, H, W, 3)
    label_a, label_b : ndarray, shape (N,)
    name_a, name_b : str
    threshold : float
        Cutoff separating "high" from "low".
    seed : int
    """
    rng = np.random.default_rng(seed)

    hi_a = label_a > threshold
    lo_a = label_a <= threshold
    hi_b = label_b > threshold
    lo_b = label_b <= threshold

    quadrants = [
        (hi_a & hi_b, f"High {name_a} + High {name_b}"),
        (hi_a & lo_b, f"High {name_a} + Low {name_b}"),
        (lo_a & hi_b, f"Low {name_a} + High {name_b}"),
        (lo_a & lo_b, f"Low {name_a} + Low {name_b}"),
    ]

    fig, _ = textwidth_figure(10)
    _.remove()
    axes = subpanels(fig, 4, 3, hspace=0.15, wspace=0.05, sharex=False)

    for row, (mask, title) in enumerate(quadrants):
        idx = np.where(mask)[0]
        if len(idx) >= 3:
            chosen = rng.choice(idx, size=3, replace=False)
        else:
            chosen = idx[:3]
        for col in range(3):
            ax = axes[row, col]
            if col < len(chosen):
                ax.imshow(images[chosen[col]])
            ax.axis("off")
        axes[row, 0].set_title(title, fontsize=LEGEND_SIZE - 1, loc="left")

    safe_a = name_a.lower().replace(" ", "_").replace("/", "_")
    safe_b = name_b.lower().replace(" ", "_").replace("/", "_")
    savefig(fig, f"fig_montage_{safe_a}_vs_{safe_b}.pdf")
    return axes


# ---------------------------------------------------------------------------
# 15. Pairwise scatter plots of correlated labels (Diagnostic 4)
# ---------------------------------------------------------------------------

def plot_pairwise_label_scatter(labels, pairs, label_descriptive_list):
    """Density scatter plots for selected label pairs.

    Parameters
    ----------
    labels : ndarray, shape (N, 37)
    pairs : list of (int, int)
        Pairs of label indices to plot.
    label_descriptive_list : list of str
    """
    import matplotlib.colors as mcolors

    n = len(pairs)
    ncols = min(n, _PAIRWISE_LABEL_MAX_COLS)
    nrows = (n + ncols - 1) // ncols

    fig, _ = textwidth_figure(3.5 * nrows)
    _.remove()
    axes = subpanels(fig, nrows, ncols, hspace=0.55, wspace=0.45, sharex=False)
    axes = np.atleast_2d(axes)

    for k, (i, j) in enumerate(pairs):
        row, col = divmod(k, ncols)
        ax = axes[row, col]

        x, y = labels[:, i], labels[:, j]

        # 2D histogram for density coloring
        counts, xedges, yedges = np.histogram2d(x, y, bins=_PAIRWISE_2D_HIST_BINS,
                                                 range=[[0, 1], [0, 1]])
        # Map each point to its bin density
        xi = np.clip(np.digitize(x, xedges) - 1, 0, _PAIRWISE_2D_HIST_BINS - 1)
        yi = np.clip(np.digitize(y, yedges) - 1, 0, _PAIRWISE_2D_HIST_BINS - 1)
        density = counts[xi, yi]

        order = np.argsort(density)
        ax.scatter(x[order], y[order], c=density[order], cmap="viridis",
                   s=SS_MICRO, alpha=ALPHA_FAINT, lw=LW_NONE,
                   rasterized=True, zorder=3,
                   norm=mcolors.LogNorm(vmin=1, vmax=density.max()))

        rho = np.corrcoef(x, y)[0, 1]
        ax.set_title(rf"$\rho = {rho:+.3f}$", fontsize=LEGEND_SIZE - 1)
        ax.set_xlabel(f"{LABEL_COLUMNS[i]}: {label_descriptive_list[i]}",
                      fontsize=LEGEND_SIZE - 1)
        ax.set_ylabel(f"{LABEL_COLUMNS[j]}: {label_descriptive_list[j]}",
                      fontsize=LEGEND_SIZE - 1)
        ax.set_xlim(-_LABEL_AXIS_PAD, 1.0 + _LABEL_AXIS_PAD)
        ax.set_ylim(-_LABEL_AXIS_PAD, 1.0 + _LABEL_AXIS_PAD)
        ax.tick_params(labelsize=LEGEND_SIZE - 1)

    # Hide unused axes
    for k in range(n, nrows * ncols):
        row, col = divmod(k, ncols)
        axes[row, col].set_visible(False)

    savefig(fig, "fig_pairwise_label_scatter.pdf")
    return axes


# ---------------------------------------------------------------------------
# 16. Effective sample size per label (Diagnostic 5)
# ---------------------------------------------------------------------------

def plot_effective_sample_size(labels, label_descriptive_list, threshold):
    """Horizontal bar chart of N_eff = count(label > threshold) per label.

    Parameters
    ----------
    labels : ndarray, shape (N, 37)
    label_descriptive_list : list of str
    threshold : float
        Minimum value to count a galaxy as "active" for that label.
    """
    n_eff = np.sum(labels > threshold, axis=0)
    order = np.argsort(n_eff)

    fig, ax = textwidth_figure(9)

    y_pos = np.arange(len(n_eff))
    colors = ["C0" if n > _NEFF_HIGH_THRESHOLD else "C3" if n < _NEFF_LOW_THRESHOLD else "C1" for n in n_eff[order]]
    ax.barh(y_pos, n_eff[order], color=colors, alpha=ALPHA_STANDARD)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label_descriptive_list[i] for i in order],
                       fontsize=LEGEND_SIZE - 1)
    ax.set_xlabel(rf"$N_{{\mathrm{{eff}}}}$ (galaxies with label $> {threshold}$)")
    ax.axvline(_NEFF_HIGH_THRESHOLD, **GUIDE_STYLE, label=r"$N_{\mathrm{eff}} = 1000$")
    ax.legend(fontsize=LEGEND_SIZE)

    savefig(fig, "fig_effective_sample_size.pdf")
    return ax


# ---------------------------------------------------------------------------
# 17. PCA eigenspectrum of label correlation matrix (Diagnostic 6)
# ---------------------------------------------------------------------------

def plot_pca_eigenspectrum(eigenvalues):
    """Scree plot of eigenvalues from the label correlation matrix.

    Parameters
    ----------
    eigenvalues : ndarray, shape (37,)
        Eigenvalues sorted in descending order.
    """
    n = len(eigenvalues)
    cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    fig, _ = textwidth_figure(4)
    _.remove()
    ax_eig, ax_cum = subpanels(fig, 2, height_ratios=(2, 1), sharex=True)

    ax_eig.bar(np.arange(n), eigenvalues, color="C0", alpha=ALPHA_STANDARD)
    ax_eig.axhline(1.0, **GUIDE_STYLE, label="Kaiser criterion")
    ax_eig.set_ylabel("Eigenvalue")
    ax_eig.legend(fontsize=LEGEND_SIZE)

    ax_cum.plot(np.arange(n), cumvar, lw=LW_STANDARD, color="C1",
                alpha=ALPHA_STANDARD, zorder=3)
    ax_cum.axhline(0.9, **GUIDE_STYLE)
    ax_cum.set_xlabel("Principal component")
    ax_cum.set_ylabel("Cumulative\nvariance")
    ax_cum.set_ylim(0, 1.05)

    savefig(fig, "fig_pca_eigenspectrum.pdf")
    return ax_eig, ax_cum


# ---------------------------------------------------------------------------
# 18. Pixel intensity statistics by morphology (Diagnostic 7)
# ---------------------------------------------------------------------------

def plot_pixel_statistics(stats_smooth, stats_disk, stat_names):
    """Overlaid histograms of pixel statistics split by morphology.

    Parameters
    ----------
    stats_smooth : dict
        Mapping stat_name -> ndarray of values for smooth galaxies.
    stats_disk : dict
        Same for disk galaxies.
    stat_names : list of str
        Which keys to plot, in order.
    """
    n = len(stat_names)
    fig, _ = textwidth_figure(3 * n)
    _.remove()
    axes = subpanels(fig, 1, n, wspace=0.3, sharex=False)
    if n == 1:
        axes = [axes]

    for i, name in enumerate(stat_names):
        ax = axes[i]
        lo = min(stats_smooth[name].min(), stats_disk[name].min())
        hi = max(stats_smooth[name].max(), stats_disk[name].max())
        bins = np.linspace(lo, hi, _PIXEL_STAT_HIST_BINS)
        ax.hist(stats_smooth[name], bins=bins, density=True,
                alpha=ALPHA_LIGHT, color="C3", lw=LW_NONE,
                label="Smooth" if i == 0 else None)
        ax.hist(stats_disk[name], bins=bins, density=True,
                alpha=ALPHA_LIGHT, color="C0", lw=LW_NONE,
                label="Disk" if i == 0 else None)
        ax.set_xlabel(name, fontsize=LABEL_SIZE)
        ax.set_ylabel("Density" if i == 0 else "")
        ax.tick_params(labelsize=LEGEND_SIZE - 1)

    axes[0].legend(fontsize=LEGEND_SIZE)

    savefig(fig, "fig_pixel_statistics.pdf")
    return axes


# ---------------------------------------------------------------------------
# 19. Bar charts — ablation, per-label RMSE, model progression
# ---------------------------------------------------------------------------

def plot_ablation_curves(names, val_losses_list, title):
    """Validation-RMSE loss curves overlaid for architecture sweep variants.

    Parameters
    ----------
    names : sequence of str
        Variant labels (one per curve, shown in the legend).
    val_losses_list : sequence of array-like
        Per-variant validation RMSE arrays of shape ``(n_epochs,)``. Arrays
        of different lengths are tolerated (each plotted against its own
        epoch range).
    title : str
        Figure title; also used to derive the filename.
    """
    # Ablation runs are short (typically 50 epochs); render at columnwidth
    # so the ablation figure pairs visually with other 50-epoch curves.
    max_epochs = max(len(np.asarray(vl)) for vl in val_losses_list)
    if max_epochs <= 50:
        fig, ax = columnwidth_figure(6.0)
    else:
        fig, ax = textwidth_figure(4)
    for name, val_losses in zip(names, val_losses_list):
        val_losses = np.asarray(val_losses)
        epochs = np.arange(1, len(val_losses) + 1)
        is_pivot = "custom cnn" in name.lower()
        if is_pivot:
            ax.plot(epochs, val_losses, label=name, color="black",
                    alpha=ALPHA_FULL, lw=LW_STANDARD, ls="-", zorder=4)
        else:
            ax.plot(epochs, val_losses, label=name, alpha=ALPHA_LIGHT,
                    lw=LW_LIGHT, ls="--", zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    # No title: each curve is identified by its legend entry.
    ax.legend(fontsize=LEGEND_SIZE - 1, loc="best")

    safe_name = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    savefig(fig, f"fig_ablation_{safe_name}.pdf")
    return ax


def plot_per_label_rmse_bar(label_names, rmse_values):
    """Horizontal bar chart of per-label RMSE, sorted descending.

    Parameters
    ----------
    label_names : sequence of str
        Descriptive label names.
    rmse_values : array-like
        RMSE per label.
    """
    order = np.argsort(rmse_values)[::-1]
    names = [label_names[i] for i in order]
    values = np.asarray(rmse_values)[order]

    fig, _ = textwidth_figure(8)
    _.remove()
    ax = subpanels(fig, 1, 1)
    y = np.arange(len(names))
    ax.barh(y, values, color="C0", alpha=ALPHA_STANDARD)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=LEGEND_SIZE - 1)
    ax.invert_yaxis()
    ax.set_xlabel("Validation RMSE")
    ax.set_title("Per-label RMSE (sorted)", loc="left", fontsize=LABEL_SIZE)

    savefig(fig, "fig_per_label_rmse.pdf")
    return ax


# ---------------------------------------------------------------------------
# 19b. Label rarity vs baseline RMSE (Diagnostic, supports NB 01b/02/06)
# ---------------------------------------------------------------------------

def plot_label_rarity_vs_rmse(n_eff, baseline_rmse, label_descriptive_list,
                              n_eff_threshold):
    """Scatter of per-label effective sample size vs mean-prediction baseline RMSE.

    Labels with low $N_{\\mathrm{eff}}$ tend to sit near the bottom of the RMSE
    axis simply because their vote-fraction distribution is concentrated near
    zero — a mean-prediction baseline already does well there. This figure
    flags the labels for which a low CNN RMSE is *not* informative.

    Parameters
    ----------
    n_eff : ndarray, shape (37,)
        Effective sample size per label (e.g. ``np.sum(labels > 0.1, axis=0)``).
    baseline_rmse : ndarray, shape (37,)
        Per-label RMSE of the mean-prediction baseline on the validation set.
    label_descriptive_list : list of str
        Descriptive names in column order.
    n_eff_threshold : float
        Vertical guide; labels with ``n_eff < threshold`` are flagged in red.
    """
    fig, ax = textwidth_figure(4)

    is_rare = np.asarray(n_eff) < n_eff_threshold
    ax.scatter(np.asarray(n_eff)[~is_rare], np.asarray(baseline_rmse)[~is_rare],
               s=20, color="C0", alpha=ALPHA_STANDARD, zorder=3,
               label="Well-sampled")
    ax.scatter(np.asarray(n_eff)[is_rare], np.asarray(baseline_rmse)[is_rare],
               s=20, color="C3", alpha=ALPHA_STANDARD, zorder=3,
               label="Rare")
    ax.axvline(n_eff_threshold, **GUIDE_STYLE)
    for i, name in enumerate(label_descriptive_list):
        ax.annotate(name, (n_eff[i], baseline_rmse[i]),
                    fontsize=LEGEND_SIZE - 3, alpha=ALPHA_LIGHT,
                    xytext=(3, 1), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_{\mathrm{eff}}$ (galaxies with label $> 0.1$)")
    ax.set_ylabel("Mean-prediction baseline RMSE")
    ax.legend(fontsize=LEGEND_SIZE, loc="best")

    savefig(fig, "fig_label_rarity_vs_rmse.pdf")
    return ax


# ---------------------------------------------------------------------------
# 20. Merger fraction vs probability threshold (Task 26)
# ---------------------------------------------------------------------------

def plot_merger_threshold_sweep(merger_probs, thresholds, lotz_band):
    """Fraction of galaxies above each merger-probability threshold, with Lotz band overlay.

    Parameters
    ----------
    merger_probs : ndarray, shape (N,)
        Predicted merger probability per galaxy in the test set.
    thresholds : ndarray, shape (T,)
        Threshold values to sweep over (e.g. ``np.linspace(0.05, 0.95, 19)``).
    lotz_band : tuple of (float, float)
        Lower and upper expected merger fraction from Lotz et al. 2011 Fig. 13,
        upper-right panel at $z \approx 0$, converted to a dimensionless fraction
        using a representative observability timescale.
    """
    fractions = np.array([float(np.mean(merger_probs > t)) for t in thresholds])

    fig, ax = textwidth_figure(3.5)
    lo, hi = lotz_band
    ax.axhspan(lo, hi, color="C2", alpha=ALPHA_EXTRA_LIGHT, zorder=1,
               label=rf"Lotz+11 $z \approx 0$: {lo*100:.1f}–{hi*100:.1f}\%")
    ax.plot(thresholds, fractions, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
            color="C0", zorder=3, label="This work")
    ax.axhline(float(np.mean(merger_probs)), **GUIDE_STYLE,
               label=rf"Mean prob = {np.mean(merger_probs)*100:.2f}\%")
    ax.set_xlabel(r"Merger-probability threshold $p_{\mathrm{cut}}$")
    ax.set_ylabel(r"$f_{\mathrm{merger}}(p > p_{\mathrm{cut}})$")
    ax.set_yscale("log")
    _decimal_log_yaxis(ax)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right")

    savefig(fig, "fig_merger_threshold_sweep.pdf")
    return ax


# ---------------------------------------------------------------------------
# 21. Derived merger rate vs Lotz et al. 2011 (Task 26)
# ---------------------------------------------------------------------------

def plot_merger_rate_vs_lotz(f_merger, f_merger_err, tau_vis_range,
                             lotz_rate, lotz_rate_err):
    """Derived major-merger rate with $\\tau_{\\mathrm{vis}}$ systematic vs Lotz value.

    Converts our merger fraction to a rate using $R = f / \\tau_{\\mathrm{vis}}$
    and propagates both the statistical uncertainty on $f$ and the systematic
    uncertainty on $\\tau_{\\mathrm{vis}}$. Overlays the Lotz et al. 2011 Fig. 13
    upper-right $z \\approx 0$ value with its quoted uncertainty.

    Parameters
    ----------
    f_merger : float
        Best-estimate merger fraction (dimensionless, e.g. 0.04).
    f_merger_err : float
        Statistical 1σ on f_merger from bootstrap.
    tau_vis_range : tuple of (float, float, float)
        $(\\tau_{\\mathrm{lo}}, \\tau_{\\mathrm{mid}}, \\tau_{\\mathrm{hi}})$ in Gyr.
    lotz_rate : float
        Lotz et al. 2011 best-fit major-merger rate at $z \\approx 0$ in Gyr$^{-1}$.
    lotz_rate_err : float
        Lotz et al. 2011 1σ on the rate.
    """
    tau_lo, tau_mid, tau_hi = tau_vis_range

    # Best estimate and its uncertainty
    r_mid = f_merger / tau_mid
    # Combine stat and tau_vis systematic in quadrature on log
    r_lo = f_merger / tau_hi
    r_hi = f_merger / tau_lo

    fig, ax = textwidth_figure(2.6)
    x = [0, 1]
    y = [r_mid, lotz_rate]
    yerr_lo = [r_mid - r_lo, lotz_rate_err]
    yerr_hi = [r_hi - r_mid, lotz_rate_err]
    ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], fmt="o", color="C0",
                lw=LW_STANDARD, capsize=4, zorder=3,
                markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([
        rf"This work" "\n" rf"($\tau_{{\rm vis}}={tau_lo}$–${tau_hi}$ Gyr)",
        rf"Lotz+11" "\n" rf"$z \approx 0$",
    ], fontsize=LEGEND_SIZE)
    ax.set_ylabel(r"$R_{\mathrm{merger}}$ (Gyr$^{-1}$)")
    ax.set_yscale("log")
    _decimal_log_yaxis(ax)
    ax.set_xlim(-0.5, 1.5)

    savefig(fig, "fig_merger_rate_vs_lotz.pdf")
    return ax


def plot_model_progression_bar(names, val_rmse):
    """Bar chart of best validation RMSE across the model progression.

    Parameters
    ----------
    names : sequence of str
        Model labels in progression order (e.g. baseline -> custom -> resnet -> ...).
    val_rmse : array-like
        Best validation RMSE per model.
    """
    fig, ax = textwidth_figure(3)
    x = np.arange(len(names))
    ax.bar(x, val_rmse, color="C2", alpha=ALPHA_STANDARD)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=LEGEND_SIZE)
    ax.set_ylabel("Best validation RMSE")
    ax.set_title("Model progression", loc="left", fontsize=LABEL_SIZE)
    for xi, v in zip(x, val_rmse):
        ax.text(xi, v, f"{v:.4f}", ha="center", va="bottom",
                fontsize=LEGEND_SIZE - 1)
    ax.set_ylim(0, max(val_rmse) * 1.15)

    savefig(fig, "fig_model_progression.pdf")
    return ax
