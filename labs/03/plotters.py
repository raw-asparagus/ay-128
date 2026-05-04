"""Lab 03 — Galaxy image classification plotters.

All functions follow the ugdatalab convention: no defaults on data
arguments, style constants from ``ugdatalab.plotting``, single
``savefig(fig, name)`` call, return axes.
"""

from pathlib import Path

import numpy as np
from matplotlib.ticker import ScalarFormatter


def _decimal_log_yaxis(ax):
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
    NEUTRAL_COLOR,
    GUIDE_STYLE,
    LABEL_SIZE,
    LEGEND_SIZE,
    textwidth_figure,
    subpanels,
)

_FIGURES_DIR = Path(__file__).parent / "report" / "figures"


def savefig(fig, name):
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

    ncols = 7
    nrows = (n_samples + ncols - 1) // ncols

    fig, _ = textwidth_figure(2 * nrows)
    _.remove()
    axes = subpanels(fig, nrows, ncols, hspace=0.42, sharex=False)

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < n_samples:
            ax.imshow(images[idx[i]])
            ax.set_title(f"ID {galaxy_ids[idx[i]]}",
                         fontsize=LEGEND_SIZE - 1, loc="left")
            ax.axis("off")
        else:
            ax.set_visible(False)

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
    ncols = 5
    nrows = (n_labels + ncols - 1) // ncols

    fig, _ = textwidth_figure(2.2 * nrows)
    _.remove()
    axes = subpanels(fig, nrows, ncols, hspace=0.56, wspace=0.28, sharex=True)

    last_row_for_col = [
        max(r for r in range(nrows) if r * ncols + c < n_labels)
        for c in range(ncols)
    ]

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < n_labels:
            class_id = int(label_names[i].split("Class")[1].split(".")[0])
            color = f"C{(class_id - 1) % 10}"
            ax.hist(labels[:, i], bins=50, density=True, color=color,
                    alpha=ALPHA_STANDARD, lw=LW_NONE)
            desc = label_descriptive.get(label_names[i], label_names[i])
            ax.set_title(desc, fontsize=LEGEND_SIZE, loc="left")
            ax.set_xlim(-0.05, 1.05)
            ax.tick_params(labelsize=LEGEND_SIZE - 1)
            ax.tick_params(labelbottom=(row == last_row_for_col[col]))
        else:
            ax.set_visible(False)

    fig.supxlabel("Label probability", fontsize=LABEL_SIZE)
    fig.supylabel("Density", fontsize=LABEL_SIZE)

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
    ncols = 4
    nrows = (n_labels + ncols - 1) // ncols

    fig, _ = textwidth_figure(2 * nrows)
    _.remove()
    axes = subpanels(fig, nrows, ncols, hspace=0.56, sharex=False)

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < n_labels:
            best = np.argmax(labels[:, i])
            ax.imshow(images[best])
            desc = label_descriptive.get(label_names[i], label_names[i])
            ax.set_title(f"{desc}\nID {galaxy_ids[best]}, {labels[best, i]:.2f}",
                         fontsize=LEGEND_SIZE - 1, loc="left")
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

def plot_image_comparison(images_before, images_after, galaxy_ids):
    """Side-by-side original vs resized images.

    Parameters
    ----------
    images_before : list of ndarray, shape (H1, W1, 3)
        Original images.
    images_after : list of ndarray, shape (H2, W2, 3)
        Cropped and resized images.
    galaxy_ids : array-like
        Galaxy IDs for titles.
    """
    n = len(images_before)

    fig, _ = textwidth_figure(3 * n)
    _.remove()
    axes = subpanels(fig, n, 2, hspace=0.3, wspace=0.1, sharex=False)
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        axes[i, 0].imshow(images_before[i])
        axes[i, 0].set_title(f"Original — {galaxy_ids[i]}", fontsize=LEGEND_SIZE)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(images_after[i])
        axes[i, 1].set_title(
            f"Resized ({images_after[i].shape[0]}x{images_after[i].shape[1]})",
            fontsize=LEGEND_SIZE,
        )
        axes[i, 1].axis("off")

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
    ncols = 6
    nrows = (n_labels + ncols - 1) // ncols

    fig, _ = textwidth_figure(2.2 * nrows)
    _.remove()
    axes = subpanels(fig, nrows, ncols, hspace=0.55, wspace=0.35, sharex=False)

    bins = np.linspace(0, 1, 30)
    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < n_labels:
            ax.hist(train_labels[:, i], bins=bins, density=True,
                    histtype="step", color="C0", lw=LW_STANDARD,
                    alpha=ALPHA_STANDARD,
                    label="Train" if i == 0 else None)
            ax.hist(val_labels[:, i], bins=bins, density=True,
                    histtype="step", color="C1", lw=LW_STANDARD,
                    alpha=ALPHA_STANDARD,
                    label="Val" if i == 0 else None)
            desc = label_descriptive.get(label_names[i], label_names[i])
            ax.set_title(desc, fontsize=LEGEND_SIZE - 1, loc="left")
            ax.set_xlim(-0.05, 1.05)
            ax.tick_params(labelsize=LEGEND_SIZE - 1)
        else:
            ax.set_visible(False)

    axes[0, 0].legend(fontsize=LEGEND_SIZE)
    fig.supxlabel("Label probability", fontsize=LABEL_SIZE)
    fig.supylabel("Density", fontsize=LABEL_SIZE)

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


# ---------------------------------------------------------------------------
# 9. Model comparison (Task 23)
# ---------------------------------------------------------------------------

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
        ax.plot(epochs, val_losses, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
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
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(
                f"{label_descriptive_list[i]}\n"
                rf"bias$={bias:+.3f}$, $\sigma={scatter_val:.3f}$",
                fontsize=LEGEND_SIZE - 1, loc="left",
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
    ncols = min(n, 4)
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
        counts, xedges, yedges = np.histogram2d(x, y, bins=60,
                                                 range=[[0, 1], [0, 1]])
        # Map each point to its bin density
        xi = np.clip(np.digitize(x, xedges) - 1, 0, 59)
        yi = np.clip(np.digitize(y, yedges) - 1, 0, 59)
        density = counts[xi, yi]

        order = np.argsort(density)
        ax.scatter(x[order], y[order], c=density[order], cmap="viridis",
                   s=SS_MICRO, alpha=ALPHA_FAINT, lw=LW_NONE,
                   rasterized=True, zorder=3,
                   norm=mcolors.LogNorm(vmin=1, vmax=density.max()))

        rho = np.corrcoef(x, y)[0, 1]
        ax.set_title(rf"$\rho = {rho:+.3f}$", fontsize=LEGEND_SIZE)
        ax.set_xlabel(label_descriptive_list[i], fontsize=LEGEND_SIZE - 1)
        ax.set_ylabel(label_descriptive_list[j], fontsize=LEGEND_SIZE - 1)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=LEGEND_SIZE - 1)

    # Hide unused axes
    for k in range(n, nrows * ncols):
        row, col = divmod(k, ncols)
        axes[row, col].set_visible(False)

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
    colors = ["C0" if n > 1000 else "C3" if n < 200 else "C1" for n in n_eff[order]]
    ax.barh(y_pos, n_eff[order], color=colors, alpha=ALPHA_STANDARD)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label_descriptive_list[i] for i in order],
                       fontsize=LEGEND_SIZE - 1)
    ax.set_xlabel(rf"$N_{{\mathrm{{eff}}}}$ (galaxies with label $> {threshold}$)")
    ax.axvline(1000, **GUIDE_STYLE, label=r"$N_{\mathrm{eff}} = 1000$")
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
        bins = np.linspace(lo, hi, 50)
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

    return axes
