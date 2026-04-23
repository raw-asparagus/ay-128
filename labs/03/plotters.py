"""Lab 03 — Galaxy image classification plotters.

All functions follow the ugdatalab convention: no defaults on data
arguments, style constants from ``ugdatalab.plotting``, single
``savefig(fig, name)`` call, return axes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ugdatalab.plotting import (
    LW_FINE,
    LW_LIGHT,
    LW_STANDARD,
    LW_MEDIUM,
    LW_NONE,
    SS_MICRO,
    SS_FINE,
    ALPHA_EXTRA_LIGHT,
    ALPHA_FAINT,
    ALPHA_LIGHT,
    ALPHA_STANDARD,
    NEUTRAL_COLOR,
    FILL_STYLE,
    GUIDE_STYLE,
    FIT_STYLE,
    MODEL_STYLE,
    SCATTER_STYLE,
    LABEL_SIZE,
    LEGEND_SIZE,
    textwidth_figure,
    columnwidth_figure,
    landscapewidth_figure,
    subpanels,
    zero_line,
    unity_line,
)

_FIGURES_DIR = Path(__file__).parent / "report" / "figures"


def savefig(fig, name):
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES_DIR / name)


# ---------------------------------------------------------------------------
# 1. Random sample images (Task 5)
# ---------------------------------------------------------------------------

def plot_random_images(images, galaxy_ids, n_rows, n_cols, seed):
    """Grid of random galaxy images from the training set.

    Parameters
    ----------
    images : ndarray, shape (N, H, W, 3)
        Image array in [0, 1].
    galaxy_ids : ndarray, shape (N,)
    n_rows, n_cols : int
        Grid layout.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n = n_rows * n_cols
    idx = rng.choice(len(images), size=n, replace=False)

    fig, _ = textwidth_figure(2 * n_rows)
    _.remove()
    axes = fig.subplots(n_rows, n_cols)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[idx[i]])
        ax.set_title(str(galaxy_ids[idx[i]]), fontsize=LEGEND_SIZE)
        ax.axis("off")

    fig.subplots_adjust(hspace=0.42)
    savefig(fig, "fig_random_images.pdf")
    return axes


# ---------------------------------------------------------------------------
# 2. Label distributions (Task 6)
# ---------------------------------------------------------------------------

def plot_label_distributions(labels, label_names, label_descriptive):
    """Normalized histograms of all 37 classification labels.

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
    axes = fig.subplots(nrows, ncols)

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < n_labels:
            ax.hist(labels[:, i], bins=50, density=True, color="C0",
                    alpha=ALPHA_STANDARD, lw=LW_NONE)
            desc = label_descriptive.get(label_names[i], label_names[i])
            ax.set_title(desc, fontsize=LEGEND_SIZE, loc="left")
            ax.set_xlim(-0.05, 1.05)
            ax.tick_params(labelsize=LEGEND_SIZE - 1)
        else:
            ax.set_visible(False)

    fig.supxlabel("Label probability", fontsize=LABEL_SIZE)
    fig.supylabel("Density", fontsize=LABEL_SIZE)
    fig.subplots_adjust(hspace=0.7, wspace=0.28)

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
    ncols = 6
    nrows = (n_labels + ncols - 1) // ncols

    fig, _ = landscapewidth_figure(2 * nrows)
    _.remove()
    axes = fig.subplots(nrows, ncols)

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

    fig.subplots_adjust(hspace=0.28)

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
    axes = fig.subplots(n, 2)
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

    fig.subplots_adjust(hspace=0.3, wspace=0.1)

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

    fig, _ = landscapewidth_figure(2.2 * nrows)
    _.remove()
    axes = fig.subplots(nrows, ncols)

    bins = np.linspace(0, 1, 30)
    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < n_labels:
            ax.hist(train_labels[:, i], bins=bins, density=True,
                    alpha=ALPHA_LIGHT, color="C0", lw=LW_NONE,
                    label="Train" if i == 0 else None)
            ax.hist(val_labels[:, i], bins=bins, density=True,
                    alpha=ALPHA_LIGHT, color="C1", lw=LW_NONE,
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
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

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

    fig, ax = columnwidth_figure(3)

    ax.plot(epochs, train_losses, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
            color="C0", label="Training", zorder=3)
    ax.plot(epochs, val_losses, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
            color="C1", label="Validation", zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_yscale("log")
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

    fig, _ = columnwidth_figure(5)
    _.remove()
    ax_loss, ax_lr = subpanels(fig, 2, height_ratios=(3, 1), sharex=True)

    ax_loss.plot(epochs, train_losses, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
                 color="C0", label="Training", zorder=3)
    ax_loss.plot(epochs, val_losses, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
                 color="C1", label="Validation", zorder=3)
    ax_loss.set_ylabel("RMSE")
    ax_loss.set_yscale("log")
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
    fig, ax = columnwidth_figure(3)

    for i, (name, val_losses) in enumerate(zip(names, val_losses_list)):
        epochs = np.arange(1, len(val_losses) + 1)
        ax.plot(epochs, val_losses, lw=LW_STANDARD, alpha=ALPHA_STANDARD,
                color=f"C{i}", label=name, zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.set_yscale("log")
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

    fig, _ = landscapewidth_figure(2.5 * nrows)
    _.remove()
    axes = fig.subplots(nrows, ncols)

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < n_labels:
            t = true_labels[:, i]
            p = pred_labels[:, i]
            resid = p - t
            bias = np.mean(resid)
            scatter_val = np.std(resid)

            ax.scatter(t, p, s=SS_MICRO, color="C0", alpha=ALPHA_FAINT,
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
    fig.subplots_adjust(hspace=0.65, wspace=0.35)

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
    axes = fig.subplots(2, 5)

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
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    safe_name = label_name.lower().replace(" ", "_").replace(":", "").replace("/", "_")
    savefig(fig, f"fig_top5_{safe_name}.pdf")
    return axes
