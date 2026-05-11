"""CNN training infrastructure for multi-label image classification."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CNNResult:
    """Result of training a CNN model.

    Attributes
    ----------
    train_losses : ndarray, shape (n_epochs,)
        Per-epoch training RMSE.
    val_losses : ndarray, shape (n_epochs,)
        Per-epoch validation RMSE.
    best_epoch : int
        Epoch (0-indexed) with lowest validation loss.
    best_val_loss : float
        Lowest validation RMSE achieved.
    model_state : dict
        ``state_dict`` of the model at ``best_epoch``.
    n_parameters : int
        Total trainable parameters.
    learning_rates : ndarray, shape (n_epochs,)
        Learning rate used at each epoch (for scheduler diagnostics).
    """
    train_losses: np.ndarray
    val_losses: np.ndarray
    best_epoch: int
    best_val_loss: float
    model_state: dict
    n_parameters: int
    learning_rates: np.ndarray


# ---------------------------------------------------------------------------
# Public utilities (loss, parameter count, baseline)
# ---------------------------------------------------------------------------


def rmse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute differentiable RMSE loss across all ``(N, K)`` entries.

    ::

        L_RMSE = sqrt( (1 / (N * K)) * sum_{i, j} (target - prediction)**2 )

    Parameters
    ----------
    predictions : Tensor, shape (N, K)
    targets : Tensor, shape (N, K)

    Returns
    -------
    Tensor
        Scalar tensor.
    """
    return torch.sqrt(torch.mean((predictions - targets) ** 2))


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def baseline_rmse(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
) -> tuple[float, float]:
    """Return RMSE of the constant-mean baseline on train and validation.

    The predictor is the per-label mean of ``train_labels`` applied to
    every sample.

    Parameters
    ----------
    train_labels : ndarray, shape (N_train, K)
    val_labels : ndarray, shape (N_val, K)

    Returns
    -------
    train_rmse : float
    val_rmse : float
    """
    mean_pred = np.mean(train_labels, axis=0)  # (K,)
    train_rmse = float(np.sqrt(np.mean((train_labels - mean_pred) ** 2)))
    val_rmse = float(np.sqrt(np.mean((val_labels - mean_pred) ** 2)))
    return train_rmse, val_rmse


# ---------------------------------------------------------------------------
# Private training loop helper
# ---------------------------------------------------------------------------


def _auto_device() -> str:
    """Return ``"cuda"`` if a CUDA device is available, else ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _run_epoch(model, loader, optimizer, device, training: bool) -> float:
    """Run one epoch of training or evaluation and return the epoch RMSE."""
    model.train() if training else model.eval()
    total_se = torch.zeros((), device=device)
    total_n = 0

    context = torch.no_grad() if not training else torch.enable_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(images)
            loss = rmse_loss(predictions, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_n = images.size(0) * labels.size(1)
            total_se = total_se + torch.sum((predictions - labels) ** 2).detach()
            total_n += batch_n

    return float(np.sqrt(total_se.item() / total_n))


# ---------------------------------------------------------------------------
# Public training and inference
# ---------------------------------------------------------------------------


def train_cnn(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    n_epochs: int,
    lr: float,
    seed: int,
    optimizer_factory=lambda params, lr: torch.optim.SGD(params, lr=lr),
    scheduler_factory=None,
    num_workers: int = 16,
) -> CNNResult:
    """Train a CNN model on a classification dataset.

    Builds DataLoaders, an Adam optimizer, and runs the training loop,
    checkpointing the model state at the epoch with the lowest
    validation RMSE.

    Parameters
    ----------
    model : nn.Module
        Model to train (e.g. from ``build_resnet18`` or ``build_custom_cnn``).
    train_dataset : Dataset
        Training data of ``(image, label)`` pairs.
    val_dataset : Dataset
        Validation data.
    batch_size : int
        Mini-batch size for both training and validation.
    n_epochs : int
        Number of training epochs.
    lr : float
        Initial learning rate passed to the optimizer factory.
    seed : int
        Random seed for reproducibility.
    optimizer_factory : callable, optional
        Called as ``optimizer_factory(model.parameters(), lr)`` to
        construct the optimizer. Defaults to a factory returning
        vanilla ``torch.optim.SGD`` — the foundational gradient-descent
        rule. Override with Adam / AdamW for adaptive per-parameter
        step sizes (typically faster convergence on deep CNNs), or
        with SGD+momentum for the classical ResNet recipe.
    scheduler_factory : callable, optional
        Called as ``scheduler_factory(optimizer)`` to create a learning
        rate scheduler whose ``step`` is invoked with the validation
        loss after each epoch.
    num_workers : int, optional
        DataLoader worker count. Default 16.

    Returns
    -------
    CNNResult
    """
    device = _auto_device()
    torch.manual_seed(seed)
    # Why: cudnn benchmark autotunes conv kernels for the given input shape;
    # huge speedup for ResNet on fixed-size inputs. We trade bit-exact
    # reproducibility for throughput — seed still controls weight init,
    # shuffling, and dropout, so results are statistically equivalent.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Use a seeded generator for DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g, pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device != "cpu"),
    )

    model = model.to(device)
    optimizer = optimizer_factory(model.parameters(), lr)

    scheduler = None
    if scheduler_factory is not None:
        scheduler = scheduler_factory(optimizer)

    n_params = count_parameters(model)
    train_losses = np.empty(n_epochs)
    val_losses = np.empty(n_epochs)
    learning_rates = np.empty(n_epochs)
    best_state = None
    best_val = float("inf")
    best_ep = 0

    for epoch in tqdm(range(n_epochs), desc="Training"):
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates[epoch] = current_lr

        train_losses[epoch] = _run_epoch(
            model, train_loader, optimizer, device, training=True,
        )
        val_losses[epoch] = _run_epoch(
            model, val_loader, None, device, training=False,
        )

        if scheduler is not None:
            scheduler.step(val_losses[epoch])

        if val_losses[epoch] < best_val:
            best_val = val_losses[epoch]
            best_ep = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return CNNResult(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_ep,
        best_val_loss=best_val,
        model_state=best_state,
        n_parameters=n_params,
        learning_rates=learning_rates,
    )


def predict_cnn(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
) -> np.ndarray:
    """Run inference on *dataset* and return concatenated predictions.

    Parameters
    ----------
    model : nn.Module
        Trained model already loaded with the desired ``state_dict``.
    dataset : Dataset
        Dataset to predict on.
    batch_size : int
        Inference batch size.

    Returns
    -------
    ndarray, shape (N, n_labels)
    """
    device = _auto_device()
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            pred = model(images)
            predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions, axis=0)
