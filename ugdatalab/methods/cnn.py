"""CNN training infrastructure for multi-label image classification.

Provides survey-agnostic engine functions following the frozen-result
pattern established by ``cannon.py`` and ``bayesian/mcmc.py``.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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


def rmse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Differentiable root mean squared error loss.

    Implements Equation (3) from the lab manual:

        L_RMSE = sqrt( (1 / (N * K)) * sum_{i,j} (l_true - l_pred)^2 )

    Parameters
    ----------
    predictions : Tensor, shape (N, K)
    targets : Tensor, shape (N, K)

    Returns
    -------
    Tensor, scalar
    """
    return torch.sqrt(torch.mean((predictions - targets) ** 2))


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def baseline_rmse(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
) -> tuple[float, float]:
    """RMSE of the mean-prediction baseline model.

    Predicts the training-set mean for every image. Returns RMSE
    evaluated on both the training and validation sets.

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


def _run_epoch(model, loader, optimizer, device, training: bool) -> float:
    """Run one epoch of training or evaluation.

    Returns the epoch RMSE.
    """
    model.train() if training else model.eval()
    total_se = 0.0
    total_n = 0

    context = torch.no_grad() if not training else torch.enable_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = rmse_loss(predictions, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_n = images.size(0) * labels.size(1)
            total_se += torch.sum((predictions - labels) ** 2).item()
            total_n += batch_n

    return float(np.sqrt(total_se / total_n))


def train_cnn(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    n_epochs: int,
    lr: float,
    device: str,
    seed: int,
    scheduler_factory=None,
    num_workers: int = 0,
) -> CNNResult:
    """Train a CNN model on a classification dataset.

    Creates DataLoaders, an Adam optimizer, and runs the training loop.
    Checkpoints the model state at the epoch with the lowest validation
    RMSE.

    Parameters
    ----------
    model : nn.Module
        Model to train (e.g. from ``build_resnet18`` or ``build_custom_cnn``).
    train_dataset : Dataset
        Training data (images + labels).
    val_dataset : Dataset
        Validation data.
    batch_size : int
        Mini-batch size for both training and validation.
    n_epochs : int
        Number of training epochs.
    lr : float
        Initial learning rate for the Adam optimizer.
    device : str
        PyTorch device string (``"cuda"`` or ``"cpu"``).
    seed : int
        Random seed for reproducibility.
    scheduler_factory : callable or None
        If not None, called as ``scheduler_factory(optimizer)`` to create
        a learning rate scheduler. The scheduler's ``step`` method is
        called with the validation loss after each epoch. Use e.g.
        ``lambda opt: ReduceLROnPlateau(opt, factor=0.5, patience=3)``.
    num_workers : int
        DataLoader worker count. Default 0 (main-process loading) is the
        safest choice; increase to 2-4 if I/O is the bottleneck.

    Returns
    -------
    CNNResult
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    from tqdm.auto import tqdm

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
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Run inference on a dataset and return predictions.

    Parameters
    ----------
    model : nn.Module
        Trained model (already loaded with the desired state_dict).
    dataset : Dataset
        Dataset to predict on.
    device : str
        PyTorch device string.
    batch_size : int
        Inference batch size.

    Returns
    -------
    ndarray, shape (N, n_labels)
    """
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
