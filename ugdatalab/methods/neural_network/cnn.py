"""CNN training infrastructure for multi-label image classification."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


Batch = tuple[torch.Tensor, torch.Tensor]


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


def _run_epoch(model, batches, optimizer, device, training: bool) -> float:
    """Run one epoch of training or evaluation and return the epoch RMSE."""
    model.train() if training else model.eval()
    total_se = torch.zeros((), device=device)
    total_n = 0

    is_cuda = device == "cuda"
    grad_context = torch.enable_grad() if training else torch.no_grad()

    with grad_context:
        for images, labels in batches:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=is_cuda):
                predictions = model(images)
                loss = rmse_loss(predictions, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_n = images.size(0) * labels.size(1)
            total_se = total_se + torch.sum(
                (predictions.detach().float() - labels) ** 2,
            )
            total_n += batch_n

    return float(np.sqrt(total_se.item() / total_n))


# ---------------------------------------------------------------------------
# Public training and inference
# ---------------------------------------------------------------------------


def train_cnn(
    model: nn.Module,
    train_batches: Iterable[Batch],
    val_batches: Iterable[Batch],
    n_epochs: int,
    lr: float,
    seed: int,
    optimizer_factory=lambda params, lr: torch.optim.SGD(params, lr=lr),
    scheduler_factory=None,
) -> CNNResult:
    """Train a CNN model on a classification dataset.

    Iterates the supplied batch sources once per epoch, checkpointing
    the model state at the epoch with the lowest validation RMSE. The
    caller owns batching: ``train_batches`` and ``val_batches`` are any
    reusable iterables that yield ``(image_tensor, label_tensor)`` pairs
    — e.g. a :class:`~ugdatalab.models.galaxy_zoo.dataset_gpu.GalaxyZooGPUDataset`,
    or a ``torch.utils.data.DataLoader`` wrapping a map-style dataset.

    Parameters
    ----------
    model : nn.Module
        Model to train (e.g. from ``build_resnet18`` or ``build_custom_cnn``).
    train_batches : Iterable[(Tensor, Tensor)]
        Reusable iterable of training batches; iterated fresh each
        epoch.
    val_batches : Iterable[(Tensor, Tensor)]
        Reusable iterable of validation batches; iterated fresh each
        epoch.
    n_epochs : int
        Number of training epochs.
    lr : float
        Initial learning rate passed to the optimizer factory.
    seed : int
        Random seed for reproducibility of model init / dropout. Batch
        ordering is controlled by whatever generator the batch source
        was constructed with.
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

    model = model.to(device)
    optimizer = optimizer_factory(model.parameters(), lr)

    # torch.compile captures forward+backward as a graph and emits fused
    # kernels. The original `model` is the parameter owner and the source of
    # truth for state_dict; `compiled_model` shares parameters and is what
    # we call in the training loop. Saving from `model` keeps checkpoint
    # keys free of the `_orig_mod.` prefix that the compiled wrapper adds.
    compiled_model = torch.compile(model) if device == "cuda" else model

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
            compiled_model, train_batches, optimizer, device, training=True,
        )
        val_losses[epoch] = _run_epoch(
            compiled_model, val_batches, None, device, training=False,
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
    batches: Iterable[Batch],
) -> np.ndarray:
    """Run inference on *batches* and return concatenated predictions.

    Parameters
    ----------
    model : nn.Module
        Trained model already loaded with the desired ``state_dict``.
    batches : Iterable[(Tensor, Tensor)]
        Reusable iterable of ``(image_batch, label_batch)`` pairs. The
        label tensor is ignored; pass dummies if labels are unavailable.

    Returns
    -------
    ndarray, shape (N, n_labels)
    """
    device = _auto_device()
    model = model.to(device)
    model.eval()

    is_cuda = device == "cuda"
    predictions = []
    with torch.no_grad(), torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=is_cuda,
    ):
        for images, _ in batches:
            images = images.to(device, non_blocking=True)
            pred = model(images)
            predictions.append(pred.float().cpu().numpy())

    return np.concatenate(predictions, axis=0)
