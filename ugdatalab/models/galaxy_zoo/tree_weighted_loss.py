"""Tree-weighted RMSE loss for the GZ2 hierarchical-label CNN.

Implements the question-presentation-weighted MSE used by the Galaxy Zoo
Challenge winners (Dieleman, Willett & Dambre 2015, MNRAS 450, 1441),
adapted to the differentiable RMSE form used by the surrounding training
loop. The key idea: weight each (galaxy, label) cell by the cumulative
product of its ancestor vote fractions in the GZ2 decision tree --- i.e.
by the probability that the question was asked of that galaxy under the
GZ2 protocol. Labels on branches the classifier never reached contribute
~0 to the loss; labels on branches that were asked contribute their full
squared error. The network still has an unconstrained 37-output sigmoid
head, so it can predict any value in [0, 1] for any label; tree-consistency
is enforced through gradient pressure rather than by clipping the output
space.
"""

from typing import Callable, Sequence

import torch


def compute_tree_weights(
    labels: torch.Tensor,
    parent_indices: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Compute per-(galaxy, label) question-presentation weights.

    For each label $j$ with parent set $P(j)$, the weight is

        w_j = 1                           if P(j) is empty (root)
        w_j = sum_{p in P(j)} f_p         otherwise,

    where $f_p$ are the corresponding parents' \emph{catalog} vote
    fractions. The parent sum is needed only for Class4.1/4.2 in the
    GZ2 tree (whose two parents Class3.1 and Class3.2 are mutually
    exclusive sibling answers to Q3 and sum by construction to the
    grandparent Class2.2 vote fraction); for all other labels
    $|P(j)| = 1$ and $w_j$ reduces to the parent's vote fraction
    directly.

    Note that GZ2's parent-normalisation already enforces
    $\\sum_{c \\in \\mathrm{siblings}(j)} f_c = f_{\\mathrm{parent}}$,
    so the catalog $f_p$ values are already cumulative joint
    probabilities of \emph{reaching} that node in the elicitation
    tree --- no recursive multiplication along the chain is needed.

    Parameters
    ----------
    labels : torch.Tensor
        Ground-truth vote fractions of shape ``(B, N_labels)``. The
        weights are derived from these targets, not from the network's
        predictions --- so each batch's weight tensor reflects which
        questions were actually asked of each galaxy by GZ2 classifiers.
    parent_indices : sequence of sequence of int
        ``parent_indices[j]`` is the list of parent label indices for
        label ``j`` in the GZ2 tree (empty for roots).

    Returns
    -------
    torch.Tensor
        Weights of shape ``(B, N_labels)`` in [0, 1]. Roots have weight
        1, deep-tree leaves on branches with low parent vote have weight
        near 0.
    """
    weights = torch.ones_like(labels)
    for j, parents in enumerate(parent_indices):
        if not parents:
            continue
        # Catalog parent-normalisation already makes f_p the cumulative
        # joint probability of reaching parent p, so we sum without
        # extra multiplication along the chain.
        weights[:, j] = sum(labels[:, p] for p in parents)
    return weights


def tree_weighted_rmse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    parent_indices: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Question-presentation-weighted differentiable RMSE.

    Computes

        L = sqrt( sum_{i, j} w_{ij} (pred - target)^2 / sum_{i, j} w_{ij} )

    where ``w_{ij}`` are the ancestor-product weights from
    :func:`compute_tree_weights`. Equivalent to a per-cell-weighted MSE
    followed by a square root; the normalisation uses the *sum of
    weights* rather than the cell count, so that branches with low
    average parent vote do not artificially deflate the reported loss.

    Parameters
    ----------
    predictions, targets : torch.Tensor
        Shape ``(B, N_labels)``.
    parent_indices : sequence of sequence of int
        Per-label parent index lists for the GZ2 tree.

    Returns
    -------
    torch.Tensor
        Scalar tensor.
    """
    weights = compute_tree_weights(targets, parent_indices)
    sq_err = (predictions - targets) ** 2
    weighted_mse = (weights * sq_err).sum() / weights.sum().clamp(min=1e-8)
    return torch.sqrt(weighted_mse)


def make_tree_weighted_rmse_loss(
    parent_indices: Sequence[Sequence[int]],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Factory returning a closure compatible with :func:`train_cnn`'s ``loss_fn``.

    The returned closure has signature ``loss_fn(predictions, targets)``,
    matching the signature expected by
    :func:`ugdatalab.methods.neural_network.cnn.train_cnn`, with the GZ2
    tree structure baked in.

    Parameters
    ----------
    parent_indices : sequence of sequence of int
        Per-label parent index lists for the GZ2 tree.

    Returns
    -------
    callable
        Closure ``loss_fn(predictions, targets)`` returning a scalar
        differentiable tensor.
    """
    def loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute tree-weighted RMSE on a (pred, target) batch."""
        return tree_weighted_rmse_loss(predictions, targets, parent_indices)
    return loss_fn
