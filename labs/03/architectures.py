"""CNN architecture factories for image classification.

Builders return standard ``torch.nn.Module`` objects ready for use
with ``train_cnn``.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

from ugdatalab.models.galaxy_zoo.constants import (
    N_LABELS,
    LABEL_COLUMNS,
    LABEL_TREE,
)


# ---------------------------------------------------------------------------
# ResNet-18 factory
# ---------------------------------------------------------------------------


def build_resnet18(n_labels: int) -> nn.Module:
    """Build a ResNet-18 for multi-label classification.

    Uses the standard torchvision ResNet-18 stem (7x7 stride-2 conv +
    3x3 stride-2 max-pool), which downsamples 96x96 input to 24x24
    before stage 1 — keeping stage-1 activations and FLOPs comparable
    to the textbook 224-input ResNet rather than the 16x more expensive
    stride-1 variant. The final FC is replaced with
    ``Linear(..., n_labels) -> Sigmoid`` for ``[0, 1]`` multi-label
    output.

    Parameters
    ----------
    n_labels : int
        Number of output labels.

    Returns
    -------
    nn.Module
    """
    model = resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, n_labels),
        nn.Sigmoid(),
    )

    return model


# ---------------------------------------------------------------------------
# Custom CNN
# ---------------------------------------------------------------------------


class _CustomCNN(nn.Module):
    """Custom CNN with configurable conv/FC layers, pooling, and dropout.

    Architecture::

        [Conv2d -> BatchNorm2d -> ReLU -> Pool] x N_conv
            -> Flatten
            -> [Linear -> ReLU -> Dropout] x N_fc
            -> Linear -> Sigmoid
    """

    def __init__(
        self,
        n_labels: int,
        n_channels_list: list[int],
        kernel_sizes: list[int],
        fc_sizes: list[int],
        dropout_rate: float | list[float],
        pool_type: str,
        input_size: int,
    ):
        """Build the conv-pool stack and FC classifier head from the layer specs."""
        super().__init__()

        pool_cls = nn.MaxPool2d if pool_type == "max" else nn.AvgPool2d

        # Build conv blocks
        conv_layers = []
        in_channels = 3
        spatial = input_size
        for out_channels, ks in zip(n_channels_list, kernel_sizes):
            padding = ks // 2
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=ks,
                          padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                pool_cls(kernel_size=2, stride=2),
            ])
            in_channels = out_channels
            spatial = spatial // 2

        self.features = nn.Sequential(*conv_layers)

        # Compute flattened feature size
        flat_size = in_channels * spatial * spatial

        # Resolve per-FC dropout rates: scalar broadcasts to every layer; list
        # gives one rate per layer and must match ``fc_sizes`` in length.
        if isinstance(dropout_rate, (int, float)):
            dropout_rates = [float(dropout_rate)] * len(fc_sizes)
        else:
            dropout_rates = [float(p) for p in dropout_rate]
            if len(dropout_rates) != len(fc_sizes):
                raise ValueError(
                    f"dropout_rate has {len(dropout_rates)} entries but "
                    f"fc_sizes has {len(fc_sizes)}"
                )

        # Build FC blocks
        fc_layers = []
        in_features = flat_size
        for out_features, p in zip(fc_sizes, dropout_rates):
            fc_layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
            ])
            in_features = out_features

        fc_layers.append(nn.Linear(in_features, n_labels))
        fc_layers.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the convolutional features and dense classifier on input ``x``."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_custom_cnn(
    n_labels: int,
    n_channels_list: list[int],
    kernel_sizes: list[int],
    fc_sizes: list[int],
    dropout_rate: float | list[float],
    pool_type: str,
    input_size: int,
) -> nn.Module:
    """Build a custom CNN for multi-label image classification.

    Parameters
    ----------
    n_labels : int
        Number of output labels.
    n_channels_list : list of int
        Output channels for each conv layer (e.g. [32, 64, 128]).
    kernel_sizes : list of int
        Kernel size for each conv layer (e.g. [5, 3, 3]).
        Must have same length as ``n_channels_list``.
    fc_sizes : list of int
        Hidden units for each fully connected layer (e.g. [256, 128]).
    dropout_rate : float or list of float
        Dropout probability for the FC layers. A scalar broadcasts to every
        FC layer (uniform dropout); a list assigns one rate per FC layer
        and must match ``fc_sizes`` in length (e.g. ``[0.25, 0.1]`` for
        tapered dropout on a two-layer head).
    pool_type : str
        ``"max"`` or ``"avg"`` — pooling layer type after each conv.
    input_size : int
        Spatial dimension of input images.

    Returns
    -------
    nn.Module
        Custom CNN with sigmoid output.
    """
    return _CustomCNN(
        n_labels=n_labels,
        n_channels_list=n_channels_list,
        kernel_sizes=kernel_sizes,
        fc_sizes=fc_sizes,
        dropout_rate=dropout_rate,
        pool_type=pool_type,
        input_size=input_size,
    )


# ---------------------------------------------------------------------------
# Tree-reweighted output wrapper (Task 22, option (a))
# ---------------------------------------------------------------------------


def _build_parent_indices():
    """Return per-label parent index lists derived from LABEL_TREE."""
    name_to_idx = {name: i for i, name in enumerate(LABEL_COLUMNS)}
    parents = [[] for _ in range(N_LABELS)]
    for parent_name, children in LABEL_TREE.items():
        p_idx = name_to_idx[parent_name]
        for child in children:
            parents[name_to_idx[child]].append(p_idx)
    return parents


def _topological_order(parents, n_labels):
    """Return label indices in an order where every parent precedes its children."""
    order = []
    placed = set()
    while len(order) < n_labels:
        progress = False
        for j in range(n_labels):
            if j in placed:
                continue
            if all(p in placed for p in parents[j]):
                order.append(j)
                placed.add(j)
                progress = True
        if not progress:
            raise ValueError("cycle detected in parent graph")
    return order


PARENT_INDICES = _build_parent_indices()
TOPO_ORDER = _topological_order(PARENT_INDICES, N_LABELS)


def gz2_tree_reweight(sig, topo=TOPO_ORDER, parents=PARENT_INDICES):
    """Convert per-label conditional sigmoid outputs to GZ2 vote-fraction predictions.

    For each label j with parent set P(j):
        f_j = sig_j                              if P(j) is empty (root)
        f_j = sig_j * sum_{p in P(j)} f_p        otherwise.

    Parameters
    ----------
    sig : torch.Tensor
        Post-sigmoid tensor of shape ``(B, N_LABELS)`` whose entries are
        interpreted as conditional probabilities of each GZ2 answer
        given that its parent question was reached.
    topo : list of int
        Topological order over labels (roots first, children after).
    parents : list of list of int
        Per-label parent index lists.

    Returns
    -------
    torch.Tensor
        Tree-consistent vote-fraction predictions of shape ``(B, N_LABELS)``.
    """
    out = [None] * sig.shape[1]
    for j in topo:
        if not parents[j]:
            out[j] = sig[:, j]
        else:
            parent_sum = out[parents[j][0]]
            for p in parents[j][1:]:
                parent_sum = parent_sum + out[p]
            out[j] = sig[:, j] * parent_sum
    return torch.stack(out, dim=1)


class TreeReweightedCNN(nn.Module):
    """Wrap a CNN so its outputs are tree-consistent GZ2 vote fractions.

    Parameters
    ----------
    base : nn.Module
        Base CNN that emits a 37-vector in ``[0, 1]`` (e.g. a Custom CNN
        with sigmoid head). Its outputs are re-interpreted as conditional
        probabilities and passed through :func:`gz2_tree_reweight`.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the base CNN and re-weight its sigmoid outputs along the GZ2 tree."""
        return gz2_tree_reweight(torch.sigmoid(self.base(x)))
