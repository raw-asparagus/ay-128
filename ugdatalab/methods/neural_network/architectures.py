"""CNN architecture factories for image classification.

Builders return standard ``torch.nn.Module`` objects ready for use
with ``train_cnn``.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18 as _resnet18_factory


def build_resnet18(n_labels: int, input_size: int) -> nn.Module:
    """Build a modified ResNet-18 for multi-label classification.

    The final FC is replaced with ``Linear(..., n_labels) -> Sigmoid``
    for ``[0, 1]`` multi-label output. For inputs smaller than 64 px,
    the initial 7x7 conv is replaced with a stride-1 3x3 conv and the
    initial max-pool is removed.

    Parameters
    ----------
    n_labels : int
        Number of output labels.
    input_size : int
        Spatial dimension of input images (``input_size x input_size``).

    Returns
    -------
    nn.Module
    """
    model = _resnet18_factory(weights=None)

    # Adapt first conv for small images
    if input_size < 64:
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False,
        )
        model.maxpool = nn.Identity()

    # Replace final FC + add sigmoid for [0, 1] output
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, n_labels),
        nn.Sigmoid(),
    )

    return model


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
        dropout_rate: float,
        pool_type: str,
        input_size: int,
    ):
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
                          padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                pool_cls(kernel_size=2, stride=2),
            ])
            in_channels = out_channels
            spatial = spatial // 2

        self.features = nn.Sequential(*conv_layers)

        # Compute flattened feature size
        flat_size = in_channels * spatial * spatial

        # Build FC blocks
        fc_layers = []
        in_features = flat_size
        for out_features in fc_sizes:
            fc_layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ])
            in_features = out_features

        fc_layers.append(nn.Linear(in_features, n_labels))
        fc_layers.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_custom_cnn(
    n_labels: int,
    n_channels_list: list[int],
    kernel_sizes: list[int],
    fc_sizes: list[int],
    dropout_rate: float,
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
    dropout_rate : float
        Dropout probability in FC layers (e.g. 0.5).
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
