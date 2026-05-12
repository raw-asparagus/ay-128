"""GPU-resident image augmentation stages for CNN training.

The class wrappers (``GpuCenterCrop``, ``GpuRandomRotation360``) operate
on a single ``(B, C, H, W)`` float tensor on any device and follow the
:class:`~ugdatalab.utils.compose.Compose` ``T -> T`` contract, so they
compose with each other identically to the CPU transforms in
:mod:`ugdatalab.methods.neural_network.augmentation`.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Per-batch helpers
# ---------------------------------------------------------------------------


def _center_crop_gpu(x: torch.Tensor, size: int) -> torch.Tensor:
    """Slice the central ``(size, size)`` region of a (B, C, H, W) tensor."""
    _, _, h, w = x.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return x[:, :, top:top + size, left:left + size]


@dataclass
class GpuCenterCrop:
    """Center-crop a (B, C, H, W) tensor to a square of ``size`` × ``size`` pixels.

    Operates on torch tensors on any device. No interpolation — slices
    out the central region.

    Parameters
    ----------
    size : int
        Output side length in pixels.
    """
    size: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the central ``size`` × ``size`` crop of *x* along H and W."""
        return _center_crop_gpu(x, self.size)


def _random_rotate_gpu(x: torch.Tensor) -> torch.Tensor:
    """Rotate each image in a (B, C, H, W) tensor by a uniform random angle in [0, 360)."""
    batch_size = x.size(0)
    angles = torch.empty(batch_size, device=x.device).uniform_(0, 2 * math.pi)
    cos, sin = angles.cos(), angles.sin()
    theta = x.new_zeros(batch_size, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(
        x, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


class GpuRandomRotation360:
    """Rotate each image in a batch by an independent uniform angle in ``[0, 360)`` degrees.

    Operates on (B, C, H, W) float tensors on any device. Uses
    ``F.affine_grid`` + ``F.grid_sample`` with bilinear interpolation,
    matching :class:`~ugdatalab.methods.neural_network.augmentation.RandomRotation360`.
    Output size equals input size; black corners may appear and should
    be removed with a downstream ``GpuCenterCrop``.

    Angles are drawn from the current default CUDA generator; seed via
    ``torch.manual_seed`` for reproducibility.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return *x* rotated per-sample by uniform random angles, bilinear."""
        return _random_rotate_gpu(x)
