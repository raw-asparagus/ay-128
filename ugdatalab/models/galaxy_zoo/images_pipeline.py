"""Galaxy Zoo per-image preprocessing helpers and ``Compose``-style stages.

The class wrappers (``CropFraction``, ``Resize``) operate on a single
``(H, W, 3)`` float32 image and follow the
:class:`~ugdatalab.utils.compose.Compose` ``T -> T`` contract, so they
can be combined with each other or with the runtime augmentation stages
in :mod:`ugdatalab.methods.neural_network.augmentation`.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from ugdatalab.models.galaxy_zoo.constants import _UINT8_MAX


# ---------------------------------------------------------------------------
# Per-image helpers
# ---------------------------------------------------------------------------


def _crop_center(image: np.ndarray, crop_fraction: float) -> np.ndarray:
    """Center-crop an (H, W, 3) image by removing ``crop_fraction`` of each side."""
    h, w = image.shape[:2]
    dy = int(h * crop_fraction)
    dx = int(w * crop_fraction)
    return image[dy:h - dy, dx:w - dx]


@dataclass
class CropFraction:
    """Center-crop an image by removing ``crop_fraction`` of each border.

    Parameters
    ----------
    crop_fraction : float
        Fraction of each side to discard. Default ``0.25`` — keeps the
        central 50% of the image; the convention used across lab 03's
        preprocessing pipeline.
    """
    crop_fraction: float = 0.25

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return a center-cropped (H', W', 3) view of *image*."""
        return _crop_center(image, self.crop_fraction)


def _resize(image: np.ndarray, target_size: int) -> np.ndarray:
    """Lanczos-resize an (H, W, 3) float32 image to ``(target_size, target_size, 3)``."""
    img = Image.fromarray((image * _UINT8_MAX).astype(np.uint8))
    img = img.resize((target_size, target_size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / _UINT8_MAX


@dataclass
class Resize:
    """Lanczos-resize an image to a square ``target_size`` × ``target_size``.

    Parameters
    ----------
    target_size : int
        Output side length in pixels.
    """
    target_size: int

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return a Lanczos-resized (target_size, target_size, 3) image."""
        return _resize(image, self.target_size)
