"""Runtime image augmentation stages for CNN training.

The class wrappers (``CenterCrop``, ``RandomRotation360``) operate on a
single ``(H, W, 3)`` uint8 image and follow the
:class:`~ugdatalab.utils.compose.Compose` ``T -> T`` contract, so they
can be combined with each other or with the per-image preprocessing
stages in :mod:`ugdatalab.models.galaxy_zoo.images_pipeline`.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Per-image helpers
# ---------------------------------------------------------------------------


def _center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Slice the central ``(size, size, 3)`` region of an (H, W, 3) image."""
    h, w = image.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    return image[top:top + size, left:left + size]


@dataclass
class CenterCrop:
    """Center-crop an image to a square of ``size`` × ``size`` pixels.

    Operates on numpy arrays in HWC format. No interpolation — slices
    out the central region. Raises if ``size`` exceeds either input
    dimension.

    Parameters
    ----------
    size : int
        Output side length in pixels.
    """
    size: int

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return the central ``size`` × ``size`` crop of *image*."""
        return _center_crop(image, self.size)


def _random_rotate(image: np.ndarray) -> np.ndarray:
    """Rotate an (H, W, 3) uint8 image by a uniform random angle in [0, 360)."""
    angle = np.random.uniform(0, 360)
    img = Image.fromarray(image)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    return np.asarray(img)


@dataclass
class RandomRotation360:
    """Rotate an image by a uniformly random angle in ``[0, 360)`` degrees.

    Operates on numpy arrays in HWC format, ``uint8`` with values in
    ``[0, 255]``. Each call samples a fresh angle and uses bilinear
    interpolation (matching :class:`GpuRandomRotation360`), keeping the
    original output size. Black corners may appear; pair with a larger
    source image and a downstream ``CenterCrop`` to avoid them.
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return *image* rotated by a uniformly random angle in [0, 360) degrees."""
        return _random_rotate(image)
