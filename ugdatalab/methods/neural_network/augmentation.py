"""Composable image transforms for CNN training (augmentation and cropping)."""

from collections.abc import Callable

import numpy as np
from PIL import Image


class Compose:
    """Sequentially apply a list of image transforms.

    Parameters
    ----------
    transforms : list of callables
        Each transform takes an HWC ``float32`` image in ``[0, 1]`` and
        returns one of the same dtype/range.
    """
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply each transform in order and return the final image."""
        for t in self.transforms:
            image = t(image)
        return image


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
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return the central ``size`` × ``size`` crop of *image*.

        Parameters
        ----------
        image : ndarray, shape (H, W, 3)

        Returns
        -------
        ndarray, shape (size, size, 3)
        """
        h, w = image.shape[:2]
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        return image[top:top + self.size, left:left + self.size]


class RandomRotation360:
    """Rotate an image by a uniformly random angle in ``[0, 360)`` degrees.

    Operates on numpy arrays in HWC format, ``float32`` with values in
    ``[0, 1]``. Each call samples a fresh angle and uses bilinear
    interpolation, keeping the original output size. Black corners may
    appear; pair with a larger source image and a downstream
    ``CenterCrop`` to avoid them.
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return *image* rotated by a fresh random angle.

        Parameters
        ----------
        image : ndarray, shape (H, W, 3), dtype float32, values in [0, 1]

        Returns
        -------
        ndarray, shape (H, W, 3), dtype float32, values in [0, 1]
        """
        angle = np.random.uniform(0, 360)
        img = Image.fromarray((image * 255).astype(np.uint8))
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        return np.asarray(img, dtype=np.float32) / 255.0
