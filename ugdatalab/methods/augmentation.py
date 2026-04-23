"""Data augmentation transforms for image classification."""

import numpy as np
from PIL import Image


class RandomRotation360:
    """Rotate an image by a uniformly random angle in [0, 360) degrees.

    Galaxy morphological labels are rotationally invariant, so rotating
    training images generates valid augmented samples without changing
    the labels. Each call produces a different random angle.

    The rotation uses bilinear interpolation and crops back to the
    original size (no padding artifacts).

    Works on numpy arrays in HWC format, float32 in [0, 1].
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply a random rotation.

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
