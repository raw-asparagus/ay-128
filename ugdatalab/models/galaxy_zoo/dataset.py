"""PyTorch ``Dataset`` adapter for Galaxy Zoo images and labels."""

from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


_UINT8_MAX = 255


class GalaxyZooDataset(Dataset):
    """PyTorch Dataset for Galaxy Zoo images and labels.

    Parameters
    ----------
    images : ndarray, shape (N, H, W, 3), dtype uint8
        Preprocessed images in [0, 255].
    labels : ndarray, shape (N, 37)
        Classification labels in [0, 1].
    transform : callable
        Transform applied to each uint8 HWC image before the
        ``/ 255.0`` cast to float and conversion to tensor. Required
        (no default); pass ``Compose([])`` for a no-op.
    """
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Callable,
    ):
        """Store uint8 images, float32 labels, and the per-sample image transform."""
        self.images = images
        self.labels = labels.astype(np.float32)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        """Return the (CHW float image tensor, label tensor) pair at *index* after applying the transform."""
        image = self.images[index]  # (H, W, 3) uint8
        label = self.labels[index]  # (37,)

        image = self.transform(image)

        # HWC uint8 → CHW float32 in [0, 1]
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
            .float()
            .div_(_UINT8_MAX)
        )
        label_tensor = torch.from_numpy(label)
        return image_tensor, label_tensor

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)
