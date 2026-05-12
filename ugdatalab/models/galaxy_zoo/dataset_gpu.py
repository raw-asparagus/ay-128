"""GPU-resident IterableDataset that yields batched (image, label) tensors.

Holds the entire image array on the GPU as ``uint8`` and produces
batches directly via ``__iter__`` — there is no CPU-side
``__getitem__``, no DataLoader workers, and no H2D copy per batch. The
per-batch float cast (``/ 255.0``) and any augmentation are performed
on the GPU.
"""

from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import IterableDataset


_UINT8_MAX = 255


class GalaxyZooGPUDataset(IterableDataset):
    """GPU-resident Galaxy Zoo dataset that yields ``(image, label)`` batches.

    Parameters
    ----------
    images : ndarray, shape (N, H, W, 3), dtype uint8
        Preprocessed images in ``[0, 255]``. Stored on the GPU as
        ``uint8`` (one-shot copy at construction).
    labels : ndarray, shape (N, K)
        Classification labels in ``[0, 1]``.
    batch_size : int
        Mini-batch size yielded by each iteration.
    transform : callable
        Transform applied to each ``(B, 3, H, W)`` float batch *after*
        the uint8→float cast. Pass ``Compose([])`` for a no-op.
    device : torch.device or str
        Device to hold the images, labels, and yielded batches on.
    shuffle : bool
        Whether to shuffle sample indices at the start of each iteration.
    generator : torch.Generator, optional
        Generator used for shuffle order. If ``None``, the default
        generator on ``device`` is used.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        transform: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device | str,
        shuffle: bool,
        generator: torch.Generator | None = None,
    ):
        """Move images and labels to *device* and store the iteration spec."""
        super().__init__()
        device = torch.device(device)

        self.images = (
            torch.from_numpy(images)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(device)
        )  # (N, 3, H, W) uint8
        self.labels = torch.from_numpy(labels.astype(np.float32)).to(device)
        self.batch_size = batch_size
        self.transform = transform
        self.device = device
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        """Yield ``(image_batch, label_batch)`` tensors until the dataset is exhausted."""
        n = self.images.size(0)
        if self.shuffle:
            order = torch.randperm(n, generator=self.generator, device=self.device)
        else:
            order = torch.arange(n, device=self.device)

        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            images = self.images.index_select(0, idx).float().div_(_UINT8_MAX)
            images = self.transform(images)
            labels = self.labels.index_select(0, idx)
            yield images, labels

    def __len__(self) -> int:
        """Return the number of batches per iteration."""
        n = self.images.size(0)
        return (n + self.batch_size - 1) // self.batch_size
