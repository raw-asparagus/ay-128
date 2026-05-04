"""Galaxy Zoo 2 classification labels, image preprocessing, and PyTorch Dataset."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ugdatalab.models.galaxy_zoo.constants import (
    LABEL_COLUMNS,
    _GALAXY_ZOO_SCHEMA,
)
from ugdatalab.utils.tables import _sanitize_dataframe


# uint8 max — denominator for normalizing 8-bit images to [0, 1].
_UINT8_MAX = 255.0


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------


def _load_image(path: Path) -> np.ndarray:
    """Read a JPEG image and return it as a float32 (H, W, 3) array in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / _UINT8_MAX


def _crop_center(image: np.ndarray, crop_fraction: float) -> np.ndarray:
    """Center-crop an (H, W, 3) image by removing ``crop_fraction`` of each side."""
    h, w = image.shape[:2]
    dy = int(h * crop_fraction)
    dx = int(w * crop_fraction)
    return image[dy:h - dy, dx:w - dx]


def _resize(image: np.ndarray, target_size: int) -> np.ndarray:
    """Lanczos-resize an (H, W, 3) float32 image to ``(target_size, target_size, 3)``."""
    img = Image.fromarray((image * _UINT8_MAX).astype(np.uint8))
    img = img.resize((target_size, target_size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / _UINT8_MAX


def _preprocess_images(
    image_dir: Path,
    galaxy_ids: np.ndarray,
    crop_fraction: float,
    target_size: int,
) -> np.ndarray:
    """Batch-load, center-crop, and resize JPEG images for the given galaxy IDs into an (N, target_size, target_size, 3) array."""
    n = len(galaxy_ids)
    images = np.empty((n, target_size, target_size, 3), dtype=np.float32)
    for i, gid in enumerate(tqdm(galaxy_ids, desc="Loading images")):
        path = image_dir / f"{gid}.jpg"
        img = _load_image(path)
        img = _crop_center(img, crop_fraction)
        img = _resize(img, target_size)
        images[i] = img
    return images


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------


@dataclass
class GalaxyZooData:
    """Load Galaxy Zoo 2 classification labels from CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to ``training_classifications.csv``.
    """
    csv_path: str | Path
    data: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        data = pd.read_csv(self.csv_path)
        _sanitize_dataframe(data, _GALAXY_ZOO_SCHEMA)
        self.data = data

    @property
    def labels(self) -> np.ndarray:
        """Classification labels, shape (N, 37), values in [0, 1]."""
        return self.data[LABEL_COLUMNS].to_numpy(dtype=np.float32)

    @property
    def galaxy_ids(self) -> np.ndarray:
        """Galaxy ID column as integer array."""
        return self.data["GalaxyID"].to_numpy()

    def __len__(self) -> int:
        """Number of galaxies in the dataset."""
        return len(self.data)


@dataclass
class GalaxyZooImages:
    """Preprocessed in-memory Galaxy Zoo image array.

    Loads all images from disk, center-crops, and resizes them to a
    uniform square size.

    Parameters
    ----------
    source : GalaxyZooData
        Data source providing galaxy IDs.
    image_dir : Path
        Directory containing JPEG images named by GalaxyID.
    crop_fraction : float
        Fraction of each border to remove before resizing (0.25 keeps
        the central 50% of the image).
    target_size : int
        Output image dimension (``target_size`` × ``target_size`` pixels).
    """
    source: GalaxyZooData
    image_dir: Path
    crop_fraction: float
    target_size: int

    images: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.image_dir = Path(self.image_dir)
        self.images = _preprocess_images(
            self.image_dir,
            self.source.galaxy_ids,
            self.crop_fraction,
            self.target_size,
        )


class GalaxyZooDataset(Dataset):
    """PyTorch Dataset for Galaxy Zoo images and labels.

    Parameters
    ----------
    images : ndarray, shape (N, H, W, 3)
        Preprocessed images in [0, 1], float32.
    labels : ndarray, shape (N, 37)
        Classification labels in [0, 1].
    transform : callable
        Transform applied to each image (numpy HWC array) before
        conversion to tensor. Required: cached images are typically
        larger than the model input, so callers must compose at minimum
        a ``CenterCrop`` to the model's input size. Pass ``Compose([])``
        for a true no-op.
    """
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Callable,
    ):
        self.images = images
        self.labels = labels.astype(np.float32)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        image = self.images[index]  # (H, W, 3)
        label = self.labels[index]  # (37,)

        image = self.transform(image)

        # HWC → CHW for PyTorch convention
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1)),
        )
        label_tensor = torch.from_numpy(label)
        return image_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.images)
