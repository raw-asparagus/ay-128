"""Galaxy Zoo 2 classification labels, image preprocessing, and PyTorch Dataset."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ugdatalab.models.base import Data
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
class GalaxyZooData(Data):
    """Load Galaxy Zoo 2 classification labels from CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to ``training_classifications.csv``.
    pipeline : Compose, keyword-only
        Inherited from :class:`~ugdatalab.models.base.Data`. Pipeline of
        cuts and column augmentations applied immediately after CSV load
        + sanitize. Default ``Compose([])`` — no transformations.

    Attributes
    ----------
    data : pandas.DataFrame
        Sanitized (and cut) classifications table.
    """
    csv_path: str | Path

    def _fetch(self) -> pd.DataFrame:
        """Read the Galaxy Zoo 2 classifications CSV into a DataFrame."""
        return pd.read_csv(self.csv_path)

    def _sanitize(self, raw: pd.DataFrame) -> None:
        """Coerce columns to the Galaxy Zoo schema in place."""
        _sanitize_dataframe(raw, _GALAXY_ZOO_SCHEMA)

    @property
    def labels(self) -> np.ndarray:
        """Classification labels, shape (N, 37), values in [0, 1]."""
        return self.data[LABEL_COLUMNS].to_numpy(dtype=np.float32)

    @property
    def galaxy_ids(self) -> np.ndarray:
        """Galaxy ID column as integer array."""
        return self.data["GalaxyID"].to_numpy()


@dataclass
class GalaxyZooImages(Data):
    """Preprocessed in-memory Galaxy Zoo image array.

    Loads all images for ``source``'s galaxy IDs from disk, center-crops,
    and resizes them to a uniform square size on construction.

    Parameters
    ----------
    source : GalaxyZooData
        Catalog providing galaxy IDs.
    image_dir : Path
        Directory containing JPEG images named by GalaxyID.
    crop_fraction : float
        Fraction of each border to remove before resizing (0.25 keeps
        the central 50% of the image).
    target_size : int
        Output image dimension (``target_size`` × ``target_size`` pixels).
    pipeline : Compose, keyword-only
        Inherited from :class:`~ugdatalab.models.base.Data`. Default
        ``Compose([])`` — no transformations.

    Attributes
    ----------
    data : ndarray, shape (N, target_size, target_size, 3)
        Preprocessed images in [0, 1], float32.
    images : ndarray
        Alias of ``data``.
    """
    source: GalaxyZooData
    image_dir: Path
    crop_fraction: float
    target_size: int

    def _fetch(self) -> np.ndarray:
        """Load, center-crop, and resize all images for the source's galaxy IDs."""
        return _preprocess_images(
            Path(self.image_dir),
            self.source.galaxy_ids,
            self.crop_fraction,
            self.target_size,
        )

    @property
    def images(self) -> np.ndarray:
        """Return the preprocessed image array (alias of ``self.data``)."""
        return self.data


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
        conversion to tensor. Required (no default); pass
        ``Compose([])`` for a no-op.
    """
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Callable,
    ):
        """Store images, float32 labels, and the per-sample image transform."""
        self.images = images
        self.labels = labels.astype(np.float32)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        """Return the (CHW image tensor, label tensor) pair at *index* after applying the transform."""
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
        """Return the number of samples in the dataset."""
        return len(self.images)
