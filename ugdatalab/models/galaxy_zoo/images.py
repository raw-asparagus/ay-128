"""Galaxy Zoo image loading, preprocessing, and PyTorch Dataset."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from ugdatalab.models.galaxy_zoo.galaxy_zoo import GalaxyZooData


def _load_image(path: Path) -> np.ndarray:
    """Read a JPEG image and return as float32 array in [0, 1].

    Parameters
    ----------
    path : Path
        Path to JPEG image file.

    Returns
    -------
    ndarray, shape (H, W, 3), dtype float32
    """
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _crop_center(image: np.ndarray, crop_fraction: float) -> np.ndarray:
    """Center-crop an image, removing ``crop_fraction`` of each border.

    Parameters
    ----------
    image : ndarray, shape (H, W, 3)
    crop_fraction : float
        Fraction of each dimension to remove from each side.
        E.g. 0.25 keeps the central 50% of the image.

    Returns
    -------
    ndarray, shape (H', W', 3)
    """
    h, w = image.shape[:2]
    dy = int(h * crop_fraction)
    dx = int(w * crop_fraction)
    return image[dy:h - dy, dx:w - dx]


def _resize(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize an image to ``(target_size, target_size)`` using bilinear interpolation.

    Parameters
    ----------
    image : ndarray, shape (H, W, 3), float32 in [0, 1]
    target_size : int
        Output dimension.

    Returns
    -------
    ndarray, shape (target_size, target_size, 3), dtype float32
    """
    img = Image.fromarray((image * 255).astype(np.uint8))
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _preprocess_images(
    image_dir: Path,
    galaxy_ids: np.ndarray,
    crop_fraction: float,
    target_size: int,
) -> np.ndarray:
    """Batch load, crop, and resize images.

    Parameters
    ----------
    image_dir : Path
        Directory containing JPEG images named ``{galaxy_id}.jpg``.
    galaxy_ids : ndarray
        Galaxy IDs to load.
    crop_fraction : float
        Fraction of border to crop from each side before resizing.
    target_size : int
        Output spatial dimension.

    Returns
    -------
    ndarray, shape (N, target_size, target_size, 3), dtype float32
    """
    n = len(galaxy_ids)
    images = np.empty((n, target_size, target_size, 3), dtype=np.float32)
    for i, gid in enumerate(tqdm(galaxy_ids, desc="Loading images")):
        path = image_dir / f"{gid}.jpg"
        img = _load_image(path)
        img = _crop_center(img, crop_fraction)
        img = _resize(img, target_size)
        images[i] = img
    return images


@dataclass
class GalaxyZooImages:
    """Preprocessed Galaxy Zoo image arrays.

    Loads all images from disk, center-crops, and resizes to a uniform
    square size. The resulting array is kept in memory for fast access.

    Parameters
    ----------
    source : GalaxyZooData
        Data source providing galaxy IDs.
    image_dir : Path
        Directory containing JPEG images named by GalaxyID.
    crop_fraction : float
        Fraction of border to crop from each side. A value of 0.25
        keeps the central 50% of the original image, removing mostly
        empty sky at the edges.
    target_size : int
        Output image dimension (target_size x target_size pixels).
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


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import Dataset as _Dataset

    class GalaxyZooDataset(_Dataset):
        """PyTorch Dataset for Galaxy Zoo images and labels.

        Parameters
        ----------
        images : ndarray, shape (N, H, W, 3)
            Preprocessed images in [0, 1], float32.
        labels : ndarray, shape (N, 37)
            Classification labels in [0, 1].
        transform : callable or None
            Optional transform applied to each image (numpy HWC array)
            before conversion to tensor. Used for data augmentation.
        """
        def __init__(
            self,
            images: np.ndarray,
            labels: np.ndarray,
            transform=None,
        ):
            self.images = images
            self.labels = labels.astype(np.float32)
            self.transform = transform

        def __getitem__(self, index: int) -> tuple:
            image = self.images[index]  # (H, W, 3)
            label = self.labels[index]  # (37,)

            if self.transform is not None:
                image = self.transform(image)

            # HWC → CHW for PyTorch convention
            image_tensor = torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1)),
            )
            label_tensor = torch.from_numpy(label)
            return image_tensor, label_tensor

        def __len__(self) -> int:
            return len(self.images)

except ImportError:
    pass
