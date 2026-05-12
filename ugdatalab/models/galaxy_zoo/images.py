"""Galaxy Zoo preprocessed-image loaders.

Per-image helpers and ``Compose``-style stages live in
:mod:`ugdatalab.models.galaxy_zoo.images_pipeline`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from ugdatalab.models.base import Data
from ugdatalab.models.galaxy_zoo.galaxy_zoo import GalaxyZooData
from ugdatalab.models.galaxy_zoo.images_pipeline import CropFraction, Resize
from ugdatalab.utils.compose import Compose


def _load_image(path: Path) -> np.ndarray:
    """Read a JPEG image and return it as a uint8 (H, W, 3) array in [0, 255]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _preprocess_images(
    image_dir: Path,
    galaxy_ids: np.ndarray,
    crop_fraction: float,
    target_size: int,
) -> np.ndarray:
    """Batch-load, center-crop, and resize JPEGs for the given galaxy IDs into an (N, target_size, target_size, 3) uint8 array."""
    pipeline = Compose([CropFraction(crop_fraction), Resize(target_size)])
    n = len(galaxy_ids)
    images = np.empty((n, target_size, target_size, 3), dtype=np.uint8)
    for i, gid in enumerate(tqdm(galaxy_ids, desc="Loading images")):
        images[i] = pipeline(_load_image(image_dir / f"{gid}.jpg"))
    return images


# ---------------------------------------------------------------------------
# GalaxyImages (directory-driven, catalog-less)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GalaxyImages:
    """Preprocessed images from a directory of JPEGs (no source catalog).

    Construct via :meth:`from_directory`. This is the catalog-less
    counterpart to :class:`GalaxyZooImages`: galaxy IDs come from the
    JPEG filename stems rather than a :class:`GalaxyZooData` catalog,
    and there is no :class:`~ugdatalab.models.base.Data` lifecycle.

    Attributes
    ----------
    images : ndarray, shape (N, target_size, target_size, 3)
        Preprocessed images in [0, 255], uint8.
    galaxy_ids : ndarray, shape (N,)
        Galaxy IDs derived from the JPEG filename stems, in the same
        order as ``images``.
    """
    images: np.ndarray
    galaxy_ids: np.ndarray

    @classmethod
    def from_directory(
        cls,
        image_dir: str | Path,
        crop_fraction: float,
        target_size: int,
    ) -> "GalaxyImages":
        """Scan ``image_dir`` for ``*.jpg`` files and preprocess them.

        Parameters
        ----------
        image_dir : str or Path
            Directory containing JPEG images named ``<GalaxyID>.jpg``.
        crop_fraction : float
            Fraction of each border to remove before resizing.
        target_size : int
            Output image dimension (``target_size`` × ``target_size`` pixels).

        Returns
        -------
        GalaxyImages
            Preprocessed images and the galaxy IDs derived from filename stems.
        """
        image_dir = Path(image_dir)
        files = sorted(image_dir.glob("*.jpg"))
        galaxy_ids = np.array([int(f.stem) for f in files])
        images = _preprocess_images(image_dir, galaxy_ids, crop_fraction, target_size)
        return cls(images=images, galaxy_ids=galaxy_ids)


# ---------------------------------------------------------------------------
# GalaxyZooImages (catalog-driven bulk loader)
# ---------------------------------------------------------------------------


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
        Preprocessed images in [0, 255], uint8.
    images : ndarray
        Alias of ``data``.

    See Also
    --------
    GalaxyImages
        Directory-driven, catalog-less counterpart for unlabeled image
        sets (e.g., the lab test set).
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
