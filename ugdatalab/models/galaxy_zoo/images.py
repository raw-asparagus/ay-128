"""Galaxy Zoo preprocessed-image loader.

Per-image helpers (load, crop, resize) live in
:mod:`ugdatalab.models.galaxy_zoo.images_pipeline`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ugdatalab.models.base import Data
from ugdatalab.models.galaxy_zoo.galaxy_zoo import GalaxyZooData
from ugdatalab.models.galaxy_zoo.images_pipeline import _preprocess_images


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
