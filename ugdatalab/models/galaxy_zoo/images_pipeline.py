"""Galaxy Zoo per-image preprocessing helpers: load, crop, resize."""

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


# uint8 max — denominator for normalizing 8-bit images to [0, 1].
_UINT8_MAX = 255.0


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
