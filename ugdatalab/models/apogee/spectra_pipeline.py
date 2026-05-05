"""APOGEE per-spectrum pipeline helpers: bitmask masking and continuum normalization.

These helpers operate on a single APOGEE spectrum's raw flux, error,
and bitmask arrays from one apStar FITS file.

They are kept as module-level functions rather than ``Compose``-style
``Table -> Table`` stage classes because they need per-spectrum
parameters (``continuum_path``-derived ``wavelength`` + ``continuum_mask``,
the ``chips`` slice list, the ``degree`` setting) and operate on 1D
arrays before stacking — neither shape fits the stateless table-level
pipeline-stage contract used by the lightcurve and Gaia pipelines.
"""

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

from ugdatalab.models.apogee.constants import (
    APOGEE_BAD_PIXMASK_BITS,
    _APOGEE_SENTINEL_ERROR,
)


# Continuum-fit pixels with errors at or above this magnitude are masked
# out before the per-chip Chebyshev fit. Matches APOGEE's filler convention.
_VALID_ERROR_MAX = 1e5


def _apply_bitmask(
    flux: np.ndarray,
    error: np.ndarray,
    bitmask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flag bad pixels by inflating their errors to the APOGEE sentinel and NaN'ing negative or missing flux."""
    flux = flux.copy()
    error = error.copy()

    bm_bad = np.zeros(len(bitmask), dtype=bool)
    for bit in APOGEE_BAD_PIXMASK_BITS:
        bm_bad |= (bitmask & (1 << bit)) != 0
    # Ignore pixels with finite errors already above the sentinel value
    needs_sentinel = bm_bad & np.isfinite(error) & (error < _APOGEE_SENTINEL_ERROR)
    error[needs_sentinel] = _APOGEE_SENTINEL_ERROR

    bad = (flux < 0) | np.isnan(flux)
    flux[bad] = np.nan
    error[bad] = np.nan
    return flux, error


def _normalize_spectrum(
    flux: np.ndarray,
    error: np.ndarray,
    wavelength: np.ndarray,
    continuum_mask: np.ndarray,
    chips: list[slice],
    degree: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize a spectrum by fitting per-chip Chebyshev continuum polynomials, returning (flux_norm, error_norm, continuum_fit)."""
    n_pix = len(flux)
    flux_norm = np.full(n_pix, np.nan)
    error_norm = np.full(n_pix, np.nan)
    continuum_fit = np.full(n_pix, np.nan)

    for chip in chips:
        valid = np.zeros(n_pix, dtype=bool)
        valid[chip] = True
        valid &= continuum_mask
        valid &= np.isfinite(error) & (error < _VALID_ERROR_MAX)

        if np.sum(valid) < degree + 1:
            error_norm[chip] = np.nan
            continue

        poly = Chebyshev.fit(
            wavelength[valid], flux[valid], deg=degree,
            w=1 / error[valid] ** 2,
        )
        cont = poly(wavelength[chip])
        continuum_fit[chip] = cont
        flux_norm[chip] = flux[chip] / cont
        error_norm[chip] = error[chip] / cont

    return flux_norm, error_norm, continuum_fit
