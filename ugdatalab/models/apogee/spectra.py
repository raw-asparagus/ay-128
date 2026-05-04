"""APOGEE apStar spectrum download, bitmask masking, and continuum normalization."""

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from numpy.polynomial.chebyshev import Chebyshev
from tqdm.auto import tqdm

from ugdatalab.utils.cache import cache_stable
from ugdatalab.models.apogee.constants import (
    APOGEE_BAD_PIXMASK_BITS,
    APOGEE_DR17_URL,
    _APOGEE_SENTINEL_ERROR
)
from ugdatalab.models.apogee.apogee import APOGEEData


# Continuum-fit pixels with errors at or above this magnitude are masked
# out before the per-chip Chebyshev fit. Matches APOGEE's filler convention.
_VALID_ERROR_MAX = 1e5


# ---------------------------------------------------------------------------
# APOGEE Spectra I/O
# ---------------------------------------------------------------------------


def _reconstruct_wavelength(header) -> np.ndarray:
    """Build the wavelength array from the apStar FITS header's log-linear WCS."""
    crval1 = header["CRVAL1"]
    cdelt1 = header["CDELT1"]
    naxis1 = header["NAXIS1"]
    return 10 ** (crval1 + cdelt1 * np.arange(naxis1))


def _identify_chips(header) -> list[slice]:
    """Return [blue, green, red] chip slices from BMIN/BMAX, GMIN/GMAX, RMIN/RMAX header keys."""
    return [
        slice(int(header["BMIN"]), int(header["BMAX"]) + 1),
        slice(int(header["GMIN"]), int(header["GMAX"]) + 1),
        slice(int(header["RMIN"]), int(header["RMAX"]) + 1),
    ]


@cache_stable(module="ugdatalab.apogee")
def _get_apstar_spectra(apogee_id: str, telescope: str, field: str) -> dict:
    """Download an apStar FITS file and extract the coadded spectrum."""
    url = f"{APOGEE_DR17_URL}/{telescope}/{field}/apStar-dr17-{apogee_id}.fits"
    resp = requests.get(url)
    resp.raise_for_status()

    hdul = fits.open(io.BytesIO(resp.content))
    flux = np.array(hdul[1].data[0], dtype=float)
    error = np.array(hdul[2].data[0], dtype=float)
    bitmask = np.array(hdul[3].data[0], dtype=np.int16)
    wavelength = _reconstruct_wavelength(hdul[1].header)
    chips = _identify_chips(hdul[0].header)
    hdul.close()

    return {
        "apogee_id": apogee_id,
        "wavelength": wavelength,
        "chips": chips,
        "flux": flux,
        "error": error,
        "bitmask": bitmask,
    }


def _fetch_apstar_batch(
    catalog, max_workers: int = 8,
) -> list[dict]:
    """Download apStar spectra for all stars in a catalog in parallel."""
    ids = catalog["apogee_id"]
    telescopes = catalog["telescope"]
    fields = catalog["field"]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_get_apstar_spectra, ids[i], telescopes[i], fields[i]): i
            for i in range(len(ids))
        }
        results = [None] * len(ids)
        for future in tqdm(as_completed(futures), total=len(futures), desc="APOGEE spectra"):
            idx = futures[future]
            results[idx] = future.result()
    return results


# ---------------------------------------------------------------------------
# Spectrum masking and normalization
# ---------------------------------------------------------------------------


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


def _fetch_normalized_spectra(
    catalog,
    continuum_path: str | Path,
    degree: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download, mask, and normalize spectra for all stars in a catalog, returning (flux, error, wavelength, continuum_mask, apogee_ids)."""
    continuum_data = np.load(continuum_path)
    wavelength = continuum_data["wavelengths"]
    continuum_mask = continuum_data["continuum"].astype(bool)

    raw_spectra = _fetch_apstar_batch(catalog)

    flux_all = []
    error_all = []
    apogee_ids = []

    for spec in raw_spectra:
        flux, error = _apply_bitmask(spec["flux"], spec["error"], spec["bitmask"])
        flux_norm, error_norm, _ = _normalize_spectrum(
            flux, error, wavelength, continuum_mask, spec["chips"], degree,
        )
        flux_all.append(flux_norm)
        error_all.append(error_norm)
        apogee_ids.append(spec["apogee_id"])

    return (
        np.array(flux_all),
        np.array(error_all),
        wavelength,
        continuum_mask,
        np.array(apogee_ids),
    )


# ---------------------------------------------------------------------------
# APOGEESpectra
# ---------------------------------------------------------------------------


@dataclass
class APOGEESpectra:
    """Continuum-normalized APOGEE spectra container with flux/error arrays of shape (N, n_pixels).

    Construct via ``from_fits`` (one apStar file) or ``from_catalog`` (bulk
    download from an APOGEEData catalog). The dataclass ``__init__`` takes
    pre-built arrays and is rarely called directly.

    Attributes
    ----------
    flux, error : ndarray, shape (N, n_pixels)
        Continuum-normalized flux and error for N spectra.
    wavelength : ndarray, shape (n_pixels,)
        Common wavelength grid.
    continuum_mask : ndarray of bool, shape (n_pixels,)
        Continuum pixel mask used for the per-chip normalization.
    apogee_ids : ndarray of str, shape (N,)
        APOGEE identifiers; the FITS-file stem when constructed from a single FITS.
    """

    flux: np.ndarray
    error: np.ndarray
    wavelength: np.ndarray
    continuum_mask: np.ndarray
    apogee_ids: np.ndarray

    def __len__(self):
        """Return the number of spectra (rows of ``flux``)."""
        return self.flux.shape[0]

    @classmethod
    def from_fits(
        cls,
        fits_path: str | Path,
        continuum_path: str | Path,
        degree: int = 4,
    ) -> "APOGEESpectra":
        """Construct from a single apStar FITS file.

        Args:
            fits_path: Path to the apStar FITS file.
            continuum_path: Path to the ``.npz`` file holding the wavelength
                grid and continuum mask.
            degree: Per-chip Chebyshev continuum polynomial degree.

        Returns:
            APOGEESpectra with N = 1.
        """
        hdul = fits.open(fits_path)
        flux_raw = np.array(hdul[1].data, dtype=float)
        error_raw = np.array(hdul[2].data, dtype=float)
        bitmask = np.array(hdul[3].data, dtype=np.int16)
        chips = _identify_chips(hdul[0].header)
        hdul.close()

        continuum_data = np.load(continuum_path)
        wavelength = continuum_data["wavelengths"]
        continuum_mask = continuum_data["continuum"].astype(bool)

        flux_masked, error_masked = _apply_bitmask(flux_raw, error_raw, bitmask)
        flux_norm, error_norm, _ = _normalize_spectrum(
            flux_masked, error_masked, wavelength, continuum_mask, chips, degree,
        )
        return cls(
            flux=flux_norm[np.newaxis, :],
            error=error_norm[np.newaxis, :],
            wavelength=wavelength,
            continuum_mask=continuum_mask,
            apogee_ids=np.array([Path(fits_path).stem]),
        )

    @classmethod
    def from_catalog(
        cls,
        source: APOGEEData,
        continuum_path: str | Path,
        degree: int = 4,
    ) -> "APOGEESpectra":
        """Construct by downloading and normalizing all spectra in a catalog.

        Args:
            source: APOGEEData catalog whose ``data`` rows drive the apStar
                download list.
            continuum_path: Path to the ``.npz`` file holding the wavelength
                grid and continuum mask.
            degree: Per-chip Chebyshev continuum polynomial degree.

        Returns:
            APOGEESpectra with N = len(source.data).
        """
        flux, error, wavelength, continuum_mask, apogee_ids = (
            _fetch_normalized_spectra(source.data, continuum_path, degree=degree)
        )
        return cls(
            flux=flux,
            error=error,
            wavelength=wavelength,
            continuum_mask=continuum_mask,
            apogee_ids=apogee_ids,
        )
