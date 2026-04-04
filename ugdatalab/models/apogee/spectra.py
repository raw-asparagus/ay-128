import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from numpy.polynomial.chebyshev import Chebyshev

from ugdatalab.models.cache import _cache_stable
from ugdatalab.models.apogee.constants import (
    APOGEE_BAD_PIXMASK_BITS,
    APOGEE_DR17_URL,
)
from ugdatalab.models.apogee.apogee import APOGEEData


def _reconstruct_wavelength(header) -> np.ndarray:
    """Build the wavelength array from the apStar FITS header's log-linear WCS.

    Returns 10**(CRVAL1 + CDELT1 * arange(NAXIS1)), shape (NAXIS1,).
    """
    crval1 = header["CRVAL1"]
    cdelt1 = header["CDELT1"]
    naxis1 = header["NAXIS1"]
    return 10 ** (crval1 + cdelt1 * np.arange(naxis1))


def _identify_chips(header) -> list[slice]:
    """Read chip boundaries from FITS header keywords.

    Returns 3 slices [blue, green, red] from BMIN/BMAX, GMIN/GMAX, RMIN/RMAX.
    """
    return [
        slice(int(header["BMIN"]), int(header["BMAX"]) + 1),
        slice(int(header["GMIN"]), int(header["GMAX"]) + 1),
        slice(int(header["RMIN"]), int(header["RMAX"]) + 1),
    ]


@_cache_stable(module="ugdatalab.apogee")
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
    catalog, max_workers: int = 4,
) -> list[dict]:
    """Download apStar spectra for all stars in a catalog in parallel."""
    from tqdm.auto import tqdm

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


def _apply_bitmask(
    flux: np.ndarray,
    error: np.ndarray,
    bitmask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flag bad pixels.

    Bitmask-flagged pixels retain their raw flux but get error = 1e6
    (APOGEE sentinel convention), giving them negligible weight in
    downstream fits while preserving the raw values for diagnostics.
    Negative flux (unflagged cosmic rays) and missing (NaN) pixels
    are set to NaN in both flux and error.
    """
    flux = flux.copy()
    error = error.copy()

    # Step 1: bitmask-flagged — keep raw flux, set error to sentinel
    bm_bad = np.zeros(len(bitmask), dtype=bool)
    for bit in APOGEE_BAD_PIXMASK_BITS:
        bm_bad |= (bitmask & (1 << bit)) != 0
    error[bm_bad] = 1e6

    # Step 2: negative flux and missing — set both to NaN
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
    """Normalize a spectrum by fitting per-chip Chebyshev continuum polynomials.

    Returns (flux_norm, error_norm, continuum_fit).
    """
    n_pix = len(flux)
    flux_norm = np.full(n_pix, np.nan)
    error_norm = np.full(n_pix, np.inf)
    continuum_fit = np.full(n_pix, np.nan)

    for chip in chips:
        valid = np.zeros(n_pix, dtype=bool)
        valid[chip] = True
        valid &= continuum_mask
        valid &= np.isfinite(error) & (error < 1e5)

        if np.sum(valid) < degree + 1:
            error_norm[chip] = np.inf
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
    degree: int = 4,
    max_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download, mask, and normalize spectra for all stars in a catalog.

    Returns (flux, error, wavelength, continuum_mask, apogee_ids) where
    flux and error have shape (N, n_pixels).
    """
    continuum_data = np.load(continuum_path)
    wavelength = continuum_data["wavelengths"]
    continuum_mask = continuum_data["continuum"].astype(bool)

    raw_spectra = _fetch_apstar_batch(catalog, max_workers=max_workers)

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


@dataclass
class Spectrum:
    """Normalized APOGEE spectrum container. Always 2D: (N, n_pixels)."""

    fits_path: str | Path | None = None
    continuum_path: str | Path = ""
    degree: int = 4

    flux: np.ndarray = field(init=False, repr=False)
    error: np.ndarray = field(init=False, repr=False)
    wavelength: np.ndarray = field(init=False, repr=False)
    continuum_mask: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        hdul = fits.open(self.fits_path)
        flux_raw = np.array(hdul[1].data, dtype=float)
        error_raw = np.array(hdul[2].data, dtype=float)
        bitmask = np.array(hdul[3].data, dtype=np.int16)
        chips = _identify_chips(hdul[0].header)
        hdul.close()

        continuum_data = np.load(self.continuum_path)
        self.wavelength = continuum_data["wavelengths"]
        self.continuum_mask = continuum_data["continuum"].astype(bool)

        flux_masked, error_masked = _apply_bitmask(flux_raw, error_raw, bitmask)
        flux_norm, error_norm, _ = _normalize_spectrum(
            flux_masked, error_masked, self.wavelength, self.continuum_mask,
            chips, self.degree,
        )
        self.flux = flux_norm[np.newaxis, :]
        self.error = error_norm[np.newaxis, :]

    def __len__(self):
        return self.flux.shape[0]


@dataclass
class APOGEESpectra(Spectrum):
    """Batch normalized spectra tied to an APOGEEData catalog."""

    source: APOGEEData | None = None

    apogee_ids: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        flux, error, wavelength, continuum_mask, apogee_ids = (
            _fetch_normalized_spectra(
                self.source.data,
                self.continuum_path,
                degree=self.degree,
            )
        )
        self.flux = flux
        self.error = error
        self.wavelength = wavelength
        self.continuum_mask = continuum_mask
        self.apogee_ids = apogee_ids

    def __getitem__(self, apogee_id: str):
        """Access a single spectrum by apogee_id string."""
        idx = int(np.where(self.apogee_ids == apogee_id)[0][0])
        return {
            "apogee_id": self.apogee_ids[idx],
            "flux": self.flux[idx],
            "error": self.error[idx],
            "wavelength": self.wavelength,
            "continuum_mask": self.continuum_mask,
        }
