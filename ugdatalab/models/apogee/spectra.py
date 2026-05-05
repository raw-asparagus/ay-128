"""APOGEE apStar spectrum loader.

Per-spectrum bitmask masking and continuum normalization helpers
(``_apply_bitmask``, ``_normalize_spectrum``) live in
:mod:`ugdatalab.models.apogee.spectra_pipeline`.
"""

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from astropy import table
from astropy.io import fits
from tqdm.auto import tqdm

from ugdatalab.models.base import Data
from ugdatalab.utils.cache import cache_stable
from ugdatalab.models.apogee.constants import APOGEE_DR17_URL
from ugdatalab.models.apogee.apogee import APOGEEData
from ugdatalab.models.apogee.spectra_pipeline import (
    _apply_bitmask,
    _normalize_spectrum,
)


# ---------------------------------------------------------------------------
# FITS-header helpers
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


# ---------------------------------------------------------------------------
# APOGEE Spectra I/O
# ---------------------------------------------------------------------------


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
# APOGEESpectrum (single-spectrum, file-driven)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class APOGEESpectrum:
    """A single continuum-normalized APOGEE spectrum loaded from a FITS file.

    Construct via :meth:`from_fits`. This is the catalog-less counterpart
    to :class:`APOGEESpectra`: shapes are 1D rather than ``(N, n_pixels)``,
    and there is no ``Data`` lifecycle, no source catalog, and no
    pipeline.

    Attributes
    ----------
    flux, error, wavelength : ndarray, shape (n_pixels,)
        Continuum-normalized 1D arrays sharing the apStar wavelength grid.
    apogee_id : str
        Identifier for the spectrum (FITS-file stem when constructed via
        :meth:`from_fits`).
    """
    flux: np.ndarray
    error: np.ndarray
    wavelength: np.ndarray
    apogee_id: str

    @classmethod
    def from_fits(
        cls,
        fits_path: str | Path,
        continuum_path: str | Path,
        degree: int = 4,
    ) -> "APOGEESpectrum":
        """Bitmask-correct and continuum-normalize a spectrum from an apStar FITS file.

        Parameters
        ----------
        fits_path : str or Path
            apStar FITS file.
        continuum_path : str or Path
            ``.npz`` file holding the wavelength grid and continuum mask.
        degree : int
            Per-chip Chebyshev continuum polynomial degree. Default ``4`` —
            matches the bulk ``APOGEESpectra`` default.

        Returns
        -------
        APOGEESpectrum
            ``flux``, ``error``, ``wavelength`` are 1D arrays of shape
            ``(n_pixels,)``; ``apogee_id`` is the FITS-file stem.
        """
        hdul = fits.open(fits_path)
        # ``np.atleast_2d(...)[0]`` accepts both apStar files (2D, coadd
        # at row 0) and pre-extracted 1D spectra without forcing the
        # caller to reshape.
        flux_raw = np.atleast_2d(np.asarray(hdul[1].data, dtype=float))[0]
        error_raw = np.atleast_2d(np.asarray(hdul[2].data, dtype=float))[0]
        bitmask = np.atleast_2d(np.asarray(hdul[3].data, dtype=np.int16))[0]
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
            flux=flux_norm,
            error=error_norm,
            wavelength=wavelength,
            apogee_id=Path(fits_path).stem,
        )


# ---------------------------------------------------------------------------
# APOGEESpectra (catalog-driven bulk loader)
# ---------------------------------------------------------------------------


@dataclass
class APOGEESpectra(Data):
    """Bulk-download and continuum-normalize APOGEE spectra for an ``APOGEEData`` catalog.

    Downloads each source's apStar FITS file (joblib-cached),
    bitmask-corrects and Chebyshev-normalizes it, then stacks the
    results into one table with vector-valued ``flux`` and ``error``
    columns of shape ``(N, n_pixels)`` and the common wavelength grid
    stored in ``meta``.

    Parameters
    ----------
    source : APOGEEData
        Catalog whose ``data`` rows drive the apStar download list
        (each row supplies ``apogee_id``, ``telescope``, ``field``).
    continuum_path : str or Path
        ``.npz`` file holding the published wavelength grid and
        continuum-pixel mask used for per-chip normalization.
    degree : int
        Per-chip Chebyshev continuum polynomial degree. Default ``4`` —
        the canonical Cannon-paper choice; raise for stiffer fields,
        lower for very noisy spectra.
    pipeline : Compose, keyword-only
        Inherited from :class:`~ugdatalab.models.base.Data`. Default
        ``Compose([])`` — no post-stack transformations.

    Attributes
    ----------
    data : astropy.table.Table
        Columns ``apogee_id`` (N,), ``flux`` (N, n_pixels), ``error``
        (N, n_pixels). ``meta["wavelength"]`` holds the common
        ``(n_pixels,)`` grid. The convenience properties ``flux``,
        ``error``, ``wavelength``, ``apogee_ids`` forward to the columns
        and meta entry.

    See Also
    --------
    APOGEESpectrum
        Single-spectrum, file-driven counterpart for the catalog-less
        case (e.g., loading a standalone apStar FITS file).
    """
    source: APOGEEData
    continuum_path: str | Path
    degree: int = 4

    def _fetch(self) -> table.Table:
        """Download, bitmask-correct, and continuum-normalize every spectrum in the source catalog."""
        continuum_data = np.load(self.continuum_path)
        wavelength = continuum_data["wavelengths"]
        continuum_mask = continuum_data["continuum"].astype(bool)

        raw_spectra = _fetch_apstar_batch(self.source.data)

        # Bitmask + normalize run here rather than as ``_required_stages``
        # callables: they operate per-spectrum on 1D arrays before
        # stacking, and need ``wavelength``/``continuum_mask``/``chips``/
        # ``degree`` parameters that don't fit the stateless
        # ``Table -> Table`` pipeline-stage contract.
        flux_all = []
        error_all = []
        apogee_ids = []
        for spec in raw_spectra:
            flux, error = _apply_bitmask(spec["flux"], spec["error"], spec["bitmask"])
            flux_norm, error_norm, _ = _normalize_spectrum(
                flux, error, wavelength, continuum_mask, spec["chips"], self.degree,
            )
            flux_all.append(flux_norm)
            error_all.append(error_norm)
            apogee_ids.append(spec["apogee_id"])

        out = table.Table({
            "apogee_id": np.array(apogee_ids),
            "flux": np.array(flux_all),
            "error": np.array(error_all),
        })
        out.meta["wavelength"] = wavelength
        return out

    @property
    def flux(self) -> np.ndarray:
        """Continuum-normalized flux, shape ``(N, n_pixels)``."""
        return np.asarray(self.data["flux"])

    @property
    def error(self) -> np.ndarray:
        """Continuum-normalized per-pixel error, shape ``(N, n_pixels)``."""
        return np.asarray(self.data["error"])

    @property
    def wavelength(self) -> np.ndarray:
        """Common wavelength grid, shape ``(n_pixels,)``."""
        return self.data.meta["wavelength"]

    @property
    def apogee_ids(self) -> np.ndarray:
        """APOGEE identifiers, shape ``(N,)``."""
        return np.asarray(self.data["apogee_id"])
