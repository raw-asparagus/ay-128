from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Any

import numpy as np
from astropy import table
from astropy.io import fits
from astropy.utils.data import download_file

from ugdatalab.models.cache import cache_stable
from ugdatalab.paths import CONTINUUM_PIXELS_PATH

_DOWNLOAD_TIMEOUT = 120  # seconds

_SAS_BASE = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars"

# APOGEE_PIXMASK bits 0-7 and 12 (SDSS DR17 data model)
#   0  BADPIX, 1  CRPIX, 2  SATPIX,  3  UNFIXABLE
#   4  BADDARK, 5  BADFLAT, 6  BADERR, 7  NOSKY
#  12  SIG_SKYLINE
APOGEE_BAD_BITS = 0xFF | (1 << 12)  # 0x10FF

# Sentinel error assigned to bad pixels — large enough to make
# their contribution to any chi^2 / likelihood negligible
BAD_PIXEL_ERR = 1e10

# APOGEE_PIXMASK bit definitions (SDSS DR17 data model)
APOGEE_PIXMASK: dict[int, tuple[str, str]] = {
    0:  ("BADPIX",      "Pixel flagged as bad in the bad-pixel mask"),
    1:  ("CRPIX",       "Cosmic-ray pixel"),
    2:  ("SATPIX",      "Saturated pixel"),
    3:  ("UNFIXABLE",   "Pixel could not be corrected by fixing bad columns"),
    4:  ("BADDARK",     "Dark current correction unreliable"),
    5:  ("BADFLAT",     "Flat-field correction unreliable"),
    6:  ("BADERR",      "Formal uncertainty is bad / negative"),
    7:  ("NOSKY",       "Sky-subtraction pixel unavailable"),
    12: ("SIG_SKYLINE", "Significant sky-line residual (>3σ)"),
}


def chip_wavelength_ranges(wavelength: np.ndarray, chips_pixels: list[tuple[Any]]) -> list[tuple[float, float]]:
    """Convert per-chip pixel bounds (from FITS header) to wavelength bounds.

    The apStar PRIMARY header stores the pixel extent of each detector chip
    in the combined spectrum via BOVERMIN/BOVERMAX, GOVERMIN/GOVERMAX, and
    ROVERMIN/ROVERMAX (0-indexed pixel positions).  This function maps those
    to (lo, hi) wavelength values using the pre-computed wavelength array.

    Parameters
    ----------
    wavelength   : (npix,) wavelength array in Angstrom
    chips_pixels : list of (pmin, pmax) pixel-index pairs, one per chip

    Returns
    -------
    List of (lo, hi) wavelength tuples, one per chip, in wavelength order.
    """
    return [(float(wavelength[pmin]), float(wavelength[pmax]))
            for pmin, pmax in chips_pixels]


# ---------------------------------------------------------------------------
# Payload extraction
# ---------------------------------------------------------------------------


def _apstar_url(telescope: str, field: str, apogee_id: str) -> str:
    return f"{_SAS_BASE}/{telescope}/{field}/apStar-dr17-{apogee_id}.fits"


def _spectra_payload(hdul: fits.HDUList) -> dict:
    """Extract the combined spectrum arrays from an apStar HDU list.

    apStar HDU layout (DR17):
      1 – FLUX    : (nvisits+1, npixels)  row 0 = combined
      2 – ERR     : same shape
      3 – MASK    : same shape
    Wavelength is log-linear: λ = 10^(CRVAL1 + CDELT1 * pixel)

    The PRIMARY header stores per-chip pixel bounds in the combined spectrum:
      BOVERMIN/BOVERMAX – blue chip
      GOVERMIN/GOVERMAX – green chip
      ROVERMIN/ROVERMAX – red chip
    These are returned as ``chips_pixels`` so callers can derive wavelength
    ranges without hardcoding values or inspecting NaN patterns.
    """
    flux_data = hdul[1].data
    err_data  = hdul[2].data
    mask_data = hdul[3].data

    # handle both single-visit (1D) and multi-visit (2D, row 0 = combined)
    if flux_data.ndim == 1:
        flux, flux_err, mask = flux_data, err_data, mask_data
    else:
        flux, flux_err, mask = flux_data[0], err_data[0], mask_data[0]

    hdr  = hdul[1].header
    npix = flux.shape[-1]
    wave = 10.0 ** (hdr["CRVAL1"] + hdr["CDELT1"] * np.arange(npix))

    flux     = np.asarray(flux,     dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    mask     = np.asarray(mask,     dtype=np.int32)

    # inflate errors on bad pixels so they are ignored in likelihood fits
    bad = (mask & APOGEE_BAD_BITS) != 0
    flux_err[bad] = BAD_PIXEL_ERR

    # Chip pixel bounds from PRIMARY header (BOVERMIN/MAX, GOVERMIN/MAX, ROVERMIN/MAX)
    phdr = hdul[0].header
    chips_pixels = [
        (phdr["BOVERMIN"], phdr["BOVERMAX"]),   # blue chip
        (phdr["GOVERMIN"], phdr["GOVERMAX"]),   # green chip
        (phdr["ROVERMIN"], phdr["ROVERMAX"]),   # red chip
    ]

    return {
        "flux":         flux,
        "flux_err":     flux_err,
        "mask":         mask,
        "wavelength":   wave,
        "chips_pixels": chips_pixels,
    }


# ---------------------------------------------------------------------------
# Empty-table helpers
# ---------------------------------------------------------------------------


def _empty_spectra_table() -> table.Table:
    return table.Table({"apogee_id": np.asarray([], dtype=str)})


def _empty_joined_table(catalog: table.Table, spectra: table.Table) -> table.Table:
    res = table.Table()
    for name in catalog.colnames:
        res[name] = np.asarray(catalog[name])[:0]
    for name in spectra.colnames:
        if name not in res.colnames:
            res[name] = np.asarray(spectra[name])[:0]
    return res


# ---------------------------------------------------------------------------
# Per-source cached download
# ---------------------------------------------------------------------------


@cache_stable(module="ugdatalab.sdss")
def _get_spectra(apogee_id: str, telescope: str, field: str) -> table.Table:
    """Download and cache the apStar spectrum for one source."""
    url        = _apstar_url(telescope, field, apogee_id)
    local_path = download_file(url, cache=True, timeout=_DOWNLOAD_TIMEOUT)
    with fits.open(local_path) as hdul:
        payload = _spectra_payload(hdul)
    return table.Table({
        "apogee_id":    [apogee_id],
        "flux":         [payload["flux"]],
        "flux_err":     [payload["flux_err"]],
        "mask":         [payload["mask"]],
        "wavelength":   [payload["wavelength"]],
        "chips_pixels": [np.array(payload["chips_pixels"])],  # (3, 2) int array
    })


# ---------------------------------------------------------------------------
# Multi-source fetch (mirrors _fetch_epoch_photometry)
# ---------------------------------------------------------------------------


def _fetch_spectra(catalog: table.Table) -> table.Table:
    """Download spectra for all sources in a catalog and stack the results."""
    chunks = []
    for row in catalog:
        chunk = _get_spectra(str(row["apogee_id"]), str(row["telescope"]), str(row["field"]))
        if len(chunk):
            chunks.append(chunk)

    if not chunks:
        return _empty_spectra_table()
    if len(chunks) == 1:
        return chunks[0].copy()
    return table.vstack(chunks)


# ---------------------------------------------------------------------------
# Join catalog with spectra (mirrors _join_catalog_with_epoch_photometry)
# ---------------------------------------------------------------------------


def _join_catalog_with_spectra(catalog: table.Table, spectra: table.Table) -> table.Table:
    """Join a source catalog to spectra on `apogee_id`."""
    if len(spectra) == 0 or "apogee_id" not in spectra.colnames:
        return _empty_joined_table(catalog, spectra)
    return table.join(catalog, spectra, keys="apogee_id")


def _fetch_joined_spectra(catalog: table.Table) -> table.Table:
    """Fetch spectra for a catalog and return the joined table."""
    spectra = _fetch_spectra(catalog)
    return _join_catalog_with_spectra(catalog, spectra)


# ---------------------------------------------------------------------------
# Bitmask analysis
# ---------------------------------------------------------------------------


def bitmask_summary(
    apogee_id: str,
    telescope: str,
    field: str,
    bitmask_schema: dict[int, tuple[str, str]] = APOGEE_PIXMASK,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Apply a bitmask schema to a spectrum and return per-bit statistics.

    Parameters
    ----------
    apogee_id      : APOGEE source identifier
    telescope      : telescope name (e.g. 'apo25m')
    field          : field name (e.g. 'M15')
    bitmask_schema : mapping from bit index to (name, description)

    Returns
    -------
    bad      : (npix,) bool — True where any APOGEE bad bit is set
    flux_err : (npix,) float — error array with bad pixels set to BAD_PIXEL_ERR
    rows     : list of dicts keyed by Bit, Name, Description,
               Flagged [pixels], Fraction [%] — one entry per schema bit
    """
    spec     = _get_spectra(apogee_id, telescope, field)
    mask     = np.asarray(spec["mask"][0],     dtype=np.int32)
    flux_err = np.asarray(spec["flux_err"][0], dtype=float)

    bad = (mask & APOGEE_BAD_BITS) != 0
    flux_err[bad] = BAD_PIXEL_ERR

    rows = []
    for bit, (name, description) in bitmask_schema.items():
        n = int(((mask & (1 << bit)) != 0).sum())
        rows.append({
            "Bit":              bit,
            "Name":             name,
            "Description":      description,
            "Flagged [pixels]": n,
            "Fraction [%]":     round(100 * n / len(mask), 2),
        })

    return bad, flux_err, rows


# ---------------------------------------------------------------------------
# Continuum normalisation
# ---------------------------------------------------------------------------

_POLY_DEG = 4          # default polynomial degree (Ness et al. 2015 §2.3)
CHIP_OVERLAP_FRAC = 0.1  # fraction of chip width to extend continuum on each side


def pseudo_continuum_normalize(
    apogee_id: str,
    telescope: str,
    field: str,
    poly_deg: int = _POLY_DEG,
) -> dict:
    """Fetch and pseudo-continuum normalise one APOGEE spectrum.

    Loads the continuum-pixel mask from CONTINUUM_PIXELS_PATH, derives chip
    boundaries from the apStar PRIMARY header, and fits a degree-`poly_deg`
    polynomial to continuum anchor pixels on each chip independently.

    Parameters
    ----------
    apogee_id : APOGEE source identifier
    telescope : telescope name (e.g. 'apo25m')
    field     : field name (e.g. 'M15')
    poly_deg  : Chebyshev polynomial degree for the per-chip continuum fit

    Returns
    -------
    dict with keys:
      wavelength : (npix,) Angstrom
      flux       : (npix,) raw flux
      flux_err   : (npix,) uncertainty (bad pixels set to BAD_PIXEL_ERR)
      norm_flux  : (npix,) normalised flux  (NaN outside chip ranges)
      norm_err   : (npix,) normalised uncertainty
      continuum  : (npix,) estimated continuum in raw-flux units
      cont_mask  : (npix,) bool — True at pseudo-continuum anchor pixels
      chips      : list of (lo, hi) wavelength boundaries, one per chip
    """
    spec       = _get_spectra(apogee_id, telescope, field)
    wavelength = np.asarray(spec["wavelength"][0], dtype=float)
    flux       = np.asarray(spec["flux"][0],       dtype=float)
    flux_err   = np.asarray(spec["flux_err"][0],   dtype=float)

    chips_pixels = [tuple(row) for row in np.asarray(spec["chips_pixels"][0])]
    chips        = chip_wavelength_ranges(wavelength, chips_pixels)

    _cont     = np.load(CONTINUUM_PIXELS_PATH)
    cont_mask = np.asarray(_cont["continuum"], dtype=bool)

    norm_flux = np.full_like(flux,     np.nan)
    norm_err       = np.full_like(flux_err, np.nan)
    continuum      = np.full_like(flux,     np.nan)
    chip_continua  = []   # per-chip (wave_eval, cont_eval) for front+back visualization

    for lo, hi in chips:
        ext      = CHIP_OVERLAP_FRAC * (hi - lo)
        chip_pix = (wavelength >= lo)       & (wavelength <= hi)       & np.isfinite(flux)
        eval_pix = (wavelength >= lo - ext) & (wavelength <= hi + ext)
        if chip_pix.sum() == 0:
            chip_continua.append((np.array([]), np.array([])))
            continue

        anchor = chip_pix & cont_mask & (flux_err < BAD_PIXEL_ERR / 2)
        if anchor.sum() < poly_deg + 1:
            chip_continua.append((np.array([]), np.array([])))
            continue

        w_mid    = wavelength[chip_pix].mean()
        w_scale  = wavelength[chip_pix].std()
        x_eval   = (wavelength[eval_pix] - w_mid) / w_scale
        x_chip   = (wavelength[chip_pix] - w_mid) / w_scale
        x_anchor = (wavelength[anchor]   - w_mid) / w_scale

        coeffs    = np.polynomial.chebyshev.chebfit(x_anchor, flux[anchor], poly_deg)
        cont_eval = np.polynomial.chebyshev.chebval(x_eval,  coeffs)
        cont_chip = np.polynomial.chebyshev.chebval(x_chip,  coeffs)

        continuum[chip_pix] = cont_chip
        chip_continua.append((wavelength[eval_pix], cont_eval))
        norm_flux[chip_pix] = flux[chip_pix]     / cont_chip
        norm_err[chip_pix]  = flux_err[chip_pix] / cont_chip

    return {
        "wavelength":    wavelength,
        "flux":          flux,
        "flux_err":      flux_err,
        "norm_flux":     norm_flux,
        "norm_err":      norm_err,
        "continuum":     continuum,
        "cont_mask":     cont_mask,
        "chips":         chips,
        "chip_continua": chip_continua,
    }
