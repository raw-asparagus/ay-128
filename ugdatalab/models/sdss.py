from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy import table
from astroquery.sdss import SDSS

from ugdatalab.models.cache import _cache_stable


_SDSS_FLOAT_COLUMNS = (
    "teff",
    "logg",
    "fe_h",
    "mg_fe",
    "si_fe",
    "snr",
)

_ASPCAP_NULL = -9999.0
_LABEL_COLS  = ("teff", "logg", "fe_h", "mg_fe", "si_fe")

_SDSS_STRING_COLUMNS = (
    "apogee_id",
    "telescope",
    "field",
    "apstar_id",
)


@_cache_stable(module="ugdatalab.sdss")
def get_sdss(query):
    return SDSS.query_sql(query)


def _sanitize_stars(data: table.Table) -> table.Table:
    """Clean raw get_sdss output: decode byte strings, strip whitespace, cast numerics."""
    out = data.copy()

    for name in out.colnames:
        col = out[name]
        if col.dtype.kind in ("S", "O"):
            try:
                decoded = np.asarray(
                    [v.decode() if isinstance(v, bytes) else str(v) for v in col]
                )
                out[name] = np.char.strip(decoded)
            except Exception:
                pass

    for name in _SDSS_STRING_COLUMNS:
        if name in out.colnames:
            out[name] = np.char.strip(np.asarray(out[name], dtype=str))

    for name in _SDSS_FLOAT_COLUMNS:
        if name in out.colnames:
            values = np.ma.asarray(out[name], dtype=float)
            out[name] = np.asarray(np.ma.filled(values, np.nan), dtype=float)

    return out


@_cache_stable(module="ugdatalab.sdss")
def _get_sdss_quality(query):
    raw  = _sanitize_stars(get_sdss(query))
    mask = np.ones(len(raw), dtype=bool)

    # drop rows where any label is ASPCAP null or NaN
    for name in _LABEL_COLS:
        if name in raw.colnames:
            col   = np.asarray(raw[name], dtype=float)
            mask &= (col != _ASPCAP_NULL) & np.isfinite(col)

    # quality and science cuts
    if "snr"  in raw.colnames: mask &= np.asarray(raw["snr"],  dtype=float) >= 50.0
    if "logg" in raw.colnames: mask &= np.asarray(raw["logg"], dtype=float) <= 4.0
    if "teff" in raw.colnames: mask &= np.asarray(raw["teff"], dtype=float) <= 5700.0
    if "fe_h" in raw.colnames: mask &= np.asarray(raw["fe_h"], dtype=float) >= -1.0

    return raw[mask]


@dataclass
class SDSSData:
    """Fetches and caches SDSS/APOGEE query results."""
    query: str
    include_spectra: bool = False
    data: table.Table = field(init=False, repr=False)
    spectra: table.Table | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.data = _get_sdss_quality(self.query)
        self._load_spectra()

    def _load_spectra(self):
        if not self.include_spectra:
            self.spectra = None
            return

        from ugdatalab.spectra import _fetch_joined_spectra

        self.spectra = _fetch_joined_spectra(self.data)
