from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import shorten

from astropy.table import Table, vstack
from astroquery.sdss import SDSS
import numpy as np

from ugdatalab.models.cache import _cache_stable
from ugdatalab.models.gaia import _as_float_array


ALLSTAR_DR17_QUERY_TEMPLATE = """
SELECT
    star.apogee_id AS APOGEE_ID,
    star.apstar_id AS APSTAR_ID,
    star.location_id AS LOCATION_ID,
    star.field AS FIELD,
    star.ra AS RA,
    star.dec AS DEC,
    star.glon AS GLON,
    star.glat AS GLAT,
    star.nvisits AS NVISITS,
    star.snr AS SNR,
    star.starflag AS STARFLAG,
    star.andflag AS ANDFLAG,
    star.vhelio_avg AS VHELIO_AVG,
    star.vscatter AS VSCATTER,
    star.verr AS VERR,
    aspcap.aspcap_id AS ASPCAP_ID,
    aspcap.aspcapflag AS ASPCAPFLAG,
    aspcap.teff AS TEFF,
    aspcap.teff_err AS TEFF_ERR,
    aspcap.logg AS LOGG,
    aspcap.logg_err AS LOGG_ERR,
    aspcap.m_h AS M_H,
    aspcap.m_h_err AS M_H_ERR,
    aspcap.fe_h AS FE_H,
    aspcap.fe_h_err AS FE_H_ERR,
    aspcap.mg_fe AS MG_FE,
    aspcap.mg_fe_err AS MG_FE_ERR,
    aspcap.si_fe AS SI_FE,
    aspcap.si_fe_err AS SI_FE_ERR
FROM apogeeStar AS star
LEFT JOIN aspcapStar AS aspcap
    ON aspcap.apstar_id = star.apstar_id
WHERE {ra_clause}
"""


_RA_CHUNKS = (
    (0.0, 90.0),
    (90.0, 180.0),
    (180.0, 270.0),
    (270.0, 360.0),
)


def _run_sdss_sql(query: str) -> Table:
    """Run an SDSS SQL query and raise a readable error on non-CSV responses."""
    response = SDSS.query_sql_async(query, data_release=17, cache=False)
    response.raise_for_status()

    text = response.text.lstrip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_data_line = next((line for line in lines if not line.startswith("#")), "")
    if not text or first_data_line.startswith("<") or "," not in first_data_line:
        snippet = shorten(" ".join(text.split()), width=240, placeholder="...")
        raise RuntimeError(f"SDSS SkyServer returned a non-CSV response: {snippet}")

    return Table.read(text, format="ascii.csv", comment="#")


def _query_sdss_allstar_chunk(ra_min: float, ra_max: float, *, inclusive_upper: bool = False) -> Table:
    """Query one RA chunk of the APOGEE DR17 catalog through astroquery."""
    upper_op = "<=" if inclusive_upper else "<"
    query = ALLSTAR_DR17_QUERY_TEMPLATE.format(
        ra_clause=f"star.ra >= {ra_min:.1f} AND star.ra {upper_op} {ra_max:.1f}"
    )
    return _run_sdss_sql(query)


def _query_sdss_allstar() -> Table:
    """Query the APOGEE DR17 catalog through astroquery/SDSS SkyServer in chunks."""
    tables = []
    for i, (ra_min, ra_max) in enumerate(_RA_CHUNKS):
        tables.append(
            _query_sdss_allstar_chunk(
                ra_min,
                ra_max,
                inclusive_upper=(i == len(_RA_CHUNKS) - 1),
            )
        )
    return vstack(tables, metadata_conflicts="silent")


def _metallicity_column_name(data: Table) -> str:
    """Return the APOGEE metallicity column used for [Fe/H]."""
    if "FE_H" in data.colnames:
        return "FE_H"
    if "M_H" in data.colnames:
        return "M_H"
    raise KeyError("SDSS allStar table is missing both 'FE_H' and 'M_H'.")


@_cache_stable(module="ugdatalab.sdss")
def get_sdss() -> Table:
    """Return the full APOGEE DR17 catalog before any cuts."""
    return _query_sdss_allstar()


@_cache_stable(module="ugdatalab.sdss")
def get_sdss_quality() -> Table:
    """
    Return APOGEE stars with labels for Teff, log g, [Fe/H], [Mg/Fe], [Si/Fe]
    and SNR >= 50, as reported in the allStar catalog.
    """
    data = get_sdss()
    feh_col = _metallicity_column_name(data)

    mask = np.ones(len(data), dtype=bool)
    for name in ("TEFF", "LOGG", feh_col, "MG_FE", "SI_FE"):
        mask &= np.isfinite(_as_float_array(data[name]))

    mask &= _as_float_array(data["SNR"]) >= 50.0
    return data[mask]


@dataclass
class SDSSData:
    """Fetches the full cached APOGEE allStar catalog."""
    data: Table = field(init=False, repr=False)

    def __post_init__(self):
        self.data = get_sdss()


class SDSSQuality(SDSSData):
    """Fetches the cached APOGEE allStar subset after the quality cuts."""

    def __post_init__(self):
        self.data = get_sdss_quality()
