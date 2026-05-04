"""APOGEE DR17 SQL query loader and Cannon-ready training-set filter."""

from dataclasses import dataclass, field

import numpy as np
from astropy import table

from ugdatalab.utils.cache import cache_stable
from ugdatalab.utils.tables import _sanitize_table
from ugdatalab.models.apogee.constants import (
    _FILLER_VALUE,
    _APOGEE_SCHEMA,
)


@cache_stable(module="ugdatalab.apogee")
def _get_apogee_allstar(query: str) -> table.Table:
    """Query APOGEE DR17 allStar catalog with a SQL query string."""
    from astroquery.sdss import SDSS

    return SDSS.query_sql(query, data_release=17)


@dataclass
class APOGEEData:
    """Fetch and cache an APOGEE DR17 SQL query result.

    Parameters
    ----------
    query : str
        SDSS DR17 ADQL/SQL query string.

    Attributes
    ----------
    data : astropy.table.Table
        Sanitized query result.
    """
    query: str
    data: table.Table = field(init=False, repr=False)

    def __post_init__(self):
        data = _get_apogee_allstar(self.query)
        _sanitize_table(data, _APOGEE_SCHEMA)
        self.data = data


class APOGEETrainingSet(APOGEEData):
    """Quality-filtered APOGEE training set for The Cannon.

    Accepts sources satisfying:
      - All 5 labels finite and != -9999
      - SNR > 50
      - logg <= 4 and Teff <= 5700 (giants only)
      - [Fe/H] >= -1

    Parameters
    ----------
    source : APOGEEData
        Catalog to filter.
    """
    def __init__(self, source: APOGEEData):
        self.query = source.query

        label_cols = ["teff", "logg", "fe_h", "mg_fe", "si_fe"]
        mask = np.ones(len(source.data), dtype=bool)
        for col in label_cols:
            vals = source.data[col]
            mask &= np.isfinite(vals) & (vals != _FILLER_VALUE)

        mask &= source.data["snr"] > 50
        mask &= source.data["logg"] <= 4
        mask &= source.data["teff"] <= 5700
        mask &= source.data["fe_h"] >= -1

        self.data = source.data[mask]
