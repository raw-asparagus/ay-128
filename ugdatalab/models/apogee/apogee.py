"""APOGEE DR17 SQL query loader; quality cuts live in ``cuts.py``."""

from dataclasses import dataclass

from astropy import table

from ugdatalab.models.base import Data
from ugdatalab.utils.cache import cache_stable
from ugdatalab.utils.tables import _sanitize_table
from ugdatalab.models.apogee.constants import _APOGEE_SCHEMA


@cache_stable(module="ugdatalab.apogee")
def _get_apogee_allstar(query: str) -> table.Table:
    """Query APOGEE DR17 allStar catalog with a SQL query string."""
    from astroquery.sdss import SDSS

    return SDSS.query_sql(query, data_release=17)


@dataclass
class APOGEEData(Data):
    """Fetch and cache an APOGEE DR17 SQL query result.

    Parameters
    ----------
    query : str
        SDSS DR17 ADQL/SQL query string.
    pipeline : Compose, keyword-only
        Inherited from :class:`~ugdatalab.models.base.Data`. Pipeline of
        quality cuts and column augmentations applied immediately after
        fetch + sanitize. Default ``Compose([])`` — no transformations.
        See :mod:`ugdatalab.models.apogee.pipeline` for available
        cuts and augmentations.

    Attributes
    ----------
    data : astropy.table.Table
        Sanitized (and cut) query result.
    """
    query: str

    def _fetch(self) -> table.Table:
        """Run the cached APOGEE DR17 SQL query."""
        return _get_apogee_allstar(self.query)

    def _sanitize(self, raw: table.Table) -> None:
        """Coerce columns to the APOGEE schema in place."""
        _sanitize_table(raw, _APOGEE_SCHEMA)
