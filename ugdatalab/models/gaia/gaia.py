"""Gaia DR3 catalog loader; pipeline stages live in ``pipeline.py``.

For epoch photometry (lightcurves), construct a separate
:class:`~ugdatalab.models.gaia.lightcurves.GaiaLightcurves` object
with this loader's instance as ``source``.
"""

from dataclasses import dataclass

from astropy import table

from ugdatalab.models.base import Data
from ugdatalab.utils.cache import cache_stable
from ugdatalab.models.gaia.constants import _GAIA_SCHEMA
from ugdatalab.models.gaia.pipeline import AttachRepresentativePeriod
from ugdatalab.utils.tables import _sanitize_table


@cache_stable(module="ugdatalab.gaia")
def _get_gaia(query):
    """Run an async Gaia ADQL query and return the result table (cached)."""
    from astroquery.gaia import Gaia

    job  = Gaia.launch_job_async(query)
    data = job.get_results()
    return data


# ---------------------------------------------------------------------------
# GaiaData
# ---------------------------------------------------------------------------

@dataclass
class GaiaData(Data):
    """Fetch and cache raw Gaia query results.

    Parameters
    ----------
    query : str
        Gaia ADQL query string.
    pipeline : Compose, keyword-only
        Inherited from :class:`~ugdatalab.models.base.Data`. Pipeline of
        quality cuts and column augmentations applied immediately after
        fetch + sanitize. Default ``Compose([])`` — no transformations.
        See :mod:`ugdatalab.models.gaia.pipeline` for available
        cuts and augmentations.

    Attributes
    ----------
    data : astropy.table.Table
        Sanitized (and cut) query result with the RR Lyrae representative
        period column attached.

    See Also
    --------
    ugdatalab.models.gaia.lightcurves.GaiaLightcurves
        Epoch-photometry loader; constructed with a :class:`GaiaData`
        instance as ``source``.
    """
    _required_stages = (AttachRepresentativePeriod(),)
    query: str

    def _fetch(self) -> table.Table:
        """Run the cached Gaia ADQL query."""
        return _get_gaia(self.query)

    def _sanitize(self, raw: table.Table) -> None:
        """Coerce columns to the Gaia schema in place."""
        _sanitize_table(raw, _GAIA_SCHEMA)
