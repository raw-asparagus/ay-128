"""Gaia/AllWISE photometric loader.

WISE-specific pipeline stages (``WISECompletenessCut``,
``MarreseOneToOneMatchCut``, ``W2PhotometryQualityCut``,
``AddW2PhotometryColumns``) live in
:mod:`ugdatalab.models.gaia.wise_pipeline`. Shared parallax/latitude
cuts live in :mod:`ugdatalab.models.gaia.pipeline`.
"""

from dataclasses import dataclass

from astropy import table

from ugdatalab.models.base import Data
from ugdatalab.models.gaia.constants import _WISE_SCHEMA
from ugdatalab.models.gaia.gaia import _get_gaia
from ugdatalab.models.gaia.pipeline import AttachRepresentativePeriod
from ugdatalab.utils.tables import _sanitize_table


@dataclass
class WISEData(Data):
    """Fetch and cache a Gaia/AllWISE join query result.

    Parameters
    ----------
    query : str
        Gaia/AllWISE join ADQL query string.
    pipeline : Compose, keyword-only
        Inherited from :class:`~ugdatalab.models.base.Data`. Pipeline of
        WISE quality cuts and column augmentations applied immediately
        after fetch + sanitize. Default ``Compose([])`` — no
        transformations. See
        :mod:`ugdatalab.models.gaia.wise_pipeline` for available cuts
        and augmentations.

    Attributes
    ----------
    data : astropy.table.Table
        Sanitized (and cut) table with the RR Lyrae representative
        period column attached.
    """
    _required_stages = (AttachRepresentativePeriod(),)
    query: str

    def _fetch(self) -> table.Table:
        """Run the cached Gaia ADQL query (Gaia/AllWISE join)."""
        return _get_gaia(self.query)

    def _sanitize(self, raw: table.Table) -> None:
        """Coerce columns to the WISE schema in place."""
        _sanitize_table(raw, _WISE_SCHEMA)
