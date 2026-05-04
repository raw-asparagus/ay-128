"""Gaia DR3 catalog loading; quality cuts live in ``cuts.py``."""

from dataclasses import dataclass, field

from astropy import table

from ugdatalab.models.base import Data
from ugdatalab.utils.cache import cache_stable
from ugdatalab.models.gaia.constants import _GAIA_SCHEMA
from ugdatalab.models.gaia.pipeline import AttachRepresentativePeriod
from ugdatalab.models.gaia.lightcurves import (
    _fetch_joined_epoch_photometry, _attach_derived_epoch_columns,
    _attach_periodogram_periods, _attach_fourier_mean_magnitudes,
)
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
    """Fetch and cache raw Gaia query results, optionally with epoch photometry.

    Parameters
    ----------
    query : str
        Gaia ADQL query string.
    include_lightcurve : bool
        If True, also download epoch photometry and attach derived
        columns. Default ``False`` — light-curve fetching is expensive
        and only needed for variability / period analysis.
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
    lightcurves : astropy.table.Table or None
        Epoch photometry joined on ``source_id`` when
        ``include_lightcurve``.
    """
    _required_stages = (AttachRepresentativePeriod(),)
    query: str
    include_lightcurve: bool = False
    lightcurves: table.Table | None = field(default=None, init=False, repr=False)

    def _fetch(self) -> table.Table:
        """Run the cached Gaia ADQL query."""
        return _get_gaia(self.query)

    def _sanitize(self, raw: table.Table) -> None:
        """Coerce columns to the Gaia schema in place."""
        _sanitize_table(raw, _GAIA_SCHEMA)

    def _post_pipeline(self) -> None:
        """Optionally fetch and process epoch photometry after the pipeline runs."""
        if not self.include_lightcurve:
            self.lightcurves = None
            return
        lightcurves = _fetch_joined_epoch_photometry(self.data)
        _attach_derived_epoch_columns(lightcurves)
        _attach_periodogram_periods(lightcurves)
        _attach_fourier_mean_magnitudes(lightcurves)
        self.lightcurves = lightcurves
