"""Gaia DR3 epoch-photometry loader.

Pipeline stages (``LightcurveCompletenessCut``,
``AttachDerivedEpochColumns``, ``AttachPeriodogramPeriods``,
``AttachFourierMeanMagnitudes``) live in
:mod:`ugdatalab.models.gaia.lightcurves_pipeline`.
"""

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

import requests
from astropy import table
from astropy.io.votable import parse as parse_votable
from tqdm.auto import tqdm

from ugdatalab.models.base import Data
from ugdatalab.models.gaia.gaia import GaiaData
from ugdatalab.models.gaia.constants import _GAIA_DATALINK_URL, _EPOCH_SCHEMA
from ugdatalab.models.gaia.lightcurves_pipeline import (
    AttachDerivedEpochColumns,
    AttachFourierMeanMagnitudes,
    AttachPeriodogramPeriods,
    LightcurveCompletenessCut,
)
from ugdatalab.utils.cache import cache_stable
from ugdatalab.utils.tables import _sanitize_table


# ---------------------------------------------------------------------------
# Epoch photometry I/O
# ---------------------------------------------------------------------------

@cache_stable(module="ugdatalab.gaia")
def _get_epoch_photometry(source_id: int) -> table.Table:
    """Download Gaia DR3 epoch photometry for one source via direct HTTP."""
    resp = requests.post(
        _GAIA_DATALINK_URL,
        data={
            "RETRIEVAL_TYPE": "EPOCH_PHOTOMETRY",
            "ID": str(source_id),
            "RELEASE": "Gaia DR3",
            "DATA_STRUCTURE": "INDIVIDUAL",
            "FORMAT": "votable",
            "USE_ZIP_ALWAYS": "true",
        },
    )
    resp.raise_for_status()

    tables = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".xml"):
                with zf.open(name) as f:
                    vot = parse_votable(io.BytesIO(f.read()))
                    tables.extend(t.to_table() for t in vot.iter_tables())

    if not tables:
        raise KeyError(f"Could not find epoch photometry for source_id={source_id}.")

    for t in tables:
        for col in t.columns.values():
            col.unit = None

    data = table.vstack(tables)
    data["source_id"] = source_id
    return data


def _fetch_epoch_photometry(
    source_ids: Iterable[int], max_workers: int = 8,
) -> table.Table:
    """Download epoch photometry for many Gaia sources in parallel and vstack."""
    source_ids = tuple(source_ids)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_get_epoch_photometry, sid): sid for sid in source_ids}
        chunks = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Epoch photometry"):
            chunks.append(future.result())
    return table.vstack(chunks)


# ---------------------------------------------------------------------------
# GaiaLightcurves loader
# ---------------------------------------------------------------------------


@dataclass
class GaiaLightcurves(Data):
    """Fetch Gaia DR3 epoch photometry for every source in a ``GaiaData`` catalog.

    Downloads epoch photometry for every ``source_id`` in
    ``source.data``, joins on that key, and attaches per-epoch error,
    Lomb-Scargle period, and Fourier-fit mean magnitude columns via
    ``_required_stages``. Per-source downloads are cached on disk.

    Parameters
    ----------
    source : GaiaData
        Source catalog whose ``data["source_id"]`` rows drive the
        epoch-photometry download list.
    pipeline : Compose, keyword-only
        Inherited from :class:`~ugdatalab.models.base.Data`. Empty by
        default; ``_required_stages`` already attaches every column the
        lab notebooks rely on. See
        :mod:`ugdatalab.models.gaia.lightcurves_pipeline` for available
        stages.

    Attributes
    ----------
    data : astropy.table.Table
        Joined epoch-photometry table sorted by ``(source_id,
        g_transit_time)`` with derived columns attached.
    """
    _required_stages = (
        LightcurveCompletenessCut(),
        AttachDerivedEpochColumns(),
        AttachPeriodogramPeriods(),
        AttachFourierMeanMagnitudes(),
    )
    source: GaiaData

    def _fetch(self) -> table.Table:
        """Download epoch photometry for every source in ``self.source.data`` and join."""
        epoch_data = _fetch_epoch_photometry(self.source.data["source_id"])
        joined = table.join(self.source.data, epoch_data, keys="source_id")
        joined.sort(["source_id", "g_transit_time"])
        return joined

    def _sanitize(self, raw: table.Table) -> None:
        """Coerce epoch columns to the epoch schema in place."""
        _sanitize_table(raw, _EPOCH_SCHEMA)
