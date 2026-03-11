from typing import Iterable

import numpy as np
from astroquery.gaia import Gaia
from astropy import table
from astropy.timeseries import LombScargle

from ugdatalab.models.cache import _cache_stable
from ugdatalab.models.gaia import ZP_ERR_G, ZP_G

DEFAULT_PERIOD_MIN = 0.2
DEFAULT_PERIOD_MAX = 1.2
EPOCH_DATA_RELEASE = "Gaia DR3"
EPOCH_FILE_FORMAT = "xml"


def _as_float_array(column) -> np.ndarray:
    values = np.ma.asarray(column, dtype=float)
    return np.asarray(np.ma.filled(values, np.nan), dtype=float)

def _epoch_payload(datalink: dict, source_id: int):
    key = f"EPOCH_PHOTOMETRY-{EPOCH_DATA_RELEASE} {int(source_id)}.{EPOCH_FILE_FORMAT}"
    if key in datalink:
        return datalink[key]

    source_id_str = str(int(source_id))
    for dl_key, payload in datalink.items():
        if source_id_str in dl_key:
            return payload
    raise KeyError(f"Could not find epoch photometry for source_id={source_id}.")


def _empty_epoch_table() -> table.Table:
    return table.Table({"source_id": np.asarray([], dtype=np.int64)})


def _empty_joined_table(catalog: table.Table, epoch_data: table.Table) -> table.Table:
    res = table.Table()
    for name in catalog.colnames:
        res[name] = np.asarray(catalog[name])[:0]
    for name in epoch_data.colnames:
        if name not in res.colnames:
            res[name] = np.asarray(epoch_data[name])[:0]
    return res


@_cache_stable(module="ugdatalab.gaia")
def get_epoch_photometry(source_id: int) -> table.Table:
    """Download Gaia epoch photometry for one source."""
    source_id = int(source_id)
    datalink = Gaia.load_data(
        ids=[source_id],
        data_release=EPOCH_DATA_RELEASE,
        retrieval_type="EPOCH_PHOTOMETRY",
        data_structure="INDIVIDUAL",
        linking_parameter="SOURCE_ID",
    )
    payload = _epoch_payload(datalink, source_id)
    data = table.vstack([chunk.to_table() for chunk in payload])
    data["source_id"] = source_id
    return data


def fetch_epoch_photometry(source_ids: Iterable[int]) -> table.Table:
    """Download epoch photometry for many Gaia sources and stack the results."""
    source_ids = tuple(int(source_id) for source_id in source_ids)
    chunks = []
    for source_id in source_ids:
        chunk = get_epoch_photometry(source_id)
        if len(chunk):
            chunks.append(chunk)

    if not chunks:
        return _empty_epoch_table()
    if len(chunks) == 1:
        return chunks[0].copy()
    return table.vstack(chunks)


def clean_epoch_photometry(data: table.Table) -> table.Table:
    """Drop rows with missing Gaia G epoch time, flux, or magnitude values."""
    if len(data) == 0:
        return data.copy()

    mask = (
        np.isfinite(_as_float_array(data["g_transit_time"]))
        & np.isfinite(_as_float_array(data["g_transit_mag"]))
        & np.isfinite(_as_float_array(data["g_transit_flux"]))
        & np.isfinite(_as_float_array(data["g_transit_flux_error"]))
    )
    return data[mask]


def join_catalog_with_epoch_photometry(catalog: table.Table, epoch_data: table.Table) -> table.Table:
    """Join a source catalog to epoch photometry on `source_id` and clean the result."""
    if len(epoch_data) == 0 or "source_id" not in epoch_data.colnames:
        return _empty_joined_table(catalog, epoch_data)
    return clean_epoch_photometry(table.join(catalog, epoch_data, keys="source_id"))


def fetch_joined_epoch_photometry(catalog: table.Table) -> table.Table:
    """Fetch epoch photometry for a catalog and return the cleaned joined table."""
    epoch_data = fetch_epoch_photometry(catalog["source_id"])
    return join_catalog_with_epoch_photometry(catalog, epoch_data)


def _add_g_transit_mag_error(data: table.Table) -> table.Table:
    """Attach `g_transit_mag_err` computed from flux errors and the G zero point."""
    flux = _as_float_array(data["g_transit_flux"])
    flux_err = _as_float_array(data["g_transit_flux_error"])
    meas_err = (2.5 / np.log(10.0)) * np.abs(flux_err / flux)
    data["g_transit_mag_err"] = np.sqrt(meas_err**2 + float(ZP_ERR_G) ** 2)
    return data


def _get_mean_mags(epoch_table: table.Table) -> tuple[float, float]:
    flux = _as_float_array(epoch_table["g_transit_flux"])
    flux_err = _as_float_array(epoch_table["g_transit_flux_error"])

    mean_flux = float(np.mean(flux))
    mean_flux_err = float(np.sqrt(np.sum(flux_err**2)) / len(flux_err))
    mean_g_mag = -2.5 * np.log10(mean_flux) + float(ZP_G)
    mean_meas_err = (2.5 / np.log(10.0)) * (mean_flux_err / mean_flux)
    mean_g_mag_err = float(np.sqrt(mean_meas_err**2 + float(ZP_ERR_G) ** 2))
    return mean_g_mag, mean_g_mag_err


def attach_flux_mean_magnitudes(data: table.Table) -> table.Table:
    """Attach per-epoch magnitude errors and repeated per-source flux means."""
    res = _add_g_transit_mag_error(data)
    source_column = np.asarray(res["source_id"], dtype=np.int64)
    source_ids = np.unique(source_column)

    lookup = {}
    for source_id in np.asarray(list(source_ids), dtype=np.int64):
        lookup[int(source_id)] = _get_mean_mags(res[source_column == source_id])

    res["mean_g_transit_mag"] = [lookup[int(source_id)][0] for source_id in res["source_id"]]
    res["mean_g_transit_mag_err"] = [lookup[int(source_id)][1] for source_id in res["source_id"]]
    return res


def attach_periodogram_periods(data: table.Table) -> table.Table:
    """Attach repeated per-source Lomb-Scargle periods to a joined epoch table."""
    source_ids, periods = estimate_periods_from_epoch_photometry(data)
    lookup = {int(source_id): float(period) for source_id, period in zip(source_ids, periods)}
    data["period_ls"] = [lookup[int(source_id)] for source_id in data["source_id"]]
    return data


def lomb_scargle_periodogram(target: table.Table) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute a Lomb-Scargle periodogram for one source's epoch photometry."""
    epoch = _as_float_array(target["g_transit_time"])
    values = _as_float_array(target["g_transit_flux"])
    value_err = _as_float_array(target["g_transit_flux_error"])

    freqs, power = LombScargle(epoch, values, value_err).autopower(
        minimum_frequency=1.0 / float(DEFAULT_PERIOD_MAX),
        maximum_frequency=1.0 / float(DEFAULT_PERIOD_MIN),
    )

    periods = 1.0 / np.asarray(freqs, dtype=float)
    power = np.asarray(power, dtype=float)
    order = np.argsort(power)[::-1]
    periods = periods[order]
    power = power[order]
    max_power = float(np.max(power))
    near_max = np.where(power >= 0.98 * max_power)[0]
    if len(near_max) == 0:
        best_period = float(periods[int(np.argmax(power))])
    else:
        best_period = float(periods[int(near_max[np.argmax(periods[near_max])])])
    return periods, power, best_period


def estimate_periods_from_epoch_photometry(
    data: table.Table,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the best Lomb-Scargle period for each source in a joined table."""
    source_column = np.asarray(data["source_id"], dtype=np.int64)
    source_ids = np.unique(source_column)
    periods = np.empty(len(source_ids), dtype=float)
    for i, source_id in enumerate(source_ids):
        _, _, periods[i] = lomb_scargle_periodogram(data[source_column == source_id])
    return source_ids, periods
