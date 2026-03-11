from dataclasses import dataclass, field
from pathlib import Path

from astroquery.gaia import Gaia
from astropy import table
from astropy.io import ascii
from ugdatalab.models.cache import _cache_stable

import numpy as np


# ---------------------------------------------------------------------------
# Zero-point data
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.parent
ZP_DIR    = _HERE / "GaiaEDR3_passbands_zeropoints_version2"
ZP_README = str(ZP_DIR / "ReadMe")

ZEROPT         = ascii.read(str(ZP_DIR / "zeropt.dat"), readme=ZP_README)
ZEROPT_VEGAMAG = ZEROPT[ZEROPT["System"] == "VEGAMAG"]

ZP_G     = float(ZEROPT_VEGAMAG["GZp"][0])
ZP_ERR_G = float(ZEROPT_VEGAMAG["e_GZp"][0])

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _add_gaia_photometry_columns(data: table.Table) -> table.Table:
    sigma_G_meas    = (2.5 / np.log(10)) * np.abs(
        data["phot_g_mean_flux_error"] / data["phot_g_mean_flux"]
    )
    data["sigma_G"] = np.sqrt(sigma_G_meas**2 + ZP_ERR_G**2)

    if "parallax" in data.colnames:
        omega            = data["parallax"]
        data["mu"]       = 10 - 5 * np.log10(omega)
        data["sigma_mu"] = 5 * data["parallax_error"] / (omega * np.log(10))
        data["M_G"]      = data["phot_g_mean_mag"] - data["mu"]
        data["sigma_M"]  = np.sqrt(data["sigma_G"]**2 + data["sigma_mu"]**2)
    return data


@_cache_stable(module="ugdatalab.gaia")
def get_gaia(query):
    job  = Gaia.launch_job_async(query)
    data = job.get_results()
    return data


@_cache_stable(module="ugdatalab.gaia")
def get_gaia_quality(query):
    raw  = get_gaia(query)
    poe  = raw["parallax_over_error"]
    b    = raw["b"]
    data = raw[(poe > 5) & (np.abs(b) > 30)]
    data = _add_gaia_photometry_columns(data)

    pf   = data["pf"].filled(np.nan)
    p1_o = data["p1_o"].filled(np.nan)

    data["period"]  = np.where(np.isfinite(pf), pf, p1_o)
    data["is_rrab"] = np.isfinite(pf)  & ~np.isfinite(p1_o)
    data["is_rrc"]  = ~np.isfinite(pf) &  np.isfinite(p1_o)
    return data

# ---------------------------------------------------------------------------
# GaiaData and subclasses
# ---------------------------------------------------------------------------

@dataclass
class GaiaData:
    """Fetches and caches raw Gaia query results."""
    query: str
    include_lightcurve: bool = False
    data: table.Table = field(init=False, repr=False)
    lightcurve_data: table.Table | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.data = get_gaia(self.query)
        self._load_lightcurves()

    def _load_lightcurves(self):
        from ugdatalab.lightcurves import fetch_joined_epoch_photometry

        if not self.include_lightcurve:
            self.lightcurve_data = None
            return

        self.lightcurve_data = fetch_joined_epoch_photometry(
            self.data,
        )


class GaiaQuality(GaiaData):
    """Fetches and caches the quality-filtered Gaia sample with derived columns."""

    def __post_init__(self):
        self.data = get_gaia_quality(self.query)
        self._load_lightcurves()


class Local(GaiaQuality):
    def __init__(self, source: GaiaQuality):
        self.query = source.query
        self.include_lightcurve = False
        self.data = source.data[source.data["parallax"] > 0.25]
        self.lightcurve_data = None


class StrictGBPRP(GaiaQuality):
    """Strict BP/RP signal-to-noise cut.

    Accepts sources satisfying:
      phot_g_mean_flux_over_error > 5
      phot_bp_mean_flux_over_error > 5
      phot_rp_mean_flux_over_error > 5
    """
    def __init__(self, source: GaiaQuality):
        self.query = source.query
        self.include_lightcurve = False
        mask       = (
            (source.data["phot_g_mean_flux_over_error"] > 5) &
            (source.data["phot_bp_mean_flux_over_error"] > 5) &
            (source.data["phot_rp_mean_flux_over_error"] > 5)
        )
        self.data  = source.data[mask]
        self.lightcurve_data = None


class Cut1(GaiaQuality):
    """RUWE quality cut.

    Accepts sources satisfying:
      ruwe < 1.2 * max(1, exp(-0.2 * (G - 19.5)))
    """
    def __init__(self, source: GaiaQuality):
        self.query = source.query
        self.include_lightcurve = False
        G          = np.asarray(source.data["phot_g_mean_mag"])
        u_max      = 1.2 * np.maximum(1, np.exp(-0.2 * (G - 19.5)))
        mask       = np.asarray(source.data["ruwe"]) < u_max
        self.data = source.data[mask]
        self.lightcurve_data = None


class Cut2(GaiaQuality):
    """BP/RP flux excess factor quality cut.

    Accepts sources satisfying:
      1.0 + 0.015*(bp_rp)^2 < phot_bp_rp_excess_factor < 1.3 + 0.06*(bp_rp)^2
    """
    def __init__(self, source: GaiaQuality):
        self.query = source.query
        self.include_lightcurve = False
        bp_rp      = np.asarray(source.data["bp_rp"])
        E          = np.asarray(source.data["phot_bp_rp_excess_factor"])
        mask       = (
            (E > 1.0 + 0.015 * bp_rp**2) &
            (E < 1.3 + 0.06  * bp_rp**2)
        )
        self.data = source.data[mask]
        self.lightcurve_data = None
