"""Gaia / WISE schemas, default period bounds, datalink URL, and zero points."""

from astropy.io import ascii
import numpy as np

from ugdatalab.data._fetch import gaia_zeropoints_dir

RRLYRAE_PERIOD_MIN = 0.2
RRLYRAE_PERIOD_MAX = 1.2

_GAIA_DATALINK_URL = "https://gea.esac.esa.int/data-server/data"

_GAIA_SCHEMA = {
    np.int64: ["source_id"],
    int: ["num_clean_epochs_g"],
    str: ["best_classification"],
    float: [
        "l", "b", "pf", "pf_error", "p1_o", "p1_o_error", "int_average_g",
        "parallax", "parallax_error", "parallax_over_error",
        "phot_g_mean_flux", "phot_g_mean_flux_error", "phot_g_mean_mag",
        "bp_rp", "phot_g_mean_flux_over_error",
        "phot_bp_mean_flux_over_error",
        "phot_rp_mean_flux_over_error", "phot_bp_rp_excess_factor", "ruwe",
    ],
}

_EPOCH_SCHEMA = {
    float: ["g_transit_time", "g_transit_mag", "g_transit_flux", "g_transit_flux_error"],
}

_WISE_SCHEMA = {
    np.int64: ["source_id"],
    str: ["best_classification", "ph_qual", "cc_flags"],
    float: [
        "l", "b", "pf", "pf_error", "p1_o", "p1_o_error",
        "parallax", "parallax_error",
        "phot_g_mean_mag", "bp_rp", "w2mpro", "w2mpro_error",
        "allwise_oid", "number_of_mates", "number_of_neighbours", "ext_flag",
    ],
}

# ---------------------------------------------------------------------------
# Gaia zero-point constants
# ---------------------------------------------------------------------------

_ZP_DIR = gaia_zeropoints_dir()
_ZEROPT = ascii.read(str(_ZP_DIR / "zeropt.dat"), readme=str(_ZP_DIR / "ReadMe"))
_ZEROPT_VEGAMAG = _ZEROPT[_ZEROPT["System"] == "VEGAMAG"]

ZP_G = float(_ZEROPT_VEGAMAG["GZp"][0])
ZP_ERR_G = float(_ZEROPT_VEGAMAG["e_GZp"][0])
