APOGEE_BAD_PIXMASK_BITS = [0, 1, 2, 3, 4, 5, 6, 7, 12]

LABEL_NAMES = ["TEFF", "LOGG", "FE_H", "MG_FE", "SI_FE"]
LABEL_LATEX = [
    r"$T_{\rm eff}$", r"$\log g$", r"$[\mathrm{Fe/H}]$",
    r"$[\mathrm{Mg/Fe}]$", r"$[\mathrm{Si/Fe}]$",
]
N_LABELS = 5
_FILLER_VALUE = -9999.0
_APOGEE_SENTINEL_ERROR = 1e6

APOGEE_DR17_URL = (
    "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars"
)

_APOGEE_SCHEMA = {
    str: ["apogee_id", "field", "telescope"],
    float: ["teff", "logg", "fe_h", "mg_fe", "si_fe", "snr", "vhelio_avg"],
    int: ["nvisits", "starflag", "aspcapflag"],
}
