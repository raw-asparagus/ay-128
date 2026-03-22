"""Gaia zero-point constants extracted to break circular imports."""

from importlib import resources

from astropy.io import ascii

_ZP_DIR = resources.files("ugdatalab.data") / "GaiaEDR3_passbands_zeropoints_version2"
_ZEROPT = ascii.read(str(_ZP_DIR / "zeropt.dat"), readme=str(_ZP_DIR / "ReadMe"))
_ZEROPT_VEGAMAG = _ZEROPT[_ZEROPT["System"] == "VEGAMAG"]

ZP_G = float(_ZEROPT_VEGAMAG["GZp"][0])
ZP_ERR_G = float(_ZEROPT_VEGAMAG["e_GZp"][0])
