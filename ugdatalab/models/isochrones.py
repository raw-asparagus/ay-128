import numpy as np
import pandas as pd

from ugdatalab.models.cache import _cache_stable


@_cache_stable(module="ugdatalab.isochrones")
def _get_mist_isochrone(age_gyr: float, feh: float) -> pd.DataFrame:
    """Fetch a MIST isochrone for the given age and metallicity.

    Parameters
    ----------
    age_gyr : float
        Stellar age in Gyr (e.g. 6.0 for 6 Gyr).
    feh : float
        Metallicity [Fe/H] in dex.

    Returns
    -------
    pd.DataFrame
        Isochrone table with columns including ``Teff``, ``logg``,
        ``initial_mass``, ``logTeff``, and ``phase``.
    """
    from isochrones import get_ichrone

    mist = get_ichrone("mist")
    return mist.isochrone(age=np.log10(age_gyr * 1e9), feh=feh)
