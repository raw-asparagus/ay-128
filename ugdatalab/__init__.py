"""ugdatalab — methods and models for astronomical data analysis.

Submodules are imported on demand to avoid pulling heavy dependencies
(PyMC, astroquery) at package import time.
"""

from ugdatalab.models.cache import cache_stable
from ugdatalab.models.gaia import (
    GaiaData,
    GaiaQuality,
    Local,
    StrictGBPRP,
    LindegrenC1,
    LindegrenC2,
)
from ugdatalab.methods.fourier import FourierFit, fourier_fit, phase_fold
from ugdatalab.methods.periodogram import PeriodogramResult, lomb_scargle
from ugdatalab.methods.cross_validate import (
    HoldoutResult,
    KFoldResult,
    ValidationResult,
    holdout_validate,
    k_fold_validate,
)
from ugdatalab.models.gaia.lightcurves import DEFAULT_PERIOD_MIN, DEFAULT_PERIOD_MAX
