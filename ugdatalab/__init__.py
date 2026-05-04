"""ugdatalab — methods and models for astronomical data analysis.

Re-exports common data models (``GaiaData``, ``GaiaSample``, ``WISEData``,
``APOGEEData``, ``GalaxyZooData`` and related sample/filter classes) and
analysis methods (``FourierFit``, ``lomb_scargle``, ``cross_validate``) at
the package top level. Import as ``from ugdatalab import GaiaData``.
"""

import importlib

_LAZY_EXPORTS = {
    # ugdatalab.models.gaia
    "GaiaData":           "ugdatalab.models.gaia",
    "GaiaSample":         "ugdatalab.models.gaia",
    "Local":              "ugdatalab.models.gaia",
    "StrictG":            "ugdatalab.models.gaia",
    "StrictBPRP":         "ugdatalab.models.gaia",
    "LindegrenC1":        "ugdatalab.models.gaia",
    "LindegrenC2":        "ugdatalab.models.gaia",
    "GaiaReddening":      "ugdatalab.models.gaia",
    "StrictReddening":    "ugdatalab.models.gaia",
    "Deoutlier":          "ugdatalab.models.gaia",
    "WISEData":           "ugdatalab.models.gaia",
    "WISESample":         "ugdatalab.models.gaia",
    "RRLYRAE_PERIOD_MIN": "ugdatalab.models.gaia.lightcurves",
    "RRLYRAE_PERIOD_MAX": "ugdatalab.models.gaia.lightcurves",
    # ugdatalab.models.apogee
    "APOGEEData":         "ugdatalab.models.apogee",
    "APOGEETrainingSet":  "ugdatalab.models.apogee",
    "APOGEESpectra":      "ugdatalab.models.apogee",
    # ugdatalab.models.galaxy_zoo
    "GalaxyZooData":      "ugdatalab.models.galaxy_zoo",
    "GalaxyZooImages":    "ugdatalab.models.galaxy_zoo",
    "GalaxyZooDataset":   "ugdatalab.models.galaxy_zoo",
    # ugdatalab.methods
    "FourierFit":         "ugdatalab.methods.fourier",
    "fourier_fit":        "ugdatalab.methods.fourier",
    "phase_fold":         "ugdatalab.methods.fourier",
    "PeriodogramResult":  "ugdatalab.methods.periodogram",
    "lomb_scargle":       "ugdatalab.methods.periodogram",
    "ValidationResult":   "ugdatalab.methods.cross_validate",
    "cross_validate":     "ugdatalab.methods.cross_validate",
}


def __getattr__(name):
    try:
        module_path = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module 'ugdatalab' has no attribute {name!r}"
        ) from exc
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(_LAZY_EXPORTS) | set(globals()))
