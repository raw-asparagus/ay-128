"""ugdatalab — methods and models for astronomical data analysis.

Re-exports common data loaders (``GaiaData``, ``WISEData``, ``APOGEEData``,
``GalaxyZooData``), composable quality cuts (Gaia/WISE/APOGEE ``*Cut`` classes,
``Compose``), and analysis methods (``FourierFit``, ``lomb_scargle``,
``cross_validate``) at the package top level. Import as
``from ugdatalab import GaiaData, Compose, HighParallaxSnrCut``.
"""

import importlib

_LAZY_EXPORTS = {
    # ugdatalab.methods.compose
    "Compose":                       "ugdatalab.methods.compose",
    # ugdatalab.models.gaia (loader + pipeline stages)
    "GaiaData":                      "ugdatalab.models.gaia",
    "AddPhotometryColumns":          "ugdatalab.models.gaia",
    "AttachInlierProbColumn":        "ugdatalab.models.gaia",
    "AttachReddening":               "ugdatalab.models.gaia",
    "AttachRepresentativePeriod":    "ugdatalab.models.gaia",
    "GaiaFiniteCut":                 "ugdatalab.models.gaia",
    "HighLatitudeCut":               "ugdatalab.models.gaia",
    "HighParallaxSnrCut":            "ugdatalab.models.gaia",
    "LindegrenC1Cut":                "ugdatalab.models.gaia",
    "LindegrenC2Cut":                "ugdatalab.models.gaia",
    "LocalCut":                      "ugdatalab.models.gaia",
    "StrictBPRPCut":                 "ugdatalab.models.gaia",
    "StrictGCut":                    "ugdatalab.models.gaia",
    "StrictReddeningCut":            "ugdatalab.models.gaia",
    "WISEData":                      "ugdatalab.models.gaia",
    "AddW2PhotometryColumns":        "ugdatalab.models.gaia",
    "MarreseOneToOneMatchCut":       "ugdatalab.models.gaia",
    "W2PhotometryQualityCut":        "ugdatalab.models.gaia",
    "WISEFiniteCut":                 "ugdatalab.models.gaia",
    "RRLYRAE_PERIOD_MIN":            "ugdatalab.models.gaia.lightcurves",
    "RRLYRAE_PERIOD_MAX":            "ugdatalab.models.gaia.lightcurves",
    # ugdatalab.models.apogee (loader + pipeline stages)
    "APOGEEData":                    "ugdatalab.models.apogee",
    "APOGEEGiantCut":                "ugdatalab.models.apogee",
    "APOGEELabelCompletenessCut":    "ugdatalab.models.apogee",
    "APOGEEMetallicityCut":          "ugdatalab.models.apogee",
    "APOGEESnrCut":                  "ugdatalab.models.apogee",
    "APOGEETempCut":                 "ugdatalab.models.apogee",
    "APOGEESpectra":                 "ugdatalab.models.apogee",
    # ugdatalab.models.galaxy_zoo
    "GalaxyZooData":                 "ugdatalab.models.galaxy_zoo",
    "GalaxyZooImages":               "ugdatalab.models.galaxy_zoo",
    "GalaxyZooDataset":              "ugdatalab.models.galaxy_zoo",
    # ugdatalab.methods
    "FourierFit":                    "ugdatalab.methods.fourier",
    "fourier_fit":                   "ugdatalab.methods.fourier",
    "phase_fold":                    "ugdatalab.methods.fourier",
    "PeriodogramResult":             "ugdatalab.methods.periodogram",
    "lomb_scargle":                  "ugdatalab.methods.periodogram",
    "ValidationResult":              "ugdatalab.methods.cross_validate",
    "cross_validate":                "ugdatalab.methods.cross_validate",
}


def __getattr__(name):
    """Lazily import and cache a top-level export by name."""
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
    """Return the union of lazy export names and currently loaded globals."""
    return sorted(set(_LAZY_EXPORTS) | set(globals()))
