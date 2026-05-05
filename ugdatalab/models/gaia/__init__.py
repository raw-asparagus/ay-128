"""Gaia DR3 catalog loaders, composable quality cuts, and WISE cross-match wrappers."""

from ugdatalab.models.gaia.gaia import GaiaData
from ugdatalab.models.gaia.pipeline import (
    AddPhotometryColumns,
    AttachInlierProbColumn,
    AttachReddening,
    AttachRepresentativePeriod,
    GaiaCompletenessCut,
    HighLatitudeCut,
    HighParallaxSnrCut,
    LindegrenC1Cut,
    LindegrenC2Cut,
    LocalCut,
    StrictBPRPCut,
    StrictGCut,
    StrictReddeningCut,
)
from ugdatalab.models.gaia.wise import WISEData
from ugdatalab.models.gaia.wise_pipeline import (
    AddW2PhotometryColumns,
    MarreseOneToOneMatchCut,
    W2PhotometryQualityCut,
    WISECompletenessCut,
)
from ugdatalab.models.gaia.constants import RRLYRAE_PERIOD_MIN, RRLYRAE_PERIOD_MAX
from ugdatalab.models.gaia.lightcurves import GaiaLightcurves
from ugdatalab.models.gaia.lightcurves_pipeline import (
    AttachDerivedEpochColumns,
    AttachFourierMeanMagnitudes,
    AttachPeriodogramPeriods,
    LightcurveCompletenessCut,
)
