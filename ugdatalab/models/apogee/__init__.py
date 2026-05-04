"""APOGEE DR17 catalog and spectrum loaders for The Cannon training."""

from ugdatalab.models.apogee.apogee import APOGEEData
from ugdatalab.models.apogee.pipeline import (
    APOGEEGiantCut,
    APOGEELabelCompletenessCut,
    APOGEEMetallicityCut,
    APOGEESnrCut,
    APOGEETempCut,
)
from ugdatalab.models.apogee.spectra import APOGEESpectra
