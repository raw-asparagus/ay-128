"""APOGEE DR17 pipeline stages: row-filter cuts and column augmentations.

Each stage is a small ``__call__``-able (dataclass when parameterized)
that takes an ``astropy.table.Table`` and returns one. Compose with
:class:`~ugdatalab.methods.compose.Compose` to build pipelines that
:class:`~ugdatalab.models.apogee.apogee.APOGEEData` runs immediately
after fetching and sanitizing.
"""

from dataclasses import dataclass, field

import numpy as np
from astropy import table

from ugdatalab.models.apogee.constants import _FILLER_VALUE


_DEFAULT_LABEL_COLUMNS = ("teff", "logg", "fe_h", "mg_fe", "si_fe")


# ---------------------------------------------------------------------------
# Row-filter cuts
# ---------------------------------------------------------------------------


@dataclass
class APOGEELabelCompletenessCut:
    """Drop rows with missing or sentinel values in any required label.

    Parameters
    ----------
    label_columns : tuple of str
        Column names that must all be finite and not equal to ``sentinel``.
        Default ``("teff", "logg", "fe_h", "mg_fe", "si_fe")`` — the five
        labels The Cannon is trained against. Override to relax the
        requirement to a subset of labels.
    sentinel : float
        Filler value APOGEE writes for missing pipeline outputs. Default
        ``-9999.0`` from ``ugdatalab.models.apogee.constants._FILLER_VALUE``;
        avoid changing unless DR conventions change.
    """
    label_columns: tuple = field(default=_DEFAULT_LABEL_COLUMNS)
    sentinel: float = _FILLER_VALUE

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows with all *label_columns* finite and not equal to *sentinel*."""
        mask = np.ones(len(data), dtype=bool)
        for col in self.label_columns:
            vals = data[col]
            mask &= np.isfinite(vals) & (vals != self.sentinel)
        return data[mask]


@dataclass
class APOGEESnrCut:
    """Spectrum signal-to-noise floor.

    Parameters
    ----------
    snr_min : float
        Minimum ``snr`` required. Default ``50`` — the canonical Cannon
        training-set threshold; loosening pulls in noisier spectra that
        degrade label-recovery scatter.
    """
    snr_min: float = 50

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows with ``snr > self.snr_min``."""
        return data[data["snr"] > self.snr_min]


@dataclass
class APOGEEGiantCut:
    """Restrict to red-giant surface gravities.

    Parameters
    ----------
    logg_max : float
        Maximum ``logg``. Default ``4.0`` — separates giants from dwarfs;
        Cannon training in this lab uses giants only because the spectral
        models are calibrated against them.
    """
    logg_max: float = 4.0

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows with ``logg <= self.logg_max``."""
        return data[data["logg"] <= self.logg_max]


@dataclass
class APOGEETempCut:
    """Effective-temperature ceiling.

    Parameters
    ----------
    teff_max : float
        Maximum ``teff`` in K. Default ``5700`` — excludes the warmest
        stars whose spectra differ qualitatively from cool giants and
        would broaden Cannon's label-prediction scatter.
    """
    teff_max: float = 5700.0

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows with ``teff <= self.teff_max``."""
        return data[data["teff"] <= self.teff_max]


@dataclass
class APOGEEMetallicityCut:
    """Metallicity floor.

    Parameters
    ----------
    fe_h_min : float
        Minimum ``fe_h``. Default ``-1.0`` — excludes the most metal-poor
        stars whose spectra are sparse-line and degrade Cannon's
        polynomial label model.
    """
    fe_h_min: float = -1.0

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows with ``fe_h >= self.fe_h_min``."""
        return data[data["fe_h"] >= self.fe_h_min]
