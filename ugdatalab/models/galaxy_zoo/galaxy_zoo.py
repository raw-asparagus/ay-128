"""Galaxy Zoo 2 classification-labels CSV loader."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ugdatalab.models.base import Data
from ugdatalab.models.galaxy_zoo.constants import (
    LABEL_COLUMNS,
    _GALAXY_ZOO_SCHEMA,
)
from ugdatalab.utils.tables import _sanitize_dataframe


@dataclass
class GalaxyZooData(Data):
    """Load Galaxy Zoo 2 classification labels from CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to ``training_classifications.csv``.
    pipeline : Compose, keyword-only
        Inherited from :class:`~ugdatalab.models.base.Data`. Pipeline of
        cuts and column augmentations applied immediately after CSV load
        + sanitize. Default ``Compose([])`` — no transformations.

    Attributes
    ----------
    data : pandas.DataFrame
        Sanitized (and cut) classifications table.
    """
    csv_path: str | Path

    def _fetch(self) -> pd.DataFrame:
        """Read the Galaxy Zoo 2 classifications CSV into a DataFrame."""
        return pd.read_csv(self.csv_path)

    def _sanitize(self, raw: pd.DataFrame) -> None:
        """Coerce columns to the Galaxy Zoo schema in place."""
        _sanitize_dataframe(raw, _GALAXY_ZOO_SCHEMA)

    @property
    def labels(self) -> np.ndarray:
        """Classification labels, shape (N, 37), values in [0, 1]."""
        return self.data[LABEL_COLUMNS].to_numpy(dtype=np.float32)

    @property
    def galaxy_ids(self) -> np.ndarray:
        """Galaxy ID column as integer array."""
        return self.data["GalaxyID"].to_numpy()
