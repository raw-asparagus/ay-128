"""Galaxy Zoo 2 classification data loading and train/validation splitting."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ugdatalab.models.galaxy_zoo.constants import (
    LABEL_COLUMNS,
    _GALAXY_ZOO_SCHEMA,
)


def _sanitize_dataframe(data: pd.DataFrame, schema: dict) -> None:
    """Cast DataFrame columns to specified dtypes.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to modify in place.
    schema : dict
        Mapping ``{dtype: [column_names]}`` — same format as
        ``_sanitize_table`` but for pandas DataFrames.
    """
    for dtype, cols in schema.items():
        for col in cols:
            if col in data.columns:
                data[col] = data[col].astype(dtype)


@dataclass
class GalaxyZooData:
    """Loads Galaxy Zoo 2 classification labels from CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to ``training_classifications.csv``.
    """
    csv_path: str | Path
    data: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        data = pd.read_csv(self.csv_path)
        _sanitize_dataframe(data, _GALAXY_ZOO_SCHEMA)
        self.data = data

    @property
    def labels(self) -> np.ndarray:
        """Classification labels, shape (N, 37), values in [0, 1]."""
        return self.data[LABEL_COLUMNS].to_numpy(dtype=np.float32)

    @property
    def galaxy_ids(self) -> np.ndarray:
        """Galaxy ID column as integer array."""
        return self.data["GalaxyID"].to_numpy()

    @property
    def n_galaxies(self) -> int:
        """Number of galaxies in the dataset."""
        return len(self.data)


class GalaxyZooSplit:
    """Random train/validation split of a GalaxyZooData source.

    Parameters
    ----------
    source : GalaxyZooData
        Full dataset to split.
    seed : int
        Random seed for reproducibility.
    train_fraction : float
        Fraction of data to use for training (e.g. 0.8).
    """
    def __init__(self, source: GalaxyZooData, seed: int, train_fraction: float):
        self.csv_path = source.csv_path
        n = source.n_galaxies
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_train = int(train_fraction * n)
        self.train_idx = np.sort(idx[:n_train])
        self.val_idx = np.sort(idx[n_train:])
        self.train_data = source.data.iloc[self.train_idx].reset_index(drop=True)
        self.val_data = source.data.iloc[self.val_idx].reset_index(drop=True)

    @property
    def train_labels(self) -> np.ndarray:
        """Training labels, shape (N_train, 37)."""
        return self.train_data[LABEL_COLUMNS].to_numpy(dtype=np.float32)

    @property
    def val_labels(self) -> np.ndarray:
        """Validation labels, shape (N_val, 37)."""
        return self.val_data[LABEL_COLUMNS].to_numpy(dtype=np.float32)

    @property
    def train_galaxy_ids(self) -> np.ndarray:
        """Training galaxy IDs."""
        return self.train_data["GalaxyID"].to_numpy()

    @property
    def val_galaxy_ids(self) -> np.ndarray:
        """Validation galaxy IDs."""
        return self.val_data["GalaxyID"].to_numpy()
