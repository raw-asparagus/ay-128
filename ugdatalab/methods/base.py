from abc import ABC, abstractmethod

import numpy as np


class Fit(ABC):
    """Base class for fitted models."""
    chi2_r: float

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray: ...
