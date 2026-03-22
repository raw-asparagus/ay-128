from typing import Protocol

import numpy as np


class Fit(Protocol):
    """Protocol for fitted models.

    Any model returned by a fit function must have:
    - ``chi2_r``: reduced chi-squared of the fit
    - ``predict(x)``: predict values at given positions
    """
    chi2_r: float

    def predict(self, x: np.ndarray) -> np.ndarray: ...
