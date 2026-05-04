"""Base classes for fitted models.

``Fit`` is a trained-model artifact applicable to new data. ``DataFit``
binds the fit to its (x, y, y_err) and provides a polymorphic
``chi2_r``.
"""

from abc import ABC, abstractmethod

import numpy as np

# Sentinel: y_err values at or above this magnitude are treated as masked.
# Matches APOGEE's filler-value convention; safe upper bound for any
# physically meaningful uncertainty in the surveys we consume.
_MASKED_ERR_THRESHOLD = 1e5


class Fit(ABC):
    """Fitted model that predicts y at arbitrary x."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the model response at *x*.

        Parameters
        ----------
        x : ndarray
            Independent-variable positions at which to evaluate the model.

        Returns
        -------
        ndarray
            Predicted y values, same shape as *x*.
        """
        ...


class DataFit(Fit):
    """Fitted model bound to the data it was fit to.

    Carries ``(x, y, y_err)`` alongside the model and exposes a
    polymorphic ``chi2_r`` derived from ``predict`` and
    ``total_variance``.
    """
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray

    def total_variance(self) -> np.ndarray:
        """Return per-point predictive variance at ``self.x``.

        Default implementation returns ``y_err ** 2``. Subclasses with
        intrinsic scatter or x-uncertainty override.
        """
        return self.y_err ** 2

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Return the number of free parameters consumed in the fit."""
        ...

    @property
    def chi2_r(self) -> float:
        """Return reduced chi-squared at the best-fit parameters.

        Masks points where ``y_err`` is non-finite or above
        ``_MASKED_ERR_THRESHOLD``, then divides the weighted sum of
        squared residuals by ``max(N_good - n_params, 1)``.
        """
        y_pred = self.predict(self.x)
        var = self.total_variance()
        good = (np.isfinite(self.y_err) & (self.y_err < _MASKED_ERR_THRESHOLD)
                & np.isfinite(self.y) & np.isfinite(y_pred))
        nu = max(int(good.sum()) - self.n_params, 1)
        resid = (self.y - y_pred)[good]
        return float(np.sum(resid ** 2 / var[good]) / nu)
