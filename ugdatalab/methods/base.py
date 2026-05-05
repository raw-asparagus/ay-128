"""Base classes for fitted models.

``Fit`` is a trained-model artifact applicable to new data. ``DataFit``
binds the fit to its (x, y, y_err) and provides a polymorphic
``chi2_r``.
"""

from abc import ABC, abstractmethod

import numpy as np


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

        Computes the weighted sum of squared residuals divided by
        ``max(len(y) - n_params, 1)``. Callers must pass already-clean
        ``(x, y, y_err)`` — non-finite or sentinel-masked entries are
        the caller's responsibility to filter out.
        """
        resid = self.y - self.predict(self.x)
        nu = max(len(self.y) - self.n_params, 1)
        return float(np.sum(resid ** 2 / self.total_variance()) / nu)
