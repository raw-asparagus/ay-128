"""Base class for fitted models bound to their data.

``Fit`` binds a fitted model to its ``(x, y, y_err)`` and provides a
polymorphic ``chi2_r`` derived from ``predict`` and ``total_variance``.
``observational_variance`` and ``total_variance`` are kept distinct:
the former is a property of the data alone (``y_err ** 2``), the
latter is the predictive variance under the fit and may add intrinsic
scatter, x-uncertainty contributions, etc. ``total_variance`` is
abstract so each subclass must explicitly declare whether the model
adds scatter beyond ``y_err``.
"""

from abc import ABC, abstractmethod

import numpy as np


class Fit(ABC):
    """Fitted model bound to the data it was fit to.

    Carries ``(x, y, y_err)`` alongside the model and exposes a
    polymorphic ``chi2_r`` derived from ``predict`` and
    ``total_variance``. Subclasses implement ``predict``,
    ``predict_std``, and ``n_params``; ``total_variance`` defaults to
    ``observational_variance`` and is overridden when the model adds
    intrinsic scatter or x-uncertainty contributions.
    """
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray

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

    @abstractmethod
    def predict_std(self, x: np.ndarray) -> np.ndarray:
        """Return the std of the mean prediction at *x* due to parameter uncertainty.

        Captures parameter-spread uncertainty only; observational and
        intrinsic scatter are not included.

        Parameters
        ----------
        x : ndarray
            Positions at which to evaluate the prediction.

        Returns
        -------
        ndarray
            Std of the mean prediction at *x*, same shape as *x*.
        """
        ...

    def observational_variance(self) -> np.ndarray:
        """Return per-point variance attributable to measurement error alone.

        Property of the data, not the fit: ``y_err ** 2``. The constant
        floor used to weight residuals if the model is taken to add no
        further noise.
        """
        return self.y_err ** 2

    def total_variance(self) -> np.ndarray:
        """Return per-point predictive variance at ``self.x`` under the fit.

        Property of the fit, not the data: the spread that explains why
        ``y_i`` differs from ``predict(x_i)`` under the fitted model.
        Defaults to :meth:`observational_variance` when the model adds
        no further noise; subclasses with intrinsic scatter or
        x-uncertainty override to augment it.
        """
        return self.observational_variance()

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
