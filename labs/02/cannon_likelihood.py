"""Cannon-model spectral likelihood, composed via the ugdatalab Bayesian framework."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ugdatalab.methods.bayesian.base import GridInlierComponent, Parameter
from ugdatalab.methods.bayesian.likelihoods import ComposedGridLikelihood
from ugdatalab.models.apogee.constants import LABEL_LATEX


class CannonInlier(GridInlierComponent):
    """Inlier strategy wrapping a trained Cannon model.

    Declares one ``pm.Normal`` per stellar label with weakly-informative
    priors (mean = training-set mean, sigma = 2 x training-set std),
    builds the Cannon design vector in PyTensor, and returns the
    per-pixel ``(mu, var)`` tensors. Variance combines the per-pixel
    observational error with the model's per-pixel intrinsic scatter.

    Parameters
    ----------
    model : CannonModel
        Trained Cannon model. Provides ``theta`` (design matrix),
        ``scatter`` (per-pixel intrinsic scatter), ``label_names``,
        ``label_means``, ``label_stds``, and ``predict``.
    good_mask : ndarray of bool, shape (n_pixels,)
        Pixel mask selecting the entries of ``model.theta`` and
        ``model.scatter`` that correspond to the filtered ``(y, y_err)``
        passed to :meth:`build_pymc`.
    """

    def __init__(self, model, good_mask: np.ndarray):
        self.model = model
        self._good_mask = good_mask
        self._theta_good = np.asarray(model.theta, dtype=float)[good_mask]
        self._scatter_good = np.asarray(model.scatter, dtype=float)[good_mask]

    @property
    def parameters(self) -> list[Parameter]:
        """Return one :class:`Parameter` per stellar label, in model order."""
        return [
            Parameter(name=name, label=LABEL_LATEX[name])
            for name in self.model.label_names
        ]

    def build_pymc(self, y, y_err):
        """Add label RVs to the active model and return ``(flux_pred, var_total)``."""
        means = self.model.label_means
        stds = self.model.label_stds
        n_labels = len(means)

        labels = [
            pm.Normal(name, mu=means[i], sigma=2 * stds[i])
            for i, name in enumerate(self.model.label_names)
        ]
        labels_vec = pt.stack(labels)
        labels_scaled = (
            labels_vec - pt.as_tensor_variable(means)
        ) / pt.as_tensor_variable(stds)

        dv = self._design_vector(labels_scaled, n_labels)
        theta_matrix = pt.as_tensor_variable(self._theta_good)
        flux_pred = pt.dot(theta_matrix, dv)

        var_total = pt.as_tensor_variable(y_err ** 2 + self._scatter_good)
        return flux_pred, var_total

    def predict_at(self, theta):
        """Predict spectrum at good pixels for label vector *theta*."""
        return self.model.predict(np.asarray(theta, dtype=float))[self._good_mask]

    def total_variance_at(self, y_err, theta):
        """Per-pixel variance: observational + intrinsic scatter.

        ``theta`` is unused; the Cannon scatter is fixed by training and
        does not depend on labels.
        """
        return y_err ** 2 + self._scatter_good

    def _design_vector(self, labels_scaled, n_labels):
        """Build the (1 + n + n*(n+1)/2,) Cannon design vector from scaled labels."""
        terms = [pt.ones(1)]
        for i in range(n_labels):
            terms.append(labels_scaled[i : i + 1])
        for i in range(n_labels):
            for j in range(i, n_labels):
                terms.append(labels_scaled[i : i + 1] * labels_scaled[j : j + 1])
        return pt.concatenate(terms)


class CannonLabelLikelihood(ComposedGridLikelihood):
    """Likelihood for fitting stellar labels given a trained Cannon model.

    Composes a :class:`CannonInlier` so that ``NUTSSampler`` can
    estimate the stellar labels via MCMC. The constructor filters
    ``(y, y_err)`` to pixels with finite, non-sentinel error and
    finite model scatter before delegating to
    :class:`ComposedGridLikelihood`. The wavelength grid is read from
    the trained model and stored as ``coords`` for plotting.

    Parameters
    ----------
    y : array-like
        Observed normalized flux (full pixel grid).
    y_err : array-like
        Per-pixel observational errors (full pixel grid).
    model : CannonModel
        Trained Cannon model.
    """

    def __init__(self, y, y_err, model):
        y_err_arr = np.asarray(y_err, dtype=float)
        good = (
            np.isfinite(y_err_arr)
            & (y_err_arr < 1e5)
            & np.isfinite(np.asarray(model.scatter, dtype=float))
        )
        super().__init__(
            coords=np.asarray(model.wavelength, dtype=float)[good],
            y=np.asarray(y, dtype=float)[good],
            y_err=y_err_arr[good],
            inlier=CannonInlier(model, good_mask=good),
        )
