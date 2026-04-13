from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ugdatalab.methods.bayesian.base import Likelihood
from ugdatalab.models.apogee.constants import LABEL_LATEX


@dataclass
class CannonLabelLikelihood(Likelihood):
    """Likelihood for fitting stellar labels given a trained Cannon model.

    Wraps a ``CannonModel`` as a ``Likelihood`` so that ``nuts_sample``
    can estimate labels via MCMC.

    Attributes
    ----------
    x : ndarray
        Wavelength array (fulfills ABC; also used by MCMCResult.predict).
    y : ndarray
        Observed normalized flux, shape (n_pixels,).
    y_err : ndarray
        Per-pixel errors, shape (n_pixels,).  1e6 sentinel for masked pixels.
    model : CannonModel
        Trained Cannon model.
    """
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray
    model: object

    @property
    def param_labels(self) -> list[str]:
        return list(LABEL_LATEX)

    def _predict(self, x, theta):
        """Predict spectrum given label vector *theta* (5,).

        *x* is ignored — wavelength is baked into the model.
        """
        return self.model.predict(np.asarray(theta, dtype=float))

    def _inlier_variance(self, theta):
        """Total per-pixel variance: measurement noise + intrinsic scatter."""
        return self.y_err ** 2 + self.model.scatter

    def _build_cannon_design_vector_pytensor(self, labels_scaled):
        """Build (21,) design vector from (5,) scaled labels in PyTensor."""
        n_labels = len(self.model.label_names)
        terms = [pt.ones(1)]
        for i in range(n_labels):
            terms.append(labels_scaled[i : i + 1])
        for i in range(n_labels):
            for j in range(i, n_labels):
                terms.append(labels_scaled[i : i + 1] * labels_scaled[j : j + 1])
        return pt.concatenate(terms)

    def build_pymc(self):
        """Build a PyMC model for NUTS sampling over 5 stellar labels.

        Priors are weakly informative Normals centered on training-set
        means with width = 2 × training-set standard deviations.
        """
        good = np.isfinite(self.y_err) & (self.y_err < 1e5) & np.isfinite(self.model.scatter)
        y_obs = pt.as_tensor_variable(self.y[good])
        sigma2_obs = pt.as_tensor_variable(self.y_err[good] ** 2)
        s2 = pt.as_tensor_variable(self.model.scatter[good])
        theta_matrix = pt.as_tensor_variable(self.model.theta[good])

        means = self.model.label_means
        stds = self.model.label_stds
        n_labels = len(means)

        with pm.Model() as model:
            labels = []
            for i in range(n_labels):
                name = self.model.label_names[i]
                labels.append(pm.Normal(name, mu=means[i], sigma=2 * stds[i]))
            labels_vec = pt.stack(labels)

            labels_scaled = (labels_vec - pt.as_tensor_variable(means)) / pt.as_tensor_variable(stds)
            dv = self._build_cannon_design_vector_pytensor(labels_scaled)
            flux_pred = pt.dot(theta_matrix, dv)

            var_total = sigma2_obs + s2
            pm.Normal("obs", mu=flux_pred, sigma=pt.sqrt(var_total), observed=y_obs)

        return model

    def build_pymc_mixture(self):
        raise NotImplementedError(
            "Mixture model not applicable for Cannon label fitting"
        )

    def inlier_probs(self, trace, model_var_names):
        raise NotImplementedError(
            "Inlier probabilities not applicable for Cannon label fitting"
        )
