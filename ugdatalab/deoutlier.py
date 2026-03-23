import numpy as np

from ugdatalab.methods.likelihoods import LinearGaussianLikelihood
from ugdatalab.methods.mixture import mixture_contamination
from ugdatalab.models.gaia import GaiaQuality


class MixtureContaminationModel:
    """Outlier rejection via a Gaussian mixture contamination model.

    For RRab and RRc separately, fits  M_G = a + b·log10(P) + ε  with an
    explicit broad-Gaussian outlier component using PyMC NUTS. Stars
    whose posterior inlier probability < prob_threshold are rejected.

    Attributes
    ----------
    mcmc_results : dict
        Keys "RRab" / "RRc", values are MixtureResult dataclasses.
    """

    def __init__(
        self,
        source: GaiaQuality,
        prob_threshold: float = 0.95,
        n_steps: int = 2000,
        n_burn: int = 1000,
        seed: int = 42,
    ):
        self.query = source.query
        self.prob_threshold = prob_threshold
        self.mcmc_results = {}

        inlier_probs = np.ones(len(source.data))
        for label, mask in [
            ("RRab", source.data["best_classification"] == "RRab"),
            ("RRc", source.data["best_classification"] == "RRc"),
        ]:
            if mask.sum() < 10:
                continue
            sub = source.data[mask]
            period_column = "pf" if label == "RRab" else "p1_o"
            x = np.log10(np.asarray(sub[period_column], dtype=float))
            y = np.asarray(sub["M_G"], dtype=float)
            sig = np.asarray(sub["sigma_M"], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(sig) & (sig > 0)
            if valid.sum() < 10:
                continue

            likelihood = LinearGaussianLikelihood(x[valid], y[valid], sig[valid])
            result = mixture_contamination(
                likelihood,
                n_steps=n_steps,
                n_burn=n_burn,
                seed=seed,
            )

            full_idx = np.where(mask)[0][valid]
            inlier_probs[full_idx] = result.inlier_prob
            self.mcmc_results[label] = result

        full_data = source.data.copy()
        full_data["inlier_prob"] = inlier_probs
        self._all_data = full_data
        self.data = full_data[inlier_probs >= prob_threshold]

    @property
    def all_data(self):
        return self._all_data
