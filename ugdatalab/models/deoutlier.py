import emcee

import numpy as np

from ugdatalab.models.gaia import GaiaQuality, rrlyrae_class_mask


class MixtureContaminationModel:
    """Outlier rejection via a Gaussian mixture contamination model.

    For RRab and RRc separately, fits  M_G = a + b·log10(P) + ε  with an
    explicit broad-Gaussian outlier component using MCMC (emcee). Stars
    whose posterior inlier probability < prob_threshold are rejected.

    Attributes
    ----------
    mcmc_results : dict
        Keys "RRab" / "RRc". Each value has keys: flat_samples, a, b,
        sig_scatter, f_in, mu_bg, sig_bg (median ± std tuples where applicable).
    """

    @staticmethod
    def _log_mix_terms(a, b, sig_scatter, f, x, y, sigma_y, mu_bg, sig_bg):
        """Per-star (log_f + ll_in, log_1mf + ll_out); all arguments broadcast."""
        var_in = sigma_y**2 + sig_scatter**2
        var_out = sigma_y**2 + sig_bg**2
        ll_in  = -0.5 * (np.log(2*np.pi*var_in) + (y - (a + b*x))**2 / var_in)
        ll_out = -0.5 * (np.log(2*np.pi*var_out) + (y - mu_bg)**2 / var_out)
        return np.log(f) + ll_in, np.log(1 - f) + ll_out

    @staticmethod
    def _log_prob(theta, x, y, sigma_y, mu_bg, sig_bg):
        """Log-posterior of the mixture model (emcee target).

        theta = [a, b, log10_sig_scatter, logit_f].
        Weakly informative priors: N(0,10) on a,b; N(0,2) on log10_sig; N(0,3) on logit_f.
        """
        a, b, log10_sig, logit_f = theta
        sig_scatter = 10.0 ** log10_sig
        f = 1.0 / (1.0 + np.exp(-logit_f))
        if not (0 < f < 1):
            return -np.inf
        log_prior = -0.5 * ((a/10)**2 + (b/10)**2 + (log10_sig/2)**2 + (logit_f/3)**2)
        t_in, t_out = MixtureContaminationModel._log_mix_terms(
            a, b, sig_scatter, f, x, y, sigma_y, mu_bg, sig_bg
        )
        return log_prior + np.sum(np.logaddexp(t_in, t_out))

    @staticmethod
    def _inlier_probs(flat_samples, x, y, sigma_y, mu_bg, sig_bg):
        """Posterior inlier probability per star, averaged over S MCMC samples.

        Vectorised: broadcasts (S,1)-shaped parameters against (N,)-shaped data
        to produce an (S, N) responsibility matrix, then averages over S.
        """
        a, b, log10_sig, logit_f = flat_samples.T           # each (S,)
        sig_scatter = (10.0 ** log10_sig)[:, None]           # (S, 1)
        f           = (1.0 / (1.0 + np.exp(-logit_f)))[:, None]
        t_in, t_out = MixtureContaminationModel._log_mix_terms(
            a[:, None], b[:, None], sig_scatter, f,
            x[None, :], y[None, :], sigma_y[None, :], mu_bg, sig_bg,
        )                                                    # each (S, N)
        return np.exp(t_in - np.logaddexp(t_in, t_out)).mean(axis=0)

    @staticmethod
    def _run_mcmc(x, y, sigma_y, n_walkers=32, n_steps=2000, seed=42):
        """Run the mixture model with an emcee ensemble sampler.

        Background fixed to mu_bg = median(y), sig_bg = 3*std(y).
        Walkers initialised near the least-squares PL solution.
        Returns (sampler, mu_bg, sig_bg).
        """
        mu_bg  = float(np.median(y))
        sig_bg = float(3.0 * np.std(y))
        A      = np.column_stack([np.ones_like(x), x])
        a0, b0 = np.linalg.lstsq(A, y, rcond=None)[0]
        p0     = np.array([a0, b0, np.log10(0.3), np.log(0.9/0.1)])
        rng    = np.random.default_rng(seed)
        p0     = p0 + 1e-2 * rng.standard_normal((n_walkers, 4))
        sampler = emcee.EnsembleSampler(
            n_walkers, 4, MixtureContaminationModel._log_prob,
            args=(x, y, sigma_y, mu_bg, sig_bg),
        )
        sampler.run_mcmc(p0, n_steps, progress=False)
        return sampler, mu_bg, sig_bg

    def __init__(self, source: GaiaQuality, prob_threshold: float = 0.95,
                 n_walkers: int = 32, n_steps: int = 2000,
                 n_burn: int = 1000, seed: int = 42):
        self.query          = source.query
        self.prob_threshold = prob_threshold
        self.mcmc_results   = {}

        inlier_probs = np.ones(len(source.data))  # default: keep
        for label, mask in [("RRab", rrlyrae_class_mask(source.data, "RRab")),
                            ("RRc", rrlyrae_class_mask(source.data, "RRc"))]:
            if mask.sum() < 10:
                continue
            sub   = source.data[mask]
            period_column = "pf" if label == "RRab" else "p1_o"
            x     = np.log10(np.asarray(sub[period_column], dtype=float))
            y     = np.asarray(sub["M_G"])
            sig   = np.asarray(sub["sigma_M"])
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(sig) & (sig > 0)
            if valid.sum() < 10:
                continue

            sampler, mu_bg, sig_bg = self._run_mcmc(
                x[valid], y[valid], sig[valid],
                n_walkers=n_walkers, n_steps=n_steps, seed=seed,
            )
            flat  = sampler.get_chain(discard=n_burn, flat=True)
            probs = self._inlier_probs(
                flat, x[valid], y[valid], sig[valid], mu_bg, sig_bg
            )

            full_idx = np.where(mask)[0][valid]
            inlier_probs[full_idx] = probs

            self.mcmc_results[label] = dict(
                flat_samples = flat,
                a      = (np.median(flat[:, 0]), np.std(flat[:, 0])),
                b      = (np.median(flat[:, 1]), np.std(flat[:, 1])),
                sig_scatter= (10**np.median(flat[:, 2]), ),
                f_in   = (float(np.mean(1/(1+np.exp(-flat[:, 3])))), ),
                mu_bg  = mu_bg,
                sig_bg = sig_bg,
            )

        full_data = source.data.copy()
        full_data["inlier_prob"] = inlier_probs
        self._all_data  = full_data
        self.data       = full_data[inlier_probs >= prob_threshold]

    @property
    def all_data(self):
        return self._all_data
