import numpy as np

class MetropolisHastings:
    """Single-chain Metropolis-Hastings sampler with an isotropic Gaussian proposal.

    Parameters
    ----------
    log_prob : callable (ndarray) -> float
        Unnormalised log-posterior or log-likelihood.
    theta0 : array-like, shape (ndim,)
        Starting position in parameter space.
    proposal_std : float
        Standard deviation of the isotropic Gaussian proposal distribution.
    seed : int or None
        RNG seed for reproducibility.
    labels : list of str or None
        Human-readable parameter names used by the diagnostic plot methods.
        Defaults to [r"$\theta_0$", r"$\theta_1$", ...].
    """

    def __init__(self, log_prob, theta0, proposal_std=0.1, seed=None, labels=None):
        self.log_prob     = log_prob
        self.theta0       = np.asarray(theta0, dtype=float)
        self.proposal_std = proposal_std
        self.rng          = np.random.default_rng(seed)

        ndim = len(self.theta0)
        self.param_labels = labels if labels is not None \
                            else [rf"$\theta_{i}$" for i in range(ndim)]

        self.samples         = None
        self.log_probs       = None
        self.acceptance_rate = None
        self.n_burn          = 0

    def run(self, n_steps, n_burn=0):
        """Run the Metropolis-Hastings chain for n_steps steps.

        Parameters
        ----------
        n_steps : int
            Total number of MCMC steps including burn-in.
        n_burn : int
            Number of initial steps to treat as burn-in. Stored as
            self.n_burn; diagnostic plots will exclude these steps.
            Acceptance rate is computed over post-burn-in steps only.

        Returns
        -------
        self
        """
        ndim      = len(self.theta0)
        samples   = np.empty((n_steps, ndim))
        log_probs = np.empty(n_steps)

        theta      = self.theta0.copy()
        lp_curr    = self.log_prob(theta)
        n_accepted = 0

        for i in range(n_steps):
            theta_prop = theta + self.rng.normal(0, self.proposal_std, size=ndim)
            lp_prop    = self.log_prob(theta_prop)
            log_alpha  = lp_prop - lp_curr

            if np.log(self.rng.uniform()) < log_alpha:
                theta   = theta_prop
                lp_curr = lp_prop
                if i >= n_burn:
                    n_accepted += 1

            samples[i]   = theta
            log_probs[i] = lp_curr

        self.samples         = samples
        self.log_probs       = log_probs
        self.n_burn          = n_burn
        self.acceptance_rate = n_accepted / max(n_steps - n_burn, 1)
        return self


class NoUTurnHamiltonian:
    """No-U-Turn Hamiltonian Monte Carlo sampler wrapping PyMC.

    Parameters
    ----------
    model : pymc.Model
        A PyMC model context containing free variables and the likelihood
        added via pm.Potential(). Encodes both priors and likelihood.
    var_names : list of str
        Names of the free variables in desired column order for self.samples.
    theta0 : array-like, shape (ndim,)
        Starting values (in user-visible parameter space) matching var_names order.
    seed : int or None
        Random seed for reproducibility.
    labels : list of str or None
        Human-readable parameter labels for diagnostic plots.
    """

    def __init__(self, model, var_names, theta0, seed=None, labels=None):
        self.model      = model
        self.var_names  = var_names
        self.theta0     = np.asarray(theta0, dtype=float)
        self.seed       = seed

        ndim = len(var_names)
        self.param_labels = labels if labels is not None \
                            else [rf"$\theta_{i}$" for i in range(ndim)]

        self.samples         = None
        self.log_probs       = None
        self.acceptance_rate = None
        self.n_burn          = 0

    def run(self, n_steps, n_burn=500):
        """Run NUTS for n_steps posterior draws with n_burn tuning steps.

        Parameters
        ----------
        n_steps : int
            Number of posterior draws (post-tuning).
        n_burn : int
            Tuning (warm-up / step-size adaptation) steps; discarded by PyMC.
            self.n_burn is set to 0 because all stored samples are post-burn-in.

        Returns
        -------
        self
        """
        import pymc as pm

        initvals = dict(zip(self.var_names, self.theta0))

        with self.model:
            trace = pm.sample(
                draws=n_steps,
                tune=n_burn,
                initvals=initvals,
                random_seed=self.seed,
                chains=1,
                progressbar=True,
            )

        # Extract samples: shape (n_steps, ndim)
        self.samples = np.column_stack(
            [trace.posterior[v].values[0] for v in self.var_names]
        )

        # PyMC discards tune steps; all stored samples are post-burn-in
        self.n_burn = 0

        # Log-posterior from sampler statistics
        self.log_probs = trace.sample_stats.lp.values[0]          # (n_steps,)

        # Mean NUTS acceptance rate (target ≈ 0.80–0.95)
        self.acceptance_rate = float(
            trace.sample_stats.acceptance_rate.values[0].mean()
        )

        return self
