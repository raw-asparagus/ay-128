import corner
import matplotlib.pyplot as plt
import numpy as np


class MCMCDiagnostics:
    """Mixin providing diagnostic plots for any sampler that sets:

      self.samples   : ndarray, shape (n_steps, ndim)
      self.log_probs : ndarray, shape (n_steps,)
      self.n_burn    : int  (optional; defaults to 0 if not set)

    Burn-in steps are excluded from all plots.
    """

    def _labels(self, override=None):
        """Return parameter labels, preferring override then stored then auto-generated."""
        if override is not None:
            return override
        ndim = self.samples.shape[1]
        return getattr(self, 'param_labels',
                       [rf"$\theta_{i}$" for i in range(ndim)])

    def plot_posterior(self, ax=None, title=None, pdf_fn=None,
                       param_idx=0, label=None):
        """Histogram of the 1-D marginal posterior for a single parameter.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Pre-existing axes. If None, a new figure is created.
        title : str, optional
            Axes title.
        pdf_fn : callable, optional
            Analytic PDF evaluated on a fine grid and overlaid as a curve.
        param_idx : int, optional
            Index of the parameter to plot (default 0).
        label : str, optional
            x-axis label. Defaults to the stored label for `param_idx`.

        Returns
        -------
        ax : matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        n_burn  = getattr(self, 'n_burn', 0)
        samples = self.samples[n_burn:, param_idx]
        xlabel  = label if label is not None else self._labels()[param_idx]
        ax.hist(samples, bins=50, density=True, alpha=0.6, label="MCMC samples")

        if pdf_fn is not None:
            margin = 0.5 * (samples.max() - samples.min())
            grid   = np.linspace(samples.min() - margin, samples.max() + margin, 500)
            ax.plot(grid, pdf_fn(grid), lw=2, label=rf"Analytic $p({xlabel}\mid x)$")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        if title is not None:
            ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_trace(self, axes=None, title=None, labels=None):
        """Trace plot: one panel per parameter plus log-probability vs. step.

        Parameters
        ----------
        axes : array-like of (ndim + 1) matplotlib Axes, optional
            Pre-existing axes (sharex). If None, a new figure is created.
        title : str, optional
            Title applied to the top panel.
        labels : list of str, optional
            Override parameter labels. Defaults to stored labels or auto-generated.

        Returns
        -------
        axes : ndarray, shape (ndim + 1,)
        """
        n_burn = getattr(self, 'n_burn', 0)
        steps  = np.arange(n_burn, len(self.samples))
        ndim   = self.samples.shape[1]
        lbls   = self._labels(labels)

        if axes is None:
            fig, axes = plt.subplots(ndim + 1, 1,
                                     figsize=(10, 3 * (ndim + 1)),
                                     dpi=300, sharex=True)

        for i, lbl in enumerate(lbls):
            axes[i].plot(steps, self.samples[n_burn:, i], lw=0.4, alpha=0.8)
            axes[i].set_ylabel(lbl)
            axes[i].grid(True, alpha=0.3)
        if title is not None:
            axes[0].set_title(title)

        axes[-1].plot(steps, self.log_probs[n_burn:], lw=0.4, alpha=0.8, color="C1")
        axes[-1].set_ylabel(r"$\ln P$")
        axes[-1].set_xlabel("Step")
        axes[-1].grid(True, alpha=0.3)

        return axes

    def plot_corner(self, fig=None, title=None, labels=None):
        """Corner plot of all parameter marginals and pairwise correlations.

        Requires the `corner` package (pip install corner).

        Parameters
        ----------
        fig : matplotlib Figure, optional
            Pre-existing figure to draw on. If None, corner creates one.
        title : str, optional
            Figure suptitle.
        labels : list of str, optional
            Override parameter labels. Defaults to stored labels or auto-generated.

        Returns
        -------
        fig : matplotlib Figure
            `corner.corner` manages an N×N grid of axes internally; returning
            `fig` (rather than `fig.axes`) is the standard usage.
        """
        n_burn = getattr(self, 'n_burn', 0)
        samples = self.samples[n_burn:]
        lbls    = self._labels(labels)

        fig = corner.corner(samples, labels=lbls, fig=fig,
                            show_titles=True, title_fmt=".3f",
                            quantiles=[0.16, 0.5, 0.84])
        if title is not None:
            fig.suptitle(title)
        return fig


class MetropolisHastings(MCMCDiagnostics):
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


class NoUTurnHamiltonian(MCMCDiagnostics):
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
