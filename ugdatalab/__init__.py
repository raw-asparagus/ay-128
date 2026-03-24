"""ugdatalab — methods and models for astronomical data analysis.

Submodules are imported on demand to avoid pulling heavy dependencies
(PyMC, astroquery) at package import time.

Usage:
    from ugdatalab.methods.fourier import fourier_fit, FourierFit
    from ugdatalab.methods.periodogram import lomb_scargle
    from ugdatalab.methods.bayesian.likelihoods import LinearGaussianLikelihood
    from ugdatalab.methods.bayesian.mcmc import nuts_sample
    from ugdatalab.methods.bayesian.mixture import mixture_contamination
    from ugdatalab.models.gaia import GaiaData, GaiaQuality
"""
