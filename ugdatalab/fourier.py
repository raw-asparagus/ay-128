from __future__ import annotations

import warnings

from .lightcurves import (
    FourierFit,
    build_fourier_matrix,
    cross_validate_harmonics,
    fourier_fit,
    fourier_mean_magnitude,
    phase_fold,
    predict_future_magnitude,
)

warnings.warn(
    "ugdatalab.fourier is deprecated; import Fourier helpers from ugdatalab.lightcurves instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FourierFit",
    "build_fourier_matrix",
    "cross_validate_harmonics",
    "fourier_fit",
    "fourier_mean_magnitude",
    "phase_fold",
    "predict_future_magnitude",
]
