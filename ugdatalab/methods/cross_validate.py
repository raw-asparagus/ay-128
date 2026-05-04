"""Cross-validation over a single scalar parameter (holdout or k-fold)."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Result of cross-validated model selection over a parameter grid.

    Attributes
    ----------
    param_values : ndarray
        Parameter values tested.
    chi2r_train : ndarray
        Reduced chi-squared on the training set per parameter value,
        averaged across folds when ``n_folds > 1``.
    chi2r_cv : ndarray
        Reduced chi-squared on the held-out set per parameter value,
        averaged across folds when ``n_folds > 1``.
    best_param : int or float
        Parameter value with the lowest validation chi-squared.
    n_folds : int
        Number of folds (``1`` for holdout).
    fold_assignments : list of (train_idx, cv_idx)
        Index arrays for each fold; length equals ``n_folds``.
    """
    param_values: np.ndarray
    chi2r_train: np.ndarray
    chi2r_cv: np.ndarray
    best_param: int | float
    n_folds: int
    fold_assignments: list[Tuple[np.ndarray, np.ndarray]]

    @property
    def train_idx(self) -> np.ndarray:
        """Return the training-set indices.

        Raises
        ------
        AttributeError
            If ``n_folds != 1``.
        """
        if self.n_folds != 1:
            raise AttributeError(
                "train_idx is only defined for single-split (holdout) "
                "validation; use fold_assignments for k-fold results."
            )
        return self.fold_assignments[0][0]

    @property
    def cv_idx(self) -> np.ndarray:
        """Return the held-out indices.

        Raises
        ------
        AttributeError
            If ``n_folds != 1``.
        """
        if self.n_folds != 1:
            raise AttributeError(
                "cv_idx is only defined for single-split (holdout) "
                "validation; use fold_assignments for k-fold results."
            )
        return self.fold_assignments[0][1]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chi2r(y_true: np.ndarray, y_pred: np.ndarray, y_err: np.ndarray) -> float:
    """Reduced chi-squared with no dof correction (divides by ``N``)."""
    return float(np.sum(((y_true - y_pred) / y_err) ** 2) / len(y_true))


def _make_folds(
    n: int,
    n_folds: int,
    cv_fraction: float,
    rng: np.random.Generator,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """Build (train_idx, cv_idx) pairs for either holdout or k-fold."""
    idx = rng.permutation(n)
    if n_folds == 1:
        n_cv = max(1, int(round(cv_fraction * n)))
        return [(idx[n_cv:], idx[:n_cv])]
    return [
        (np.setdiff1d(idx, fold, assume_unique=True), fold)
        for fold in np.array_split(idx, n_folds)
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    fit_fn: Callable,
    param_values: np.ndarray,
    *,
    n_folds: int = 1,
    cv_fraction: float = 0.2,
    seed: int = 346,
) -> ValidationResult:
    """Validate a model over a single parameter, by holdout or k-fold.

    Searches over one scalar parameter at a time. For each value in
    *param_values*, calls::

        model = fit_fn(x[train], y[train], y_err[train], param_value)

    then evaluates ``model.predict(x[cv])`` against ``y[cv]``.

    Parameters
    ----------
    x, y, y_err : array-like
        Full dataset.
    fit_fn : callable
        ``fit_fn(x, y, y_err, param_value) -> Fit``. Extra arguments
        must be bound before passing.
    param_values : array-like
        Grid of parameter values to search over.
    n_folds : int
        Number of cross-validation folds. Default ``1`` — a single
        holdout split, the cheapest option and adequate when the dataset
        is large enough that one held-out subset is representative.
        Override to ``5``–``10`` for small datasets where holdout variance
        across splits is large, or whenever you want a CV-error estimate
        with its own uncertainty.
    cv_fraction : float
        Fraction of data held out for validation in the holdout case.
        Ignored when ``n_folds > 1``. Default ``0.2`` — the standard
        80/20 train/validation split, which keeps most of the data for
        fitting while leaving enough rows for a stable validation score.
        Lower it on very small datasets to preserve training power; raise
        it if validation scores look noisy from a too-small CV set.
    seed : int
        Random seed for the split / fold assignment. Default ``346`` —
        an arbitrary fixed value for reproducibility; override to test
        sensitivity of the chosen ``best_param`` to the random split.

    Returns
    -------
    ValidationResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)
    param_values = np.asarray(param_values)

    rng = np.random.default_rng(seed)
    folds = _make_folds(len(x), n_folds, cv_fraction, rng)

    chi2r_train = np.full(len(param_values), np.nan, dtype=float)
    chi2r_cv = np.full(len(param_values), np.nan, dtype=float)

    for i, pv in enumerate(param_values):
        per_fold_train = []
        per_fold_cv = []
        for train_idx, cv_idx in folds:
            try:
                model = fit_fn(x[train_idx], y[train_idx], y_err[train_idx], int(pv))
            except (ValueError, np.linalg.LinAlgError):
                continue
            per_fold_train.append(model.chi2_r)
            per_fold_cv.append(_chi2r(y[cv_idx], model.predict(x[cv_idx]), y_err[cv_idx]))

        if per_fold_train:
            chi2r_train[i] = float(np.mean(per_fold_train))
            chi2r_cv[i] = float(np.mean(per_fold_cv))

    best_param = param_values[int(np.nanargmin(chi2r_cv))]

    return ValidationResult(
        param_values=param_values,
        chi2r_train=chi2r_train,
        chi2r_cv=chi2r_cv,
        best_param=best_param,
        n_folds=n_folds,
        fold_assignments=folds,
    )
