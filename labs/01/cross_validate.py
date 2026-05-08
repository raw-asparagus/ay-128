"""Cross-validation over a single scalar parameter (holdout or k-fold)."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Result of cross-validated model selection over a parameter grid.

    Common base for :class:`HoldoutResult` and :class:`KFoldResult`. Use
    :func:`cross_validate` to construct one of the concrete subclasses
    rather than instantiating this directly.

    Attributes
    ----------
    param_values : ndarray
        Parameter values tested.
    chi2r_train : ndarray
        Reduced chi-squared (χ² / (N - n_params)) on the training set
        per parameter value, averaged across folds when ``n_folds > 1``.
        ``np.nan`` for parameter values where every fold failed.
    mean_chi2_cv : ndarray
        Mean χ² per held-out point (χ² / N_cv, no dof correction —
        parameters were not fit to the held-out data) per parameter
        value, averaged across folds when ``n_folds > 1``. ``np.nan``
        for parameter values where every fold failed.
    best_param : int or float
        Parameter value with the lowest validation chi-squared.
    fold_assignments : list of (train_idx, cv_idx)
        Index arrays for each fold; length equals ``n_folds``.
    failed_param_values : ndarray
        Parameter values where at least one fold raised during
        ``fit_fn``. Empty when every fold succeeded.
    """
    param_values: np.ndarray
    chi2r_train: np.ndarray
    mean_chi2_cv: np.ndarray
    best_param: int | float
    fold_assignments: list[tuple[np.ndarray, np.ndarray]]
    failed_param_values: np.ndarray

    @property
    def n_folds(self) -> int:
        """Return the number of folds."""
        return len(self.fold_assignments)


@dataclass(frozen=True)
class HoldoutResult(ValidationResult):
    """Cross-validation result from a single train/holdout split.

    Adds :attr:`train_idx` / :attr:`cv_idx` shortcuts that are only
    well-defined for the one-fold case.
    """

    @property
    def train_idx(self) -> np.ndarray:
        """Return the training-set indices."""
        return self.fold_assignments[0][0]

    @property
    def cv_idx(self) -> np.ndarray:
        """Return the held-out indices."""
        return self.fold_assignments[0][1]


@dataclass(frozen=True)
class KFoldResult(ValidationResult):
    """Cross-validation result from a k-fold split.

    Per-fold indices are available via :attr:`fold_assignments`.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mean_chi2(y_true: np.ndarray, y_pred: np.ndarray, y_err: np.ndarray) -> float:
    """Mean χ² per point (divides by ``N``, no dof correction).

    For held-out CV evaluation: parameters were not fit to these
    points, so each residual carries a full degree of freedom and the
    ``-n_params`` correction does not apply.
    """
    return float(np.sum(((y_true - y_pred) / y_err) ** 2) / len(y_true))


def _make_folds(
    n: int,
    n_folds: int,
    cv_fraction: float,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
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

    then evaluates ``model.predict(x[cv])`` against ``y[cv]``. Folds
    that raise ``ValueError`` or ``np.linalg.LinAlgError`` during
    fitting are skipped; parameter values for which any fold fails are
    recorded in :attr:`ValidationResult.failed_param_values` and a
    warning is issued.

    Parameters
    ----------
    x, y, y_err : array-like
        Full dataset.
    fit_fn : callable
        ``fit_fn(x, y, y_err, param_value) -> Fit``. Extra arguments
        must be bound before passing. ``param_value`` is forwarded
        unchanged from *param_values* — the callable is responsible for
        any type coercion it requires.
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
    HoldoutResult
        When ``n_folds == 1``.
    KFoldResult
        When ``n_folds > 1``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)
    param_values = np.asarray(param_values)

    rng = np.random.default_rng(seed)
    folds = _make_folds(len(x), n_folds, cv_fraction, rng)

    chi2r_train = np.full(len(param_values), np.nan, dtype=float)
    mean_chi2_cv = np.full(len(param_values), np.nan, dtype=float)
    failed = []

    for i, pv in enumerate(param_values):
        per_fold_train = []
        per_fold_cv = []
        had_failure = False
        for train_idx, cv_idx in folds:
            try:
                model = fit_fn(x[train_idx], y[train_idx], y_err[train_idx], pv)
            except (ValueError, np.linalg.LinAlgError):
                had_failure = True
                continue
            per_fold_train.append(model.chi2_r)
            per_fold_cv.append(_mean_chi2(y[cv_idx], model.predict(x[cv_idx]), y_err[cv_idx]))

        if per_fold_train:
            chi2r_train[i] = float(np.mean(per_fold_train))
            mean_chi2_cv[i] = float(np.mean(per_fold_cv))
        if had_failure:
            failed.append(pv)

    failed_param_values = np.array(failed, dtype=param_values.dtype)
    if np.all(np.isnan(mean_chi2_cv)):
        raise RuntimeError(
            f"cross_validate: every fit failed across the entire parameter "
            f"grid {param_values.tolist()}. Check the data, the grid range, "
            f"or fit_fn for a deeper problem."
        )
    if failed_param_values.size:
        warnings.warn(
            f"cross_validate: {failed_param_values.size} parameter value(s) had "
            f"≥1 fold fail during fitting: {failed_param_values.tolist()}. "
            f"See `failed_param_values` on the returned result.",
            stacklevel=2,
        )

    best_param = param_values[int(np.nanargmin(mean_chi2_cv))]

    cls = HoldoutResult if n_folds == 1 else KFoldResult
    return cls(
        param_values=param_values,
        chi2r_train=chi2r_train,
        mean_chi2_cv=mean_chi2_cv,
        best_param=best_param,
        fold_assignments=folds,
        failed_param_values=failed_param_values,
    )
