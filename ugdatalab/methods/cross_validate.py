from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidationResult:
    """Base result of cross-validated model selection over a parameter grid.

    Attributes
    ----------
    param_values : ndarray
        The parameter values tested (e.g., harmonic orders).
    chi2r_train : ndarray
        Reduced chi-squared on the training set for each parameter value.
    chi2r_cv : ndarray
        Reduced chi-squared on the validation set for each parameter value.
    best_param : int or float
        The parameter value with the lowest validation chi-squared.
    """
    param_values: np.ndarray
    chi2r_train: np.ndarray
    chi2r_cv: np.ndarray
    best_param: int | float


@dataclass(frozen=True)
class HoldoutResult(ValidationResult):
    """Result from a single train/validation split."""
    train_idx: np.ndarray
    cv_idx: np.ndarray


@dataclass(frozen=True)
class KFoldResult(ValidationResult):
    """Result from k-fold cross-validation."""
    n_folds: int


def _chi2r(y_true: np.ndarray, y_pred: np.ndarray, y_err: np.ndarray) -> float:
    return float(np.sum(((y_true - y_pred) / y_err) ** 2) / len(y_true))


def holdout_validate(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    fit_fn: Callable,
    param_values: np.ndarray,
    cv_fraction: float = 0.2,
    seed: int = 346,
) -> HoldoutResult:
    """Validate a model over a single parameter using a holdout split.

    Searches over one scalar parameter at a time.  For each value in
    *param_values*, calls::

        model = fit_fn(x[train], y[train], y_err[train], param_value)

    then evaluates ``model.predict(x[cv])`` against ``y[cv]``.

    Parameters
    ----------
    x, y, y_err : array-like
        Full dataset.
    fit_fn : callable
        ``fit_fn(x, y, y_err, param_value) -> Fit``.  The caller is
        responsible for binding any additional arguments (e.g., period)
        before passing the function here.
    param_values : array-like
        Grid of parameter values to search over.
    cv_fraction : float
        Fraction of data held out for validation (default 0.2).
    seed : int
        Random seed for the train/validation split (default 346).

    Returns
    -------
    HoldoutResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)
    param_values = np.asarray(param_values)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_cv = max(1, int(round(cv_fraction * len(x))))
    cv_idx = idx[:n_cv]
    train_idx = idx[n_cv:]

    chi2r_train = np.full(len(param_values), np.nan, dtype=float)
    chi2r_cv = np.full(len(param_values), np.nan, dtype=float)

    for i, pv in enumerate(param_values):
        try:
            model = fit_fn(x[train_idx], y[train_idx], y_err[train_idx], int(pv))
        except (ValueError, np.linalg.LinAlgError):
            continue

        chi2r_train[i] = model.chi2_r
        chi2r_cv[i] = _chi2r(y[cv_idx], model.predict(x[cv_idx]), y_err[cv_idx])

    best_param = param_values[int(np.nanargmin(chi2r_cv))]

    return HoldoutResult(
        param_values=param_values,
        chi2r_train=chi2r_train,
        chi2r_cv=chi2r_cv,
        best_param=best_param,
        train_idx=train_idx,
        cv_idx=cv_idx,
    )


def k_fold_validate(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    fit_fn: Callable,
    param_values: np.ndarray,
    n_folds: int = 5,
    seed: int = 346,
) -> KFoldResult:
    """Validate a model over a single parameter using k-fold cross-validation.

    Parameters
    ----------
    x, y, y_err : array-like
        Full dataset.
    fit_fn : callable
        ``fit_fn(x, y, y_err, param_value) -> Fit``.
    param_values : array-like
        Grid of parameter values to search over.
    n_folds : int
        Number of folds (default 5).
    seed : int
        Random seed for the fold assignment (default 346).

    Returns
    -------
    KFoldResult
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)
    param_values = np.asarray(param_values)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    folds = np.array_split(idx, n_folds)

    chi2r_train = np.full(len(param_values), np.nan, dtype=float)
    chi2r_cv = np.full(len(param_values), np.nan, dtype=float)

    for i, pv in enumerate(param_values):
        fold_chi2r_train = []
        fold_chi2r_cv = []

        for fold_idx in folds:
            train_idx = np.setdiff1d(idx, fold_idx)

            try:
                model = fit_fn(x[train_idx], y[train_idx], y_err[train_idx], int(pv))
            except (ValueError, np.linalg.LinAlgError):
                continue

            fold_chi2r_train.append(model.chi2_r)
            fold_chi2r_cv.append(_chi2r(y[fold_idx], model.predict(x[fold_idx]), y_err[fold_idx]))

        if fold_chi2r_train:
            chi2r_train[i] = float(np.mean(fold_chi2r_train))
            chi2r_cv[i] = float(np.mean(fold_chi2r_cv))

    best_param = param_values[int(np.nanargmin(chi2r_cv))]

    return KFoldResult(
        param_values=param_values,
        chi2r_train=chi2r_train,
        chi2r_cv=chi2r_cv,
        best_param=best_param,
        n_folds=n_folds,
    )
