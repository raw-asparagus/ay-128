from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar, least_squares

from ugdatalab.methods.base import Fit


def _build_cannon_design_matrix(labels: np.ndarray) -> np.ndarray:
    """Build design matrix from centered/scaled labels.

    Parameters
    ----------
    labels : ndarray, shape (N, L)
        Centered and scaled label array.

    Returns
    -------
    ndarray, shape (N, 1 + L + L*(L+1)/2)
        Design matrix with constant, linear, and quadratic cross terms.
    """
    N, L = labels.shape
    terms = [np.ones(N)]
    for i in range(L):
        terms.append(labels[:, i])
    for i in range(L):
        for j in range(i, L):
            terms.append(labels[:, i] * labels[:, j])
    return np.column_stack(terms)


def _build_cannon_design_vector(labels: np.ndarray) -> np.ndarray:
    """Build design vector for a single star.

    Parameters
    ----------
    labels : ndarray, shape (L,)
        Centered and scaled label vector.

    Returns
    -------
    ndarray, shape (1 + L + L*(L+1)/2,)
    """
    return _build_cannon_design_matrix(labels.reshape(1, -1)).ravel()


def _train_pixel(flux_pixel, error_pixel, design_matrix) -> tuple:
    """Train one pixel via profile likelihood with a scatter floor.

    For fixed s², theta is WLS: theta = (X^T W X)^{-1} X^T W f
    where W = diag(1 / (sigma² + s²)).

    Optimizes over log(s²) via bounded scalar minimization, solving WLS
    analytically at each evaluation. The lower bound on s² is set to the
    median measurement variance at this pixel, enforcing that model
    inadequacy is at least comparable to the measurement noise floor.

    Parameters
    ----------
    flux_pixel : ndarray, shape (N,)
    error_pixel : ndarray, shape (N,)
    design_matrix : ndarray, shape (N, n_terms)

    Returns
    -------
    theta : ndarray, shape (n_terms,)
    s2 : float
        Optimized intrinsic scatter variance.
    """
    valid = np.isfinite(error_pixel)
    n_valid = np.sum(valid)
    n_terms = design_matrix.shape[1]

    if n_valid < max(25, n_terms + 1):
        return np.zeros(n_terms), np.inf

    f = flux_pixel[valid]
    sigma2 = error_pixel[valid] ** 2
    X = design_matrix[valid]

    # Scatter floor: median measurement variance at this pixel
    log_s2_floor = np.log(np.median(sigma2))

    def _neg_log_likelihood(log_s2):
        s2 = np.exp(log_s2)
        var = sigma2 + s2
        w = 1.0 / var
        XtW = X.T * w
        A = XtW @ X
        b = XtW @ f
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e30
        resid = f - X @ theta
        return 0.5 * np.sum(np.log(var) + resid ** 2 / var)

    result = minimize_scalar(
        _neg_log_likelihood, bounds=(log_s2_floor, log_s2_floor + 10),
        method="bounded",
    )
    s2_opt = np.exp(result.x)

    var = sigma2 + s2_opt
    w = 1.0 / var
    XtW = X.T * w
    A = XtW @ X
    b = XtW @ f
    try:
        theta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A, b, rcond=None)[0]

    return theta, s2_opt


@dataclass(frozen=True)
class CannonModel(Fit):
    """Result of training The Cannon spectral model.

    Attributes
    ----------
    theta : ndarray, shape (n_pixels, n_terms)
        Per-pixel polynomial coefficients.
    scatter : ndarray, shape (n_pixels,)
        Per-pixel intrinsic scatter variance s².
    label_names : list of str
        Label column names.
    label_means : ndarray, shape (n_labels,)
        Centering constants from the training set.
    label_stds : ndarray, shape (n_labels,)
        Scaling constants from the training set.
    wavelength : ndarray, shape (n_pixels,)
        Wavelength grid.
    chi2_r : float
        Mean reduced chi-squared across the training set.
    """
    theta: np.ndarray
    scatter: np.ndarray
    label_names: list
    label_means: np.ndarray
    label_stds: np.ndarray
    wavelength: np.ndarray
    chi2_r: float

    @property
    def _trained_mask(self) -> np.ndarray:
        """Boolean mask of pixels that were actually trained (finite scatter)."""
        return np.isfinite(self.scatter)

    def predict(self, labels: np.ndarray) -> np.ndarray:
        """Predict spectrum for a single star.

        Parameters
        ----------
        labels : ndarray, shape (n_labels,)
            Stellar labels in original (unscaled) space.

        Returns
        -------
        ndarray, shape (n_pixels,)
            NaN at untrained pixels (chip gaps, edges).
        """
        scaled = (labels - self.label_means) / self.label_stds
        dv = _build_cannon_design_vector(scaled)
        result = self.theta @ dv
        result[~self._trained_mask] = np.nan
        return result

    def predict_batch(self, labels: np.ndarray) -> np.ndarray:
        """Predict spectra for N stars.

        Parameters
        ----------
        labels : ndarray, shape (N, n_labels)
            Stellar labels in original (unscaled) space.

        Returns
        -------
        ndarray, shape (N, n_pixels)
            NaN at untrained pixels (chip gaps, edges).
        """
        scaled = (labels - self.label_means) / self.label_stds
        X = _build_cannon_design_matrix(scaled)
        result = X @ self.theta.T
        result[:, ~self._trained_mask] = np.nan
        return result

    def gradient(self, label_idx: int) -> np.ndarray:
        """Gradient spectrum df/dl_i at the training-set mean.

        Computed analytically from the polynomial structure. Returns the
        derivative with respect to the *unscaled* label (divides by the
        label's standard deviation).

        Parameters
        ----------
        label_idx : int
            Index of the label (0-based).

        Returns
        -------
        ndarray, shape (n_pixels,)
        """
        n_labels = len(self.label_means)
        # At the training-set mean, scaled labels are all zero.
        # d(design_vector)/d(scaled_l_i) evaluated at zero:
        #   - constant term: 0
        #   - linear term l_i: 1, others: 0
        #   - quadratic l_i^2: 2*l_i = 0 at zero
        #   - cross term l_i*l_j (i!=j): l_j = 0 at zero
        # So the gradient in scaled space is simply theta[:, 1 + label_idx].
        grad_scaled = self.theta[:, 1 + label_idx]
        return grad_scaled / self.label_stds[label_idx]


def train_cannon(
    flux: np.ndarray,
    error: np.ndarray,
    labels: np.ndarray,
    wavelength: np.ndarray,
    label_names: list | None = None,
) -> CannonModel:
    """Train The Cannon spectral model.

    Parameters
    ----------
    flux : ndarray, shape (N, n_pixels)
        Normalized training spectra.
    error : ndarray, shape (N, n_pixels)
        Per-pixel errors (1e6 sentinel for masked pixels).
    labels : ndarray, shape (N, n_labels)
        Stellar labels for each training star.
    wavelength : ndarray, shape (n_pixels,)
        Wavelength grid.
    label_names : list of str, optional
        Label column names.

    Returns
    -------
    CannonModel
    """
    N, n_pixels = flux.shape
    n_labels = labels.shape[1]

    label_means = np.mean(labels, axis=0)
    label_stds = np.std(labels, axis=0)
    label_stds[label_stds == 0] = 1.0
    scaled = (labels - label_means) / label_stds
    X = _build_cannon_design_matrix(scaled)
    n_terms = X.shape[1]

    if label_names is None:
        label_names = [f"label_{i}" for i in range(n_labels)]

    theta_all = np.zeros((n_pixels, n_terms))
    scatter_all = np.full(n_pixels, np.inf)

    # Skip pixels that fail either of two trainability criteria:
    #
    # 1. Data completeness: >= 99.7% of training stars must have finite
    #    errors.  Chip boundaries vary slightly across stars, creating a
    #    ramp-down zone at each chip edge; this threshold excludes the
    #    transition region where many stars have NaN.
    #
    # 2. Median per-pixel SNR >= 3.  After continuum normalization, the
    #    last few pixels before a chip gap can have finite but enormous
    #    errors (the continuum polynomial extrapolates toward small values
    #    at the chip edge, inflating normalized errors).  These pixels
    #    pass the completeness check but carry negligible information —
    #    the median training-star SNR is < 1 even though valid_frac ≈ 1.
    #    The SNR criterion catches the 2 pixels (at ~16430 Å) that leak
    #    through the completeness check alone.
    valid_frac = np.mean(np.isfinite(error), axis=0)
    finite_mask = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    with np.errstate(invalid="ignore"):
        snr = np.where(finite_mask, np.abs(flux) / error, np.nan)
    median_snr = np.nanmedian(snr, axis=0)
    trainable = (valid_frac >= 0.997) & (median_snr >= 3)

    for pix in range(n_pixels):
        if not trainable[pix]:
            continue
        theta_all[pix], scatter_all[pix] = _train_pixel(
            flux[:, pix], error[:, pix], X,
        )

    # Training chi2_r: mean across stars of mean per-pixel chi2
    predicted = X @ theta_all.T
    resid = flux - predicted
    var = error ** 2 + scatter_all
    valid = np.isfinite(error) & (error < 1e5) & np.isfinite(scatter_all)
    chi2_terms = np.where(valid, resid ** 2 / var, 0.0)
    n_valid_per_star = np.sum(valid, axis=1)
    chi2_per_star = np.sum(chi2_terms, axis=1) / np.maximum(
        n_valid_per_star - n_terms, 1,
    )
    chi2_r = float(np.mean(chi2_per_star[n_valid_per_star > n_terms]))

    return CannonModel(
        theta=theta_all,
        scatter=scatter_all,
        label_names=list(label_names),
        label_means=label_means,
        label_stds=label_stds,
        wavelength=wavelength,
        chi2_r=chi2_r,
    )


# ---------------------------------------------------------------------------
# Label fitting (inverse problem)
# ---------------------------------------------------------------------------

_LABEL_BOUNDS_LOWER = np.array([2800.0, -0.5, -1.5, -0.5, -0.7])
_LABEL_BOUNDS_UPPER = np.array([6000.0,  4.5,  0.8,  0.7,  0.6])


def _design_vector_jacobian(scaled_labels, n_labels):
    """Jacobian of the design vector w.r.t. scaled labels.

    Parameters
    ----------
    scaled_labels : ndarray, shape (n_labels,)
    n_labels : int

    Returns
    -------
    ndarray, shape (n_terms, n_labels)
        J[t, i] = d(design_vector[t]) / d(scaled_label[i])
    """
    n_terms = 1 + n_labels + n_labels * (n_labels + 1) // 2
    J = np.zeros((n_terms, n_labels))

    # Linear terms: d(l_i)/d(l_i) = 1
    for i in range(n_labels):
        J[1 + i, i] = 1.0

    # Quadratic and cross terms
    idx = 1 + n_labels
    for i in range(n_labels):
        for j in range(i, n_labels):
            if i == j:
                J[idx, i] = 2.0 * scaled_labels[i]
            else:
                J[idx, i] = scaled_labels[j]
                J[idx, j] = scaled_labels[i]
            idx += 1

    return J


def fit_star_labels(
    model: CannonModel,
    flux: np.ndarray,
    error: np.ndarray,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Fit labels for a single star by minimizing chi-squared.

    Parameters
    ----------
    model : CannonModel
        Trained Cannon model.
    flux : ndarray, shape (n_pixels,)
        Observed normalized flux.
    error : ndarray, shape (n_pixels,)
        Per-pixel errors (1e6 sentinel for masked pixels).
    x0 : ndarray, shape (n_labels,), optional
        Initial guess in original label space.  Defaults to training-set means.

    Returns
    -------
    ndarray, shape (n_labels,)
        Fitted labels in original (unscaled) space.
    """
    n_labels = len(model.label_means)
    valid = np.isfinite(error) & (error < 1e5) & np.isfinite(model.scatter)
    f = flux[valid]
    sigma2 = error[valid] ** 2
    s2 = model.scatter[valid]
    inv_std = 1.0 / np.sqrt(sigma2 + s2)
    theta_valid = model.theta[valid]

    if x0 is None:
        x0 = model.label_means.copy()

    # Work in scaled space for the optimizer
    x0_scaled = (x0 - model.label_means) / model.label_stds
    lb_scaled = (np.maximum(_LABEL_BOUNDS_LOWER, model.label_means - 5 * model.label_stds) - model.label_means) / model.label_stds
    ub_scaled = (np.minimum(_LABEL_BOUNDS_UPPER, model.label_means + 5 * model.label_stds) - model.label_means) / model.label_stds

    def _residuals(scaled_labels):
        dv = _build_cannon_design_vector(scaled_labels)
        pred = theta_valid @ dv
        return (f - pred) * inv_std

    def _jacobian(scaled_labels):
        dv_jac = _design_vector_jacobian(scaled_labels, n_labels)
        dpred_dl = theta_valid @ dv_jac
        return -dpred_dl * inv_std[:, None]

    result = least_squares(
        _residuals, x0_scaled, jac=_jacobian, method="trf",
        bounds=(lb_scaled, ub_scaled),
    )

    return result.x * model.label_stds + model.label_means


def fit_labels_batch(
    model: CannonModel,
    flux: np.ndarray,
    error: np.ndarray,
) -> np.ndarray:
    """Fit labels for N stars.

    Parameters
    ----------
    model : CannonModel
    flux : ndarray, shape (N, n_pixels)
    error : ndarray, shape (N, n_pixels)

    Returns
    -------
    ndarray, shape (N, n_labels)
        Fitted labels in original (unscaled) space.
    """
    from tqdm.auto import tqdm

    N = flux.shape[0]
    n_labels = len(model.label_means)
    fitted = np.empty((N, n_labels))
    for i in tqdm(range(N), desc="Fitting labels"):
        fitted[i] = fit_star_labels(model, flux[i], error[i])
    return fitted
