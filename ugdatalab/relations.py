from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from astropy import table

from ugdatalab.mcmc import MetropolisHastings, NoUTurnHamiltonian
from ugdatalab.models.gaia import _as_float_array, rrlyrae_class_mask


@dataclass
class RelationData:
    rr_class: str
    relation_kind: str
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    x_label: str
    y_label: str
    data_label: str


@dataclass(frozen=True)
class OpticalPLComparisonData:
    rr_class: str
    x_obs: np.ndarray
    y_obs: np.ndarray
    sigma_obs: np.ndarray
    x_grid: np.ndarray
    median_mean: np.ndarray
    predictive_q16: np.ndarray
    predictive_q84: np.ndarray
    slope_q16: float
    slope_q50: float
    slope_q84: float
    intercept_q16: float
    intercept_q50: float
    intercept_q84: float
    sigma_scatter_q16: float
    sigma_scatter_q50: float
    sigma_scatter_q84: float


@dataclass(frozen=True)
class OpticalPCComparisonData:
    rr_class: str
    x_obs: np.ndarray
    y_obs: np.ndarray
    sigma_obs: np.ndarray
    x_grid: np.ndarray
    median_mean: np.ndarray
    predictive_q16: np.ndarray
    predictive_q84: np.ndarray
    slope_q16: float
    slope_q50: float
    slope_q84: float
    intercept_q16: float
    intercept_q50: float
    intercept_q84: float
    sigma_scatter_q16: float
    sigma_scatter_q50: float
    sigma_scatter_q84: float
    slope_median: float
    slope_std: float
    intercept_median: float
    intercept_std: float
    intrinsic_sigma_median: float
    intrinsic_sigma_std: float


_RELATION_META = {
    "pl": {
        "y_label": r"$M_G$ [mag]",
        "data_label": "period-luminosity",
        "labels": [
            r"$a$",
            r"$b$",
            r"$\log_{10}\sigma_\mathrm{scatter}$",
        ],
        "default_proposal_std": np.array([0.05, 0.15, 0.08]),
    },
    "pc": {
        "y_label": r"$G_\mathrm{BP} - G_\mathrm{RP}$ [mag]",
        "data_label": "period-color",
        "labels": [
            r"$a_c$",
            r"$b_c$",
            r"$\log_{10}\sigma_c$",
        ],
        "default_proposal_std": np.array([0.03, 0.05, 0.06]),
    },
}


def _class_mask(data, rr_class: str) -> np.ndarray:
    if rr_class not in {"RRab", "RRc"}:
        raise ValueError(f"Unsupported RR Lyrae class: {rr_class}")
    return rrlyrae_class_mask(data, rr_class)


def _period_column(data, rr_class: str) -> np.ndarray:
    if rr_class == "RRab":
        return _as_float_array(data["pf"])
    return _as_float_array(data["p1_o"])


def relation_parameter_labels(relation_kind: str) -> list[str]:
    if relation_kind not in _RELATION_META:
        raise ValueError(f"Unsupported relation kind: {relation_kind}")
    return list(_RELATION_META[relation_kind]["labels"])


def _subsample_draws(values: np.ndarray, *, max_draws: int = 400, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    step = max(len(values) // max_draws, 1)
    sample_pool = np.asarray(values[::step], dtype=float)
    if len(sample_pool) > max_draws:
        keep_idx = rng.choice(len(sample_pool), size=max_draws, replace=False)
        sample_pool = sample_pool[keep_idx]
    return sample_pool


def _predictive_summary(
    ctx: Any,
    values: np.ndarray,
    *,
    sigma_transform=lambda arr: arr,
    seed: int = 42,
    n_grid: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_obs = _as_float_array(ctx.x_raw)
    sigma_obs = _as_float_array(ctx.sigma)
    x_mean = float(ctx.x_mean)
    if len(x_obs) == 0:
        raise ValueError(f"No observations available for {getattr(ctx, 'class_label', 'relation')}.")

    x_grid = np.linspace(float(np.min(x_obs)), float(np.max(x_obs)), int(n_grid))
    order = np.argsort(x_obs)
    sigma_logp = getattr(ctx, "sigma_logp", np.zeros_like(x_obs))
    sigma_logp = _as_float_array(sigma_logp)
    sigma_obs_grid = np.interp(x_grid, x_obs[order], sigma_obs[order])
    sigma_x_grid = np.interp(x_grid, x_obs[order], sigma_logp[order])

    rng = np.random.default_rng(seed)
    sample_pool = _subsample_draws(values, max_draws=400, seed=seed)
    mean_draws = np.empty((len(sample_pool), len(x_grid)), dtype=float)
    predictive_draws = np.empty_like(mean_draws)
    for i, (slope_draw, intercept_draw, sigma_draw) in enumerate(sample_pool):
        sigma_scatter_draw = sigma_transform(sigma_draw)
        mu_grid = slope_draw * (x_grid - x_mean) + intercept_draw
        sigma_pred = np.sqrt(
            sigma_obs_grid**2 + sigma_scatter_draw**2 + (slope_draw * sigma_x_grid) ** 2
        )
        mean_draws[i] = mu_grid
        predictive_draws[i] = rng.normal(mu_grid, sigma_pred)

    return (
        x_grid,
        np.quantile(mean_draws, 0.50, axis=0),
        np.quantile(predictive_draws, 0.16, axis=0),
        np.quantile(predictive_draws, 0.84, axis=0),
    )


def build_optical_pl_comparison_data(
    ctx: Any,
    samples: np.ndarray,
    *,
    n_grid: int = 300,
) -> OpticalPLComparisonData:
    """Summarize native-PyMC optical PL samples into the values plotted in Lab 1-04."""
    rr_class = getattr(ctx, "class_label", None)
    if rr_class not in {"RRab", "RRc"}:
        raise ValueError("Expected `ctx.class_label` to be `RRab` or `RRc`.")

    x_obs = _as_float_array(ctx.x_raw)
    y_obs = _as_float_array(ctx.y)
    sigma_obs = _as_float_array(ctx.sigma)
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected posterior samples with shape (n_samples, 3).")

    x_grid, median_mean, predictive_q16, predictive_q84 = _predictive_summary(
        ctx,
        values,
        seed=42,
        n_grid=n_grid,
    )
    q16, q50, q84 = np.percentile(values, [16, 50, 84], axis=0)
    return OpticalPLComparisonData(
        rr_class=rr_class,
        x_obs=x_obs,
        y_obs=y_obs,
        sigma_obs=sigma_obs,
        x_grid=x_grid,
        median_mean=median_mean,
        predictive_q16=predictive_q16,
        predictive_q84=predictive_q84,
        slope_q16=float(q16[0]),
        slope_q50=float(q50[0]),
        slope_q84=float(q84[0]),
        intercept_q16=float(q16[1]),
        intercept_q50=float(q50[1]),
        intercept_q84=float(q84[1]),
        sigma_scatter_q16=float(q16[2]),
        sigma_scatter_q50=float(q50[2]),
        sigma_scatter_q84=float(q84[2]),
    )


def build_infrared_pl_comparison_data(
    ctx: Any,
    samples: np.ndarray,
    *,
    n_grid: int = 300,
) -> OpticalPLComparisonData:
    """Alias for the WISE `W2` period-luminosity handoff artifact."""
    return build_optical_pl_comparison_data(ctx, samples, n_grid=n_grid)


def build_optical_pc_comparison_data(
    ctx: Any,
    samples: np.ndarray,
    *,
    n_grid: int = 300,
) -> OpticalPCComparisonData:
    """Summarize centered `[slope, intercept, log10_sigma]` PC samples for plotting and dust analysis."""
    rr_class = getattr(ctx, "class_label", None)
    if rr_class not in {"RRab", "RRc"}:
        raise ValueError("Expected `ctx.class_label` to be `RRab` or `RRc`.")

    x_obs = _as_float_array(ctx.x_raw)
    y_obs = _as_float_array(ctx.y)
    sigma_obs = _as_float_array(ctx.sigma)
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected posterior samples with shape (n_samples, 3).")

    x_grid, median_mean, predictive_q16, predictive_q84 = _predictive_summary(
        ctx,
        values,
        sigma_transform=lambda arr: 10.0 ** arr,
        seed=42,
        n_grid=n_grid,
    )
    q16, q50, q84 = np.percentile(values, [16, 50, 84], axis=0)
    raw_intercepts = values[:, 1] - values[:, 0] * float(ctx.x_mean)
    intercept_q16, intercept_q50, intercept_q84 = np.percentile(raw_intercepts, [16, 50, 84])
    sigma_q16, sigma_q50, sigma_q84 = 10.0 ** q16[2], 10.0 ** q50[2], 10.0 ** q84[2]
    sigma_values = 10.0 ** values[:, 2]
    return OpticalPCComparisonData(
        rr_class=rr_class,
        x_obs=x_obs,
        y_obs=y_obs,
        sigma_obs=sigma_obs,
        x_grid=x_grid,
        median_mean=median_mean,
        predictive_q16=predictive_q16,
        predictive_q84=predictive_q84,
        slope_q16=float(q16[0]),
        slope_q50=float(q50[0]),
        slope_q84=float(q84[0]),
        intercept_q16=float(intercept_q16),
        intercept_q50=float(intercept_q50),
        intercept_q84=float(intercept_q84),
        sigma_scatter_q16=float(sigma_q16),
        sigma_scatter_q50=float(sigma_q50),
        sigma_scatter_q84=float(sigma_q84),
        slope_median=float(np.median(values[:, 0])),
        slope_std=float(np.std(values[:, 0])),
        intercept_median=float(np.median(raw_intercepts)),
        intercept_std=float(np.std(raw_intercepts)),
        intrinsic_sigma_median=float(10.0 ** np.median(values[:, 2])),
        intrinsic_sigma_std=float(np.std(sigma_values)),
    )


def save_optical_pl_comparison_data(
    path: str | Path,
    comparison_map: Mapping[str, OpticalPLComparisonData],
) -> Path:
    """Save the Lab 1 optical PL comparison handoff artifact."""
    payload: dict[str, np.ndarray | float] = {}
    for rr_class, comparison in comparison_map.items():
        prefix = rr_class.lower()
        payload[f"{prefix}_x_obs"] = np.asarray(comparison.x_obs, dtype=float)
        payload[f"{prefix}_y_obs"] = np.asarray(comparison.y_obs, dtype=float)
        payload[f"{prefix}_sigma_obs"] = np.asarray(comparison.sigma_obs, dtype=float)
        payload[f"{prefix}_x_grid"] = np.asarray(comparison.x_grid, dtype=float)
        payload[f"{prefix}_median_mean"] = np.asarray(comparison.median_mean, dtype=float)
        payload[f"{prefix}_predictive_q16"] = np.asarray(comparison.predictive_q16, dtype=float)
        payload[f"{prefix}_predictive_q84"] = np.asarray(comparison.predictive_q84, dtype=float)
        payload[f"{prefix}_slope_q16"] = float(comparison.slope_q16)
        payload[f"{prefix}_slope_q50"] = float(comparison.slope_q50)
        payload[f"{prefix}_slope_q84"] = float(comparison.slope_q84)
        payload[f"{prefix}_intercept_q16"] = float(comparison.intercept_q16)
        payload[f"{prefix}_intercept_q50"] = float(comparison.intercept_q50)
        payload[f"{prefix}_intercept_q84"] = float(comparison.intercept_q84)
        payload[f"{prefix}_sigma_scatter_q16"] = float(comparison.sigma_scatter_q16)
        payload[f"{prefix}_sigma_scatter_q50"] = float(comparison.sigma_scatter_q50)
        payload[f"{prefix}_sigma_scatter_q84"] = float(comparison.sigma_scatter_q84)

    output_path = Path(path)
    np.savez(output_path, **payload)
    return output_path


def save_infrared_pl_comparison_data(
    path: str | Path,
    comparison_map: Mapping[str, OpticalPLComparisonData],
) -> Path:
    """Save the Lab 1 infrared PL comparison handoff artifact."""
    return save_optical_pl_comparison_data(path, comparison_map)


def save_optical_pc_comparison_data(
    path: str | Path,
    comparison_map: Mapping[str, OpticalPCComparisonData],
) -> Path:
    """Save the Lab 1 optical PC comparison handoff artifact."""
    payload: dict[str, np.ndarray | float] = {}
    for rr_class, comparison in comparison_map.items():
        prefix = rr_class.lower()
        payload[f"{prefix}_x_obs"] = np.asarray(comparison.x_obs, dtype=float)
        payload[f"{prefix}_y_obs"] = np.asarray(comparison.y_obs, dtype=float)
        payload[f"{prefix}_sigma_obs"] = np.asarray(comparison.sigma_obs, dtype=float)
        payload[f"{prefix}_x_grid"] = np.asarray(comparison.x_grid, dtype=float)
        payload[f"{prefix}_median_mean"] = np.asarray(comparison.median_mean, dtype=float)
        payload[f"{prefix}_predictive_q16"] = np.asarray(comparison.predictive_q16, dtype=float)
        payload[f"{prefix}_predictive_q84"] = np.asarray(comparison.predictive_q84, dtype=float)
        payload[f"{prefix}_slope_q16"] = float(comparison.slope_q16)
        payload[f"{prefix}_slope_q50"] = float(comparison.slope_q50)
        payload[f"{prefix}_slope_q84"] = float(comparison.slope_q84)
        payload[f"{prefix}_intercept_q16"] = float(comparison.intercept_q16)
        payload[f"{prefix}_intercept_q50"] = float(comparison.intercept_q50)
        payload[f"{prefix}_intercept_q84"] = float(comparison.intercept_q84)
        payload[f"{prefix}_sigma_scatter_q16"] = float(comparison.sigma_scatter_q16)
        payload[f"{prefix}_sigma_scatter_q50"] = float(comparison.sigma_scatter_q50)
        payload[f"{prefix}_sigma_scatter_q84"] = float(comparison.sigma_scatter_q84)
        payload[f"{prefix}_slope_median"] = float(comparison.slope_median)
        payload[f"{prefix}_slope_std"] = float(comparison.slope_std)
        payload[f"{prefix}_intercept_median"] = float(comparison.intercept_median)
        payload[f"{prefix}_intercept_std"] = float(comparison.intercept_std)
        payload[f"{prefix}_intrinsic_sigma_median"] = float(comparison.intrinsic_sigma_median)
        payload[f"{prefix}_intrinsic_sigma_std"] = float(comparison.intrinsic_sigma_std)

    output_path = Path(path)
    np.savez(output_path, **payload)
    return output_path


def load_optical_pl_comparison_data(
    path: str | Path,
) -> dict[str, OpticalPLComparisonData]:
    """Load the summarized Lab 1 optical PL comparison handoff artifact."""
    comparison_map: dict[str, OpticalPLComparisonData] = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        for rr_class in ("RRab", "RRc"):
            prefix = rr_class.lower()
            comparison_map[rr_class] = OpticalPLComparisonData(
                rr_class=rr_class,
                x_obs=np.asarray(archive[f"{prefix}_x_obs"], dtype=float),
                y_obs=np.asarray(archive[f"{prefix}_y_obs"], dtype=float),
                sigma_obs=np.asarray(archive[f"{prefix}_sigma_obs"], dtype=float),
                x_grid=np.asarray(archive[f"{prefix}_x_grid"], dtype=float),
                median_mean=np.asarray(archive[f"{prefix}_median_mean"], dtype=float),
                predictive_q16=np.asarray(archive[f"{prefix}_predictive_q16"], dtype=float),
                predictive_q84=np.asarray(archive[f"{prefix}_predictive_q84"], dtype=float),
                slope_q16=float(archive[f"{prefix}_slope_q16"]),
                slope_q50=float(archive[f"{prefix}_slope_q50"]),
                slope_q84=float(archive[f"{prefix}_slope_q84"]),
                intercept_q16=float(archive[f"{prefix}_intercept_q16"]),
                intercept_q50=float(archive[f"{prefix}_intercept_q50"]),
                intercept_q84=float(archive[f"{prefix}_intercept_q84"]),
                sigma_scatter_q16=float(archive[f"{prefix}_sigma_scatter_q16"]),
                sigma_scatter_q50=float(archive[f"{prefix}_sigma_scatter_q50"]),
                sigma_scatter_q84=float(archive[f"{prefix}_sigma_scatter_q84"]),
            )
    return comparison_map


def load_infrared_pl_comparison_data(
    path: str | Path,
) -> dict[str, OpticalPLComparisonData]:
    """Load the Lab 1 infrared PL comparison handoff artifact."""
    return load_optical_pl_comparison_data(path)


def load_optical_pc_comparison_data(
    path: str | Path,
) -> dict[str, OpticalPCComparisonData]:
    """Load the summarized Lab 1 optical PC comparison handoff artifact."""
    comparison_map: dict[str, OpticalPCComparisonData] = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        for rr_class in ("RRab", "RRc"):
            prefix = rr_class.lower()
            comparison_map[rr_class] = OpticalPCComparisonData(
                rr_class=rr_class,
                x_obs=np.asarray(archive[f"{prefix}_x_obs"], dtype=float),
                y_obs=np.asarray(archive[f"{prefix}_y_obs"], dtype=float),
                sigma_obs=np.asarray(archive[f"{prefix}_sigma_obs"], dtype=float),
                x_grid=np.asarray(archive[f"{prefix}_x_grid"], dtype=float),
                median_mean=np.asarray(archive[f"{prefix}_median_mean"], dtype=float),
                predictive_q16=np.asarray(archive[f"{prefix}_predictive_q16"], dtype=float),
                predictive_q84=np.asarray(archive[f"{prefix}_predictive_q84"], dtype=float),
                slope_q16=float(archive[f"{prefix}_slope_q16"]),
                slope_q50=float(archive[f"{prefix}_slope_q50"]),
                slope_q84=float(archive[f"{prefix}_slope_q84"]),
                intercept_q16=float(archive[f"{prefix}_intercept_q16"]),
                intercept_q50=float(archive[f"{prefix}_intercept_q50"]),
                intercept_q84=float(archive[f"{prefix}_intercept_q84"]),
                sigma_scatter_q16=float(archive[f"{prefix}_sigma_scatter_q16"]),
                sigma_scatter_q50=float(archive[f"{prefix}_sigma_scatter_q50"]),
                sigma_scatter_q84=float(archive[f"{prefix}_sigma_scatter_q84"]),
                slope_median=float(archive[f"{prefix}_slope_median"]),
                slope_std=float(archive[f"{prefix}_slope_std"]),
                intercept_median=float(archive[f"{prefix}_intercept_median"]),
                intercept_std=float(archive[f"{prefix}_intercept_std"]),
                intrinsic_sigma_median=float(archive[f"{prefix}_intrinsic_sigma_median"]),
                intrinsic_sigma_std=float(archive[f"{prefix}_intrinsic_sigma_std"]),
            )
    return comparison_map


def prepare_relation_data(source, rr_class: str, relation_kind: str) -> RelationData:
    if relation_kind not in _RELATION_META:
        raise ValueError(f"Unsupported relation kind: {relation_kind}")

    data = source.data
    mask = _class_mask(data, rr_class)
    sub = data[mask]

    period = _period_column(sub, rr_class)
    x = np.log10(period)

    if relation_kind == "pl":
        y = _as_float_array(sub["M_G"])
        sigma = _as_float_array(sub["sigma_M"])
    else:
        y = _as_float_array(sub["bp_rp"])
        snr_bp = _as_float_array(sub["phot_bp_mean_flux_over_error"])
        snr_rp = _as_float_array(sub["phot_rp_mean_flux_over_error"])
        sigma = (2.5 / np.log(10)) * np.sqrt(1.0 / snr_bp**2 + 1.0 / snr_rp**2)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(ok):
        raise ValueError(
            f"No valid points available for {rr_class} {relation_kind} relation."
        )

    meta = _RELATION_META[relation_kind]
    return RelationData(
        rr_class=rr_class,
        relation_kind=relation_kind,
        x=x[ok],
        y=y[ok],
        sigma=sigma[ok],
        x_label=r"$\log_{10}(P\,/\,\mathrm{day})$",
        y_label=meta["y_label"],
        data_label=f"{rr_class} {meta['data_label']}",
    )


def _log_prior(theta: np.ndarray) -> float:
    if theta.shape != (3,):
        raise ValueError("Expected a 3-parameter relation vector [slope, intercept, log10_sigma].")
    if not np.all(np.isfinite(theta)):
        return -np.inf
    if not (-3.0 < theta[2] < 1.0):
        return -np.inf
    return 0.0


def _log_likelihood(theta: np.ndarray, data: RelationData) -> float:
    slope, intercept, log10_sig = theta
    sig2 = data.sigma**2 + (10.0**log10_sig) ** 2
    model = slope * data.x + intercept
    return -0.5 * np.sum(np.log(2 * np.pi * sig2) + (data.y - model) ** 2 / sig2)


def make_relation_log_posterior(data: RelationData):
    def log_posterior(theta: np.ndarray) -> float:
        lp = _log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + _log_likelihood(theta, data)

    return log_posterior


def _wls_theta0(data: RelationData) -> np.ndarray:
    A = np.column_stack([data.x, np.ones_like(data.x)])
    w = 1.0 / data.sigma
    beta, _, _, _ = np.linalg.lstsq(A * w[:, None], data.y * w, rcond=None)
    slope, intercept = beta

    resid = data.y - (slope * data.x + intercept)
    scatter = float(np.std(resid))
    sigma0 = float(np.clip(max(scatter, np.median(data.sigma), 1e-3), 1e-3, 10.0))
    return np.array([slope, intercept, np.log10(sigma0)], dtype=float)


def estimate_initial_theta0(
    source,
    rr_class: str,
    relation_kind: str,
    data: RelationData | None = None,
) -> np.ndarray:
    if data is None:
        data = prepare_relation_data(source, rr_class, relation_kind)

    if relation_kind == "pl" and hasattr(source, "mcmc_results"):
        class_result = getattr(source, "mcmc_results", {}).get(rr_class)
        if class_result is not None:
            intercept = float(class_result["a"][0])
            slope = float(class_result["b"][0])
            sigma0 = float(class_result["sig_scatter"][0])
            sigma0 = float(np.clip(sigma0, 1e-3, 10.0))
            return np.array([slope, intercept, np.log10(sigma0)], dtype=float)

    return _wls_theta0(data)


def _tune_proposal_std(
    log_prob,
    theta0: np.ndarray,
    proposal_std: np.ndarray,
    labels: list[str],
    seed: int | None,
    target_range: tuple[float, float] = (0.15, 0.50),
    pilot_steps: int = 4000,
    pilot_burn: int = 1000,
    max_rounds: int = 6,
) -> np.ndarray:
    tuned = np.asarray(proposal_std, dtype=float)
    low, high = target_range

    for round_idx in range(max_rounds):
        pilot_seed = None if seed is None else seed + round_idx
        pilot = MetropolisHastings(
            log_prob=log_prob,
            theta0=theta0,
            proposal_std=tuned,
            seed=pilot_seed,
            labels=labels,
        )
        pilot.run(n_steps=pilot_steps, n_burn=pilot_burn)
        acceptance = pilot.acceptance_rate

        if low <= acceptance <= high:
            break
        if acceptance < low:
            tuned = tuned * (0.5 if acceptance < low / 2 else 0.75)
        else:
            tuned = tuned * (1.5 if acceptance > high * 1.5 else 1.25)

    return tuned


def fit_relation_mh(
    data: RelationData,
    theta0: np.ndarray,
    proposal_std: np.ndarray | None = None,
    n_steps: int = 30000,
    n_burn: int = 5000,
    seed: int | None = 42,
    tune: bool = True,
) -> MetropolisHastings:
    labels = relation_parameter_labels(data.relation_kind)
    log_prob = make_relation_log_posterior(data)
    if proposal_std is None:
        proposal_std = np.array(
            _RELATION_META[data.relation_kind]["default_proposal_std"],
            dtype=float,
        )
    else:
        proposal_std = np.asarray(proposal_std, dtype=float)

    if tune:
        proposal_std = _tune_proposal_std(
            log_prob=log_prob,
            theta0=np.asarray(theta0, dtype=float),
            proposal_std=proposal_std,
            labels=labels,
            seed=seed,
        )

    sampler = MetropolisHastings(
        log_prob=log_prob,
        theta0=theta0,
        proposal_std=proposal_std,
        seed=seed,
        labels=labels,
    )
    sampler.run(n_steps=n_steps, n_burn=n_burn)
    sampler.proposal_std = proposal_std
    return sampler


def _build_pymc_model(data: RelationData, model_kind: str):
    import pymc as pm
    import pytensor.tensor as pt

    x = pt.as_tensor_variable(data.x)
    y = pt.as_tensor_variable(data.y)
    sigma_obs = pt.as_tensor_variable(data.sigma)

    with pm.Model() as model:
        slope = pm.Flat("a")
        intercept = pm.Flat("b")
        log10_sig = pm.Uniform("log10_sig", lower=-3, upper=1)
        sigma_tot = pt.sqrt(sigma_obs**2 + (10.0**log10_sig) ** 2)
        mu = slope * x + intercept

        if model_kind == "potential":
            loglike = -0.5 * pt.sum(
                pt.log(2 * np.pi * sigma_tot**2) + (y - mu) ** 2 / sigma_tot**2
            )
            pm.Potential("loglike_relation", loglike)
        elif model_kind == "native":
            pm.Normal("obs", mu=mu, sigma=sigma_tot, observed=data.y)
        else:
            raise ValueError(f"Unsupported NUTS model kind: {model_kind}")

    return model


def fit_relation_nuts(
    data: RelationData,
    theta0: np.ndarray,
    model_kind: str = "native",
    n_steps: int = 30000,
    n_burn: int = 5000,
    seed: int | None = 42,
) -> NoUTurnHamiltonian:
    labels = relation_parameter_labels(data.relation_kind)
    model = _build_pymc_model(data, model_kind=model_kind)
    sampler = NoUTurnHamiltonian(
        model=model,
        var_names=["a", "b", "log10_sig"],
        theta0=theta0,
        seed=seed,
        labels=labels,
    )
    sampler.run(n_steps=n_steps, n_burn=n_burn)
    return sampler


# ---------------------------------------------------------------------------
# Context builders and NUTS fit wrappers (Lab 1 notebooks 02-02, 03, 05)
# ---------------------------------------------------------------------------


def build_pl_context(
    data: table.Table,
    class_label: str,
    *,
    slope_prior_sigma: float = 5.0,
    intercept_prior_sigma: float = 1.0,
    sigma_scatter_scale: float = 0.5,
) -> SimpleNamespace:
    """Returns SimpleNamespace with PL fitting context for NUTS/MH samplers."""
    period_column = "pf" if class_label == "RRab" else "p1_o"
    period_error_column = "pf_error" if class_label == "RRab" else "p1_o_error"

    period = _as_float_array(data[period_column])
    period_error = _as_float_array(data[period_error_column])
    x_raw = np.log10(period)
    y = _as_float_array(data["M_G"])
    sigma = _as_float_array(data["sigma_M"])
    sigma_logp = period_error / (period * np.log(10.0))

    mask = (
        np.isfinite(x_raw)
        & np.isfinite(y)
        & np.isfinite(sigma)
        & (sigma > 0)
        & np.isfinite(sigma_logp)
        & (sigma_logp >= 0)
    )
    period = period[mask]
    period_error = period_error[mask]
    x_raw = x_raw[mask]
    y = y[mask]
    sigma = sigma[mask]
    sigma_logp = sigma_logp[mask]

    x_mean = float(np.mean(x_raw))
    x_centered = x_raw - x_mean
    intercept_prior_mean = float(np.median(y))

    w = 1.0 / sigma ** 2
    A = np.column_stack([x_centered, np.ones_like(x_centered)])
    a0, b0 = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    resid0 = y - (a0 * x_centered + b0)
    log10_sigma0 = np.log10(max(float(np.std(resid0)), float(np.median(sigma)), 1e-3))

    return SimpleNamespace(
        class_label=class_label,
        period=period,
        period_error=period_error,
        x_raw=x_raw,
        x_centered=x_centered,
        x_mean=x_mean,
        y=y,
        sigma=sigma,
        sigma_logp=sigma_logp,
        theta0=np.array([a0, b0, log10_sigma0], dtype=float),
        intercept_prior_mean=intercept_prior_mean,
        slope_prior_sigma=slope_prior_sigma,
        intercept_prior_sigma=intercept_prior_sigma,
        sigma_scatter_scale=sigma_scatter_scale,
        n=int(np.sum(mask)),
    )


def build_pc_context(
    data: table.Table,
    class_label: str,
    *,
    slope_prior_sigma: float = 5.0,
    intercept_prior_sigma: float = 1.0,
    sigma_scatter_scale: float = 0.5,
) -> SimpleNamespace:
    """Returns PL context for BP-RP period-color relation."""
    period_column = "pf" if class_label == "RRab" else "p1_o"
    period_error_column = "pf_error" if class_label == "RRab" else "p1_o_error"

    period = _as_float_array(data[period_column])
    period_error = _as_float_array(data[period_error_column])
    x_raw = np.log10(period)
    y = _as_float_array(data["bp_rp"])
    snr_bp = _as_float_array(data["phot_bp_mean_flux_over_error"])
    snr_rp = _as_float_array(data["phot_rp_mean_flux_over_error"])
    sigma = (2.5 / np.log(10.0)) * np.sqrt(1.0 / snr_bp ** 2 + 1.0 / snr_rp ** 2)
    sigma_logp = period_error / (period * np.log(10.0))

    mask = (
        np.isfinite(period)
        & (period > 0)
        & np.isfinite(period_error)
        & (period_error >= 0)
        & np.isfinite(x_raw)
        & np.isfinite(y)
        & np.isfinite(sigma)
        & (sigma > 0)
        & np.isfinite(sigma_logp)
        & (sigma_logp >= 0)
    )
    period = period[mask]
    period_error = period_error[mask]
    x_raw = x_raw[mask]
    y = y[mask]
    sigma = sigma[mask]
    sigma_logp = sigma_logp[mask]

    x_mean = float(np.mean(x_raw))
    x_centered = x_raw - x_mean
    intercept_prior_mean = float(np.median(y))

    w = 1.0 / sigma ** 2
    A = np.column_stack([x_centered, np.ones_like(x_centered)])
    a0, b0 = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    resid0 = y - (a0 * x_centered + b0)
    log10_sigma0 = np.log10(max(float(np.std(resid0)), float(np.median(sigma)), 1e-3))

    return SimpleNamespace(
        class_label=class_label,
        period=period,
        period_error=period_error,
        x_raw=x_raw,
        x_centered=x_centered,
        x_mean=x_mean,
        y=y,
        sigma=sigma,
        sigma_logp=sigma_logp,
        theta0=np.array([a0, b0, log10_sigma0], dtype=float),
        intercept_prior_mean=intercept_prior_mean,
        slope_prior_sigma=slope_prior_sigma,
        intercept_prior_sigma=intercept_prior_sigma,
        sigma_scatter_scale=sigma_scatter_scale,
        x_label=r"$\log_{10}(P\,/\,\mathrm{day})$",
        y_label=r"$G_\mathrm{BP} - G_\mathrm{RP}$ [mag]",
        n=int(np.sum(mask)),
    )


def build_w2_context(
    data: table.Table,
    class_label: str,
    *,
    slope_prior_sigma: float = 5.0,
    intercept_prior_sigma: float = 1.0,
    sigma_scatter_scale: float = 0.5,
) -> SimpleNamespace:
    """Returns PL context for WISE W2 period-luminosity relation.
    Note: theta0[2] is sigma0 (not log10(sigma0)) — preserve exactly.
    """
    period_column = "pf" if class_label == "RRab" else "p1_o"
    period_error_column = "pf_error" if class_label == "RRab" else "p1_o_error"

    period = _as_float_array(data[period_column])
    period_error = _as_float_array(data[period_error_column])
    x_raw = np.log10(period)
    y = _as_float_array(data["M_W2"])
    sigma = _as_float_array(data["sigma_M_W2"])
    sigma_logp = period_error / (period * np.log(10.0))

    mask = (
        np.isfinite(period)
        & (period > 0)
        & np.isfinite(period_error)
        & (period_error >= 0)
        & np.isfinite(x_raw)
        & np.isfinite(y)
        & np.isfinite(sigma)
        & (sigma > 0)
        & np.isfinite(sigma_logp)
        & (sigma_logp >= 0)
    )
    period = period[mask]
    period_error = period_error[mask]
    x_raw = x_raw[mask]
    y = y[mask]
    sigma = sigma[mask]
    sigma_logp = sigma_logp[mask]

    x_mean = float(np.mean(x_raw))
    x_centered = x_raw - x_mean
    intercept_prior_mean = float(np.median(y))

    w = 1.0 / sigma ** 2
    A = np.column_stack([x_centered, np.ones_like(x_centered)])
    a0, b0 = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    resid0 = y - (a0 * x_centered + b0)
    sigma0 = max(float(np.std(resid0)), float(np.median(sigma)), 1e-3)

    return SimpleNamespace(
        class_label=class_label,
        period=period,
        period_error=period_error,
        x_raw=x_raw,
        x_centered=x_centered,
        x_mean=x_mean,
        y=y,
        sigma=sigma,
        sigma_logp=sigma_logp,
        theta0=np.array([a0, b0, sigma0], dtype=float),  # theta0[2] = sigma0, not log10
        intercept_prior_mean=intercept_prior_mean,
        slope_prior_sigma=slope_prior_sigma,
        intercept_prior_sigma=intercept_prior_sigma,
        sigma_scatter_scale=sigma_scatter_scale,
        n=int(np.sum(mask)),
    )


def _build_pl_pymc_model_normal_priors(ctx: SimpleNamespace, model_kind: str):
    """Build a PyMC model for the PL/PC/W2 relation with Normal/HalfNormal priors."""
    import pymc as pm
    import pytensor.tensor as pt

    x_centered = pt.as_tensor_variable(ctx.x_centered)
    y = pt.as_tensor_variable(ctx.y)
    sigma_obs = pt.as_tensor_variable(ctx.sigma)
    sigma_logp = pt.as_tensor_variable(ctx.sigma_logp)

    with pm.Model() as model:
        a = pm.Normal("a", mu=0.0, sigma=ctx.slope_prior_sigma)
        b = pm.Normal("b", mu=ctx.intercept_prior_mean, sigma=ctx.intercept_prior_sigma)
        sigma_scatter = pm.HalfNormal("sigma_scatter", sigma=ctx.sigma_scatter_scale)

        mu = a * x_centered + b
        sigma_tot = pt.sqrt(sigma_obs ** 2 + sigma_scatter ** 2 + (a * sigma_logp) ** 2)

        if model_kind == "potential":
            loglike = -0.5 * pt.sum(
                pt.log(2.0 * np.pi * sigma_tot ** 2) + (y - mu) ** 2 / sigma_tot ** 2
            )
            pm.Potential("pl_loglike", loglike)
        elif model_kind == "native":
            pm.Normal("obs", mu=mu, sigma=sigma_tot, observed=ctx.y)
        else:
            raise ValueError(f"Unsupported model kind: {model_kind}")

    return model


def _run_pl_nuts(ctx: SimpleNamespace, model_kind: str, n_steps: int, n_burn: int, seed: int) -> SimpleNamespace:
    display_labels = [r"$a$", r"$b$", r"$\sigma_\mathrm{scatter}$"]
    model = _build_pl_pymc_model_normal_priors(ctx, model_kind)
    sampler_raw = NoUTurnHamiltonian(
        model=model,
        var_names=["a", "b", "sigma_scatter"],
        theta0=np.array([ctx.theta0[0], ctx.theta0[1], 10.0 ** ctx.theta0[2]], dtype=float),
        seed=seed,
        labels=display_labels,
    )
    sampler_raw.run(n_steps=n_steps, n_burn=n_burn)
    samples = np.asarray(sampler_raw.samples, dtype=float).copy()
    log_probs = np.asarray(sampler_raw.log_probs, dtype=float).copy()
    display = SimpleNamespace(
        samples=samples,
        log_probs=log_probs,
        n_burn=sampler_raw.n_burn,
        param_labels=display_labels,
    )
    return SimpleNamespace(raw=sampler_raw, display=display, samples=samples, acceptance_rate=sampler_raw.acceptance_rate)


def fit_pl_nuts_native(ctx: SimpleNamespace, *, n_steps: int = 3000, n_burn: int = 1500, seed: int = 42) -> SimpleNamespace:
    """NUTS fit for PL relation using native PyMC Normal likelihood."""
    return _run_pl_nuts(ctx, "native", n_steps, n_burn, seed)


def fit_pl_nuts_potential(ctx: SimpleNamespace, *, n_steps: int = 3000, n_burn: int = 1500, seed: int = 42) -> SimpleNamespace:
    """NUTS fit for PL relation using pm.Potential log-likelihood."""
    return _run_pl_nuts(ctx, "potential", n_steps, n_burn, seed)


def fit_w2_nuts(ctx: SimpleNamespace, *, n_steps: int = 2000, n_burn: int = 1000, seed: int = 42) -> SimpleNamespace:
    """NUTS fit for WISE W2 PL relation. ctx.theta0[2] is sigma0 (not log10)."""
    display_labels = [r"$a$", r"$b$", r"$\sigma_\mathrm{scatter}$"]
    model = _build_pl_pymc_model_normal_priors(ctx, "native")
    sampler_raw = NoUTurnHamiltonian(
        model=model,
        var_names=["a", "b", "sigma_scatter"],
        theta0=np.asarray(ctx.theta0, dtype=float),  # theta0[2] already in sigma units
        seed=seed,
        labels=display_labels,
    )
    sampler_raw.run(n_steps=n_steps, n_burn=n_burn)
    samples = np.asarray(sampler_raw.samples, dtype=float).copy()
    return SimpleNamespace(
        raw=sampler_raw,
        samples=samples,
        log_probs=np.asarray(sampler_raw.log_probs, dtype=float).copy(),
        n_burn=sampler_raw.n_burn,
        param_labels=display_labels,
        acceptance_rate=sampler_raw.acceptance_rate,
    )


def fit_pc_nuts(ctx: SimpleNamespace, *, n_steps: int = 10000, n_burn: int = 2000, seed: int = 42) -> SimpleNamespace:
    """NUTS fit for period-color relation. Returns samples in [a_c, b_c, sigma_c_scatter] space."""
    import pymc as pm
    import pytensor.tensor as pt

    display_labels = [r"$a_c$", r"$b_c$", r"$\sigma_c$"]
    x_centered = pt.as_tensor_variable(ctx.x_centered)
    sigma_obs = pt.as_tensor_variable(ctx.sigma)
    sigma_logp = pt.as_tensor_variable(ctx.sigma_logp)

    with pm.Model() as model:
        a_var = pm.Normal("a_c", mu=0.0, sigma=ctx.slope_prior_sigma)
        b_var = pm.Normal("b_c", mu=ctx.intercept_prior_mean, sigma=ctx.intercept_prior_sigma)
        sigma_scatter_var = pm.HalfNormal("sigma_c_scatter", sigma=ctx.sigma_scatter_scale)

        mu = a_var * x_centered + b_var
        sigma_tot = pt.sqrt(sigma_obs ** 2 + sigma_scatter_var ** 2 + (a_var * sigma_logp) ** 2)
        pm.Normal("obs", mu=mu, sigma=sigma_tot, observed=ctx.y)

    sampler_raw = NoUTurnHamiltonian(
        model=model,
        var_names=["a_c", "b_c", "sigma_c_scatter"],
        theta0=np.array([ctx.theta0[0], ctx.theta0[1], 10.0 ** ctx.theta0[2]], dtype=float),
        seed=seed,
        labels=display_labels,
    )
    sampler_raw.run(n_steps=n_steps, n_burn=n_burn)

    raw_samples = np.asarray(sampler_raw.samples, dtype=float).copy()
    post_burn = raw_samples[sampler_raw.n_burn:]
    comparison_samples = post_burn.copy()
    comparison_samples[:, 2] = np.log10(comparison_samples[:, 2])

    display = SimpleNamespace(
        samples=raw_samples,
        log_probs=np.asarray(sampler_raw.log_probs, dtype=float).copy(),
        n_burn=sampler_raw.n_burn,
        param_labels=display_labels,
    )
    return SimpleNamespace(
        raw=sampler_raw,
        display=display,
        samples=post_burn,
        comparison_samples=comparison_samples,
        acceptance_rate=sampler_raw.acceptance_rate,
    )


def pl_posterior_summary_row(
    method: str, class_label: str, samples: np.ndarray, acceptance_rate: float
) -> dict:
    """Posterior summary row for PL fits (slope a, intercept b, sigma_scatter)."""
    q16, q50, q84 = np.percentile(np.asarray(samples, dtype=float), [16, 50, 84], axis=0)
    return {
        "class": class_label,
        "method": method,
        "acceptance": round(float(acceptance_rate), 3),
        "a": round(float(q50[0]), 4),
        "a_minus": round(float(q50[0] - q16[0]), 4),
        "a_plus": round(float(q84[0] - q50[0]), 4),
        "b": round(float(q50[1]), 4),
        "b_minus": round(float(q50[1] - q16[1]), 4),
        "b_plus": round(float(q84[1] - q50[1]), 4),
        "sigma_scatter": round(float(q50[2]), 4),
        "sigma_minus": round(float(q50[2] - q16[2]), 4),
        "sigma_plus": round(float(q84[2] - q50[2]), 4),
    }


def w2_posterior_summary_row(
    class_label: str, band_label: str, sampler: Any, n_fit: int
) -> dict:
    """Posterior summary row for WISE W2 fits."""
    samples = np.asarray(sampler.samples, dtype=float)
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    return {
        "class": class_label,
        "band": band_label,
        "N_fit": int(n_fit),
        "acceptance": round(float(sampler.acceptance_rate), 3),
        "a": round(float(q50[0]), 4),
        "a_minus": round(float(q50[0] - q16[0]), 4),
        "a_plus": round(float(q84[0] - q50[0]), 4),
        "b": round(float(q50[1]), 4),
        "b_minus": round(float(q50[1] - q16[1]), 4),
        "b_plus": round(float(q84[1] - q50[1]), 4),
        "sigma_scatter": round(float(q50[2]), 4),
        "sigma_minus": round(float(q50[2] - q16[2]), 4),
        "sigma_plus": round(float(q84[2] - q50[2]), 4),
    }


def pc_posterior_summary_row(
    class_label: str, samples: np.ndarray, acceptance_rate: float
) -> dict:
    """Posterior summary row for PC fits (a_c, b_c, sigma_c)."""
    q16, q50, q84 = np.percentile(np.asarray(samples, dtype=float), [16, 50, 84], axis=0)
    return {
        "class": class_label,
        "acceptance": round(float(acceptance_rate), 3),
        "a_c": round(float(q50[0]), 4),
        "a_c_minus": round(float(q50[0] - q16[0]), 4),
        "a_c_plus": round(float(q84[0] - q50[0]), 4),
        "b_c": round(float(q50[1]), 4),
        "b_c_minus": round(float(q50[1] - q16[1]), 4),
        "b_c_plus": round(float(q84[1] - q50[1]), 4),
        "sigma_c": round(float(q50[2]), 4),
        "sigma_c_minus": round(float(q50[2] - q16[2]), 4),
        "sigma_c_plus": round(float(q84[2] - q50[2]), 4),
    }


def build_band_summary(
    ctx: SimpleNamespace, samples: np.ndarray, band_label: str, *, n_grid: int = 300, seed: int = 42
) -> SimpleNamespace:
    """Predictive summary for a single band given context + samples."""
    values = np.asarray(samples, dtype=float)
    x_obs = _as_float_array(ctx.x_raw)
    y_obs = _as_float_array(ctx.y)
    sigma_obs = _as_float_array(ctx.sigma)
    sigma_logp = _as_float_array(ctx.sigma_logp)

    x_grid = np.linspace(float(np.min(x_obs)), float(np.max(x_obs)), int(n_grid))
    order = np.argsort(x_obs)
    sigma_obs_grid = np.interp(x_grid, x_obs[order], sigma_obs[order])
    sigma_logp_grid = np.interp(x_grid, x_obs[order], sigma_logp[order])

    rng = np.random.default_rng(seed)
    step = max(len(values) // 400, 1)
    sample_pool = np.asarray(values[::step], dtype=float)
    if len(sample_pool) > 400:
        keep_idx = rng.choice(len(sample_pool), size=400, replace=False)
        sample_pool = sample_pool[keep_idx]

    mean_draws = np.empty((len(sample_pool), len(x_grid)), dtype=float)
    predictive_draws = np.empty_like(mean_draws)
    for i, (slope_draw, intercept_draw, sigma_scatter_draw) in enumerate(sample_pool):
        mu_grid = slope_draw * (x_grid - ctx.x_mean) + intercept_draw
        sigma_pred = np.sqrt(
            sigma_obs_grid ** 2 + sigma_scatter_draw ** 2 + (slope_draw * sigma_logp_grid) ** 2
        )
        mean_draws[i] = mu_grid
        predictive_draws[i] = rng.normal(mu_grid, sigma_pred)

    q16, q50, q84 = np.percentile(values, [16, 50, 84], axis=0)
    return SimpleNamespace(
        class_label=ctx.class_label,
        band_label=band_label,
        n=len(x_obs),
        x_obs=x_obs,
        y_obs=y_obs,
        sigma_obs=sigma_obs,
        x_grid=x_grid,
        median_mean=np.quantile(mean_draws, 0.50, axis=0),
        predictive_q16=np.quantile(predictive_draws, 0.16, axis=0),
        predictive_q84=np.quantile(predictive_draws, 0.84, axis=0),
        slope_q16=float(q16[0]),
        slope_q50=float(q50[0]),
        slope_q84=float(q84[0]),
        intercept_q16=float(q16[1]),
        intercept_q50=float(q50[1]),
        intercept_q84=float(q84[1]),
        sigma_scatter_q16=float(q16[2]),
        sigma_scatter_q50=float(q50[2]),
        sigma_scatter_q84=float(q84[2]),
    )


def pl_scatter_metrics(data: table.Table, label: str) -> dict:
    """PL residual scatter metrics (sigma_MAD, RMS, outlier counts).

    Keys: sample, N, sigma_MAD [mag], RMS [mag], N(|ΔM|>1 mag), N(|ΔM|>2 mag).
    """
    from ugdatalab.models.gaia import rrlyrae_representative_period
    periods = _as_float_array(rrlyrae_representative_period(data))
    x = np.log10(periods)
    y = _as_float_array(data["M_G"])
    coeff = np.polyfit(x, y, 1)
    resid = y - np.polyval(coeff, x)
    abs_resid = np.abs(resid)
    return {
        "sample": label,
        "N": len(data),
        "sigma_MAD [mag]": round(1.4826 * float(np.median(np.abs(resid - np.median(resid)))), 3),
        "RMS [mag]": round(float(np.sqrt(np.mean(resid ** 2))), 3),
        "N(|ΔM| > 1 mag)": int(np.count_nonzero(abs_resid > 1.0)),
        "N(|ΔM| > 2 mag)": int(np.count_nonzero(abs_resid > 2.0)),
    }
