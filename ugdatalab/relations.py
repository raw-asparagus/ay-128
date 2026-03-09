from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ugdatalab.mcmc import MetropolisHastings, NoUTurnHamiltonian


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


def _as_float_array(column: Any) -> np.ndarray:
    data = np.asarray(column)
    if hasattr(data, "filled"):
        data = data.filled(np.nan)
    return np.asarray(data, dtype=float)


def _class_mask(data, rr_class: str) -> np.ndarray:
    if rr_class not in {"RRab", "RRc"}:
        raise ValueError(f"Unsupported RR Lyrae class: {rr_class}")

    if rr_class == "RRab":
        if "is_rrab" in data.colnames:
            return np.asarray(data["is_rrab"], dtype=bool)
        if "pf" in data.colnames and "p1_o" in data.colnames:
            pf = _as_float_array(data["pf"])
            p1_o = _as_float_array(data["p1_o"])
            return np.isfinite(pf) & ~np.isfinite(p1_o)
    else:
        if "is_rrc" in data.colnames:
            return np.asarray(data["is_rrc"], dtype=bool)
        if "pf" in data.colnames and "p1_o" in data.colnames:
            pf = _as_float_array(data["pf"])
            p1_o = _as_float_array(data["p1_o"])
            return ~np.isfinite(pf) & np.isfinite(p1_o)

    raise ValueError(
        "Could not determine RR Lyrae class mask; expected class flags or pf/p1_o columns."
    )


def _period_column(data, rr_class: str) -> np.ndarray:
    if "period" in data.colnames:
        return _as_float_array(data["period"])

    if rr_class == "RRab" and "pf" in data.colnames:
        return _as_float_array(data["pf"])
    if rr_class == "RRc" and "p1_o" in data.colnames:
        return _as_float_array(data["p1_o"])

    raise ValueError(
        "Could not determine period column; expected `period` or class-specific Gaia period columns."
    )


def relation_parameter_labels(relation_kind: str) -> list[str]:
    if relation_kind not in _RELATION_META:
        raise ValueError(f"Unsupported relation kind: {relation_kind}")
    return list(_RELATION_META[relation_kind]["labels"])


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
