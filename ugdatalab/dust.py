from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy import table
from astropy.coordinates import SkyCoord

from ugdatalab.models.gaia import get_gaia


FULL_RRLYRAE_GAIA_SOURCE_QUERY = """
SELECT *
FROM gaiadr3.vari_rrlyrae AS vr
JOIN gaiadr3.gaia_source AS gs
    ON vr.source_id = gs.source_id
""".strip()


def _as_float_array(column: Any) -> np.ndarray:
    values = np.asarray(column)
    if hasattr(values, "filled"):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class RelationPosteriorSummary:
    slope_median: float
    slope_std: float
    intercept_median: float
    intercept_std: float
    intrinsic_sigma_median: float
    intrinsic_sigma_std: float


@dataclass
class ExtinctionResiduals:
    mask: np.ndarray
    catalog: np.ndarray
    empirical: np.ndarray
    residuals: np.ndarray


def build_rrlyrae_gaia_source_query(
    *,
    columns: Sequence[str] | str = "*",
    where: Sequence[str] | None = None,
    order_by: str | None = None,
    limit: int | None = None,
) -> str:
    """Build the shared RR Lyrae + `gaia_source` join used in Lab 1."""
    if isinstance(columns, str):
        column_text = columns
    else:
        column_text = ", ".join(columns)

    query_lines = ["SELECT"]
    if limit is not None:
        query_lines.append(f"    TOP {int(limit)} {column_text}")
    else:
        query_lines.append(f"    {column_text}")
    query_lines.append("FROM gaiadr3.vari_rrlyrae AS vr")
    query_lines.append("JOIN gaiadr3.gaia_source AS gs")
    query_lines.append("    ON vr.source_id = gs.source_id")
    if where:
        query_lines.append("WHERE " + "\n  AND ".join(where))
    if order_by is not None:
        query_lines.append(f"ORDER BY {order_by}")
    return "\n".join(query_lines)


def prepare_rrlyrae_class_columns(data: table.Table, *, copy: bool = True) -> table.Table:
    """Attach the period, RRab, and RRc columns expected by later analysis steps."""
    out = data.copy(copy_data=True) if copy else data
    pf = _as_float_array(out["pf"])
    p1_o = _as_float_array(out["p1_o"])

    out["period"] = np.where(np.isfinite(pf), pf, p1_o)
    out["is_rrab"] = np.isfinite(pf) & ~np.isfinite(p1_o)
    out["is_rrc"] = ~np.isfinite(pf) & np.isfinite(p1_o)
    return out


def prepare_full_rrlyrae_table(data: table.Table, copy: bool = True) -> table.Table:
    """Alias matching the notebook-oriented naming for class-column preparation."""
    return prepare_rrlyrae_class_columns(data, copy=copy)


def rrlyrae_class_masks(data: table.Table) -> dict[str, np.ndarray]:
    """Return boolean masks for the RRab and RRc subclasses."""
    prepared = prepare_rrlyrae_class_columns(data, copy=False)
    return {
        "RRab": np.asarray(prepared["is_rrab"], dtype=bool),
        "RRc": np.asarray(prepared["is_rrc"], dtype=bool),
    }


def get_full_rrlyrae_catalog(query: str = FULL_RRLYRAE_GAIA_SOURCE_QUERY) -> table.Table:
    """Fetch the full Gaia RR Lyrae cross-match and attach class columns."""
    return prepare_rrlyrae_class_columns(get_gaia(query), copy=True)


def summarize_relation_samples(samples: np.ndarray) -> RelationPosteriorSummary:
    """Summarize `[slope, intercept, log10_sigma]` posterior samples."""
    values = np.asarray(samples, dtype=float)
    return RelationPosteriorSummary(
        slope_median=float(np.median(values[:, 0])),
        slope_std=float(np.std(values[:, 0])),
        intercept_median=float(np.median(values[:, 1])),
        intercept_std=float(np.std(values[:, 1])),
        intrinsic_sigma_median=float(10.0 ** np.median(values[:, 2])),
        intrinsic_sigma_std=float(np.std(10.0 ** values[:, 2])),
    )


def summarize_relation_posteriors(
    paths: Mapping[str, str | Path],
) -> dict[str, RelationPosteriorSummary]:
    """Load and summarize saved posterior sample arrays keyed by RR Lyrae class."""
    _, summaries = load_relation_posteriors(paths)
    return summaries


def load_relation_posteriors(
    paths: Mapping[str, str | Path],
) -> tuple[dict[str, np.ndarray], dict[str, RelationPosteriorSummary]]:
    """Load saved posterior sample arrays and summarize them by RR Lyrae class."""
    arrays: dict[str, np.ndarray] = {}
    summaries: dict[str, RelationPosteriorSummary] = {}
    for rr_class, path in paths.items():
        samples = np.load(Path(path))
        arrays[rr_class] = samples
        summaries[rr_class] = summarize_relation_samples(samples)
    return arrays, summaries


def _coerce_summary(summary: RelationPosteriorSummary | Mapping[str, float]) -> RelationPosteriorSummary:
    if isinstance(summary, RelationPosteriorSummary):
        return summary
    return RelationPosteriorSummary(
        slope_median=float(summary["slope_median"]),
        slope_std=float(summary["slope_std"]),
        intercept_median=float(summary["intercept_median"]),
        intercept_std=float(summary["intercept_std"]),
        intrinsic_sigma_median=float(summary["intrinsic_sigma_median"]),
        intrinsic_sigma_std=float(summary.get("intrinsic_sigma_std", 0.0)),
    )


def compute_empirical_extinction(
    data: table.Table,
    period_color_models: Mapping[str, RelationPosteriorSummary | Mapping[str, float]],
    *,
    R_G: float = 2.0,
    copy: bool = True,
) -> table.Table:
    """Compute class-specific intrinsic color, color excess, and empirical extinction."""
    out = prepare_rrlyrae_class_columns(data, copy=copy)

    period = _as_float_array(out["period"])
    log10_period = np.full(len(out), np.nan, dtype=float)
    positive_period = np.isfinite(period) & (period > 0)
    log10_period[positive_period] = np.log10(period[positive_period])

    snr_bp = _as_float_array(out["phot_bp_mean_flux_over_error"])
    snr_rp = _as_float_array(out["phot_rp_mean_flux_over_error"])
    sigma_color_obs = (2.5 / np.log(10.0)) * np.sqrt(1.0 / snr_bp**2 + 1.0 / snr_rp**2)

    color_int = np.full(len(out), np.nan, dtype=float)
    sigma_coeff = np.full(len(out), np.nan, dtype=float)
    sigma_intrinsic = np.full(len(out), np.nan, dtype=float)

    for rr_class, summary_like in period_color_models.items():
        summary = _coerce_summary(summary_like)
        mask = np.asarray(out["is_rrab"] if rr_class == "RRab" else out["is_rrc"], dtype=bool)
        mask &= np.isfinite(log10_period)

        color_int[mask] = summary.slope_median * log10_period[mask] + summary.intercept_median
        sigma_coeff[mask] = np.sqrt(
            (summary.slope_std * log10_period[mask]) ** 2 + summary.intercept_std**2
        )
        sigma_intrinsic[mask] = summary.intrinsic_sigma_median

    color_obs = _as_float_array(out["bp_rp"])
    E_bprp = color_obs - color_int
    A_G_calc = float(R_G) * E_bprp
    sigma_E = np.sqrt(sigma_color_obs**2 + sigma_coeff**2 + sigma_intrinsic**2)

    out["log10_period"] = log10_period
    out["sigma_color_obs"] = sigma_color_obs
    out["color_int"] = color_int
    out["sigma_coeff"] = sigma_coeff
    out["sigma_intrinsic"] = sigma_intrinsic
    out["E_bprp"] = E_bprp
    out["A_G_calc"] = A_G_calc
    out["sigma_E"] = sigma_E
    return out


def compute_period_color_extinction(
    data: table.Table,
    summaries: Mapping[str, RelationPosteriorSummary | Mapping[str, float]],
    *,
    r_g: float = 2.0,
    copy: bool = True,
) -> table.Table:
    """Alias matching the notebook terminology for the dust-stage calculation."""
    return compute_empirical_extinction(data, summaries, R_G=r_g, copy=copy)


def empirical_vs_catalog_extinction(
    data: table.Table,
    *,
    empirical_column: str = "A_G_calc",
    catalog_column: str = "g_absorption",
) -> ExtinctionResiduals:
    """Return the finite comparison subset for Lab 1 part 26."""
    empirical = _as_float_array(data[empirical_column])
    catalog = _as_float_array(data[catalog_column])
    mask = np.isfinite(empirical) & np.isfinite(catalog)
    return ExtinctionResiduals(
        mask=mask,
        catalog=catalog[mask],
        empirical=empirical[mask],
        residuals=empirical[mask] - catalog[mask],
    )


def extinction_residuals(
    data: table.Table,
    *,
    empirical_column: str = "A_G_calc",
    catalog_column: str = "g_absorption",
) -> tuple[np.ndarray, np.ndarray]:
    """Compact tuple return for simple residual workflows."""
    result = empirical_vs_catalog_extinction(
        data,
        empirical_column=empirical_column,
        catalog_column=catalog_column,
    )
    return result.mask, result.residuals


def build_reddening_quality_mask(
    data: table.Table,
    *,
    min_bp_snr: float = 5.0,
    min_rp_snr: float = 5.0,
    apply_bp_rp_excess_cut: bool = True,
    max_sigma_E: float | None = None,
    max_abs_E: float | None = None,
) -> np.ndarray:
    """Construct a practical quality mask for the empirical reddening map."""
    mask = (
        np.isfinite(_as_float_array(data["E_bprp"]))
        & np.isfinite(_as_float_array(data["A_G_calc"]))
        & np.isfinite(_as_float_array(data["bp_rp"]))
        & (_as_float_array(data["phot_bp_mean_flux_over_error"]) > float(min_bp_snr))
        & (_as_float_array(data["phot_rp_mean_flux_over_error"]) > float(min_rp_snr))
    )

    if apply_bp_rp_excess_cut:
        bp_rp = _as_float_array(data["bp_rp"])
        excess = _as_float_array(data["phot_bp_rp_excess_factor"])
        mask &= (excess > 1.0 + 0.015 * bp_rp**2) & (excess < 1.3 + 0.06 * bp_rp**2)

    if max_sigma_E is not None:
        mask &= _as_float_array(data["sigma_E"]) <= float(max_sigma_E)
    if max_abs_E is not None:
        mask &= np.abs(_as_float_array(data["E_bprp"])) <= float(max_abs_E)
    return mask


def apply_reddening_quality_mask(data: table.Table, **kwargs: Any) -> table.Table:
    """Filter a reddening catalog with `build_reddening_quality_mask`."""
    return data[build_reddening_quality_mask(data, **kwargs)]


def _load_sfd_query():
    try:
        from dustmaps.sfd import SFDQuery
    except ImportError as exc:
        raise ImportError(
            "dustmaps is required for SFD comparisons. Install the optional "
            "`ugdatalab[dust]` dependency or add `dustmaps` to your environment."
        ) from exc
    return SFDQuery()


def sample_sfd_ebv(
    data: table.Table,
    *,
    query: Any | None = None,
) -> np.ndarray:
    """Sample the SFD map at the Galactic coordinates in a Gaia RR Lyrae table."""
    if query is None:
        query = _load_sfd_query()

    coords = SkyCoord(
        l=_as_float_array(data["l"]) * u.deg,
        b=_as_float_array(data["b"]) * u.deg,
        frame="galactic",
    )
    return np.asarray(query(coords), dtype=float)


def attach_sfd_ebv(
    data: table.Table,
    *,
    query: Any | None = None,
    column_name: str = "sfd_ebv",
    copy: bool = True,
) -> table.Table:
    """Attach SFD `E(B-V)` values to a Gaia RR Lyrae table."""
    out = data.copy(copy_data=True) if copy else data
    out[column_name] = sample_sfd_ebv(out, query=query)
    return out
