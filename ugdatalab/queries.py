from __future__ import annotations

from textwrap import dedent


def build_rrlyrae_top_n_query(
    limit: int = 100,
    min_clean_epochs_g: int = 40,
    require_pf: bool = True,
) -> str:
    """ADQL for the top RR Lyrae sample used in Lab 1 parts 1-4."""
    where = [f"num_clean_epochs_g > {int(min_clean_epochs_g)}"]
    if require_pf:
        where.append("pf IS NOT NULL")

    return dedent(
        f"""
        SELECT TOP {int(limit)} *
        FROM gaiadr3.vari_rrlyrae
        WHERE {" AND ".join(where)}
        ORDER BY num_clean_epochs_g DESC
        """
    ).strip()


def build_rrlyrae_class_lightcurve_query(
    rr_class: str,
    limit: int = 3,
    max_int_average_g: float = 15.0,
    min_clean_epochs_g: int = 80,
) -> str:
    """ADQL for the bright RRab/RRc comparison sample in Lab 1 part 10."""
    if rr_class not in {"RRab", "RRc"}:
        raise ValueError("rr_class must be 'RRab' or 'RRc'.")

    return dedent(
        f"""
        SELECT TOP {int(limit)} *
        FROM gaiadr3.vari_rrlyrae
        WHERE best_classification = '{rr_class}'
          AND int_average_g < {float(max_int_average_g):.3f}
          AND num_clean_epochs_g > {int(min_clean_epochs_g)}
        ORDER BY num_clean_epochs_g DESC
        """
    ).strip()


def build_local_rrlyrae_query(
    max_fractional_parallax_error: float = 0.2,
    min_abs_b_deg: float = 30.0,
    max_distance_kpc: float = 4.0,
) -> str:
    """ADQL for the full low-dust calibration sample used in Lab 1 parts 12-18."""
    if max_fractional_parallax_error <= 0:
        raise ValueError("max_fractional_parallax_error must be positive.")
    if min_abs_b_deg < 0:
        raise ValueError("min_abs_b_deg must be non-negative.")
    if max_distance_kpc <= 0:
        raise ValueError("max_distance_kpc must be positive.")

    min_parallax_mas = 1.0 / float(max_distance_kpc)
    min_parallax_over_error = 1.0 / float(max_fractional_parallax_error)

    return dedent(
        f"""
        SELECT *
        FROM gaiadr3.vari_rrlyrae AS vr
        JOIN gaiadr3.gaia_source AS gs
          ON vr.source_id = gs.source_id
        WHERE gs.parallax IS NOT NULL
          AND gs.parallax_error > 0
          AND gs.parallax_over_error > {min_parallax_over_error:.6f}
          AND ABS(gs.b) > {float(min_abs_b_deg):.3f}
          AND gs.parallax > {min_parallax_mas:.6f}
        """
    ).strip()


def build_full_rrlyrae_join_query() -> str:
    """ADQL for the full RR Lyrae catalog cross-matched to Gaia source data."""
    return dedent(
        """
        SELECT *
        FROM gaiadr3.vari_rrlyrae AS vr
        JOIN gaiadr3.gaia_source AS gs
          ON vr.source_id = gs.source_id
        """
    ).strip()
