import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from astropy.table import Table

from ugdatalab import (
    build_infrared_pl_comparison_data,
    build_optical_pc_comparison_data,
    estimate_initial_theta0,
    load_infrared_pl_comparison_data,
    load_optical_pc_comparison_data,
    prepare_relation_data,
    save_infrared_pl_comparison_data,
    save_optical_pc_comparison_data,
)


class RelationHelperTests(unittest.TestCase):
    def test_prepare_relation_data_rrab_pl_filters_by_class_and_valid_rows(self):
        table = Table(
            {
                "best_classification": ["RRab", "RRc", "RRab"],
                "pf": [0.60, np.nan, 0.70],
                "p1_o": [np.nan, 0.32, np.nan],
                "M_G": [0.55, 0.20, np.nan],
                "sigma_M": [0.08, 0.03, 0.09],
                "bp_rp": [0.72, 0.31, 0.80],
                "phot_bp_mean_flux_over_error": [30.0, 40.0, 50.0],
                "phot_rp_mean_flux_over_error": [25.0, 35.0, 45.0],
            }
        )
        source = SimpleNamespace(data=table)

        data = prepare_relation_data(source, rr_class="RRab", relation_kind="pl")

        self.assertEqual(data.rr_class, "RRab")
        self.assertEqual(data.relation_kind, "pl")
        np.testing.assert_allclose(data.x, np.log10([0.60]))
        np.testing.assert_allclose(data.y, [0.55])
        np.testing.assert_allclose(data.sigma, [0.08])

    def test_prepare_relation_data_rrc_pc_uses_first_overtone_period_and_color_sigma(self):
        table = Table(
            {
                "best_classification": ["RRab", "RRc", "RRc"],
                "pf": [0.61, np.nan, np.nan],
                "p1_o": [np.nan, 0.33, 0.28],
                "bp_rp": [0.70, 0.42, 0.39],
                "phot_bp_mean_flux_over_error": [20.0, 40.0, 25.0],
                "phot_rp_mean_flux_over_error": [18.0, 32.0, 20.0],
            }
        )
        source = SimpleNamespace(data=table)

        data = prepare_relation_data(source, rr_class="RRc", relation_kind="pc")

        expected_x = np.log10([0.33, 0.28])
        expected_sigma = (2.5 / np.log(10)) * np.sqrt(
            1.0 / np.array([40.0, 25.0]) ** 2 + 1.0 / np.array([32.0, 20.0]) ** 2
        )

        np.testing.assert_allclose(data.x, expected_x)
        np.testing.assert_allclose(data.y, [0.42, 0.39])
        np.testing.assert_allclose(data.sigma, expected_sigma)

    def test_estimate_initial_theta0_maps_mixture_results_to_sampler_order(self):
        table = Table(
            {
                "best_classification": ["RRab", "RRab"],
                "pf": [0.62, 0.71],
                "p1_o": [np.nan, np.nan],
                "M_G": [0.49, 0.41],
                "sigma_M": [0.07, 0.08],
            }
        )
        source = SimpleNamespace(
            data=table,
            mcmc_results={
                "RRab": {
                    "a": (0.42, 0.01),
                    "b": (-2.10, 0.05),
                    "sig_scatter": (0.18,),
                }
            },
        )
        data = prepare_relation_data(source, rr_class="RRab", relation_kind="pl")

        theta0 = estimate_initial_theta0(
            source,
            rr_class="RRab",
            relation_kind="pl",
            data=data,
        )

        np.testing.assert_allclose(theta0, [-2.10, 0.42, np.log10(0.18)])

    def test_infrared_pl_comparison_round_trip_uses_shared_shape(self):
        ctx = SimpleNamespace(
            class_label="RRab",
            x_raw=np.array([-0.20, -0.10, 0.00], dtype=float),
            x_mean=-0.10,
            y=np.array([0.5, 0.4, 0.3], dtype=float),
            sigma=np.array([0.05, 0.06, 0.07], dtype=float),
            sigma_logp=np.array([0.01, 0.01, 0.02], dtype=float),
        )
        samples = np.array(
            [
                [-2.30, -0.50, 0.10],
                [-2.10, -0.45, 0.12],
                [-2.20, -0.48, 0.11],
            ]
        )

        comparison = build_infrared_pl_comparison_data(ctx, samples, n_grid=16)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "infrared_pl.npz"
            save_infrared_pl_comparison_data(path, {"RRab": comparison, "RRc": comparison})
            loaded = load_infrared_pl_comparison_data(path)

        self.assertEqual(set(loaded), {"RRab", "RRc"})
        np.testing.assert_allclose(loaded["RRab"].x_grid.shape, (16,))
        self.assertAlmostEqual(loaded["RRab"].slope_q50, comparison.slope_q50, places=10)

    def test_optical_pc_comparison_round_trip_preserves_dust_summary_fields(self):
        ctx = SimpleNamespace(
            class_label="RRc",
            x_raw=np.array([-0.5, -0.4, -0.3], dtype=float),
            x_mean=-0.4,
            y=np.array([0.35, 0.38, 0.41], dtype=float),
            sigma=np.array([0.02, 0.02, 0.03], dtype=float),
            sigma_logp=np.array([0.01, 0.01, 0.01], dtype=float),
        )
        samples = np.array(
            [
                [0.10, 0.45, np.log10(0.04)],
                [0.12, 0.47, np.log10(0.05)],
                [0.08, 0.44, np.log10(0.03)],
            ]
        )

        comparison = build_optical_pc_comparison_data(ctx, samples, n_grid=12)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "optical_pc.npz"
            save_optical_pc_comparison_data(path, {"RRab": comparison, "RRc": comparison})
            loaded = load_optical_pc_comparison_data(path)

        self.assertEqual(set(loaded), {"RRab", "RRc"})
        self.assertGreater(loaded["RRc"].intrinsic_sigma_median, 0.0)
        self.assertAlmostEqual(loaded["RRc"].slope_std, comparison.slope_std, places=10)
        expected_raw_intercepts = samples[:, 1] - samples[:, 0] * ctx.x_mean
        self.assertAlmostEqual(
            loaded["RRc"].intercept_median,
            float(np.median(expected_raw_intercepts)),
            places=10,
        )
        self.assertAlmostEqual(
            loaded["RRc"].intrinsic_sigma_median,
            comparison.intrinsic_sigma_median,
            places=10,
        )

    def test_optical_pc_comparison_predictive_width_uses_sigma_logp_when_present(self):
        base_ctx = SimpleNamespace(
            class_label="RRab",
            x_raw=np.array([-0.35, -0.20, -0.05], dtype=float),
            x_mean=-0.20,
            y=np.array([0.72, 0.69, 0.66], dtype=float),
            sigma=np.array([0.02, 0.02, 0.02], dtype=float),
            sigma_logp=np.zeros(3, dtype=float),
        )
        wide_ctx = SimpleNamespace(
            class_label="RRab",
            x_raw=base_ctx.x_raw,
            x_mean=base_ctx.x_mean,
            y=base_ctx.y,
            sigma=base_ctx.sigma,
            sigma_logp=np.full(3, 0.03, dtype=float),
        )
        samples = np.array(
            [
                [-0.25, 0.68, np.log10(0.04)],
                [-0.22, 0.69, np.log10(0.05)],
                [-0.28, 0.67, np.log10(0.04)],
            ]
        )

        base = build_optical_pc_comparison_data(base_ctx, samples, n_grid=24)
        wide = build_optical_pc_comparison_data(wide_ctx, samples, n_grid=24)

        base_width = float(np.mean(base.predictive_q84 - base.predictive_q16))
        wide_width = float(np.mean(wide.predictive_q84 - wide.predictive_q16))
        self.assertGreater(wide_width, base_width)


if __name__ == "__main__":
    unittest.main()
