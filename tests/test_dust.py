import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.table import Table

from ugdatalab import (
    RelationPosteriorSummary,
    apply_reddening_quality_mask,
    attach_sfd_ebv,
    build_reddening_quality_mask,
    build_rrlyrae_gaia_source_query,
    compute_empirical_extinction,
    empirical_vs_catalog_extinction,
    load_relation_posteriors,
    prepare_rrlyrae_class_columns,
    sample_sfd_ebv,
    summarize_relation_samples,
)


class DustHelperTests(unittest.TestCase):
    def test_build_rrlyrae_gaia_source_query_supports_where_limit_and_order(self):
        query = build_rrlyrae_gaia_source_query(
            where=["ABS(gs.b) > 30", "gs.parallax_over_error > 5"],
            limit=25,
            order_by="gs.parallax_over_error DESC",
        )

        self.assertIn("TOP 25", query)
        self.assertIn("FROM gaiadr3.vari_rrlyrae AS vr", query)
        self.assertIn("JOIN gaiadr3.gaia_source AS gs", query)
        self.assertIn("ABS(gs.b) > 30", query)
        self.assertIn("ORDER BY gs.parallax_over_error DESC", query)

    def test_prepare_rrlyrae_class_columns_assigns_period_and_class_masks(self):
        data = Table(
            {
                "best_classification": ["RRab", "RRc", "RRd"],
                "pf": [0.6, np.nan, 0.74],
                "p1_o": [np.nan, 0.32, 0.41],
            }
        )

        result = prepare_rrlyrae_class_columns(data)

        np.testing.assert_allclose(result["period"], [0.6, 0.32, 0.41])
        self.assertEqual(list(result["is_rrab"]), [True, False, False])
        self.assertEqual(list(result["is_rrc"]), [False, True, False])

    def test_summarize_relation_samples_and_load_relation_posteriors(self):
        samples = np.array(
            [
                [-2.1, 0.45, np.log10(0.08)],
                [-2.0, 0.47, np.log10(0.09)],
                [-2.2, 0.44, np.log10(0.07)],
            ]
        )

        summary = summarize_relation_samples(samples)

        self.assertIsInstance(summary, RelationPosteriorSummary)
        self.assertTrue(summary.intrinsic_sigma_median > 0)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rrab.npy"
            np.save(path, samples)
            arrays, summaries = load_relation_posteriors({"RRab": path})

        np.testing.assert_allclose(arrays["RRab"], samples)
        self.assertAlmostEqual(
            summaries["RRab"].slope_median,
            summary.slope_median,
            places=10,
        )

    def test_compute_empirical_extinction_adds_expected_columns(self):
        data = Table(
            {
                "best_classification": ["RRab", "RRc"],
                "pf": [0.62, np.nan],
                "p1_o": [np.nan, 0.33],
                "bp_rp": [0.70, 0.41],
                "phot_bp_mean_flux_over_error": [30.0, 40.0],
                "phot_rp_mean_flux_over_error": [25.0, 32.0],
                "phot_bp_rp_excess_factor": [1.05, 1.10],
                "g_absorption": [0.20, 0.10],
                "l": [10.0, 20.0],
                "b": [35.0, -42.0],
            }
        )
        models = {
            "RRab": {
                "slope_median": 0.20,
                "slope_std": 0.02,
                "intercept_median": 0.80,
                "intercept_std": 0.03,
                "intrinsic_sigma_median": 0.05,
            },
            "RRc": {
                "slope_median": 0.10,
                "slope_std": 0.01,
                "intercept_median": 0.45,
                "intercept_std": 0.02,
                "intrinsic_sigma_median": 0.04,
            },
        }

        result = compute_empirical_extinction(data, models)

        for name in (
            "log10_period",
            "color_int",
            "sigma_coeff",
            "sigma_intrinsic",
            "E_bprp",
            "A_G_calc",
            "sigma_E",
        ):
            self.assertIn(name, result.colnames)

        expected_rrab_color = 0.20 * np.log10(0.62) + 0.80
        expected_rrc_color = 0.10 * np.log10(0.33) + 0.45
        self.assertAlmostEqual(result["color_int"][0], expected_rrab_color, places=10)
        self.assertAlmostEqual(result["color_int"][1], expected_rrc_color, places=10)
        self.assertAlmostEqual(result["A_G_calc"][0], 2.0 * (0.70 - expected_rrab_color), places=10)

    def test_empirical_vs_catalog_extinction_filters_finite_rows(self):
        data = Table(
            {
                "A_G_calc": [0.20, np.nan, 0.45],
                "g_absorption": [0.18, 0.30, np.nan],
            }
        )

        result = empirical_vs_catalog_extinction(data)

        self.assertEqual(result.mask.sum(), 1)
        np.testing.assert_allclose(result.catalog, [0.18])
        np.testing.assert_allclose(result.empirical, [0.20])
        np.testing.assert_allclose(result.residuals, [0.02])

    def test_reddening_quality_mask_applies_snr_excess_and_sigma_cuts(self):
        data = Table(
            {
                "E_bprp": [0.4, 0.3, np.nan],
                "A_G_calc": [0.8, 0.6, 0.1],
                "bp_rp": [0.5, 0.7, 0.8],
                "phot_bp_mean_flux_over_error": [10.0, 3.0, 12.0],
                "phot_rp_mean_flux_over_error": [10.0, 6.0, 12.0],
                "phot_bp_rp_excess_factor": [1.02, 1.80, 1.10],
                "sigma_E": [0.10, 0.50, 0.05],
            }
        )

        mask = build_reddening_quality_mask(data, max_sigma_E=0.2, max_abs_E=1.0)
        filtered = apply_reddening_quality_mask(data, max_sigma_E=0.2, max_abs_E=1.0)

        self.assertEqual(mask.tolist(), [True, False, False])
        self.assertEqual(len(filtered), 1)

    def test_sample_sfd_ebv_and_attach_sfd_ebv_use_injected_query(self):
        data = Table({"l": [10.0, 25.0], "b": [30.0, -12.0]})
        calls = []

        def fake_query(coords):
            calls.append(coords)
            return np.array([0.11, 0.22])

        samples = sample_sfd_ebv(data, query=fake_query)
        result = attach_sfd_ebv(data, query=fake_query)

        np.testing.assert_allclose(samples, [0.11, 0.22])
        np.testing.assert_allclose(result["sfd_ebv"], [0.11, 0.22])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
