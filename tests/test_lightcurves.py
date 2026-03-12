import unittest
from unittest.mock import patch

import numpy as np
from astropy.table import Table

from ugdatalab import (
    attach_flux_mean_magnitudes,
    attach_periodogram_periods,
    build_fourier_matrix,
    build_rrlyrae_class_lightcurve_query,
    build_rrlyrae_top_n_query,
    cross_validate_harmonics,
    fetch_epoch_photometry,
    fourier_fit,
    fourier_mean_magnitude,
    get_epoch_photometry,
    join_catalog_with_epoch_photometry,
    lomb_scargle_periodogram,
    predict_future_magnitude,
)


class LightcurveHelperTests(unittest.TestCase):
    def setUp(self):
        get_epoch_photometry.clear()
        self.addCleanup(get_epoch_photometry.clear)

    def test_build_rrlyrae_top_n_query_matches_lab_filters(self):
        query = build_rrlyrae_top_n_query()

        self.assertIn("TOP 100", query)
        self.assertIn("FROM gaiadr3.vari_rrlyrae", query)
        self.assertIn("pf IS NOT NULL", query)
        self.assertIn("num_clean_epochs_g > 40", query)
        self.assertIn("ORDER BY num_clean_epochs_g DESC", query)

    def test_build_rrlyrae_class_lightcurve_query_uses_rr_class_brightness_and_epoch_filters(self):
        query = build_rrlyrae_class_lightcurve_query("RRc")

        self.assertIn("best_classification = 'RRc'", query)
        self.assertIn("int_average_g < 15", query)
        self.assertIn("num_clean_epochs_g > 80", query)

    def test_attach_flux_mean_magnitudes_adds_source_level_columns(self):
        data = Table(
            {
                "source_id": [1, 1, 2, 2],
                "g_transit_flux": [1000.0, 1020.0, 900.0, 920.0],
                "g_transit_flux_error": [10.0, 11.0, 12.0, 12.0],
            }
        )

        result = attach_flux_mean_magnitudes(data)

        self.assertIn("g_transit_mag_err", result.colnames)
        self.assertIn("mean_g_transit_mag", result.colnames)
        self.assertIn("mean_g_transit_mag_err", result.colnames)
        self.assertTrue(np.all(np.asarray(result["g_transit_mag_err"]) > 0.0))
        self.assertTrue(np.all(np.isfinite(np.asarray(result["mean_g_transit_mag"], dtype=float))))
        self.assertTrue(np.all(np.asarray(result["mean_g_transit_mag_err"]) > 0.0))
        self.assertEqual(result["mean_g_transit_mag"][0], result["mean_g_transit_mag"][1])
        self.assertEqual(result["mean_g_transit_mag"][2], result["mean_g_transit_mag"][3])

    def test_attach_periodogram_periods_adds_repeated_best_period_per_source(self):
        period_1 = 0.61
        period_2 = 0.47
        rng = np.random.default_rng(17)
        epochs_1 = np.sort(rng.uniform(0.0, 20.0, 120))
        epochs_2 = np.sort(rng.uniform(0.0, 18.0, 120))
        data = Table(
            {
                "source_id": np.concatenate(
                    [
                        np.full(len(epochs_1), 1, dtype=int),
                        np.full(len(epochs_2), 2, dtype=int),
                    ]
                ),
                "g_transit_time": np.concatenate([epochs_1, epochs_2]),
                "g_transit_flux": np.concatenate(
                    [
                        1.0 + 0.2 * np.sin(2.0 * np.pi * epochs_1 / period_1),
                        1.0 + 0.15 * np.sin(2.0 * np.pi * epochs_2 / period_2),
                    ]
                ),
                "g_transit_flux_error": np.full(len(epochs_1) + len(epochs_2), 0.02),
            }
        )

        result = attach_periodogram_periods(data)

        self.assertIn("period_ls", result.colnames)
        self.assertEqual(len(np.unique(result["period_ls"][result["source_id"] == 1])), 1)
        self.assertEqual(len(np.unique(result["period_ls"][result["source_id"] == 2])), 1)
        self.assertAlmostEqual(float(result["period_ls"][0]), period_1, places=2)
        self.assertAlmostEqual(float(result["period_ls"][-1]), period_2, places=2)

    def test_fetch_epoch_photometry_raises_for_missing_sources(self):
        epochs1 = Table(
            {
                "source_id": [1, 1],
                "g_transit_time": [0.2, 0.8],
                "g_transit_mag": [15.0, 15.1],
                "g_transit_flux": [1010.0, 1000.0],
                "g_transit_flux_error": [10.0, 10.0],
            }
        )
        epochs3 = Table(
            {
                "source_id": [3],
                "g_transit_time": [0.5],
                "g_transit_mag": [16.0],
                "g_transit_flux": [900.0],
                "g_transit_flux_error": [9.0],
            }
        )

        def fake_get_epoch_photometry(source_id):
            if source_id == 2:
                raise KeyError("missing")
            return epochs1 if source_id == 1 else epochs3

        with patch("ugdatalab.lightcurves.get_epoch_photometry", side_effect=fake_get_epoch_photometry):
            with self.assertRaises(KeyError):
                fetch_epoch_photometry([1, 2, 3])

    def test_join_catalog_with_epoch_photometry_repeats_catalog_columns(self):
        catalog = Table(
            {
                "source_id": [1, 2],
                "period": [0.55, 0.62],
            }
        )
        epochs = Table(
            {
                "source_id": [2, 1, 1],
                "g_transit_time": [0.5, 0.8, 0.2],
                "g_transit_mag": [16.0, 15.1, 15.0],
                "g_transit_flux": [900.0, 1000.0, 1010.0],
                "g_transit_flux_error": [9.0, 10.0, 10.0],
            }
        )

        joined = join_catalog_with_epoch_photometry(catalog, epochs)

        self.assertEqual(len(joined), 3)
        np.testing.assert_allclose(joined["period"][joined["source_id"] == 1], [0.55, 0.55])
        np.testing.assert_allclose(joined["period"][joined["source_id"] == 2], [0.62])

    def test_lomb_scargle_periodogram_recovers_known_period(self):
        true_period = 0.61
        rng = np.random.default_rng(11)
        epochs = np.sort(rng.uniform(0.0, 20.0, 120))
        flux = 1.0 + 0.2 * np.sin(2.0 * np.pi * epochs / true_period)
        target = Table(
            {
                "g_transit_time": epochs,
                "g_transit_flux": flux,
                "g_transit_flux_error": np.full(len(epochs), 0.02),
            }
        )

        periods, power, best_period = lomb_scargle_periodogram(target)

        self.assertAlmostEqual(best_period, true_period, places=2)
        self.assertGreaterEqual(power[0], power[-1])

    def test_build_fourier_matrix_has_expected_shape(self):
        X = build_fourier_matrix([0.0, 0.5, 1.0], omega=2.0 * np.pi, k=2)
        self.assertEqual(X.shape, (3, 5))
        np.testing.assert_allclose(X[:, 0], 1.0)

    def test_fourier_fit_recovers_noise_free_signal(self):
        period = 0.73
        epochs = np.linspace(0.0, 4.0 * period, 80, endpoint=False)
        mags = 15.0 + 0.3 * np.cos(2.0 * np.pi * epochs / period) + 0.1 * np.sin(
            4.0 * np.pi * epochs / period
        )
        target = Table(
            {
                "source_id": np.ones(len(epochs), dtype=int),
                "g_transit_time": epochs,
                "g_transit_mag": mags,
                "g_transit_mag_err": np.full(len(epochs), 0.02),
            }
        )

        fit = fourier_fit(target, period, k=2)

        self.assertLess(fit.chi2_r, 1e-10)
        np.testing.assert_allclose(fit.predict(epochs), mags, atol=1e-10)

    def test_cross_validate_harmonics_returns_partition_and_best_k(self):
        period = 0.58
        epochs = np.linspace(0.0, 5.0 * period, 100, endpoint=False)
        mags = 15.1 + 0.25 * np.cos(2.0 * np.pi * epochs / period)
        rng = np.random.default_rng(7)
        mags = mags + rng.normal(0.0, 0.02, size=len(mags))
        target = Table(
            {
                "source_id": np.ones(len(epochs), dtype=int),
                "g_transit_time": epochs,
                "g_transit_mag": mags,
                "g_transit_mag_err": np.full(len(epochs), 0.02),
            }
        )

        result = cross_validate_harmonics(target, period)
        k_values, chi2r_train, chi2r_cv, best_k, _, _ = result
        expected_k_values = tuple(range(1, 26))

        self.assertIn(best_k, expected_k_values)
        self.assertEqual(tuple(k_values), expected_k_values)
        self.assertEqual(len(chi2r_train), len(expected_k_values))
        self.assertEqual(len(chi2r_cv), len(expected_k_values))

    def test_fourier_mean_helpers_match_constant_signal(self):
        period = 0.6
        epochs = np.linspace(0.0, period, 40, endpoint=False)
        mags = np.full(len(epochs), 15.0)
        target = Table(
            {
                "source_id": np.ones(len(epochs), dtype=int),
                "g_transit_time": epochs,
                "g_transit_mag": mags,
                "g_transit_mag_err": np.full(len(epochs), 0.01),
            }
        )

        fit = fourier_fit(target, period, k=1)
        mean_mag = fourier_mean_magnitude(fit)
        epoch_pred, mag_pred = predict_future_magnitude(fit)

        self.assertAlmostEqual(mean_mag, 15.0, places=6)
        self.assertAlmostEqual(mag_pred, 15.0, places=6)
        self.assertAlmostEqual(epoch_pred, float(np.max(epochs) + 10.0), places=6)


if __name__ == "__main__":
    unittest.main()
