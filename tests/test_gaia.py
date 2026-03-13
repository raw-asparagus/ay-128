import unittest
from unittest.mock import patch

import numpy as np
from astropy.table import MaskedColumn, Table

import ugdatalab.models.gaia as gaia


def _raw_gaia_table():
    return Table(
        {
            "source_id": [1, 2, 3],
            "num_clean_epochs_g": [90, 80, 70],
            "l": [120.0, 130.0, 140.0],
            "b": [40.0, 50.0, 20.0],
            "best_classification": ["RRab", "RRc", "RRd"],
            "parallax": [0.50, 0.40, 0.30],
            "parallax_error": [0.05, 0.06, 0.04],
            "parallax_over_error": [6.0, 4.0, 7.0],
            "phot_g_mean_flux": [1000.0, 1100.0, 1200.0],
            "phot_g_mean_flux_error": [10.0, 11.0, 12.0],
            "phot_g_mean_mag": [15.0, 16.0, 17.0],
            "bp_rp": [0.7, 0.8, 0.9],
            "phot_g_mean_flux_over_error": [30.0, 25.0, 20.0],
            "phot_bp_mean_flux_over_error": [20.0, 18.0, 16.0],
            "phot_rp_mean_flux_over_error": [18.0, 17.0, 15.0],
            "phot_bp_rp_excess_factor": [1.05, 1.08, 1.12],
            "ruwe": [1.0, 1.1, 1.0],
            "pf": MaskedColumn([0.57, 0.0, 0.62], mask=[False, True, False]),
            "p1_o": MaskedColumn([0.0, 0.31, 0.46], mask=[True, False, False]),
        }
    )


def _epoch_table():
    return Table(
        {
            "source_id": [2, 1, 1],
            "g_transit_time": [0.5, 0.8, 0.2],
            "g_transit_mag": [16.0, 15.1, 15.0],
            "g_transit_flux": [900.0, 1000.0, 1010.0],
            "g_transit_flux_error": [9.0, 10.0, 10.0],
        }
    )


class GaiaCatalogTests(unittest.TestCase):
    def setUp(self):
        gaia.get_gaia.clear()
        gaia._get_gaia_quality.clear()
        self.addCleanup(gaia.get_gaia.clear)
        self.addCleanup(gaia._get_gaia_quality.clear)

    def test__get_gaia_quality_filters_and_caches_result(self):
        raw = _raw_gaia_table()
        calls = 0

        def fake_get_gaia(query):
            nonlocal calls
            calls += 1
            return raw

        with patch.object(gaia, "get_gaia", side_effect=fake_get_gaia):
            result1 = gaia._get_gaia_quality("SELECT *")
            result2 = gaia._get_gaia_quality("SELECT *")

        self.assertEqual(calls, 1)
        self.assertEqual(list(result1["source_id"]), [1])
        self.assertIn("M_G", result1.colnames)
        self.assertIn("sigma_M", result1.colnames)
        self.assertNotIn("period", result1.colnames)
        self.assertNotIn("is_rrab", result1.colnames)
        self.assertNotIn("is_rrc", result1.colnames)
        self.assertEqual(list(result2["source_id"]), [1])

    def test_rrlyrae_class_mask_uses_best_classification(self):
        data = _raw_gaia_table()

        rrab = gaia.rrlyrae_class_mask(data, "RRab")
        rrc = gaia.rrlyrae_class_mask(data, "RRc")
        rrd = gaia.rrlyrae_class_mask(data, "RRd")

        self.assertEqual(rrab.tolist(), [True, False, False])
        self.assertEqual(rrc.tolist(), [False, True, False])
        self.assertEqual(rrd.tolist(), [False, False, True])

    def test_rrlyrae_representative_period_uses_rrd_first_overtone(self):
        data = _raw_gaia_table()

        period = gaia.rrlyrae_representative_period(data)

        np.testing.assert_allclose(period[:2], [0.57, 0.31])
        self.assertAlmostEqual(float(period[2]), 0.46)

    def test__sanitize_vari_rrlyrae_table_normalizes_masked_columns(self):
        raw = Table(
            {
                "source_id": [1, 2],
                "num_clean_epochs_g": [90, 80],
                "best_classification": ["RRab", "RRd"],
                "pf": MaskedColumn([0.57, 0.74], mask=[False, False]),
                "pf_error": MaskedColumn([0.01, 0.02], mask=[False, False]),
                "p1_o": MaskedColumn([0.0, 0.55], mask=[True, False]),
                "p1_o_error": MaskedColumn([0.0, 0.01], mask=[True, False]),
                "int_average_g": MaskedColumn([15.1, 15.4], mask=[False, False]),
            }
        )

        result = gaia._sanitize_vari_rrlyrae_table(raw)

        self.assertEqual(result["source_id"].dtype.kind, "i")
        self.assertEqual(result["num_clean_epochs_g"].dtype.kind, "i")
        self.assertEqual(result["best_classification"].dtype.kind, "U")
        self.assertEqual(result["pf"].dtype.kind, "f")
        self.assertEqual(result["pf_error"].dtype.kind, "f")
        self.assertEqual(result["p1_o"].dtype.kind, "f")
        self.assertTrue(np.isnan(float(result["p1_o"][0])))
        self.assertAlmostEqual(float(result["p1_o"][1]), 0.55)

    def test__add_gaia_photometry_columns_adds_distance_and_absolute_mag_columns(self):
        data = Table(
            {
                "phot_g_mean_flux": [1000.0],
                "phot_g_mean_flux_error": [10.0],
                "phot_g_mean_mag": [15.0],
                "parallax": [0.5],
                "parallax_error": [0.05],
            }
        )

        result = gaia._add_gaia_photometry_columns(data)

        self.assertIn("sigma_G", result.colnames)
        self.assertIn("mu", result.colnames)
        self.assertIn("sigma_mu", result.colnames)
        self.assertIn("M_G", result.colnames)
        self.assertIn("sigma_M", result.colnames)

    def test_gaia_data_instantiation_uses_raw_getter(self):
        raw = _raw_gaia_table()

        with patch.object(gaia, "get_gaia", return_value=raw) as mock_get_gaia:
            data = gaia.GaiaData("SELECT raw")

        mock_get_gaia.assert_called_once_with("SELECT raw")
        self.assertEqual(list(data.data["source_id"]), [1, 2, 3])
        self.assertIsNone(data.lightcurves)

    def test_gaia_data_can_load_lightcurves_at_instantiation(self):
        raw = _raw_gaia_table()
        lightcurves = _epoch_table()
        expected = gaia._sanitize_vari_rrlyrae_table(raw)

        with patch.object(gaia, "get_gaia", return_value=raw):
            with patch("ugdatalab.lightcurves._fetch_joined_epoch_photometry", return_value=lightcurves) as mock_fetch:
                data = gaia.GaiaData("SELECT raw", include_lightcurve=True)

        mock_fetch.assert_called_once()
        np.testing.assert_array_equal(mock_fetch.call_args.args[0]["source_id"], expected["source_id"])
        self.assertEqual(list(data.lightcurves["source_id"]), [2, 1, 1])

    def test_get_gaia_uses_async_query(self):
        raw = _raw_gaia_table()

        class _FakeJob:
            def get_results(self):
                return raw

        with patch.object(gaia.Gaia, "launch_job_async", return_value=_FakeJob()) as mock_async:
            result = gaia.get_gaia("SELECT raw")

        mock_async.assert_called_once_with("SELECT raw")
        self.assertEqual(list(result["source_id"]), [1, 2, 3])

    def test_gaia_quality_instantiation_uses_quality_getter(self):
        quality = _raw_gaia_table()[:1]

        with patch.object(gaia, "_get_gaia_quality", return_value=quality) as mock_get_gaia_quality:
            data = gaia.GaiaQuality("SELECT quality")

        mock_get_gaia_quality.assert_called_once_with("SELECT quality")
        self.assertEqual(list(data.data["source_id"]), [1])
        self.assertIsNone(data.lightcurves)

    def test_local_uses_quality_gaia_data(self):
        quality = _raw_gaia_table()[:1]
        with patch.object(gaia, "_get_gaia_quality", return_value=quality):
            source = gaia.GaiaQuality("SELECT quality")
            local = gaia.Local(source)

        self.assertEqual(list(local.data["source_id"]), [1])

    def test_filtered_gaia_objects_drop_lightcurves(self):
        quality = _raw_gaia_table()
        quality["parallax"][1] = 0.2

        with patch.object(gaia, "_get_gaia_quality", return_value=quality):
            source = gaia.GaiaQuality("SELECT quality")

        source.include_lightcurve = True
        source.lightcurves = _epoch_table()
        local = gaia.Local(source)

        self.assertEqual(list(local.data["source_id"]), [1, 3])
        self.assertFalse(local.include_lightcurve)
        self.assertIsNone(local.lightcurves)


if __name__ == "__main__":
    unittest.main()
