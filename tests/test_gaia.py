import unittest
from unittest.mock import patch

from astropy.table import MaskedColumn, Table

import ugdatalab.models.gaia as gaia


def _raw_gaia_table():
    return Table(
        {
            "source_id": [1, 2, 3],
            "l": [120.0, 130.0, 140.0],
            "b": [40.0, 50.0, 20.0],
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
            "pf": MaskedColumn([0.57, 0.48, 0.62], mask=[False, False, False]),
            "p1_o": MaskedColumn([0.0, 0.0, 0.0], mask=[True, True, True]),
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
        gaia.get_gaia_quality.clear()
        self.addCleanup(gaia.get_gaia.clear)
        self.addCleanup(gaia.get_gaia_quality.clear)

    def test_get_gaia_quality_filters_and_caches_result(self):
        raw = _raw_gaia_table()
        calls = 0

        def fake_get_gaia(query):
            nonlocal calls
            calls += 1
            return raw

        with patch.object(gaia, "get_gaia", side_effect=fake_get_gaia):
            result1 = gaia.get_gaia_quality("SELECT *")
            result2 = gaia.get_gaia_quality("SELECT *")

        self.assertEqual(calls, 1)
        self.assertEqual(list(result1["source_id"]), [1])
        self.assertIn("M_G", result1.colnames)
        self.assertIn("sigma_M", result1.colnames)
        self.assertEqual(float(result1["period"][0]), 0.57)
        self.assertTrue(bool(result1["is_rrab"][0]))
        self.assertFalse(bool(result1["is_rrc"][0]))
        self.assertEqual(list(result2["source_id"]), [1])

    def test_add_gaia_photometry_columns_skips_parallax_outputs_when_missing(self):
        data = Table(
            {
                "phot_g_mean_flux": [1000.0],
                "phot_g_mean_flux_error": [10.0],
                "phot_g_mean_mag": [15.0],
            }
        )

        result = gaia._add_gaia_photometry_columns(data)

        self.assertIn("sigma_G", result.colnames)
        self.assertNotIn("mu", result.colnames)
        self.assertNotIn("sigma_mu", result.colnames)
        self.assertNotIn("M_G", result.colnames)
        self.assertNotIn("sigma_M", result.colnames)

    def test_gaia_data_instantiation_uses_raw_getter(self):
        raw = _raw_gaia_table()

        with patch.object(gaia, "get_gaia", return_value=raw) as mock_get_gaia:
            data = gaia.GaiaData("SELECT raw")

        mock_get_gaia.assert_called_once_with("SELECT raw")
        self.assertEqual(list(data.data["source_id"]), [1, 2, 3])
        self.assertIsNone(data.lightcurve_data)

    def test_gaia_data_can_load_lightcurves_at_instantiation(self):
        raw = _raw_gaia_table()
        lightcurves = _epoch_table()

        with patch.object(gaia, "get_gaia", return_value=raw):
            with patch("ugdatalab.lightcurves.fetch_joined_epoch_photometry", return_value=lightcurves) as mock_fetch:
                data = gaia.GaiaData("SELECT raw", include_lightcurve=True)

        mock_fetch.assert_called_once_with(raw)
        self.assertEqual(list(data.lightcurve_data["source_id"]), [2, 1, 1])

    def test_gaia_quality_instantiation_uses_quality_getter(self):
        quality = _raw_gaia_table()[:1]

        with patch.object(gaia, "get_gaia_quality", return_value=quality) as mock_get_gaia_quality:
            data = gaia.GaiaQuality("SELECT quality")

        mock_get_gaia_quality.assert_called_once_with("SELECT quality")
        self.assertEqual(list(data.data["source_id"]), [1])
        self.assertIsNone(data.lightcurve_data)

    def test_local_uses_quality_gaia_data(self):
        quality = _raw_gaia_table()[:1]
        with patch.object(gaia, "get_gaia_quality", return_value=quality):
            source = gaia.GaiaQuality("SELECT quality")
            local = gaia.Local(source)

        self.assertEqual(list(local.data["source_id"]), [1])

    def test_filtered_gaia_objects_drop_lightcurves(self):
        quality = _raw_gaia_table()
        quality["parallax"][1] = 0.2

        with patch.object(gaia, "get_gaia_quality", return_value=quality):
            source = gaia.GaiaQuality("SELECT quality")

        source.include_lightcurve = True
        source.lightcurve_data = _epoch_table()
        local = gaia.Local(source)

        self.assertEqual(list(local.data["source_id"]), [1, 3])
        self.assertFalse(local.include_lightcurve)
        self.assertIsNone(local.lightcurve_data)


if __name__ == "__main__":
    unittest.main()
