import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from astropy.table import MaskedColumn, Table

import ugdatalab.models.sdss as sdss
from ugdatalab import SDSSData, SDSSQuality, get_sdss, get_sdss_quality


class SDSSCatalogTests(unittest.TestCase):
    def setUp(self):
        get_sdss.clear()
        get_sdss_quality.clear()
        self.addCleanup(get_sdss.clear)
        self.addCleanup(get_sdss_quality.clear)

    def test_get_sdss_returns_full_catalog_and_caches_result(self):
        table = Table(
            {
                "APOGEE_ID": ["star-1", "star-2"],
                "TEFF": [4500.0, 4800.0],
            }
        )

        calls = 0

        def fake_read():
            nonlocal calls
            calls += 1
            return table

        with patch.object(sdss, "_query_sdss_allstar", side_effect=fake_read):
            result1 = get_sdss()
            result2 = get_sdss()

        self.assertEqual(calls, 1)
        self.assertEqual(result1.colnames, table.colnames)
        self.assertEqual(list(result1["APOGEE_ID"]), ["star-1", "star-2"])
        self.assertEqual(list(result2["APOGEE_ID"]), ["star-1", "star-2"])

    def test_get_sdss_quality_filters_invalid_labels_and_low_snr(self):
        table = Table(
            {
                "APOGEE_ID": [
                    "good",
                    "low_snr",
                    "masked_teff",
                    "bad_logg",
                    "bad_mg",
                    "bad_si",
                    "bad_feh",
                ],
                "TEFF": MaskedColumn(
                    [4500.0, 4600.0, 4700.0, 4800.0, 4900.0, 5000.0, 5100.0],
                    mask=[False, False, True, False, False, False, False],
                ),
                "LOGG": [2.0, 2.1, 2.2, -9999.0, 2.3, 2.4, 2.5],
                "FE_H": [-0.2, -0.1, -0.3, -0.2, -0.1, -0.4, -9999.0],
                "MG_FE": [0.10, 0.11, 0.12, 0.13, np.nan, 0.14, 0.15],
                "SI_FE": [0.05, 0.05, 0.06, 0.07, 0.08, -9999.0, 0.09],
                "SNR": [80.0, 40.0, 90.0, 90.0, 90.0, 90.0, 90.0],
            }
        )

        with patch.object(sdss, "_query_sdss_allstar", return_value=table):
            result = get_sdss_quality()

        self.assertEqual(list(result["APOGEE_ID"]), ["good"])

    def test_get_sdss_quality_uses_m_h_when_fe_h_is_missing(self):
        table = Table(
            {
                "APOGEE_ID": ["good", "missing_m_h"],
                "TEFF": [4500.0, 4700.0],
                "LOGG": [2.0, 2.2],
                "M_H": [-0.2, -9999.0],
                "MG_FE": [0.10, 0.11],
                "SI_FE": [0.05, 0.06],
                "SNR": [70.0, 80.0],
            }
        )

        with patch.object(sdss, "_query_sdss_allstar", return_value=table):
            result = get_sdss_quality()

        self.assertEqual(list(result["APOGEE_ID"]), ["good"])

    def test_sdss_data_instantiation_uses_full_catalog_getter(self):
        table = Table({"APOGEE_ID": ["full-1", "full-2"]})

        with patch.object(sdss, "get_sdss", return_value=table) as mock_get_sdss:
            data = SDSSData()

        mock_get_sdss.assert_called_once_with()
        self.assertEqual(list(data.data["APOGEE_ID"]), ["full-1", "full-2"])

    def test_sdss_quality_instantiation_uses_quality_getter(self):
        table = Table({"APOGEE_ID": ["quality-1"]})

        with patch.object(sdss, "get_sdss_quality", return_value=table) as mock_get_sdss_quality:
            data = SDSSQuality()

        mock_get_sdss_quality.assert_called_once_with()
        self.assertEqual(list(data.data["APOGEE_ID"]), ["quality-1"])

    def test_run_sdss_sql_raises_readable_error_on_non_csv_response(self):
        response = SimpleNamespace(
            text="Incorrect syntax near 'target_id'.",
            raise_for_status=lambda: None,
        )

        with patch.object(sdss.SDSS, "query_sql_async", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "non-CSV response"):
                sdss._run_sdss_sql("SELECT broken")

    def test_run_sdss_sql_accepts_comment_prefixed_csv(self):
        response = SimpleNamespace(
            text="#Table1\nAPOGEE_ID,TEFF\nstar-1,4500.0\nstar-2,4800.0\n",
            raise_for_status=lambda: None,
        )

        with patch.object(sdss.SDSS, "query_sql_async", return_value=response):
            result = sdss._run_sdss_sql("SELECT ok")

        self.assertEqual(list(result["APOGEE_ID"]), ["star-1", "star-2"])


if __name__ == "__main__":
    unittest.main()
