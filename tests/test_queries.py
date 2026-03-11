import unittest

from ugdatalab import (
    build_full_rrlyrae_join_query,
    build_local_rrlyrae_query,
    build_rrlyrae_class_lightcurve_query,
    build_rrlyrae_top_n_query,
)


class QueryBuilderTests(unittest.TestCase):
    def test_build_rrlyrae_top_n_query_includes_expected_filters(self):
        query = build_rrlyrae_top_n_query(limit=50, min_clean_epochs_g=60)

        self.assertIn("SELECT TOP 50 *", query)
        self.assertIn("FROM gaiadr3.vari_rrlyrae", query)
        self.assertIn("pf IS NOT NULL", query)
        self.assertIn("num_clean_epochs_g > 60", query)
        self.assertIn("ORDER BY num_clean_epochs_g DESC", query)

    def test_build_rrlyrae_class_lightcurve_query_targets_requested_class(self):
        query = build_rrlyrae_class_lightcurve_query(
            "RRc",
            limit=5,
            max_int_average_g=14.5,
            min_clean_epochs_g=120,
        )

        self.assertIn("SELECT TOP 5 *", query)
        self.assertIn("best_classification = 'RRc'", query)
        self.assertIn("int_average_g < 14.500", query)
        self.assertIn("num_clean_epochs_g > 120", query)

    def test_build_local_rrlyrae_query_encodes_lab_constraints(self):
        query = build_local_rrlyrae_query(
            max_fractional_parallax_error=0.2,
            min_abs_b_deg=30.0,
            max_distance_kpc=4.0,
        )

        self.assertIn("JOIN gaiadr3.gaia_source AS gs", query)
        self.assertIn("gs.parallax_over_error > 5.000000", query)
        self.assertIn("ABS(gs.b) > 30.000", query)
        self.assertIn("gs.parallax > 0.250000", query)

    def test_build_full_rrlyrae_join_query_has_expected_join(self):
        query = build_full_rrlyrae_join_query()

        self.assertIn("FROM gaiadr3.vari_rrlyrae AS vr", query)
        self.assertIn("JOIN gaiadr3.gaia_source AS gs", query)
        self.assertIn("vr.source_id = gs.source_id", query)


if __name__ == "__main__":
    unittest.main()
