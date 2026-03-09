import unittest
from types import SimpleNamespace

import numpy as np
from astropy.table import Table

from ugdatalab import estimate_initial_theta0, prepare_relation_data


class RelationHelperTests(unittest.TestCase):
    def test_prepare_relation_data_rrab_pl_filters_by_class_and_valid_rows(self):
        table = Table(
            {
                "period": [0.60, 0.32, 0.70],
                "M_G": [0.55, 0.20, np.nan],
                "sigma_M": [0.08, 0.03, 0.09],
                "bp_rp": [0.72, 0.31, 0.80],
                "phot_bp_mean_flux_over_error": [30.0, 40.0, 50.0],
                "phot_rp_mean_flux_over_error": [25.0, 35.0, 45.0],
                "is_rrab": [True, False, True],
                "is_rrc": [False, True, False],
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
                "period": [0.62, 0.71],
                "M_G": [0.49, 0.41],
                "sigma_M": [0.07, 0.08],
                "is_rrab": [True, True],
                "is_rrc": [False, False],
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


if __name__ == "__main__":
    unittest.main()
