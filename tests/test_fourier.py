import unittest

import numpy as np
from astropy.table import Table

from ugdatalab import (
    build_fourier_matrix,
    cross_validate_harmonics,
    fourier_fit,
    fourier_mean_magnitude,
    phase_fold,
    predict_future_magnitude,
)


class FourierHelperTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.period = 0.58
        self.times = np.sort(rng.uniform(0.0, 12.0, size=160))
        omega = 2.0 * np.pi / self.period
        self.true_model = (
            15.2
            + 0.25 * np.cos(omega * self.times)
            - 0.12 * np.sin(omega * self.times)
            + 0.07 * np.cos(2 * omega * self.times)
        )
        self.mag_err = np.full(len(self.times), 0.03)
        self.mags = self.true_model + rng.normal(0.0, self.mag_err[0], size=len(self.times))
        self.target = Table(
            {
                "g_transit_time": self.times,
                "g_transit_mag": self.mags,
                "g_transit_mag_err": self.mag_err,
            }
        )

    def test_phase_fold_and_design_matrix_are_periodic(self):
        phases = phase_fold(np.array([0.1, 0.1 + self.period, 0.1 + 3 * self.period]), self.period)
        np.testing.assert_allclose(phases, [0.1 / self.period] * 3)

        omega = 2.0 * np.pi / self.period
        X0 = build_fourier_matrix([0.25], omega, 2)
        X1 = build_fourier_matrix([0.25 + 10 * self.period], omega, 2)
        np.testing.assert_allclose(X0, X1)

    def test_fourier_fit_recovers_training_series(self):
        fit = fourier_fit(self.target, period=self.period, k_harmonics=2)
        pred = fit.predict(self.times)

        self.assertLess(fit.chi2_r, 3.0)
        self.assertEqual(fit.beta.shape, (5,))
        self.assertLess(np.sqrt(np.mean((pred - self.true_model) ** 2)), 0.03)

    def test_cross_validate_harmonics_finds_low_order_model(self):
        result = cross_validate_harmonics(
            self.target,
            period=self.period,
            k_values=range(1, 6),
            cv_fraction=0.2,
            seed=3,
        )

        self.assertIn(result.best_k, {2, 3})
        self.assertTrue(np.all(np.isfinite(result.chi2r_train[np.isfinite(result.chi2r_train)])))
        self.assertTrue(np.isfinite(result.chi2r_cv).any())

    def test_predict_future_magnitude_and_fourier_mean_magnitude_are_finite(self):
        fit = fourier_fit(self.target, period=self.period, k_harmonics=2)
        epoch_pred, mag_pred = predict_future_magnitude(fit, days_after_last=10.0)
        mean_mag = fourier_mean_magnitude(fit, n_phase_samples=2048)

        self.assertGreater(epoch_pred, self.times.max())
        self.assertTrue(np.isfinite(mag_pred))
        self.assertTrue(np.isfinite(mean_mag))


if __name__ == "__main__":
    unittest.main()
