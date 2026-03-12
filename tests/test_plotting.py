import os
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from ugdatalab.plotting import (
    LW_GRID,
    figure,
    figure_names,
    plot_corner,
    plot_fourier_cross_validation,
    plot_fourier_harmonic_fits,
    plot_fourier_extrapolation,
    plot_fourier_normalized_residual_histograms,
    plot_fourier_train_cv_phase_comparison,
    plot_hr,
    plot_inlier_prob_map,
    plot_lomb_scargle_periodogram,
    plot_mean_g_catalog_comparison,
    plot_mollweide,
    plot_mollweide_diff,
    plot_period_abs_mag,
    plot_period_mean_g,
    plot_vari_rrlyrae_period_comparison,
    plot_period_luminosity_diff,
    plot_posterior,
    plot_raw_phase_folded_lightcurve,
    plot_trace,
)
from ugdatalab.lightcurves import fourier_fit


def _gaia_quality_table():
    return Table(
        {
            "source_id": [1, 2],
            "l": [120.0, 130.0],
            "b": [40.0, 45.0],
            "period": [0.57, 0.62],
            "is_rrc": [False, True],
            "M_G": [0.6, 0.4],
            "sigma_M": [0.08, 0.07],
            "bp_rp": [0.72, 0.39],
            "inlier_prob": [0.95, 0.80],
        }
    )


def _lightcurve_table():
    flux = 10.0 ** (-0.4 * (np.asarray([15.1, 15.4, 15.2]) - 25.6874))
    return Table(
        {
            "source_id": [1, 1, 1],
            "g_transit_time": [0.1, 0.4, 0.7],
            "g_transit_mag": [15.1, 15.4, 15.2],
            "g_transit_mag_err": [0.03, 0.04, 0.03],
            "g_transit_flux": flux,
            "g_transit_flux_error": 0.03 * flux,
            "period_ls": [0.5, 0.5, 0.5],
            "best_classification": ["RRab", "RRab", "RRab"],
            "pf": [0.5, 0.5, 0.5],
            "p1_o": [np.nan, np.nan, np.nan],
        }
    )


def _fourier_lightcurve_table():
    period = 0.5
    phase = np.linspace(0.0, 1.0, 40, endpoint=False)
    epoch = phase * period
    mag = 15.2 + 0.25 * np.cos(2.0 * np.pi * phase) - 0.1 * np.sin(4.0 * np.pi * phase)
    flux = 10.0 ** (-0.4 * (mag - 25.6874))
    return Table(
        {
            "source_id": np.ones(len(epoch), dtype=int),
            "g_transit_time": epoch,
            "g_transit_mag": mag,
            "g_transit_mag_err": np.full(len(epoch), 0.02),
            "g_transit_flux": flux,
            "g_transit_flux_error": np.full(len(epoch), 0.02) * flux,
            "period_ls": np.full(len(epoch), period),
            "best_classification": np.full(len(epoch), "RRab"),
            "pf": np.full(len(epoch), period),
            "p1_o": np.full(len(epoch), np.nan),
        }
    )


def _period_abs_mag_table():
    return Table(
        {
            "source_id": [1, 2],
            "period": [0.55, 0.62],
            "period_ls": [0.81, 0.93],
            "is_rrc": [False, True],
            "M_G": [0.6, 0.4],
            "sigma_M": [0.08, 0.07],
            "M_G_ls": [0.1, -0.2],
            "sigma_M_ls": [0.03, 0.04],
        }
    )


def _period_mean_g_table():
    return Table(
        {
            "best_classification": ["RRab", "RRc", "RRd"],
            "best_period": [0.81, 0.93, 0.74],
            "mean_apparent_g": [15.2, 15.6, 15.35],
            "mean_apparent_g_err": [0.03, 0.04, 0.05],
        }
    )


def _vari_rrlyrae_period_table():
    return Table(
        {
            "best_classification": ["RRab", "RRc", "RRd"],
            "pf": [0.55, np.nan, 0.74],
            "pf_error": [0.01, np.nan, 0.02],
            "best_period": [0.57, 0.31, 0.55],
            "p1_o": [np.nan, np.nan, 0.55],
            "p1_o_error": [np.nan, np.nan, 0.01],
        }
    )


def _cross_validation_series():
    Ks = np.arange(1, 6, dtype=int)
    chi2r_train = np.array([2.4, 1.4, 0.95, 0.82, 0.76], dtype=float)
    chi2r_cv = np.array([2.6, 1.2, 0.88, 0.97, 1.15], dtype=float)
    return Ks, chi2r_train, chi2r_cv


def _sampler_view():
    return SimpleNamespace(
        samples=np.array([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2], [0.3, 1.3]], dtype=float),
        log_probs=np.array([-2.0, -1.5, -1.2, -1.0], dtype=float),
        n_burn=1,
        param_labels=[r"$a$", r"$b$"],
    )


class PlottingHelperTests(unittest.TestCase):
    @contextmanager
    def writable_cache_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mplconfigdir = os.path.join(tmpdir, "matplotlib")
            xdg_cache_home = os.path.join(tmpdir, "xdg-cache")
            os.makedirs(mplconfigdir, exist_ok=True)
            os.makedirs(xdg_cache_home, exist_ok=True)
            with patch.dict(
                os.environ,
                {
                    **os.environ,
                    "MPLCONFIGDIR": mplconfigdir,
                    "XDG_CACHE_HOME": xdg_cache_home,
                },
                clear=True,
            ):
                yield

    def test_gaia_plot_helpers_return_axes(self):
        data = _gaia_quality_table()
        subset = data[:1]

        ax1 = plot_mollweide(data)
        ax2 = plot_mollweide_diff(data, subset)
        ax3 = plot_period_abs_mag(data)
        ax4 = plot_period_luminosity_diff(data, subset)
        ax5 = plot_hr(data)

        for ax in (ax1, ax2, ax3, ax4, ax5):
            self.assertIsNotNone(ax)
            plt.close(ax.figure)

    def test_rcparams_use_visual_weight_tokens(self):
        self.assertEqual(mpl.rcParams["grid.linewidth"], LW_GRID)

    def test_plot_inlier_prob_map_uses_all_data(self):
        source = SimpleNamespace(all_data=_gaia_quality_table())

        ax = plot_inlier_prob_map(source)

        self.assertIsNotNone(ax)
        plt.close(ax.figure)

    def test_plot_raw_phase_folded_lightcurve_returns_axes(self):
        axes = plot_raw_phase_folded_lightcurve(_lightcurve_table(), "RRab", 0.5)

        self.assertEqual(len(axes), 2)
        for ax in axes:
            self.assertIsNotNone(ax)
        self.assertFalse(axes[0].get_shared_y_axes().joined(axes[0], axes[1]))
        self.assertEqual(axes[1].get_ylabel(), "G")
        self.assertTrue(any(label.get_visible() for label in axes[1].get_yticklabels()))
        self.assertEqual(axes[0].get_title(), "")
        self.assertEqual(axes[1].get_title(), "")
        self.assertEqual(axes[0].get_legend().texts[0].get_text(), "Raw light curve")
        self.assertEqual(axes[1].get_legend().texts[0].get_text(), "Phase-folded light curve")
        phase_color = mpl.colors.to_rgba("C1", alpha=0.55)
        np.testing.assert_allclose(axes[1].collections[-1].get_facecolors()[0], phase_color)
        self.assertEqual(len(axes[0].containers), 1)
        self.assertEqual(len(axes[1].containers), 1)
        plt.close(axes[0].figure)

    def test_plot_lomb_scargle_periodogram_marks_selected_period(self):
        ax = plot_lomb_scargle_periodogram(_fourier_lightcurve_table(), "RRab")

        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), r"$P$ [days]")
        self.assertTrue(
            any(
                len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [0.5, 0.5])
                for line in ax.lines
            )
        )
        plt.close(ax.figure)

    def test_plot_fourier_harmonic_fits_returns_grid_of_axes(self):
        axes = plot_fourier_harmonic_fits(_fourier_lightcurve_table(), "RRab", 0.5, [1, 3])
        within_gap = axes[0, 0].get_position().y0 - axes[0, 1].get_position().y1
        between_gap = axes[0, 1].get_position().y0 - axes[1, 0].get_position().y1

        self.assertEqual(np.asarray(axes).shape, (2, 2))
        self.assertEqual(axes[0, 0].get_ylabel(), r"$G$ [mag]")
        self.assertEqual(axes[0, 1].get_ylabel(), "Res.")
        self.assertEqual(axes[0, 0].get_subplotspec().colspan.start, 0)
        self.assertEqual(axes[0, 1].get_subplotspec().colspan.start, 0)
        self.assertEqual(axes[0, 0].get_subplotspec().rowspan.stop - axes[0, 0].get_subplotspec().rowspan.start, 1)
        self.assertEqual(axes[0, 1].get_subplotspec().rowspan.stop - axes[0, 1].get_subplotspec().rowspan.start, 1)
        self.assertAlmostEqual(axes[0, 0].get_position().x0, axes[0, 1].get_position().x0, places=3)
        self.assertAlmostEqual(axes[0, 0].get_position().x1, axes[0, 1].get_position().x1, places=3)
        self.assertGreater(axes[0, 0].get_position().height, 3.0 * axes[0, 1].get_position().height)
        self.assertLess(within_gap, 1e-6)
        self.assertGreater(between_gap, 1e-2)
        self.assertEqual(axes[0, 1].get_xlabel(), "Phase")
        self.assertEqual(axes[1, 1].get_xlabel(), "Phase")
        plt.close(axes[0, 0].figure)

    def test_plot_fourier_cross_validation_returns_axis(self):
        Ks, chi2r_train, chi2r_cv = _cross_validation_series()

        ax = plot_fourier_cross_validation(Ks, chi2r_train, chi2r_cv, best_K=3, target_id=1, n_train=32, n_cv=8)

        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(ax.get_xlabel(), r"$K$ (number of Fourier harmonics)")
        self.assertEqual(ax.get_ylabel(), r"$\chi_r^2$")
        self.assertTrue(any(len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [3, 3]) for line in ax.lines))
        self.assertTrue(any(len(line.get_ydata()) == 2 and np.allclose(line.get_ydata(), [0.88, 0.88]) for line in ax.lines))
        plt.close(ax.figure)

    def test_plot_fourier_normalized_residual_histograms_returns_two_axes(self):
        axes = plot_fourier_normalized_residual_histograms(
            train_norm_low=np.array([-1.6, -0.8, 0.2, 1.1]),
            cv_norm_low=np.array([-1.9, -0.6, 0.5, 1.7]),
            train_norm_best=np.array([-1.0, -0.2, 0.1, 0.9]),
            cv_norm_best=np.array([-1.1, -0.4, 0.2, 1.0]),
            low_K=1,
            best_K=3,
        )

        self.assertEqual(len(axes), 2)
        self.assertFalse(axes[0].get_shared_y_axes().joined(axes[0], axes[1]))
        self.assertEqual(axes[0].get_ylabel(), "Density")
        self.assertEqual(axes[1].get_ylabel(), "Density")
        self.assertEqual(axes[0].get_xlabel(), r"$(G - G_{\rm model})/\sigma$")
        self.assertEqual(axes[1].get_xlabel(), r"$(G - G_{\rm model})/\sigma$")
        self.assertEqual(len(axes[0].get_legend().texts), 3)
        plt.close(axes[0].figure)

    def test_plot_fourier_train_cv_phase_comparison_returns_two_axes(self):
        phase_grid = np.linspace(0.0, 1.0, 100, endpoint=False)
        model_best = 15.0 + 0.2 * np.cos(2.0 * np.pi * phase_grid)
        model_high = model_best + 0.04 * np.sin(10.0 * np.pi * phase_grid)

        axes = plot_fourier_train_cv_phase_comparison(
            train_phase=np.array([0.1, 0.3, 0.6, 0.8]),
            train_mags=np.array([15.1, 14.9, 15.2, 15.0]),
            cv_phase=np.array([0.2, 0.5, 0.7]),
            cv_mags=np.array([15.0, 15.15, 15.05]),
            phase_grid=phase_grid,
            model_mag_best=model_best,
            model_mag_high=model_high,
            best_K=3,
            high_K=5,
        )

        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].get_ylabel(), r"$G$ [mag]")
        self.assertEqual(axes[0].get_xlabel(), "Phase")
        self.assertEqual(axes[1].get_xlabel(), "Phase")
        plt.close(axes[0].figure)

    def test_plot_fourier_extrapolation_returns_axis(self):
        target = _fourier_lightcurve_table()
        fit = fourier_fit(target, period=0.5, k=2)
        epoch_grid = np.linspace(9.8, 22.6, 100)
        mag_grid = 15.05 + 0.1 * np.cos(2.0 * np.pi * (epoch_grid - 10.0) / 0.5)

        ax = plot_fourier_extrapolation(
            fit=fit,
            epoch_grid=epoch_grid,
            mag_grid=mag_grid,
            epoch_pred=20.6,
            mag_pred=15.02,
            time_label="Time [d]",
        )

        self.assertEqual(ax.get_xlabel(), "Time [d]")
        self.assertEqual(ax.get_ylabel(), r"$G$ [mag]")
        self.assertEqual(len(ax.containers), 1)
        plt.close(ax.figure)

    def test_plot_mean_g_catalog_comparison_returns_grid_of_axes(self):
        gaia_int_average_g = np.array([15.0, 15.2, 15.4], dtype=float)
        simple_mean_g = np.array([15.02, 15.19, 15.46], dtype=float)
        fourier_mean_g = np.array([15.01, 15.20, 15.41], dtype=float)
        resid_simple = simple_mean_g - gaia_int_average_g
        resid_fourier = fourier_mean_g - gaia_int_average_g

        axes = plot_mean_g_catalog_comparison(
            gaia_int_average_g=gaia_int_average_g,
            simple_mean_g=simple_mean_g,
            fourier_mean_g=fourier_mean_g,
            resid_simple=resid_simple,
            resid_fourier=resid_fourier,
            best_K=3,
        )

        self.assertEqual(np.asarray(axes).shape, (2, 2))
        self.assertEqual(axes[0, 0].get_ylabel(), r"Part (3) $\langle G \rangle$ [mag]")
        self.assertEqual(axes[0, 1].get_ylabel(), r"Fourier $\langle G \rangle$ [mag]")
        self.assertEqual(axes[1, 0].get_ylabel(), "Res.")
        self.assertEqual(axes[1, 1].get_ylabel(), "Res.")
        plt.close(axes[0, 0].figure)

    def test_plot_period_abs_mag_can_use_periodogram_columns(self):
        ax = plot_period_abs_mag(_period_abs_mag_table(), use_periodogram=True)

        self.assertIsNotNone(ax)
        scatter_collections = [
            c for c in ax.collections if hasattr(c, "get_offsets") and len(c.get_offsets()) == 1
        ]
        offsets = np.vstack([c.get_offsets() for c in scatter_collections[-2:]])
        np.testing.assert_allclose(np.sort(offsets[:, 0]), [0.81, 0.93])
        np.testing.assert_allclose(np.sort(offsets[:, 1]), [-0.2, 0.1])
        plt.close(ax.figure)

    def test_plot_period_mean_g_returns_axes(self):
        data = _period_mean_g_table()
        ax = plot_period_mean_g(data, np.asarray(data["best_classification"]).astype(str))

        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_xlabel(), r"$P$ [days]")
        self.assertEqual(ax.get_ylabel(), r"$\langle G \rangle$ [mag]")
        self.assertEqual(len(ax.get_legend().texts), 3)
        self.assertEqual(len(ax.containers), 1)
        rrab_color = mpl.colors.to_rgba("C0", alpha=0.55)
        rrd_color = mpl.colors.to_rgba("C1", alpha=0.55)
        scatter_collections = [
            c
            for c in ax.collections
            if hasattr(c, "get_offsets")
            and len(c.get_offsets()) == 1
            and len(c.get_facecolors()) == 1
        ]
        np.testing.assert_allclose(scatter_collections[0].get_facecolors()[0], rrab_color)
        np.testing.assert_allclose(scatter_collections[2].get_facecolors()[0], rrd_color)
        plt.close(ax.figure)

    def test_plot_vari_rrlyrae_period_comparison_returns_two_axes(self):
        data = _vari_rrlyrae_period_table()
        axes = plot_vari_rrlyrae_period_comparison(data, np.asarray(data["best_classification"]).astype(str))
        finite = np.isfinite(np.asarray(data["pf"], dtype=float)) & np.isfinite(np.asarray(data["best_period"], dtype=float))
        data_min = float(
            np.nanmin(
                np.concatenate(
                    [
                        np.asarray(data["pf"], dtype=float)[finite],
                        np.asarray(data["best_period"], dtype=float)[finite],
                    ]
                )
            )
        )
        data_max = float(
            np.nanmax(
                np.concatenate(
                    [
                        np.asarray(data["pf"], dtype=float)[finite],
                        np.asarray(data["best_period"], dtype=float)[finite],
                    ]
                )
            )
        )

        self.assertEqual(len(axes), 2)
        self.assertEqual(
            axes[0].get_xlabel(),
            r"\texttt{vari\_rrlyrae} fundamental period $P_{\rm F}$ [days]",
        )
        self.assertEqual(
            axes[1].get_xlabel(),
            r"\texttt{vari\_rrlyrae} first-overtone period $P_{1\rm O}$ [days]",
        )
        self.assertEqual(axes[0].get_ylabel(), r"L-S period $P_{\rm LS}$ [days]")
        self.assertEqual(axes[1].get_ylabel(), r"L-S period $P_{\rm LS}$ [days]")
        self.assertAlmostEqual(axes[0].get_xlim()[0], axes[0].get_ylim()[0])
        self.assertAlmostEqual(axes[0].get_xlim()[1], axes[0].get_ylim()[1])
        self.assertLess(axes[0].get_xlim()[0], data_min)
        self.assertGreater(axes[0].get_xlim()[1], data_max)
        self.assertEqual(axes[0].get_aspect(), 1.0)
        self.assertEqual(len(axes[0].containers), 2)
        self.assertEqual(len(axes[1].containers), 1)
        self.assertEqual(len(axes[0].get_legend().texts), 4)
        self.assertEqual(len(axes[1].get_legend().texts), 1)
        rrab_color = mpl.colors.to_rgba("C0", alpha=0.55)
        rrd_color = mpl.colors.to_rgba("C1", alpha=0.55)
        left_scatters = [
            c
            for c in axes[0].collections
            if hasattr(c, "get_offsets")
            and len(c.get_offsets()) == 1
            and len(c.get_facecolors()) == 1
        ]
        right_scatters = [
            c
            for c in axes[1].collections
            if hasattr(c, "get_offsets")
            and len(c.get_offsets()) == 1
            and len(c.get_facecolors()) == 1
        ]
        np.testing.assert_allclose(left_scatters[0].get_facecolors()[0], rrab_color)
        np.testing.assert_allclose(left_scatters[1].get_facecolors()[0], rrd_color)
        np.testing.assert_allclose(right_scatters[0].get_facecolors()[0], rrd_color)
        plt.close(axes[0].figure)

    def test_mcmc_plot_helpers_return_axes(self):
        sampler = _sampler_view()

        ax = plot_posterior(sampler, param_idx=0)
        axes = plot_trace(sampler)

        self.assertIsNotNone(ax)
        self.assertEqual(len(axes), 3)
        plt.close(ax.figure)
        plt.close(axes[0].figure)

    def test_plot_corner_returns_figure(self):
        sampler = _sampler_view()

        with self.writable_cache_env():
            fig = plot_corner(sampler)

        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_figure_helpers_access_named_figures(self):
        result = SimpleNamespace(figures={"b": object(), "a": object()})

        self.assertEqual(figure_names(result), ["a", "b"])
        self.assertIs(figure(result, "a"), result.figures["a"])


if __name__ == "__main__":
    unittest.main()
