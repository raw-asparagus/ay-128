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
    plot_fourier_cv_normalized_residual_histograms,
    plot_fourier_cv_phase_comparison,
    plot_fourier_harmonic_fits,
    plot_rrlyrae_shape_comparison,
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
from ugdatalab.lightcurves import HarmonicCrossValidationResult, cross_validate_harmonics, fourier_fit


def _gaia_quality_table():
    return Table(
        {
            "source_id": [1, 2],
            "l": [120.0, 130.0],
            "b": [40.0, 45.0],
            "best_classification": ["RRab", "RRc"],
            "pf": [0.57, np.nan],
            "p1_o": [np.nan, 0.62],
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


def _dense_fourier_lightcurve_table():
    period = 0.5
    phase = np.linspace(0.0, 1.0, 120, endpoint=False)
    epoch = phase * (6.0 * period)
    mag = 15.2 + 0.25 * np.cos(2.0 * np.pi * epoch / period) - 0.1 * np.sin(4.0 * np.pi * epoch / period)
    flux = 10.0 ** (-0.4 * (mag - 25.6874))
    return Table(
        {
            "source_id": np.full(len(epoch), 1, dtype=int),
            "g_transit_time": epoch,
            "g_transit_mag": mag,
            "g_transit_mag_err": np.full(len(epoch), 0.03),
            "g_transit_flux": flux,
            "g_transit_flux_error": 0.03 * flux,
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
            "best_classification": ["RRab", "RRc"],
            "pf": [0.55, np.nan],
            "p1_o": [np.nan, 0.62],
            "period_ls": [0.81, 0.93],
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


def _shape_comparison_panels():
    phase_grid = np.linspace(0.0, 1.0, 100, endpoint=False)

    def make_panel(source_id, rr_class, period, best_K, phase_shift):
        phase_data = np.linspace(0.02, 0.98, 24)
        if rr_class == "RRc":
            mag_centered = 0.18 * np.cos(2.0 * np.pi * (phase_data - phase_shift))
            model_centered = 0.18 * np.cos(2.0 * np.pi * (phase_grid - phase_shift))
        else:
            mag_centered = (
                0.24 * np.cos(2.0 * np.pi * (phase_data - phase_shift))
                - 0.10 * np.sin(4.0 * np.pi * (phase_data - phase_shift))
            )
            model_centered = (
                0.24 * np.cos(2.0 * np.pi * (phase_grid - phase_shift))
                - 0.10 * np.sin(4.0 * np.pi * (phase_grid - phase_shift))
            )
        return {
            "source_id": source_id,
            "rr_class": rr_class,
            "phase_data": phase_data,
            "mag_centered": mag_centered,
            "phase_grid": phase_grid,
            "model_centered": model_centered,
            "best_K": best_K,
            "period": period,
            "n_epochs": len(phase_data),
        }

    return [
        make_panel(101, "RRc", 0.3124, 3, 0.00),
        make_panel(102, "RRc", 0.3281, 4, 0.03),
        make_panel(103, "RRc", 0.3417, 3, 0.05),
        make_panel(201, "RRab", 0.5571, 6, 0.00),
        make_panel(202, "RRab", 0.5884, 7, 0.04),
        make_panel(203, "RRab", 0.6128, 6, 0.06),
    ]


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
        lightcurve = _lightcurve_table()
        axes = plot_raw_phase_folded_lightcurve(lightcurve["source_id"][0], lightcurve)

        self.assertEqual(len(axes), 2)
        for ax in axes:
            self.assertIsNotNone(ax)
        self.assertFalse(axes[0].get_shared_y_axes().joined(axes[0], axes[1]))
        self.assertEqual(axes[0].get_xlabel(), "Time [days]")
        self.assertEqual(axes[0].get_ylabel(), "G [mag]")
        self.assertEqual(axes[1].get_ylabel(), "G [mag]")
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
        data = _fourier_lightcurve_table()
        ax = plot_lomb_scargle_periodogram(int(data["source_id"][0]), data)

        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), r"$P_{\rm LS}$ [days]")
        self.assertEqual(ax.get_ylim(), (0.0, 1.0))
        self.assertTrue(
            any(
                len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [0.5, 0.5])
                for line in ax.lines
            )
        )
        plt.close(ax.figure)

    def test_plot_fourier_harmonic_fits_returns_grid_of_axes(self):
        data = _fourier_lightcurve_table()
        axes = plot_fourier_harmonic_fits(int(data["source_id"][0]), data, [1, 3])
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
        result = HarmonicCrossValidationResult(
            source_id=1,
            period=0.5,
            Ks=Ks,
            chi2r_train=chi2r_train,
            chi2r_cv=chi2r_cv,
            best_K=3,
            train_idx=np.arange(32, dtype=int),
            cv_idx=np.arange(32, 40, dtype=int),
        )

        ax = plot_fourier_cross_validation(result)

        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(ax.get_xlabel(), r"$K$ (number of Fourier harmonics)")
        self.assertEqual(ax.get_ylabel(), r"$\chi_r^2$")
        self.assertTrue(any(len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [3, 3]) for line in ax.lines))
        self.assertTrue(any(len(line.get_ydata()) == 2 and np.allclose(line.get_ydata(), [0.88, 0.88]) for line in ax.lines))
        plt.close(ax.figure)

    def test_plot_fourier_cv_normalized_residual_histograms_returns_two_axes(self):
        data = _dense_fourier_lightcurve_table()
        result = cross_validate_harmonics(data)

        axes = plot_fourier_cv_normalized_residual_histograms(data, result)

        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].get_ylabel(), "Density")
        self.assertEqual(axes[1].get_ylabel(), "Density")
        self.assertEqual(axes[0].get_xlabel(), r"$(G - G_{\rm model})/\sigma$")
        self.assertEqual(axes[1].get_xlabel(), r"$(G - G_{\rm model})/\sigma$")
        plt.close(axes[0].figure)

    def test_plot_fourier_cv_phase_comparison_returns_two_axes(self):
        data = _dense_fourier_lightcurve_table()
        result = cross_validate_harmonics(data)

        axes = plot_fourier_cv_phase_comparison(data, result)

        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].get_xlabel(), "Phase")
        self.assertEqual(axes[1].get_xlabel(), "Phase")
        self.assertEqual(axes[0].get_ylabel(), r"$G$ [mag]")
        self.assertEqual(axes[1].get_ylabel(), r"$G$ [mag]")
        plt.close(axes[0].figure)

    def test_plot_rrlyrae_shape_comparison_returns_three_by_two_axes(self):
        axes = plot_rrlyrae_shape_comparison(_shape_comparison_panels())

        self.assertEqual(np.asarray(axes).shape, (3, 2))
        self.assertEqual(axes[0, 0].get_xlim(), (0.0, 1.0))
        self.assertEqual(axes[0, 1].get_xlim(), (0.0, 1.0))
        self.assertEqual(axes[0, 0].get_ylabel(), r"$G - \langle G \rangle_{\rm Fourier}$")
        self.assertEqual(axes[0, 1].get_ylabel(), r"$G - \langle G \rangle_{\rm Fourier}$")
        for row in range(3):
            self.assertEqual(axes[row, 0].get_ylim(), axes[row, 1].get_ylim())
        left_points = axes[0, 0].collections[0]
        right_points = axes[0, 1].collections[0]
        left_line = axes[0, 0].lines[0]
        right_line = axes[0, 1].lines[0]
        np.testing.assert_allclose(left_points.get_facecolors()[0], mpl.colors.to_rgba("C1", alpha=0.55))
        np.testing.assert_allclose(right_points.get_facecolors()[0], mpl.colors.to_rgba("C0", alpha=0.55))
        np.testing.assert_allclose(mpl.colors.to_rgba(left_line.get_color()), mpl.colors.to_rgba("C7"))
        np.testing.assert_allclose(mpl.colors.to_rgba(right_line.get_color()), mpl.colors.to_rgba("C7"))
        plt.close(axes[0, 0].figure)

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
        self.assertEqual(axes[1].get_ylabel(), r"$G$ [mag]")
        self.assertFalse(axes[0].get_shared_y_axes().joined(axes[0], axes[1]))
        self.assertEqual(axes[0].get_xlabel(), "Phase")
        self.assertEqual(axes[1].get_xlabel(), "Phase")
        self.assertAlmostEqual(axes[0].collections[0].get_alpha(), 0.5)
        self.assertAlmostEqual(axes[0].collections[1].get_alpha(), 0.6)
        cv_color = mpl.colors.to_rgba("C1", alpha=0.6)
        model_color = mpl.colors.to_rgba("C7")
        np.testing.assert_allclose(axes[0].collections[1].get_facecolors()[0], cv_color)
        np.testing.assert_allclose(mpl.colors.to_rgba(axes[0].lines[0].get_color()), model_color)
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
        self.assertEqual(axes[0, 0].get_ylabel(), r"$\langle G \rangle_{\rm epoch}$ [mag]")
        self.assertEqual(axes[0, 1].get_ylabel(), r"$\langle G \rangle_{\rm Fourier}$ [mag]")
        self.assertEqual(axes[1, 0].get_xlabel(), r"Gaia $\mathtt{int\_average\_g}$ [mag]")
        self.assertEqual(axes[1, 1].get_xlabel(), r"Gaia $\mathtt{int\_average\_g}$ [mag]")
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
        ax = plot_period_mean_g(data)

        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_xlabel(), r"L-S period $P_{\rm LS}$ [days]")
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
        axes = plot_vari_rrlyrae_period_comparison(data)
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
