import os
import tempfile
import unittest
from contextlib import contextmanager
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from ugdatalab.plotting import (
    A4_USABLE_HEIGHT_IN,
    A4_USABLE_WIDTH_IN,
    ALPHA_DENSE,
    ALPHA_DIM,
    ALPHA_EXTRA_LIGHT,
    ALPHA_SHADE,
    ALPHA_MUTED,
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
    plot_mollweide_period_abs_mag_overview,
    plot_mollweide_diff,
    plot_period_abs_mag,
    plot_period_abs_mag_ls,
    plot_period_abs_mag_comparison,
    plot_period_mean_g,
    plot_vari_rrlyrae_period_comparison,
    plot_period_luminosity_diff,
    plot_inlier_prob_period_luminosity_comparison,
    plot_optical_pl_literature_comparison,
    plot_pl_posterior_predictive,
    plot_pl_posterior_predictive_comparison,
    plot_pl_sampler_comparison_corner,
    plot_pc_posterior_predictive,
    plot_pc_posterior_predictive_comparison,
    plot_posterior,
    plot_raw_phase_folded_lightcurve,
    plot_trace,
)
from ugdatalab.lightcurves import HarmonicCrossValidationResult, cross_validate_harmonics, fourier_fit
from ugdatalab.models.gaia import rrlyrae_representative_period


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
    chi2r_train = np.array([60.0, 30.0, 20.0, 15.0, 12.0], dtype=float)
    chi2r_cv = np.array([55.0, 25.0, 10.0, 18.0, 30.0], dtype=float)
    return Ks, chi2r_train, chi2r_cv


def _shape_comparison_source(rr_class, source_ids, periods, phase_shifts):
    data = Table(
        {
            "source_id": source_ids,
            "best_classification": [rr_class] * len(source_ids),
            "pf": periods if rr_class == "RRab" else [np.nan] * len(source_ids),
            "pf_error": [0.01] * len(source_ids) if rr_class == "RRab" else [np.nan] * len(source_ids),
            "p1_o": periods if rr_class == "RRc" else [np.nan] * len(source_ids),
            "p1_o_error": [0.005] * len(source_ids) if rr_class == "RRc" else [np.nan] * len(source_ids),
        }
    )

    lightcurve_rows = []
    for source_id, period, phase_shift in zip(source_ids, periods, phase_shifts):
        epoch = np.linspace(0.0, 20.0 * period, 72, endpoint=False)
        phase = (epoch % period) / period
        if rr_class == "RRc":
            mag = 15.2 + 0.18 * np.cos(2.0 * np.pi * (phase - phase_shift))
        else:
            mag = (
                15.1
                + 0.24 * np.cos(2.0 * np.pi * (phase - phase_shift))
                - 0.10 * np.sin(4.0 * np.pi * (phase - phase_shift))
            )
        flux = 10.0 ** (-0.4 * (mag - 25.6874))
        flux_err = 0.02 * flux
        for t, m, f, ferr in zip(epoch, mag, flux, flux_err):
            lightcurve_rows.append((source_id, t, m, f, ferr))

    lightcurves = Table(
        rows=lightcurve_rows,
        names=("source_id", "g_transit_time", "g_transit_mag", "g_transit_flux", "g_transit_flux_error"),
    )
    lightcurves["best_classification"] = [rr_class] * len(lightcurves)
    return SimpleNamespace(data=data, lightcurves=lightcurves)


def _sampler_view():
    return SimpleNamespace(
        samples=np.array([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2], [0.3, 1.3]], dtype=float),
        log_probs=np.array([-2.0, -1.5, -1.2, -1.0], dtype=float),
        n_burn=1,
        param_labels=[r"$a$", r"$b$"],
    )


def _pl_context(class_label: str):
    x_centered = np.linspace(-0.25, 0.25, 8)
    slope = -2.2 if class_label == "RRab" else -2.0
    intercept = 0.62 if class_label == "RRab" else 0.56
    sigma = np.full(len(x_centered), 0.05)
    sigma_logp = np.full(len(x_centered), 0.01)
    y = slope * x_centered + intercept
    return SimpleNamespace(
        class_label=class_label,
        x_centered=x_centered,
        y=y,
        sigma=sigma,
        sigma_logp=sigma_logp,
    )


def _pl_samples(slope: float, intercept: float, scatter: float):
    offsets = np.linspace(-0.03, 0.03, 60)
    return np.column_stack(
        [
            slope + offsets,
            intercept + 0.5 * offsets,
            scatter + 0.2 * offsets,
        ]
    )


def _optical_pl_comparison(rr_class: str, slope: float, intercept: float):
    x_obs = np.linspace(-0.35, 0.05, 8)
    y_obs = slope * x_obs + intercept
    sigma_obs = np.full(len(x_obs), 0.08)
    x_grid = np.linspace(float(np.min(x_obs)), float(np.max(x_obs)), 120)
    median = slope * x_grid + intercept
    return SimpleNamespace(
        rr_class=rr_class,
        x_obs=x_obs,
        y_obs=y_obs,
        sigma_obs=sigma_obs,
        x_grid=x_grid,
        median_mean=median,
        predictive_q16=median - 0.12,
        predictive_q84=median + 0.12,
    )


def _pc_context(class_label: str):
    x_centered = np.linspace(-0.20, 0.20, 8)
    slope = 0.18 if class_label == "RRab" else 0.11
    intercept = 0.72 if class_label == "RRab" else 0.42
    sigma = np.full(len(x_centered), 0.02)
    sigma_logp = np.full(len(x_centered), 0.01)
    y = slope * x_centered + intercept
    return SimpleNamespace(
        class_label=class_label,
        x_centered=x_centered,
        y=y,
        sigma=sigma,
        sigma_logp=sigma_logp,
        y_label=r"$G_\mathrm{BP} - G_\mathrm{RP}$ [mag]",
    )


def _pc_samples(slope: float, intercept: float, scatter: float):
    offsets = np.linspace(-0.02, 0.02, 60)
    return np.column_stack(
        [
            slope + offsets,
            intercept + 0.3 * offsets,
            scatter + 0.1 * offsets,
        ]
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
        periods = rrlyrae_representative_period(data)
        classifications = np.asarray(data["best_classification"], dtype=str)

        ax1 = plot_mollweide(
            np.asarray(data["l"], dtype=float),
            np.asarray(data["b"], dtype=float),
        )
        ax2 = plot_mollweide_diff(data, subset)
        ax3 = plot_period_abs_mag(
            periods,
            np.asarray(data["M_G"], dtype=float),
            np.asarray(data["sigma_M"], dtype=float),
            classifications,
        )
        ax4 = plot_period_luminosity_diff(data, subset)
        ax5 = plot_hr(data)

        self.assertEqual(ax3.get_xlabel(), r"Catalog period $P$ [days]")
        self.assertTrue(ax3.yaxis_inverted())
        self.assertTrue(ax4.yaxis_inverted())
        self.assertTrue(ax5.yaxis_inverted())
        for ax in (ax1, ax2, ax3, ax4, ax5):
            self.assertIsNotNone(ax)
            plt.close(ax.figure)

    def test_plot_period_abs_mag_comparison_returns_two_axes(self):
        data = _period_abs_mag_table()
        left = data[:1]
        right = data[1:]

        axes = plot_period_abs_mag_comparison(
            rrlyrae_representative_period(left),
            np.asarray(left["M_G"], dtype=float),
            np.asarray(left["sigma_M"], dtype=float),
            np.asarray(left["best_classification"], dtype=str),
            rrlyrae_representative_period(right),
            np.asarray(right["M_G"], dtype=float),
            np.asarray(right["sigma_M"], dtype=float),
            np.asarray(right["best_classification"], dtype=str),
        )

        self.assertEqual(len(axes), 2)
        self.assertTrue(axes[0].get_shared_x_axes().joined(axes[0], axes[1]))
        self.assertTrue(axes[0].get_shared_y_axes().joined(axes[0], axes[1]))
        self.assertEqual(axes[0].get_xlabel(), r"Catalog period $P$ [days]")
        self.assertEqual(axes[1].get_xlabel(), r"Catalog period $P$ [days]")
        plt.close(axes[0].figure)

    def test_rcparams_use_visual_weight_tokens(self):
        self.assertEqual(mpl.rcParams["grid.linewidth"], LW_GRID)

    def test_plot_inlier_prob_map_uses_all_data(self):
        source = SimpleNamespace(all_data=_gaia_quality_table())

        ax = plot_inlier_prob_map(source)

        self.assertIsNotNone(ax)
        self.assertTrue(ax.yaxis_inverted())
        plt.close(ax.figure)

    def test_plot_inlier_prob_period_luminosity_comparison_returns_two_axes(self):
        data = _gaia_quality_table()
        subset = data[:1]
        prob_source = SimpleNamespace(all_data=data)

        axes = plot_inlier_prob_period_luminosity_comparison(
            prob_source,
            data,
            subset,
        )

        self.assertEqual(len(axes), 2)
        self.assertTrue(axes[0].yaxis_inverted())
        self.assertTrue(axes[1].yaxis_inverted())
        self.assertGreaterEqual(len(axes[0].figure.axes), 3)
        plt.close(axes[0].figure)

    def test_plot_mollweide_period_abs_mag_overview_returns_two_axes(self):
        data = _gaia_quality_table()
        axes = plot_mollweide_period_abs_mag_overview(
            np.asarray(data["l"], dtype=float),
            np.asarray(data["b"], dtype=float),
            rrlyrae_representative_period(data),
            np.asarray(data["M_G"], dtype=float),
            np.asarray(data["sigma_M"], dtype=float),
            np.asarray(data["best_classification"], dtype=str),
        )

        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[1].get_xlabel(), r"Catalog period $P$ [days]")
        plt.close(axes[0].figure)

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
        self.assertEqual(axes[0].get_title(loc="left"), "")
        self.assertEqual(axes[1].get_title(loc="left"), "")
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
        self.assertEqual(ax.get_legend().texts[0].get_text(), r"$P=0.5000$ d")
        plt.close(ax.figure)

    def test_plot_fourier_harmonic_fits_returns_grid_of_axes(self):
        data = _fourier_lightcurve_table()
        axes = plot_fourier_harmonic_fits(data, [1, 3])
        within_gap = axes[0, 0].get_position().y0 - axes[0, 1].get_position().y1
        between_gap = axes[0, 1].get_position().y0 - axes[1, 0].get_position().y1

        self.assertEqual(np.asarray(axes).shape, (2, 2))
        np.testing.assert_allclose(
            axes[0, 0].figure.get_size_inches(),
            np.array([A4_USABLE_WIDTH_IN, A4_USABLE_HEIGHT_IN]),
            atol=1e-2,
        )
        self.assertEqual(axes[0, 0].get_ylabel(), r"$G$ [mag]")
        self.assertEqual(axes[0, 1].get_ylabel(), "Res.")
        self.assertEqual(axes[0, 0].get_subplotspec().colspan.start, 0)
        self.assertEqual(axes[0, 1].get_subplotspec().colspan.start, 0)
        self.assertEqual(axes[0, 0].get_subplotspec().rowspan.stop - axes[0, 0].get_subplotspec().rowspan.start, 1)
        self.assertEqual(axes[0, 1].get_subplotspec().rowspan.stop - axes[0, 1].get_subplotspec().rowspan.start, 1)
        self.assertAlmostEqual(axes[0, 0].get_position().x0, axes[0, 1].get_position().x0, places=3)
        self.assertAlmostEqual(axes[0, 0].get_position().x1, axes[0, 1].get_position().x1, places=3)
        self.assertGreater(axes[0, 0].get_position().height, 2.0 * axes[0, 1].get_position().height)
        self.assertLess(within_gap, 1e-6)
        self.assertGreater(between_gap, 1e-2)
        self.assertEqual(axes[0, 1].get_xlabel(), "")
        self.assertEqual(axes[1, 1].get_xlabel(), "Phase")
        self.assertTrue(axes[0, 0].spines["bottom"].get_visible())
        self.assertTrue(axes[0, 1].spines["bottom"].get_visible())
        self.assertTrue(axes[1, 0].spines["bottom"].get_visible())
        self.assertTrue(axes[1, 1].spines["bottom"].get_visible())
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
            classification="RRab",
        )

        ax = plot_fourier_cross_validation(result)
        ax.figure.canvas.draw()

        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(ax.get_xlabel(), r"$K$ (number of Fourier harmonics)")
        self.assertEqual(ax.get_ylabel(), r"$\chi_r^2$")
        self.assertTrue(any(len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [3, 3]) for line in ax.lines))
        self.assertTrue(any(len(line.get_ydata()) == 2 and np.allclose(line.get_ydata(), [10.0, 10.0]) for line in ax.lines))
        ytick_text = [
            tick.get_text()
            for tick in [*ax.get_yticklabels(minor=False), *ax.get_yticklabels(minor=True)]
            if tick.get_text()
        ]
        for label in ["10", "20", "30", "40", "60"]:
            self.assertIn(label, ytick_text)
        self.assertTrue(all("^" not in text and "$" not in text for text in ytick_text))
        plt.close(ax.figure)

    def test_plot_fourier_cv_normalized_residual_histograms_returns_two_axes(self):
        data = _dense_fourier_lightcurve_table()
        result = cross_validate_harmonics(data)
        train_lightcurve = data[result.train_idx]
        cv_lightcurve = data[result.cv_idx]
        low_fit = fourier_fit(train_lightcurve, period=float(result.period), k=int(result.Ks[0]))
        best_fit = fourier_fit(train_lightcurve, period=float(result.period), k=int(result.best_K))

        axes = plot_fourier_cv_normalized_residual_histograms(train_lightcurve, cv_lightcurve, low_fit, best_fit)

        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].get_ylabel(), "Density")
        self.assertEqual(axes[1].get_ylabel(), "Density")
        self.assertEqual(axes[0].get_xlabel(), r"$(G - G_{\rm model})/\sigma$")
        self.assertEqual(axes[1].get_xlabel(), r"$(G - G_{\rm model})/\sigma$")
        self.assertIn(r"Low $K = 1$", axes[0].get_title(loc="left"))
        self.assertIn(rf"Best $K = {int(result.best_K)}$", axes[1].get_title(loc="left"))
        plt.close(axes[0].figure)

    def test_plot_fourier_cv_phase_comparison_returns_two_axes(self):
        data = _dense_fourier_lightcurve_table()
        result = cross_validate_harmonics(data)
        train_lightcurve = data[result.train_idx]
        cv_lightcurve = data[result.cv_idx]
        best_fit = fourier_fit(train_lightcurve, period=float(result.period), k=int(result.best_K))
        high_fit = fourier_fit(train_lightcurve, period=float(result.period), k=int(result.Ks[-1]))

        axes = plot_fourier_cv_phase_comparison(train_lightcurve, cv_lightcurve, best_fit, high_fit)

        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].get_xlabel(), "Phase")
        self.assertEqual(axes[1].get_xlabel(), "Phase")
        self.assertEqual(axes[0].get_ylabel(), r"$G$ [mag]")
        self.assertEqual(axes[1].get_ylabel(), r"$G$ [mag]")
        self.assertIn(rf"Best $K = {int(result.best_K)}$", axes[0].get_title(loc="left"))
        self.assertIn(rf"High $K = {int(result.Ks[-1])}$", axes[1].get_title(loc="left"))
        plt.close(axes[0].figure)

    def test_plot_rrlyrae_shape_comparison_returns_three_by_two_axes(self):
        rrab = _shape_comparison_source("RRab", [201, 202, 203], [0.5571, 0.5884, 0.6128], [0.00, 0.04, 0.06])
        rrc = _shape_comparison_source("RRc", [101, 102, 103], [0.3124, 0.3281, 0.3417], [0.00, 0.03, 0.05])
        axes = plot_rrlyrae_shape_comparison(rrab, rrc)

        self.assertEqual(np.asarray(axes).shape, (6, 2))
        self.assertEqual(axes[0, 0].get_xlim(), (0.0, 1.0))
        self.assertEqual(axes[0, 1].get_xlim(), (0.0, 1.0))
        self.assertEqual(axes[0, 0].get_ylabel(), r"$G - \langle G \rangle_{\rm Fourier}$")
        self.assertEqual(axes[0, 1].get_ylabel(), r"$G - \langle G \rangle_{\rm Fourier}$")
        self.assertEqual(axes[1, 0].get_ylabel(), "Res.")
        self.assertEqual(axes[1, 1].get_ylabel(), "Res.")
        left_title = axes[0, 0].get_title(loc="left")
        right_title = axes[0, 1].get_title(loc="left")
        self.assertIn("201", left_title)
        self.assertIn("101", right_title)
        self.assertIn("RRab", left_title)
        self.assertIn("RRc", right_title)
        self.assertIn(r"$K=", left_title)
        self.assertIn(r"\chi_\nu^2", left_title)
        self.assertNotIn(r"\mathrm{RMS}_{\rm norm}", right_title)
        for row in range(3):
            self.assertEqual(len(axes[2 * row, 0].containers), 1)
            self.assertEqual(len(axes[2 * row, 1].containers), 1)
            self.assertEqual(len(axes[2 * row + 1, 0].containers), 1)
            self.assertEqual(len(axes[2 * row + 1, 1].containers), 1)
            self.assertEqual(axes[2 * row + 1, 0].get_xlabel(), "Phase")
            self.assertEqual(axes[2 * row + 1, 1].get_xlabel(), "Phase")
        left_points = axes[0, 0].containers[0].lines[0]
        right_points = axes[0, 1].containers[0].lines[0]
        left_line = axes[0, 0].lines[-1]
        right_line = axes[0, 1].lines[-1]
        np.testing.assert_allclose(
            mpl.colors.to_rgba(left_points.get_color(), alpha=left_points.get_alpha()),
            mpl.colors.to_rgba("C0", alpha=ALPHA_MUTED),
        )
        np.testing.assert_allclose(
            mpl.colors.to_rgba(right_points.get_color(), alpha=right_points.get_alpha()),
            mpl.colors.to_rgba("C1", alpha=ALPHA_MUTED),
        )
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
        self.assertGreater(len(axes[0].patches), 0)
        self.assertAlmostEqual(axes[0].patches[0].get_alpha(), ALPHA_DIM)
        train_handle, cv_handle, gaussian_handle = axes[0].get_legend().legend_handles
        self.assertAlmostEqual(train_handle.get_alpha(), ALPHA_DENSE)
        self.assertAlmostEqual(cv_handle.get_alpha(), ALPHA_DIM)
        self.assertEqual(gaussian_handle.get_linestyle(), "--")
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
        self.assertIn(r"Best $K = 3$", axes[0].get_title(loc="left"))
        self.assertIn(r"High $K = 5$", axes[1].get_title(loc="left"))
        self.assertAlmostEqual(axes[0].collections[0].get_alpha(), ALPHA_EXTRA_LIGHT)
        self.assertAlmostEqual(axes[0].collections[1].get_alpha(), ALPHA_MUTED)
        cv_color = mpl.colors.to_rgba("C1", alpha=ALPHA_MUTED)
        model_color = mpl.colors.to_rgba("C7")
        np.testing.assert_allclose(axes[0].collections[1].get_facecolors()[0], cv_color)
        np.testing.assert_allclose(mpl.colors.to_rgba(axes[0].lines[0].get_color()), model_color)
        plt.close(axes[0].figure)

    def test_plot_fourier_extrapolation_returns_axis(self):
        target = _fourier_lightcurve_table()
        result = cross_validate_harmonics(target)
        fit = fourier_fit(target, period=float(result.period), k=int(result.best_K))

        ax = plot_fourier_extrapolation(result, fit)

        self.assertEqual(ax.get_xlabel(), "Time [days]")
        self.assertEqual(ax.get_ylabel(), r"$G$ [mag]")
        self.assertEqual(len(ax.containers), 1)
        self.assertTrue(any(np.isclose(collection.get_alpha() or 0.0, ALPHA_SHADE) for collection in ax.collections))
        plt.close(ax.figure)

    def test_plot_mean_g_catalog_comparison_returns_grid_of_axes(self):
        summary = Table(
            {
                "source_id": [1],
                "mean_apparent_g": [15.2],
                "mean_apparent_g_err": [0.03],
                "fourier_mean_apparent_g": [15.18],
                "fourier_mean_apparent_g_err": [0.02],
                "int_average_g": [15.16],
                "int_average_g_error": [0.01],
            }
        )
        axes = plot_mean_g_catalog_comparison(summary)

        self.assertEqual(np.asarray(axes).shape, (2, 2))
        self.assertEqual(axes[0, 0].get_ylabel(), r"$\langle G \rangle_{\rm epoch}$ [mag]")
        self.assertEqual(axes[0, 1].get_ylabel(), r"$\langle G \rangle_{\rm Fourier}$ [mag]")
        self.assertEqual(axes[1, 0].get_xlabel(), r"Gaia $\mathtt{int\_average\_g}$ [mag]")
        self.assertEqual(axes[1, 1].get_xlabel(), r"Gaia $\mathtt{int\_average\_g}$ [mag]")
        self.assertEqual(axes[1, 0].get_ylabel(), "Res.")
        self.assertEqual(axes[1, 1].get_ylabel(), "Res.")
        self.assertTrue(axes[0, 0].xaxis_inverted())
        self.assertTrue(axes[0, 0].yaxis_inverted())
        self.assertTrue(axes[0, 1].xaxis_inverted())
        self.assertTrue(axes[0, 1].yaxis_inverted())
        self.assertTrue(axes[1, 0].xaxis_inverted())
        self.assertTrue(axes[1, 1].xaxis_inverted())
        self.assertEqual(len(axes[0, 0].containers), 1)
        self.assertEqual(len(axes[1, 0].containers), 1)
        self.assertEqual(len(axes[0, 1].containers), 1)
        self.assertEqual(len(axes[1, 1].containers), 1)
        plt.close(axes[0, 0].figure)

    def test_plot_pl_posterior_predictive_returns_axis(self):
        ctx = _pl_context("RRab")
        samples = _pl_samples(-2.2, 0.62, 0.18)

        ax = plot_pl_posterior_predictive(ctx, samples)

        self.assertEqual(
            ax.get_xlabel(),
            r"$\log_{10}(P/\mathrm{day}) - \langle \log_{10}P \rangle_{\rm class}$",
        )
        self.assertEqual(ax.get_ylabel(), r"$M_{\rm G}$ [mag]")
        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertIn("RRab data", legend_texts)
        self.assertIn("95% predictive envelope", legend_texts)
        self.assertIn("68% predictive envelope", legend_texts)
        self.assertIn("Posterior median", legend_texts)
        plt.close(ax.figure)

    def test_plot_pl_posterior_predictive_comparison_returns_axis(self):
        rrab_ctx = _pl_context("RRab")
        rrc_ctx = _pl_context("RRc")
        rrab_samples = _pl_samples(-2.2, 0.62, 0.18)
        rrc_samples = _pl_samples(-2.0, 0.56, 0.17)

        ax = plot_pl_posterior_predictive_comparison(
            rrab_ctx,
            rrab_samples,
            rrc_ctx,
            rrc_samples,
        )

        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertIn("RRab data", legend_texts)
        self.assertIn("RRab posterior median", legend_texts)
        self.assertIn("RRc data", legend_texts)
        self.assertIn("RRc posterior median", legend_texts)
        self.assertEqual(ax.get_ylabel(), r"$M_{\rm G}$ [mag]")
        plt.close(ax.figure)

    def test_plot_pc_posterior_predictive_returns_axis(self):
        ctx = _pc_context("RRab")
        samples = _pc_samples(0.18, 0.72, 0.05)

        ax = plot_pc_posterior_predictive(ctx, samples)

        self.assertEqual(
            ax.get_xlabel(),
            r"$\log_{10}(P/\mathrm{day}) - \langle \log_{10}P \rangle_{\rm class}$",
        )
        self.assertEqual(ax.get_ylabel(), r"$G_\mathrm{BP} - G_\mathrm{RP}$ [mag]")
        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertIn("RRab data", legend_texts)
        self.assertIn("95% predictive envelope", legend_texts)
        self.assertIn("68% predictive envelope", legend_texts)
        self.assertIn("Posterior median", legend_texts)
        plt.close(ax.figure)

    def test_plot_pc_posterior_predictive_comparison_returns_axis(self):
        rrab_ctx = _pc_context("RRab")
        rrc_ctx = _pc_context("RRc")
        rrab_samples = _pc_samples(0.18, 0.72, 0.05)
        rrc_samples = _pc_samples(0.11, 0.42, 0.04)

        ax = plot_pc_posterior_predictive_comparison(
            rrab_ctx,
            rrab_samples,
            rrc_ctx,
            rrc_samples,
        )

        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertIn("RRab data", legend_texts)
        self.assertIn("RRab posterior median", legend_texts)
        self.assertIn("RRc data", legend_texts)
        self.assertIn("RRc posterior median", legend_texts)
        self.assertEqual(ax.get_ylabel(), r"$G_\mathrm{BP} - G_\mathrm{RP}$ [mag]")
        plt.close(ax.figure)

    def test_plot_optical_pl_literature_comparison_returns_two_panel_figure(self):
        rrab = _optical_pl_comparison("RRab", -2.2, 0.62)
        rrc = _optical_pl_comparison("RRc", -2.0, 0.55)

        fig = plot_optical_pl_literature_comparison(rrab, rrc)

        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(fig.axes[0].get_xlabel(), r"$\log_{10}(P/\mathrm{day})$")
        self.assertEqual(fig.axes[0].get_ylabel(), r"Absolute magnitude [mag]")
        self.assertEqual(fig.axes[1].get_xlabel(), r"$\log_{10}(P/\mathrm{day})$")
        self.assertTrue(fig.axes[0].yaxis_inverted())
        self.assertTrue(fig.axes[1].yaxis_inverted())
        self.assertIn("RRab", fig.axes[0].get_title())
        self.assertIn("RRc", fig.axes[1].get_title())
        left_text = "\n".join(text.get_text() for text in fig.axes[0].texts)
        self.assertIn("Klein+Bloom 2014", left_text)
        self.assertIn("Beaton+2018", left_text)
        legend_texts = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        self.assertIn("RRab Gaia data", legend_texts)
        self.assertIn(r"This work: median $M_{\rm G}$ fit", legend_texts)
        self.assertIn("This work: 68% predictive band", legend_texts)
        plt.close(fig)

    def test_plot_pl_sampler_comparison_corner_returns_figure(self):
        sample_map = {
            "Metropolis-Hastings": _pl_samples(-2.2, 0.62, 0.18),
            "NUTS + Potential": _pl_samples(-2.18, 0.61, 0.17),
            "Native PyMC NUTS": _pl_samples(-2.21, 0.63, 0.16),
        }

        with self.writable_cache_env():
            fig = plot_pl_sampler_comparison_corner(sample_map)

        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.legends), 1)
        legend_texts = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertIn("Metropolis-Hastings", legend_texts)
        self.assertIn("NUTS + Potential", legend_texts)
        self.assertIn("Native PyMC NUTS", legend_texts)
        plt.close(fig)

    def test_plot_save_wrapper_writes_requested_pdf(self):
        data = _period_abs_mag_table()
        target_dir = Path(tempfile.mkdtemp())

        with patch("ugdatalab.plotting.FIGURES_DIR", target_dir), patch(
            "ugdatalab.plotting.ensure_output_dirs",
            lambda: target_dir.mkdir(parents=True, exist_ok=True),
        ):
            ax = plot_period_abs_mag(
                np.asarray(data["period_ls"], dtype=float),
                np.asarray(data["M_G_ls"], dtype=float),
                np.asarray(data["sigma_M_ls"], dtype=float),
                np.asarray(data["best_classification"], dtype=str),
                save_name="test-period-abs-mag.pdf",
            )

        self.assertTrue((target_dir / "test-period-abs-mag.pdf").exists())
        plt.close(ax.figure)

    def test_plot_period_abs_mag_ls_uses_ls_label(self):
        data = _period_abs_mag_table()
        ax = plot_period_abs_mag_ls(
            np.asarray(data["period_ls"], dtype=float),
            np.asarray(data["M_G_ls"], dtype=float),
            np.asarray(data["sigma_M_ls"], dtype=float),
            np.asarray(data["best_classification"], dtype=str),
        )

        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), r"L-S period $P_{\rm LS}$ [days]")
        self.assertTrue(ax.yaxis_inverted())
        scatter_collections = [
            c for c in ax.collections if isinstance(c, mpl.collections.PathCollection)
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
        self.assertEqual(ax.get_ylabel(), r"$\langle G \rangle_{\rm epoch}$ [mag]")
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

        ax = plot_posterior(
            sampler.samples,
            sampler.param_labels,
            sampler.n_burn,
            param_idx=0,
        )
        axes = plot_trace(
            sampler.samples,
            sampler.log_probs,
            sampler.param_labels,
            sampler.n_burn,
        )

        self.assertIsNotNone(ax)
        self.assertEqual(len(axes), 3)
        np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), np.array([0, 1, 2]))
        np.testing.assert_array_equal(axes[-1].lines[0].get_xdata(), np.array([0, 1, 2]))
        plt.close(ax.figure)
        plt.close(axes[0].figure)

    def test_plot_posterior_handles_math_label_in_analytic_legend(self):
        sampler = _sampler_view()

        ax = plot_posterior(
            sampler.samples,
            [r"$\mu$", r"$b$"],
            sampler.n_burn,
            param_idx=0,
            pdf_fn=lambda x: np.exp(-0.5 * x**2),
        )
        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]

        self.assertIn(r"Analytic $p(\mu\mid x)$", legend_texts)
        ax.figure.canvas.draw()
        plt.close(ax.figure)

    def test_plot_corner_returns_figure(self):
        sampler = _sampler_view()

        with self.writable_cache_env():
            fig = plot_corner(sampler)

        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_public_plot_api_strips_presentation_overrides(self):
        forbidden = {"title", "x_label", "show_panel_titles", "best_panel_title", "high_panel_title", "fig"}
        signatures = {
            "plot_mollweide": plot_mollweide,
            "plot_mollweide_diff": plot_mollweide_diff,
            "plot_period_abs_mag": plot_period_abs_mag,
            "plot_period_abs_mag_ls": plot_period_abs_mag_ls,
            "plot_period_abs_mag_comparison": plot_period_abs_mag_comparison,
            "plot_period_luminosity_diff": plot_period_luminosity_diff,
            "plot_inlier_prob_period_luminosity_comparison": plot_inlier_prob_period_luminosity_comparison,
            "plot_hr": plot_hr,
            "plot_inlier_prob_map": plot_inlier_prob_map,
            "plot_pl_posterior_predictive": plot_pl_posterior_predictive,
            "plot_pl_posterior_predictive_comparison": plot_pl_posterior_predictive_comparison,
            "plot_pl_sampler_comparison_corner": plot_pl_sampler_comparison_corner,
            "plot_pc_posterior_predictive": plot_pc_posterior_predictive,
            "plot_pc_posterior_predictive_comparison": plot_pc_posterior_predictive_comparison,
            "plot_posterior": plot_posterior,
            "plot_trace": plot_trace,
            "plot_corner": plot_corner,
        }
        allowed_ax = {
            "plot_mollweide_diff",
            "plot_period_abs_mag",
            "plot_period_abs_mag_ls",
            "plot_period_luminosity_diff",
            "plot_hr",
            "plot_inlier_prob_map",
        }

        for name, fn in signatures.items():
            params = inspect.signature(fn).parameters
            self.assertTrue(forbidden.isdisjoint(params), name)
            if name == "plot_mollweide":
                self.assertNotIn("ax", params)
            elif name in allowed_ax:
                self.assertIn("ax", params)

    def test_figure_helpers_access_named_figures(self):
        result = SimpleNamespace(figures={"b": object(), "a": object()})

        self.assertEqual(figure_names(result), ["a", "b"])
        self.assertIs(figure(result, "a"), result.figures["a"])


if __name__ == "__main__":
    unittest.main()
