import os
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from ugdatalab.plotting import (
    figure,
    figure_names,
    plot_corner,
    plot_hr,
    plot_inlier_prob_map,
    plot_lomb_scargle_periodogram,
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
    return Table(
        {
            "source_id": [1, 1, 1],
            "g_transit_time": [0.1, 0.4, 0.7],
            "g_transit_mag": [15.1, 15.4, 15.2],
            "best_classification": ["RRab", "RRab", "RRab"],
            "pf": [0.5, 0.5, 0.5],
            "p1_o": [np.nan, np.nan, np.nan],
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
            "best_classification": ["RRab", "RRc"],
            "best_period": [0.81, 0.93],
            "mean_apparent_g": [15.2, 15.6],
            "mean_apparent_g_err": [0.03, 0.04],
        }
    )


def _vari_rrlyrae_period_table():
    return Table(
        {
            "best_classification": ["RRab", "RRc", "RRd"],
            "pf": [0.55, np.nan, 0.74],
            "best_period": [0.57, 0.31, 0.55],
            "p1_o": [np.nan, np.nan, 0.55],
        }
    )


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

    def test_plot_inlier_prob_map_uses_all_data(self):
        source = SimpleNamespace(all_data=_gaia_quality_table())

        ax = plot_inlier_prob_map(source)

        self.assertIsNotNone(ax)
        plt.close(ax.figure)

    def test_plot_raw_phase_folded_lightcurve_returns_axes(self):
        axes = plot_raw_phase_folded_lightcurve(_lightcurve_table(), source_id=1)

        self.assertEqual(len(axes), 2)
        for ax in axes:
            self.assertIsNotNone(ax)
        self.assertTrue(axes[0].get_shared_y_axes().joined(axes[0], axes[1]))
        self.assertEqual(axes[1].get_ylabel(), "")
        self.assertFalse(any(label.get_visible() for label in axes[1].get_yticklabels()))
        plt.close(axes[0].figure)

    def test_plot_lomb_scargle_periodogram_marks_selected_period(self):
        periodogram = (
            np.array([0.2, 0.4, 0.6, 0.8], dtype=float),
            np.array([0.1, 0.4, 0.8, 0.3], dtype=float),
            0.6,
        )

        ax = plot_lomb_scargle_periodogram(_lightcurve_table(), periodogram=periodogram, period=0.6)

        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), r"$P$ [days]")
        self.assertTrue(
            any(
                len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [0.6, 0.6])
                for line in ax.lines
            )
        )
        plt.close(ax.figure)

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
        ax = plot_period_mean_g(_period_mean_g_table())

        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_xlabel(), r"$P$ [days]")
        self.assertEqual(ax.get_ylabel(), r"$\langle G \rangle$ [mag]")
        self.assertEqual(len(ax.get_legend().texts), 2)
        plt.close(ax.figure)

    def test_plot_vari_rrlyrae_period_comparison_returns_two_axes(self):
        axes = plot_vari_rrlyrae_period_comparison(_vari_rrlyrae_period_table())

        self.assertEqual(len(axes), 2)
        self.assertEqual(
            axes[0].get_xlabel(),
            r"Catalog fundamental period $P_{\rm F}$ [days]",
        )
        self.assertEqual(
            axes[1].get_xlabel(),
            r"Catalog first-overtone period $P_{1\rm O}$ [days]",
        )
        self.assertEqual(axes[0].get_ylabel(), r"L-S period $P_{\rm LS}$ [days]")
        self.assertEqual(axes[1].get_ylabel(), r"L-S period $P_{\rm LS}$ [days]")
        self.assertEqual(len(axes[0].get_legend().texts), 4)
        self.assertEqual(len(axes[1].get_legend().texts), 2)
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
