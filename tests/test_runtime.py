import importlib
import os
import sys
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np


class PackageSmokeTests(unittest.TestCase):
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

    def test_public_api_imports(self):
        module = importlib.import_module("ugdatalab")
        self.assertTrue(hasattr(module, "GaiaData"))
        self.assertTrue(hasattr(module, "GaiaQuality"))
        self.assertTrue(hasattr(module, "fit_relation_nuts"))
        self.assertTrue(hasattr(module, "_cache_stable"))
        self.assertTrue(hasattr(module, "get_gaia"))
        self.assertTrue(hasattr(module, "get_gaia_quality"))
        self.assertTrue(hasattr(module, "sanitize_vari_rrlyrae_table"))
        self.assertTrue(hasattr(module, "SDSSData"))
        self.assertTrue(hasattr(module, "SDSSQuality"))
        self.assertTrue(hasattr(module, "get_sdss"))
        self.assertTrue(hasattr(module, "get_sdss_quality"))
        self.assertTrue(hasattr(module, "build_rrlyrae_top_n_query"))
        self.assertTrue(hasattr(module, "fourier_fit"))
        self.assertTrue(hasattr(module, "compute_empirical_extinction"))
        self.assertTrue(hasattr(module, "sample_sfd_ebv"))
        self.assertTrue(hasattr(module, "plot_mollweide"))
        self.assertTrue(hasattr(module, "plot_lomb_scargle_periodogram"))
        self.assertTrue(hasattr(module, "plot_raw_phase_folded_lightcurve"))
        self.assertTrue(hasattr(module, "plot_corner"))
        self.assertTrue(hasattr(module, "attach_periodogram_periods"))

    def test_fourier_module_import_emits_deprecation_warning(self):
        sys.modules.pop("ugdatalab.fourier", None)

        with self.assertWarnsRegex(
            DeprecationWarning,
            r"ugdatalab\.fourier is deprecated; import Fourier helpers from ugdatalab\.lightcurves instead\.",
        ):
            module = importlib.import_module("ugdatalab.fourier")

        self.assertTrue(hasattr(module, "fourier_fit"))

    def test_plot_corner_works(self):
        from ugdatalab import MetropolisHastings, plot_corner

        with self.writable_cache_env():
            sampler = MetropolisHastings(
                log_prob=lambda theta: -0.5 * np.dot(theta, theta),
                theta0=np.array([0.0, 0.0]),
                seed=7,
            )
            sampler.run(n_steps=40, n_burn=10)
            fig = plot_corner(sampler)

        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_build_pymc_model_works(self):
        from ugdatalab.relations import RelationData, _build_pymc_model

        data = RelationData(
            rr_class="RRab",
            relation_kind="pc",
            x=np.array([-0.3, -0.2, -0.1], dtype=float),
            y=np.array([0.30, 0.35, 0.41], dtype=float),
            sigma=np.array([0.02, 0.02, 0.03], dtype=float),
            x_label="x",
            y_label="y",
            data_label="test",
        )

        with self.writable_cache_env():
            model = _build_pymc_model(data, model_kind="native")
        self.assertIsNotNone(model)

    def test_cache_stable_uses_fixed_shared_cache(self):
        from ugdatalab import _cache_stable

        calls = []
        module_name = f"lab02.cache.{uuid.uuid4().hex}"

        @_cache_stable(module=module_name)
        def cached_square(x):
            calls.append(x)
            return x * x

        self.assertEqual(cached_square(3), 9)
        self.assertEqual(cached_square(3), 9)
        self.assertEqual(calls, [3])
        self.assertIn("lab02", cached_square.func_id)

    def test_cache_stable_without_module_keeps_original_namespace(self):
        from ugdatalab import _cache_stable

        calls = []
        value = (uuid.uuid4().int % 997) + 2

        @_cache_stable
        def cached_cube(x):
            calls.append(x)
            return x**3

        self.assertEqual(cached_cube(value), value**3)
        self.assertEqual(cached_cube(value), value**3)
        self.assertEqual(calls, [value])
        self.assertTrue(cached_cube.func_id.endswith("/cached_cube"))


if __name__ == "__main__":
    unittest.main()
