import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.table import Table

from ugdatalab import load_table_npz, load_or_create_table_npz, save_table_npz


class TableArtifactTests(unittest.TestCase):
    def test_save_and_load_table_npz_round_trip(self):
        data = Table(
            {
                "source_id": np.array([1, 2], dtype=np.int64),
                "best_classification": np.array(["RRab", "RRc"], dtype=str),
                "bp_rp": np.array([0.72, 0.41], dtype=float),
            }
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table.npz"
            save_table_npz(path, data)
            loaded = load_table_npz(path)

        self.assertEqual(loaded.colnames, data.colnames)
        np.testing.assert_array_equal(loaded["source_id"], data["source_id"])
        np.testing.assert_array_equal(loaded["best_classification"], data["best_classification"])
        np.testing.assert_allclose(loaded["bp_rp"], data["bp_rp"])

    def test_load_or_create_table_npz_uses_existing_archive(self):
        data = Table({"value": [1.0, 2.0]})

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.npz"
            save_table_npz(path, data)

            built = []

            def builder():
                built.append(True)
                return Table({"value": [9.0]})

            loaded, status = load_or_create_table_npz(path, builder)

        self.assertEqual(status, "loaded")
        self.assertEqual(built, [])
        np.testing.assert_allclose(loaded["value"], [1.0, 2.0])

    def test_load_table_npz_supports_legacy_archives_without_colnames(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.npz"
            np.savez(path, source_id=np.array([1, 2]), bp_rp=np.array([0.7, 0.4]))
            loaded = load_table_npz(path)

        self.assertEqual(set(loaded.colnames), {"source_id", "bp_rp"})
        np.testing.assert_array_equal(loaded["source_id"], [1, 2])


if __name__ == "__main__":
    unittest.main()
