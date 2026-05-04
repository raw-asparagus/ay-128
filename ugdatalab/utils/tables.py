"""Schema-driven column casting for astropy tables and pandas DataFrames.

A schema is a ``dict[type, list[str]]`` mapping a target dtype to the column
names that should be cast to it. Columns absent from the table are skipped;
columns present in the table but not listed in the schema are left untouched.

Example
-------
::

    _GAIA_SCHEMA = {
        np.int64: ["source_id"],
        int:      ["num_clean_epochs_g"],
        str:      ["best_classification"],
        float:    ["parallax", "parallax_error", "phot_g_mean_mag"],
    }
"""

from typing import Callable, Iterable

import numpy as np


def _sanitize_columns(
    column_names: Iterable[str],
    set_column: Callable[[str, type], None],
    schema: dict,
) -> None:
    """Cast columns named in *schema* in place via *set_column*, skipping absent names."""
    present = set(column_names)
    for dtype, names in schema.items():
        for name in names:
            if name in present:
                set_column(name, dtype)


def _sanitize_table(data, schema: dict) -> None:
    """Cast columns of an ``astropy.table.Table`` according to *schema*."""
    def _set_col(name, dtype):
        """Cast the named astropy table column to *dtype* in place."""
        data[name] = np.asarray(data[name], dtype=dtype)
    _sanitize_columns(data.colnames, _set_col, schema)


def _sanitize_dataframe(data, schema: dict) -> None:
    """Cast columns of a ``pandas.DataFrame`` according to *schema*."""
    def _set_col(name, dtype):
        """Cast the named DataFrame column to *dtype* in place."""
        data[name] = data[name].astype(dtype)
    _sanitize_columns(data.columns, _set_col, schema)
