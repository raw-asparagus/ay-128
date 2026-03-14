from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from astropy import table


def _archive_ready_column(column: Any) -> np.ndarray:
    values = np.asarray(column)
    if values.dtype.kind != "O":
        if np.ma.isMaskedArray(column):
            masked_values = np.ma.asarray(column)
            if masked_values.dtype.kind in {"i", "u", "f", "b"}:
                return np.asarray(np.ma.filled(masked_values.astype(float), np.nan), dtype=float)
            return np.asarray(np.ma.filled(masked_values.astype(str), ""), dtype=str)
        if values.dtype.kind in {"U", "S", "O"}:
            return values.astype(str)
        return values

    first = next((value for value in values if value is not None), None)
    if isinstance(first, (np.ndarray, np.ma.MaskedArray, list, tuple)):
        row_arrays = []
        max_len = max(len(np.atleast_1d(np.ma.asarray(value))) for value in values)
        numeric_like = np.ma.asarray(first).dtype.kind in {"i", "u", "f", "b"}
        fill_value = np.nan if numeric_like else ""
        target_dtype = float if numeric_like else str

        for value in values:
            row = np.atleast_1d(np.ma.asarray(value))
            if numeric_like:
                row = np.asarray(np.ma.filled(row.astype(float), np.nan), dtype=float)
            else:
                row = np.asarray(np.ma.filled(row.astype(str), ""), dtype=str)
            padded = np.full(max_len, fill_value, dtype=target_dtype)
            padded[: len(row)] = row
            row_arrays.append(padded)
        return np.stack(row_arrays)

    return values.astype(str)


def save_table_npz(path: str | Path, data: table.Table) -> Path:
    output_path = Path(path)
    archive_cols = {name: _archive_ready_column(data[name]) for name in data.colnames}
    np.savez(
        output_path,
        colnames=np.asarray(data.colnames, dtype=str),
        **archive_cols,
    )
    return output_path


def load_table_npz(path: str | Path) -> table.Table:
    with np.load(Path(path), allow_pickle=False) as archive:
        if "colnames" not in archive.files:
            return table.Table({name: archive[name] for name in archive.files})
        colnames = [str(name) for name in archive["colnames"]]
        return table.Table({name: archive[name] for name in colnames})


def load_or_create_table_npz(
    path: str | Path,
    builder: Callable[[], table.Table],
) -> tuple[table.Table, str]:
    archive_path = Path(path)
    if archive_path.exists():
        return load_table_npz(archive_path), "loaded"

    data = builder()
    save_table_npz(archive_path, data)
    return data, "created"
