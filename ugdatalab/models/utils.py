import numpy as np


def _sanitize_table(data, schema: dict) -> None:
    """Cast columns in *data* according to *schema*.

    *schema* maps a type to a list of column names, e.g.::

        {
            np.int64: ["source_id"],
            str: ["best_classification"],
            float: ["l", "b", "pf", ...],
        }
    """
    for dtype, names in schema.items():
        for name in names:
            if name in data.colnames:
                data[name] = np.asarray(data[name], dtype=dtype)


def _char_at(column, index: int) -> np.ndarray:
    """Extract the character at *index* from each string in *column*."""
    return np.array([s[index] if len(s) > index else "" for s in np.asarray(column, dtype=str)])
