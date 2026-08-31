from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from t4perceval.typing import NDArrayI64, SelectionLike

__all__ = ("normalize_selection",)


def normalize_selection(selection: SelectionLike, *, length: int) -> NDArrayI64:
    """Normalize any accepted selection into a 1-D array of non-negative row indices.

    Accepted inputs:
        * ``slice`` -- resolved against ``length``.
        * integer array / sequence -- negative indices are resolved, bounds are checked.
          Duplicate and non-monotonic indices are allowed.
        * boolean array / sequence -- must have exactly ``length`` elements.

    Args:
        selection: The selection to normalize.
        length: Number of rows the selection applies to.

    Returns:
        Contiguous ``int64`` array of row indices in ``[0, length)``.
    """
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")

    if isinstance(selection, slice):
        return np.arange(length, dtype=np.int64)[selection]

    array = np.asarray(selection)

    if array.ndim != 1:
        raise ValueError(f"Selection must be one-dimensional, got shape {array.shape}")

    if array.dtype == np.bool_:
        if array.shape[0] != length:
            raise ValueError(
                f"Boolean selection has length {array.shape[0]}, expected {length}",
            )
        return np.ascontiguousarray(np.flatnonzero(array).astype(np.int64, copy=False))

    if array.size == 0:
        # An empty list has dtype float64; treat it as an empty index array.
        return np.empty(0, dtype=np.int64)

    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"Selection must be a slice, an integer array, or a boolean array, got dtype {array.dtype}",
        )

    indices = array.astype(np.int64, copy=True)
    negative = indices < 0
    if negative.any():
        indices[negative] += length

    out_of_range = (indices < 0) | (indices >= length)
    if out_of_range.any():
        first = int(array[out_of_range][0])
        raise IndexError(f"Selection index {first} is out of range for length {length}")

    return np.ascontiguousarray(indices)
