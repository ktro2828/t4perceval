from __future__ import annotations

import numpy as np
from attrs import define

from t4perceval.core.component import ColumnarComponent

__all__ = ("BatchMask",)


@define(frozen=True, slots=True)
class BatchMask(ColumnarComponent):
    """A boolean column with shape ``(N,)``.

    Filter systems emit their verdict as a mask instead of dropping rows, so that the
    reason a row was excluded stays inspectable downstream.
    """

    DTYPE = np.bool_

    @property
    def num_selected(self) -> int:
        """Return how many rows are ``True``."""
        return int(np.count_nonzero(self.values))

    def indices(self) -> np.ndarray:
        """Return the indices of the ``True`` rows."""
        return np.flatnonzero(self.values).astype(np.int64, copy=False)
