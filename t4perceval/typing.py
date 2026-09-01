from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    "ArrayLike",
    "NDArrayBool",
    "NDArrayF64",
    "NDArrayI8",
    "NDArrayI32",
    "NDArrayI64",
    "Selection",
    "SelectionLike",
)

# NOTE: These aliases intentionally duplicate `t4_devkit.typing` so that
# `t4perceval.core` stays independent of the dataset devkit. Only the dataset
# loader is allowed to depend on `t4_devkit`.
ArrayLike: TypeAlias = Union[NDArray[Any], "Sequence[Any]"]

NDArrayBool: TypeAlias = NDArray[np.bool_]
NDArrayF64: TypeAlias = NDArray[np.float64]
NDArrayI8: TypeAlias = NDArray[np.int8]
NDArrayI32: TypeAlias = NDArray[np.int32]
NDArrayI64: TypeAlias = NDArray[np.int64]

#: Normalized selection: always a one-dimensional array of row indices.
Selection: TypeAlias = NDArrayI64

#: Anything accepted by the public ``select()`` APIs before normalization.
SelectionLike: TypeAlias = Union[slice, NDArrayI64, NDArrayBool, "Sequence[int]", "Sequence[bool]"]
