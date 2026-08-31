from __future__ import annotations

from enum import IntEnum

import numpy as np
from attrs import define

from t4perceval.core.component import ColumnarComponent

__all__ = (
    "BatchClassId",
    "BatchConfidence",
    "BatchInstanceId",
    "BatchNumPoints",
    "BatchVisibility",
    "VisibilityLevel",
)


class VisibilityLevel(IntEnum):
    """Ordered visibility levels.

    Numeric and ordered so that a ``>=`` threshold works directly on the column, unlike
    the string enum used by the dataset schema. ``UNAVAILABLE`` sorts below every real
    level so that it is never accidentally accepted by a threshold.
    """

    UNAVAILABLE = -1
    NONE = 0
    """90-100% occluded; nothing visible in the label."""
    PARTIAL = 1
    """Occluded by more than 50%."""
    MOST = 2
    """Occluded by less than 50%."""
    FULL = 3
    """Not occluded."""


@define(frozen=True, slots=True)
class BatchClassId(ColumnarComponent):
    """Class indices with shape ``(N,)``.

    The mapping from a class index to a human readable name lives in
    :class:`t4perceval.label.LabelRegistry`, logged as static data.
    """

    DTYPE = np.int32


@define(frozen=True, slots=True)
class BatchConfidence(ColumnarComponent):
    """Confidence scores with shape ``(N,)``, constrained to ``[0, 1]``."""

    VALUE_RANGE = (0.0, 1.0)


@define(frozen=True, slots=True)
class BatchInstanceId(ColumnarComponent):
    """Persistent per-object identifiers with shape ``(N,)``.

    String UUIDs from the dataset are mapped to integers by
    :class:`t4perceval.label.InstanceRegistry`.
    """

    DTYPE = np.int64


@define(frozen=True, slots=True)
class BatchNumPoints(ColumnarComponent):
    """Number of sensor points inside each 3D box, with shape ``(N,)``."""

    DTYPE = np.int32


@define(frozen=True, slots=True)
class BatchVisibility(ColumnarComponent):
    """Visibility level per object with shape ``(N,)``.

    Values are :class:`VisibilityLevel` members; higher means better visibility.
    """

    DTYPE = np.int8
