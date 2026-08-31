from __future__ import annotations

from enum import IntEnum

import numpy as np
from attrs import define

from t4perceval.core.component import ColumnarComponent

__all__ = ("BatchMatchStatus", "BatchMatchingScore", "BatchRowIndex", "MatchStatus")


class MatchStatus(IntEnum):
    """Outcome of matching one estimation against the ground truth."""

    TP = 0
    """A matched pair within the matching threshold."""
    FP = 1
    """An estimation with no acceptable ground truth."""
    FN = 2
    """A ground-truth object no estimation claimed."""


@define(frozen=True, slots=True)
class BatchRowIndex(ColumnarComponent):
    """Row indices into another entity's chunk, with shape ``(N,)``.

    ``-1`` marks "no counterpart", which is how an unmatched estimation or ground-truth
    object is represented.
    """

    DTYPE = np.int64


@define(frozen=True, slots=True)
class BatchMatchingScore(ColumnarComponent):
    """The score a matching system produced for each pair, with shape ``(N,)``.

    The score is whatever the producing system measures -- a centre distance in metres,
    an IoU, a plane distance. Its meaning is carried by the entity path the column is
    logged to, e.g. ``/matching/center_distance``. ``NaN`` marks an unmatched row.
    """

    DTYPE = np.float64


@define(frozen=True, slots=True)
class BatchMatchStatus(ColumnarComponent):
    """TP / FP / FN verdict per row, with shape ``(N,)``.

    Values are :class:`MatchStatus` members.
    """

    DTYPE = np.int8
