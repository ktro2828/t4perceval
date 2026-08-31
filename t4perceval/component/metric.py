from __future__ import annotations

import numpy as np
from attrs import define

from t4perceval.core.component import ColumnarComponent

__all__ = ("ALL_CLASSES", "BatchMetricValue", "BatchSupport", "BatchThreshold")

#: ``class_id`` marking a row that aggregates over every class.
#:
#: This is the same integer as :data:`t4perceval.label.UNKNOWN_CLASS_ID`, and it is
#: unambiguous in a metrics table: per-class rows are only ever emitted for classes the
#: :class:`~t4perceval.label.LabelRegistry` knows, so a ``-1`` row can only be an
#: aggregate.
ALL_CLASSES = -1


@define(frozen=True, slots=True)
class BatchThreshold(ColumnarComponent):
    """The threshold a value was produced at, with shape ``(N,)``.

    ``NaN`` means the row has no threshold -- either the metric does not take one, or the
    row aggregates across thresholds.
    """

    DTYPE = np.float64


@define(frozen=True, slots=True)
class BatchMetricValue(ColumnarComponent):
    """A metric value with shape ``(N,)``.

    ``NaN`` means undefined: there was nothing to measure. The original package used
    ``inf`` for this, which reads as "infinitely good or bad"; ``NaN`` says undefined, and
    the accompanying :class:`BatchSupport` says why.
    """

    DTYPE = np.float64


@define(frozen=True, slots=True)
class BatchSupport(ColumnarComponent):
    """How many ground-truth objects a value rests on, with shape ``(N,)``.

    ``0`` is what turns an undefined value from a mystery into a fact: the metric is
    ``NaN`` because there was nothing of that class to find.
    """

    DTYPE = np.int64
