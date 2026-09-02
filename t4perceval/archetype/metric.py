from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import (
    ALL_CLASSES,
    BatchClassId,
    BatchMetricValue,
    BatchSupport,
    BatchThreshold,
)
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import CLASS_ID, METRIC_VALUE, SUPPORT, THRESHOLD

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

__all__ = ("MetricValues",)


@define(frozen=True, slots=True)
class MetricValues(Archetype):
    """One metric's values, broken down by class and threshold.

    Every metric shares this shape, and the metric's *name* is the entity path it is
    logged to -- the same rule that gives each matching mode its own path. A reader can
    therefore treat ``/metrics/ap`` and ``/metrics/mota`` identically, and comparing two
    runs is one query rather than a bespoke traversal per metric family.

    ``class_id == ALL_CLASSES`` marks a row aggregated over classes, and ``threshold`` is
    ``NaN`` when the row has none.
    """

    class_id = component_field(CLASS_ID, BatchClassId)
    threshold = component_field(THRESHOLD, BatchThreshold)
    value = component_field(METRIC_VALUE, BatchMetricValue)
    support = component_field(SUPPORT, BatchSupport)

    @classmethod
    def empty(cls) -> Self:
        """Return a metric with no rows."""
        return cls(
            class_id=np.empty(0, dtype=np.int32),
            threshold=np.empty(0, dtype=np.float64),
            value=np.empty(0, dtype=np.float64),
            support=np.empty(0, dtype=np.int64),
        )

    @classmethod
    def from_rows(cls, rows: Sequence[tuple[int, float, float, int]]) -> Self:
        """Build from ``(class_id, threshold, value, support)`` tuples."""
        if not rows:
            return cls.empty()
        class_ids, thresholds, values, supports = zip(*rows)
        return cls(
            class_id=np.asarray(class_ids, dtype=np.int32),
            threshold=np.asarray(thresholds, dtype=np.float64),
            value=np.asarray(values, dtype=np.float64),
            support=np.asarray(supports, dtype=np.int64),
        )

    def of_class(self, class_id: int) -> float:
        """Return the single value recorded for ``class_id``.

        Raises when the class has no row, or more than one -- a metric with several
        thresholds per class must be read through :attr:`threshold` instead.
        """
        matching = np.flatnonzero(self.class_id.values == class_id)
        if matching.size != 1:
            raise KeyError(
                f"Expected exactly one row for class {class_id}, found {matching.size}",
            )
        return float(self.value.values[matching[0]])

    @property
    def aggregate(self) -> float:
        """Return the value of the row aggregating over classes and thresholds."""
        rows = np.flatnonzero(
            (self.class_id.values == ALL_CLASSES) & np.isnan(self.threshold.values),
        )
        if rows.size != 1:
            raise KeyError(f"Expected exactly one aggregate row, found {rows.size}")
        return float(self.value.values[rows[0]])
