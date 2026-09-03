from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import (
    ALL_CLASSES,
    BACKGROUND_CLASS_ID,
    BatchClassId,
    BatchCount,
    BatchMetricValue,
    BatchSupport,
    BatchThreshold,
)
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import (
    CLASS_ID,
    COUNT,
    ESTIMATION_CLASS_ID,
    GROUND_TRUTH_CLASS_ID,
    METRIC_VALUE,
    SUPPORT,
    THRESHOLD,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

__all__ = ("ConfusionMatrix", "MetricValues")


@define(frozen=True, slots=True)
class ConfusionMatrix(Archetype):
    """Long-form detection confusion matrix.

    Each row is one ``(ground_truth_class_id, estimation_class_id, count)`` cell.
    :data:`~t4perceval.component.BACKGROUND_CLASS_ID` on the estimation axis represents
    a false negative; the same value on the ground-truth axis represents a false
    positive.  The background/background cell is present for a complete square matrix
    but is always zero.
    """

    ground_truth_class_id = component_field(GROUND_TRUTH_CLASS_ID, BatchClassId)
    estimation_class_id = component_field(ESTIMATION_CLASS_ID, BatchClassId)
    count = component_field(COUNT, BatchCount)

    @classmethod
    def empty(cls) -> Self:
        """Return a confusion matrix with no cells."""
        return cls(
            ground_truth_class_id=np.empty(0, dtype=np.int32),
            estimation_class_id=np.empty(0, dtype=np.int32),
            count=np.empty(0, dtype=np.int64),
        )

    @classmethod
    def from_rows(cls, rows: Sequence[tuple[int, int, int]]) -> Self:
        """Build from ``(ground_truth_class_id, estimation_class_id, count)`` tuples."""
        if not rows:
            return cls.empty()
        ground_truth, estimation, counts = zip(*rows)
        return cls(
            ground_truth_class_id=np.asarray(ground_truth, dtype=np.int32),
            estimation_class_id=np.asarray(estimation, dtype=np.int32),
            count=np.asarray(counts, dtype=np.int64),
        )

    def at(self, ground_truth_class_id: int, estimation_class_id: int) -> int:
        """Return the count of one cell, requiring it to occur exactly once."""
        matching = np.flatnonzero(
            (self.ground_truth_class_id.values == ground_truth_class_id)
            & (self.estimation_class_id.values == estimation_class_id)
        )
        if matching.size != 1:
            raise KeyError(
                "Expected exactly one confusion-matrix cell for "
                f"({ground_truth_class_id}, {estimation_class_id}), found {matching.size}",
            )
        return int(self.count.values[matching[0]])

    def as_matrix(
        self,
        class_ids: Sequence[int] | None = None,
        *,
        include_background: bool = True,
    ) -> np.ndarray:
        """Return a dense matrix ordered by ``class_ids``, then background.

        When ``class_ids`` is omitted, non-background ids present on either axis are
        sorted.  Duplicate long-form cells are added, making this safe on concatenated
        chunks as well as a single metric result.
        """
        if class_ids is None:
            observed = np.concatenate(
                (self.ground_truth_class_id.values, self.estimation_class_id.values)
            )
            ordered = sorted(int(value) for value in np.unique(observed) if value >= 0)
        else:
            ordered = [int(value) for value in class_ids]
            if len(set(ordered)) != len(ordered):
                raise ValueError("class_ids must not contain duplicates")
            if BACKGROUND_CLASS_ID in ordered:
                raise ValueError("class_ids must not contain the background class")

        axes = ordered + ([BACKGROUND_CLASS_ID] if include_background else [])
        slots = {class_id: index for index, class_id in enumerate(axes)}
        matrix = np.zeros((len(axes), len(axes)), dtype=np.int64)
        for ground_truth, estimation, count in zip(
            self.ground_truth_class_id.values,
            self.estimation_class_id.values,
            self.count.values,
            strict=True,
        ):
            row = slots.get(int(ground_truth))
            column = slots.get(int(estimation))
            if row is not None and column is not None:
                matrix[row, column] += int(count)
        return matrix


@define(frozen=True, slots=True)
class MetricValues(Archetype):
    """One metric's values, broken down by class and threshold.

    Every scalar metric shares this shape, and the metric's *name* is the entity path it
    is logged to -- the same rule that gives each matching mode its own path. A reader
    can therefore treat ``/metrics/ap`` and ``/metrics/mota`` identically, and comparing
    two runs is one query rather than a bespoke traversal per metric family. Structured
    metrics such as :class:`ConfusionMatrix` use their own explicit axes.

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
