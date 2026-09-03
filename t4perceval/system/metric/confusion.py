"""Detection confusion matrix, including false-positive and false-negative background."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define

from t4perceval.archetype.metric import ConfusionMatrix
from t4perceval.component import BACKGROUND_CLASS_ID, MatchStatus
from t4perceval.core.timeline import TimePoint, TimeRange
from t4perceval.descriptors import CLASS_ID, EST_INDEX, GT_INDEX, MATCH_STATUS
from t4perceval.system.base import require
from t4perceval.system.join import MatchJoin
from t4perceval.system.metric.base import MetricSystem

if TYPE_CHECKING:
    from collections.abc import Iterable

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.system.base import SystemContext

__all__ = ("ConfusionMatrixSystem",)


@define(slots=True)
class ConfusionMatrixSystem(MetricSystem):
    """Count matched labels, false negatives and false positives by class pair.

    The matrix uses ground-truth classes as rows and estimation classes as columns.
    False negatives occupy the background column and false positives the background row.

    Off-diagonal class-confusion cells require a class-agnostic matcher.  A class-aware
    matcher has already separated a misclassification into an FP and FN, so no later
    metric can recover which objects originally belonged together.
    """

    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = ConfusionMatrix.required_descriptors()
    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (
        EST_INDEX,
        GT_INDEX,
        MATCH_STATUS,
    )
    REQUIRES_ESTIMATION: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)
    REQUIRES_GROUND_TRUTH: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)
    METRIC_NAME: ClassVar[str] = "confusion_matrix"

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        matching, estimation, ground_truth = self.sources
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)
        join = MatchJoin.of(
            ctx.store,
            matching,
            estimation,
            ground_truth,
            timeline=ctx.timeline,
            time_range=time_range,
        )

        if len(join.matches):
            require(join.matches, *self.REQUIRES)
        if len(join.estimation):
            require(join.estimation, *self.REQUIRES_ESTIMATION)
        if len(join.ground_truth):
            require(join.ground_truth, *self.REQUIRES_GROUND_TRUTH)

        result = self.compute_matrix(join, ctx)
        reporting_time = self._reporting_time(join, time_range)
        return (
            result.to_chunk(
                self.target,
                at=TimePoint(((ctx.timeline, reporting_time),)),
            ),
        )

    def compute_matrix(self, join: MatchJoin, ctx: SystemContext) -> ConfusionMatrix:
        """Build the complete long-form count matrix for the selected time range."""
        classes = [int(class_id) for class_id in self.classes(ctx, join)]
        if not classes:
            return ConfusionMatrix.empty()

        axes = [*classes, BACKGROUND_CLASS_ID]
        counts = {(ground_truth, estimation): 0 for ground_truth, estimation in product(axes, axes)}

        if len(join.matches):
            status = join.match_component(MATCH_STATUS).astype(np.int8, copy=False)
            both = join.has_ground_truth & join.has_estimation
            only_estimation = ~join.has_ground_truth & join.has_estimation
            only_ground_truth = join.has_ground_truth & ~join.has_estimation
            valid = (
                ((status == int(MatchStatus.TP)) & both)
                | ((status == int(MatchStatus.FP)) & only_estimation)
                | ((status == int(MatchStatus.FN)) & only_ground_truth)
            )
            if not np.all(valid):
                raise ValueError(
                    "Match status is inconsistent with estimation/ground-truth indices"
                )

            estimation_class = join.est_component(CLASS_ID, fill=BACKGROUND_CLASS_ID).astype(
                np.int32,
                copy=False,
            )
            ground_truth_class = join.gt_component(CLASS_ID, fill=BACKGROUND_CLASS_ID).astype(
                np.int32,
                copy=False,
            )
            pairs, frequencies = np.unique(
                np.column_stack((ground_truth_class, estimation_class)),
                axis=0,
                return_counts=True,
            )
            for (ground_truth, estimation), frequency in zip(pairs, frequencies, strict=True):
                key = (int(ground_truth), int(estimation))
                # Without a registry, classes come from the joined entities and are in
                # ``axes``.  With one, an unregistered data value is malformed rather
                # than a cell that should silently disappear.
                if key not in counts:
                    raise ValueError(
                        "Confusion matrix encountered a class id absent from the label registry: "
                        f"{key}",
                    )
                counts[key] += int(frequency)

        return ConfusionMatrix.from_rows(
            [
                (ground_truth, estimation, counts[(ground_truth, estimation)])
                for ground_truth, estimation in product(axes, axes)
            ]
        )
