"""Detection metrics: average precision, with and without heading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from attrs import define, field

from t4perceval import geometry
from t4perceval.archetype.metric import MetricValues
from t4perceval.component import ALL_CLASSES, MatchStatus
from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import TimePoint, TimeRange
from t4perceval.descriptors import (
    CLASS_ID,
    CONFIDENCE,
    EST_INDEX,
    GT_INDEX,
    MATCH_STATUS,
    METRIC_VALUE,
    QUATERNION,
    SUPPORT,
    THRESHOLD,
)
from t4perceval.system.base import EntitySystem, SystemContext, require
from t4perceval.system.matching.base import confidence_order
from t4perceval.system.metric.base import (
    MetricRow,
    MetricSystem,
    latest_time,
    nan_mean,
    reporting_time,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from typing_extensions import Self

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.core.view import EntityView
    from t4perceval.system.matching.join import MatchJoin
    from t4perceval.typing import NDArrayBool, NDArrayF64, NDArrayI64

__all__ = (
    "AveragePrecisionHeadingSystem",
    "AveragePrecisionSystem",
    "MeanAveragePrecisionSystem",
)


def _average_precision(
    tp_weight: NDArrayF64,
    confidence: NDArrayF64,
    est_rows: NDArrayI64,
    num_ground_truth: int,
    *,
    min_recall: float,
    min_precision: float,
    num_recall_points: int,
) -> float:
    """Return the nuScenes-style average precision of one class.

    Estimations are ranked by confidence, so a detector that is confident about its
    mistakes scores worse than one that is not. The low-recall and low-precision corners
    are then cut away and the remainder rescaled, which is what stops a detector from
    banking score on the part of the curve nobody operates in.

    ``tp_weight`` is how much each estimation counts as a true positive: 1.0 for AP, and
    the heading agreement for APH. A false positive has weight 0. ``est_rows`` are the
    estimations' own rows, which break confidence ties the way greedy matching does --
    not the order the verdicts happen to be laid out in.
    """
    if num_ground_truth == 0 and tp_weight.size == 0:
        # Nothing to find and nothing claimed: the metric is undefined, not perfect.
        return float("nan")
    if tp_weight.size == 0:
        return 0.0

    order = confidence_order(confidence, est_rows)
    true_positive = np.cumsum(tp_weight[order])
    false_positive = np.cumsum(np.where(tp_weight[order] > 0.0, 0.0, 1.0))

    claimed = true_positive + false_positive
    precision = np.divide(
        true_positive,
        claimed,
        out=np.zeros_like(true_positive),
        where=claimed > 0.0,
    )
    recall = (
        true_positive / num_ground_truth if num_ground_truth > 0 else np.zeros_like(true_positive)
    )

    # Precision is only ever reported at its best achievable value further along the
    # curve, then sampled on a fixed recall grid so classes are comparable.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0.0, 1.0, num_recall_points)
    sampled = np.interp(grid, recall, envelope, right=0.0)

    first = int(round((num_recall_points - 1) * min_recall)) + 1
    trimmed = np.maximum(sampled[first:] - min_precision, 0.0)
    return float(trimmed.mean()) / (1.0 - min_precision)


@define(slots=True)
class AveragePrecisionSystem(MetricSystem):
    """Average precision per class, at the threshold the matching was done at.

    A class's pool is the estimations *labelled* as that class, which is how the original
    package grouped them; a true positive additionally has to agree with its ground truth
    about the class, so a class-agnostic matcher cannot inflate the score.

    The last row, class ``-1``, is the mean of the defined class APs -- the mAP at this
    threshold. Unlike the other metrics it does not pool the objects: an AP over every
    class at once would rank one class's confidences against another's.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (
        EST_INDEX,
        GT_INDEX,
        MATCH_STATUS,
        THRESHOLD,
    )
    REQUIRES_ESTIMATION: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID, CONFIDENCE)
    REQUIRES_GROUND_TRUTH: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)
    METRIC_NAME: ClassVar[str] = "ap"

    min_recall: float = field(default=0.1, kw_only=True)
    min_precision: float = field(default=0.1, kw_only=True)
    num_recall_points: int = field(default=101, kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        for value, name in ((self.min_recall, "min_recall"), (self.min_precision, "min_precision")):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be within [0, 1), got {value}")
        if self.num_recall_points < 2:
            raise ValueError(
                f"num_recall_points must be at least 2, got {self.num_recall_points}",
            )

    def tp_weight(self, join: MatchJoin, is_true_positive: NDArrayBool) -> NDArrayF64:
        """Return how much each match row counts as a true positive."""
        return np.where(is_true_positive, 1.0, 0.0)

    def compute(self, join: MatchJoin, ctx: SystemContext) -> dict[EntityPath, list[MetricRow]]:
        rows: list[MetricRow] = []
        if not len(join.matches):
            return {
                self.target: [
                    (c, float("nan"), float("nan"), 0) for c in self.reported_classes(ctx, join)
                ]
            }

        status = join.match_component(MATCH_STATUS)
        est_class = join.est_component(CLASS_ID)
        confidence = join.est_component(CONFIDENCE)
        thresholds = join.match_component(THRESHOLD)
        is_true_positive = (status == int(MatchStatus.TP)) & join.is_label_correct()
        weight = self.tp_weight(join, is_true_positive)

        gt_class = (
            join.ground_truth.component(CLASS_ID).values
            if len(join.ground_truth)
            else np.empty(0, dtype=np.int32)
        )

        for class_id in self.classes(ctx, join):
            pool = join.has_estimation & (est_class == class_id)
            num_ground_truth = int(np.count_nonzero(gt_class == class_id))
            value = _average_precision(
                weight[pool],
                confidence[pool],
                join.est_rows[pool],
                num_ground_truth,
                min_recall=self.min_recall,
                min_precision=self.min_precision,
                num_recall_points=self.num_recall_points,
            )
            # Every row of a class shares its threshold, so any row mentioning the class
            # names it -- including a false negative, which has no estimation.
            mentions = (join.has_estimation & (est_class == class_id)) | (
                join.has_ground_truth & (join.gt_component(CLASS_ID) == class_id)
            )
            class_thresholds = thresholds[mentions]
            threshold = float(class_thresholds[0]) if class_thresholds.size else float("nan")
            rows.append((int(class_id), threshold, value, num_ground_truth))

        # NaN threshold: the all-class row is read by `MetricValues.aggregate`.
        rows.append(
            (
                ALL_CLASSES,
                float("nan"),
                nan_mean([value for _, _, value, _ in rows]),
                len(gt_class),
            )
        )
        return {self.target: rows}


@define(slots=True)
class AveragePrecisionHeadingSystem(AveragePrecisionSystem):
    """Average precision weighted by heading agreement (APH).

    A true positive counts as ``1 - |heading error| / pi``, so a box in the right place
    facing the wrong way earns almost nothing. Used by the Waymo Open Dataset.
    """

    REQUIRES_ESTIMATION: ClassVar[tuple[ComponentDescriptor, ...]] = (
        CLASS_ID,
        CONFIDENCE,
        QUATERNION,
    )
    REQUIRES_GROUND_TRUTH: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID, QUATERNION)
    METRIC_NAME: ClassVar[str] = "aph"

    def tp_weight(self, join: MatchJoin, is_true_positive: NDArrayBool) -> NDArrayF64:
        if not is_true_positive.any():
            return np.zeros(len(join), dtype=np.float64)

        error = geometry.heading_errors(
            join.est_component(QUATERNION),
            join.gt_component(QUATERNION),
        )
        similarity = 1.0 - error / np.pi
        return np.where(is_true_positive, np.nan_to_num(similarity, nan=0.0), 0.0)


@define(slots=True)
class MeanAveragePrecisionSystem(EntitySystem):
    """Average precision averaged across thresholds and classes.

    Reads the per-threshold AP entities produced upstream and writes the whole cross-tab
    into one table, which the uniform metric schema makes free:

    * ``(class_id=c, threshold=NaN)`` -- class ``c`` averaged over thresholds
    * ``(class_id=-1, threshold=t)`` -- threshold ``t`` averaged over classes
    * ``(class_id=-1, threshold=NaN)`` -- the mAP itself

    Undefined values are skipped rather than propagated, so one class with no ground truth
    does not erase the score. Point the same system at APH entities to get mAPH.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = MetricValues.required_descriptors()
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = MetricValues.required_descriptors()

    def __attrs_post_init__(self) -> None:
        if not self.sources:
            raise ValueError(f"{type(self).__name__} needs at least one source")

    @classmethod
    def of(
        cls,
        sources: Sequence[EntityPathLike],
        *,
        target: EntityPathLike = "/metrics/map",
        **params: Any,
    ) -> Self:
        """Average the metric entities at ``sources`` into ``target``."""
        return cls(tuple(sources), target, **params)

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)

        per_class: dict[int, list[float]] = {}
        per_threshold: dict[float, list[float]] = {}
        support: dict[int, int] = {}
        views: list[EntityView] = []

        for source in self.sources:
            view = ctx.store.range(source, timeline=ctx.timeline, time_range=time_range)
            if not len(view):
                continue
            require(view, *self.REQUIRES)
            views.append(view)

            class_ids = view.component(CLASS_ID).values
            thresholds = view.component(THRESHOLD).values
            values = view.component(METRIC_VALUE).values
            supports = view.component(SUPPORT).values

            for class_id, threshold, value, count in zip(
                class_ids,
                thresholds,
                values,
                supports,
            ):
                if int(class_id) == ALL_CLASSES:
                    continue
                per_class.setdefault(int(class_id), []).append(float(value))
                if not np.isnan(threshold):
                    # A class with nothing to score has no threshold, so it cannot join a
                    # per-threshold average -- and must not create a NaN-keyed bucket that
                    # would collide with the overall aggregate row.
                    per_threshold.setdefault(float(threshold), []).append(float(value))
                support[int(class_id)] = max(support.get(int(class_id), 0), int(count))

        rows: list[MetricRow] = [
            (class_id, float("nan"), nan_mean(values), support.get(class_id, 0))
            for class_id, values in sorted(per_class.items())
        ]
        total_support = sum(support.values())
        rows.extend(
            (ALL_CLASSES, threshold, nan_mean(values), total_support)
            for threshold, values in sorted(per_threshold.items())
        )
        rows.append(
            (
                ALL_CLASSES,
                float("nan"),
                nan_mean([nan_mean(values) for values in per_class.values()]),
                total_support,
            ),
        )

        at_time = reporting_time(latest_time(views, ctx.timeline), time_range)
        return (
            MetricValues.from_rows(rows).to_chunk(
                as_entity_path(self.target),
                at=TimePoint(((ctx.timeline, at_time),)),
            ),
        )
