"""Heading flip rate: how often a true positive faces the wrong way.

Corner displacement forgives a half-turn heading error because the footprint it draws is
the same, and APH only discounts it. Yet a reversed heading inverts the object's velocity
direction and breaks tracking, so it is counted on its own: every true positive whose
absolute wrapped yaw error exceeds :attr:`~HeadingFlipRateSystem.flip_threshold` is one flip.
The metric is reported per class as the fraction of flipped true positives of one matching.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define, field

from t4perceval import geometry
from t4perceval.component import MatchStatus
from t4perceval.descriptors import (
    CLASS_ID,
    EST_INDEX,
    GT_INDEX,
    MATCH_STATUS,
    QUATERNION,
    THRESHOLD,
)
from t4perceval.system.metric.base import MetricRow, MetricSystem

if TYPE_CHECKING:
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath
    from t4perceval.system.base import SystemContext
    from t4perceval.system.matching.join import MatchJoin

__all__ = ("HeadingFlipRateSystem",)


@define(slots=True)
class HeadingFlipRateSystem(MetricSystem):
    """Fraction of true positives with a reversed heading, per class.

    For every true positive whose classes agree, the yaw of the estimation is compared with
    that of its ground truth as the shortest angular distance in ``[0, pi]`` (see
    :func:`~t4perceval.geometry.heading_errors`). A true positive whose error is strictly
    above :attr:`flip_threshold` is a flip, and the rate is the flips divided by the true
    positives. The absolute number of flips is not reported: it is the rate times the
    true-positive count, which the matching entity already carries.

    A class with no true positive reports ``NaN`` with its ground-truth count as support.
    The threshold column carries the threshold the matching was run at.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (
        EST_INDEX,
        GT_INDEX,
        MATCH_STATUS,
        THRESHOLD,
    )
    REQUIRES_ESTIMATION: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID, QUATERNION)
    REQUIRES_GROUND_TRUTH: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID, QUATERNION)
    METRIC_NAME: ClassVar[str] = "heading_flip_rate"

    #: Absolute wrapped yaw error in radians above which a true positive counts as flipped.
    flip_threshold: float = field(default=math.pi / 2.0, kw_only=True, converter=float)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if not 0.0 < self.flip_threshold <= math.pi:
            raise ValueError(f"flip_threshold must lie in (0, pi], got {self.flip_threshold}")

    def compute(self, join: MatchJoin, ctx: SystemContext) -> dict[EntityPath, list[MetricRow]]:
        classes = self.classes(ctx, join)
        rows: list[MetricRow] = []

        if not len(join.matches):
            rows.extend((int(c), float("nan"), float("nan"), 0) for c in classes)
            return {self.target: rows}

        status = join.match_component(MATCH_STATUS)
        thresholds = join.match_component(THRESHOLD)
        est_class = join.est_component(CLASS_ID)
        gt_class = join.gt_component(CLASS_ID)
        is_true_positive = (status == int(MatchStatus.TP)) & join.is_label_correct()
        is_false_positive = (status == int(MatchStatus.FP)) & join.has_estimation

        gt_classes_all = (
            join.ground_truth.component(CLASS_ID).values
            if len(join.ground_truth)
            else np.empty(0, dtype=np.int32)
        )

        errors = np.full(len(join), np.nan, dtype=np.float64)
        if is_true_positive.any():
            errors[is_true_positive] = geometry.heading_errors(
                join.est_component(QUATERNION)[is_true_positive],
                join.gt_component(QUATERNION)[is_true_positive],
            )

        for class_id in classes:
            true_positive = is_true_positive & (gt_class == class_id)
            num_ground_truth = int(np.count_nonzero(gt_classes_all == class_id))

            mentions = (
                true_positive
                | (is_false_positive & (est_class == class_id))
                | (join.has_ground_truth & (gt_class == class_id))
            )
            class_thresholds = thresholds[mentions]
            threshold = float(class_thresholds[0]) if class_thresholds.size else float("nan")

            values = errors[true_positive]
            values = values[~np.isnan(values)]
            num_flips = int(np.count_nonzero(values > self.flip_threshold))
            rate = num_flips / values.size if values.size else float("nan")

            rows.append((int(class_id), threshold, rate, num_ground_truth))

        return {self.target: rows}
