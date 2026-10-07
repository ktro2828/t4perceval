"""Corner displacement error: how far a true positive's footprint corners sit from the truth.

Centre distance is blind to a box that is the right place but the wrong shape, and yaw
error alone says nothing about how much a wrong heading moves the outline. Corner
displacement couples position, size and yaw into one distance in metres, measured where
a planner would collide with the object: at the box outline. It is reported per class
over the true positives of one matching, as a mean, the requested percentiles and a max.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define, field

from t4perceval import geometry
from t4perceval.component import MatchStatus
from t4perceval.core import as_entity_path
from t4perceval.descriptors import (
    CLASS_ID,
    EST_INDEX,
    GT_INDEX,
    MATCH_STATUS,
    POSITION,
    QUATERNION,
    SIZE,
    THRESHOLD,
)
from t4perceval.system.metric.base import MetricRow, MetricSystem

if TYPE_CHECKING:
    from collections.abc import Iterable

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath
    from t4perceval.system.base import SystemContext
    from t4perceval.system.matching.join import MatchJoin

__all__ = ("CornerErrorSystem",)


def _as_percentiles(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(p) for p in values)


def _number_token(value: float) -> str:
    """Spell a percentile for an entity path: ``95.0`` -> ``95``, ``97.5`` -> ``97.5``."""
    return f"{value:g}"


@define(slots=True)
class CornerErrorSystem(MetricSystem):
    """Mean, percentile and max true-positive corner displacement per class, in metres.

    For every true positive whose classes agree, the four footprint corners of the
    estimation are compared with those of its ground truth under the best cyclic corner
    assignment (see :func:`~t4perceval.geometry.corner_displacements`). The per-class
    statistics of those distances are written to one entity each under :attr:`target`:
    ``mean``, one ``p<N>`` per entry of :attr:`percentiles`, and ``max``.

    A class with no true positive reports ``NaN`` with its ground-truth count as support.
    The threshold column carries the threshold the matching was run at.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (
        EST_INDEX,
        GT_INDEX,
        MATCH_STATUS,
        THRESHOLD,
    )
    REQUIRES_ESTIMATION: ClassVar[tuple[ComponentDescriptor, ...]] = (
        CLASS_ID,
        POSITION,
        QUATERNION,
        SIZE,
    )
    REQUIRES_GROUND_TRUTH: ClassVar[tuple[ComponentDescriptor, ...]] = (
        CLASS_ID,
        POSITION,
        QUATERNION,
        SIZE,
    )
    METRIC_NAME: ClassVar[str] = "corner_error"

    #: Percentiles in ``[0, 100]`` reported next to the mean and the max.
    percentiles: tuple[float, ...] = field(default=(95.0,), kw_only=True, converter=_as_percentiles)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        for percentile in self.percentiles:
            if not 0.0 <= percentile <= 100.0:
                raise ValueError(f"percentiles must lie in [0, 100], got {percentile}")
        # Each percentile writes its own entity, so check if all entities become unique
        tokens = [_number_token(p) for p in self.percentiles]
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"percentiles must map to distinct paths, got {self.percentiles}")

    @property
    def targets(self) -> tuple[EntityPath, ...]:
        root = as_entity_path(self.target)
        percentiles = tuple(root / f"p{_number_token(p)}" for p in self.percentiles)
        return (root / "mean", *percentiles, root / "max")

    def compute(self, join: MatchJoin, ctx: SystemContext) -> dict[EntityPath, list[MetricRow]]:
        classes = self.classes(ctx, join)
        results: dict[EntityPath, list[MetricRow]] = {target: [] for target in self.targets}

        if not len(join.matches):
            for rows in results.values():
                rows.extend((int(c), float("nan"), float("nan"), 0) for c in classes)
            return results

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
            errors[is_true_positive] = geometry.corner_displacements(
                join.est_component(POSITION)[is_true_positive],
                join.est_component(QUATERNION)[is_true_positive],
                join.est_component(SIZE)[is_true_positive],
                join.gt_component(POSITION)[is_true_positive],
                join.gt_component(QUATERNION)[is_true_positive],
                join.gt_component(SIZE)[is_true_positive],
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
            if values.size == 0:
                statistics = [float("nan")] * len(results)
            else:
                statistics = [
                    float(values.mean()),
                    *(float(np.percentile(values, p)) for p in self.percentiles),
                    float(values.max()),
                ]

            for rows, value in zip(results.values(), statistics, strict=True):
                rows.append((int(class_id), threshold, value, num_ground_truth))

        return results
