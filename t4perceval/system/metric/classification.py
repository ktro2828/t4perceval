"""Classification metrics: accuracy, precision, recall and F1."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define

from t4perceval.component import MatchStatus
from t4perceval.core.entity import as_entity_path
from t4perceval.descriptors import CLASS_ID, EST_INDEX, GT_INDEX, MATCH_STATUS
from t4perceval.system.metric.base import MetricRow, MetricSystem

if TYPE_CHECKING:
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath
    from t4perceval.system.base import SystemContext
    from t4perceval.system.join import MatchJoin

__all__ = ("ClassificationSystem",)


def _ratio(numerator: float, denominator: float) -> float:
    """Return the ratio, or ``NaN`` when there is nothing to divide by."""
    return numerator / denominator if denominator else float("nan")


@define(slots=True)
class ClassificationSystem(MetricSystem):
    """Accuracy, precision, recall and F1 per class.

    All four come from the same three counts, so one system produces them together and
    writes each to its own entity under :attr:`target`.

    Where the original package returned ``inf`` for an undefined ratio, this returns
    ``NaN`` and lets ``support`` say why -- ``inf`` reads as a score, ``NaN`` reads as
    "there was nothing to measure".
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (EST_INDEX, GT_INDEX, MATCH_STATUS)
    REQUIRES_ESTIMATION: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)
    REQUIRES_GROUND_TRUTH: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)
    METRIC_NAME: ClassVar[str] = "classification"

    @property
    def targets(self) -> tuple[EntityPath, ...]:
        root = as_entity_path(self.target)
        return (root / "accuracy", root / "precision", root / "recall", root / "f1")

    def compute(self, join: MatchJoin, ctx: SystemContext) -> dict[EntityPath, list[MetricRow]]:
        accuracy_target, precision_target, recall_target, f1_target = self.targets
        rows: dict[EntityPath, list[MetricRow]] = {
            accuracy_target: [],
            precision_target: [],
            recall_target: [],
            f1_target: [],
        }

        classes = self.classes(ctx, join)
        est_classes_all = (
            join.estimation.component(CLASS_ID).values
            if len(join.estimation)
            else np.empty(0, dtype=np.int32)
        )
        gt_classes_all = (
            join.ground_truth.component(CLASS_ID).values
            if len(join.ground_truth)
            else np.empty(0, dtype=np.int32)
        )

        if len(join.matches):
            status = join.match_component(MATCH_STATUS)
            gt_class = join.gt_component(CLASS_ID)
            correct = (status == int(MatchStatus.TP)) & join.is_label_correct()
        else:
            correct = np.zeros(0, dtype=bool)
            gt_class = np.zeros(0, dtype=np.float64)

        for class_id in classes:
            num_true_positive = (
                int(np.count_nonzero(correct & (gt_class == class_id))) if correct.size else 0
            )
            num_estimated = int(np.count_nonzero(est_classes_all == class_id))
            num_ground_truth = int(np.count_nonzero(gt_classes_all == class_id))

            precision = _ratio(num_true_positive, num_estimated)
            recall = _ratio(num_true_positive, num_ground_truth)
            accuracy = _ratio(
                num_true_positive,
                num_estimated + num_ground_truth - num_true_positive,
            )
            f1 = (
                _ratio(2.0 * precision * recall, precision + recall)
                if not (np.isnan(precision) or np.isnan(recall))
                else float("nan")
            )

            for target, value in (
                (accuracy_target, accuracy),
                (precision_target, precision),
                (recall_target, recall),
                (f1_target, f1),
            ):
                rows[target].append((int(class_id), float("nan"), value, num_ground_truth))

        return rows
