"""Tracking metrics: the CLEAR family."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define

from t4perceval.component import MatchStatus
from t4perceval.core.entity import as_entity_path
from t4perceval.descriptors import (
    CLASS_ID,
    EST_INDEX,
    GT_INDEX,
    INSTANCE_ID,
    MATCH_STATUS,
    MATCHING_SCORE,
    THRESHOLD,
)
from t4perceval.system.metric.base import MetricRow, MetricSystem

if TYPE_CHECKING:
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath
    from t4perceval.system.base import SystemContext
    from t4perceval.system.join import MatchJoin

__all__ = ("ClearSystem",)


@define(slots=True)
class ClearSystem(MetricSystem):
    """MOTA, MOTP and ID switches, from one pass over the tracked identities.

    The three share their expensive part -- following which estimation held each
    ground-truth identity from frame to frame -- so one system computes them together and
    writes each to its own entity under :attr:`target`.

    An **ID switch** is a ground-truth object that was held by one estimation in the
    previous frame and by a different one now. Counting it needs the instance ids of both
    sides, which is why this metric requires more than a detection metric does.

    Note that **MOTP inherits the direction of its matching mode**: it is the mean matching
    score over true positives, so it is better when small for a distance mode and better
    when large for an IoU mode. The entity path of the matching it read says which.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (
        EST_INDEX,
        GT_INDEX,
        MATCH_STATUS,
        MATCHING_SCORE,
        THRESHOLD,
    )
    REQUIRES_ESTIMATION: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID, INSTANCE_ID)
    REQUIRES_GROUND_TRUTH: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID, INSTANCE_ID)
    METRIC_NAME: ClassVar[str] = "clear"

    @property
    def targets(self) -> tuple[EntityPath, ...]:
        root = as_entity_path(self.target)
        return (root / "mota", root / "motp", root / "id_switch")

    def compute(self, join: MatchJoin, ctx: SystemContext) -> dict[EntityPath, list[MetricRow]]:
        mota_target, motp_target, switch_target = self.targets
        mota: list[MetricRow] = []
        motp: list[MetricRow] = []
        switches: list[MetricRow] = []

        classes = self.classes(ctx, join)
        if not len(join.matches):
            empty = [(int(c), float("nan"), float("nan"), 0) for c in classes]
            return {mota_target: empty, motp_target: list(empty), switch_target: list(empty)}

        status = join.match_component(MATCH_STATUS)
        score = join.match_component(MATCHING_SCORE)
        thresholds = join.match_component(THRESHOLD)
        times = join.matches.times(join.timeline)
        est_class = join.est_component(CLASS_ID)
        gt_class = join.gt_component(CLASS_ID)
        est_instance = join.est_component(INSTANCE_ID)
        gt_instance = join.gt_component(INSTANCE_ID)
        is_true_positive = (status == int(MatchStatus.TP)) & join.is_label_correct()
        is_false_positive = status == int(MatchStatus.FP)

        gt_classes_all = (
            join.ground_truth.component(CLASS_ID).values
            if len(join.ground_truth)
            else np.empty(0, dtype=np.int32)
        )
        est_classes_all = (
            join.estimation.component(CLASS_ID).values
            if len(join.estimation)
            else np.empty(0, dtype=np.int32)
        )

        for class_id in classes:
            true_positive = is_true_positive & (gt_class == class_id)
            false_positive = is_false_positive & join.has_estimation & (est_class == class_id)

            num_true_positive = int(np.count_nonzero(true_positive))
            num_false_positive = int(np.count_nonzero(false_positive))
            score_sum = float(np.nansum(score[true_positive])) if num_true_positive else 0.0
            num_switch = self._count_switches(
                times[true_positive],
                gt_instance[true_positive],
                est_instance[true_positive],
            )
            num_ground_truth = int(np.count_nonzero(gt_classes_all == class_id))

            mentions = (
                true_positive | false_positive | (join.has_ground_truth & (gt_class == class_id))
            )
            class_thresholds = thresholds[mentions]
            threshold = float(class_thresholds[0]) if class_thresholds.size else float("nan")

            mota_value = (
                max(
                    0.0,
                    (num_true_positive - num_false_positive - num_switch) / num_ground_truth,
                )
                if num_ground_truth
                else float("nan")
            )
            motp_value = score_sum / num_true_positive if num_true_positive else float("nan")

            # ID switches are a count, not a ratio, so they survive a zero support -- but a
            # class that was never seen at all reports nothing rather than a reassuring 0.
            observed = num_ground_truth or int(np.count_nonzero(est_classes_all == class_id))
            switch_value = float(num_switch) if observed else float("nan")

            mota.append((int(class_id), threshold, mota_value, num_ground_truth))
            motp.append((int(class_id), threshold, motp_value, num_ground_truth))
            switches.append((int(class_id), threshold, switch_value, num_ground_truth))

        return {mota_target: mota, motp_target: motp, switch_target: switches}

    @staticmethod
    def _count_switches(
        times: np.ndarray,
        gt_instance: np.ndarray,
        est_instance: np.ndarray,
    ) -> int:
        """Count ground-truth objects that changed hands between consecutive frames.

        Only the previous frame is compared, matching the original package: an identity
        that is lost and later recovered by the original estimation is not a switch.
        """
        if times.size == 0:
            return 0

        order = np.argsort(times, kind="stable")
        switches = 0
        previous: dict[int, int] = {}
        current: dict[int, int] = {}
        current_time: int | None = None

        for index in order:
            time = int(times[index])
            if current_time is None:
                current_time = time
            elif time != current_time:
                previous, current, current_time = current, {}, time

            ground_truth = int(gt_instance[index])
            estimation = int(est_instance[index])
            held_by = previous.get(ground_truth)
            if held_by is not None and held_by != estimation:
                switches += 1
            current[ground_truth] = estimation

        return switches
