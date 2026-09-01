"""Prediction metrics: displacement between predicted and observed futures."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define, field

from t4perceval.component import MatchStatus
from t4perceval.core.entity import as_entity_path
from t4perceval.descriptors import (
    CLASS_ID,
    EST_INDEX,
    GT_INDEX,
    MATCH_STATUS,
    MODE_CONFIDENCE,
    WAYPOINTS,
)
from t4perceval.system.metric.base import MetricRow, MetricSystem

if TYPE_CHECKING:
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath
    from t4perceval.system.base import SystemContext
    from t4perceval.system.join import MatchJoin
    from t4perceval.typing import NDArrayF64

__all__ = ("KERNELS", "PathDisplacementSystem")

#: Ways of collapsing several predicted modes into the error that gets reported.
KERNELS = (None, "min", "max", "highest")


def _align_timesteps(waypoints: NDArrayF64, num_timesteps: int) -> NDArrayF64:
    """Make the predicted horizon the same length as the observed one.

    A prediction shorter than the ground truth is held at its last state rather than
    dropped, so a model that predicts less far ahead is penalised for the gap instead of
    being excused from it.
    """
    available = waypoints.shape[2]
    if available == num_timesteps:
        return waypoints
    if available > num_timesteps:
        return waypoints[:, :, :num_timesteps, :]

    padding = np.repeat(waypoints[:, :, -1:, :], num_timesteps - available, axis=2)
    return np.concatenate((waypoints, padding), axis=2)


@define(slots=True)
class PathDisplacementSystem(MetricSystem):
    """ADE, FDE and miss rate between predicted trajectories and what happened.

    Only rows whose classes agree are scored, so a trajectory attached to a
    misclassified object does not contribute.

    :attr:`kernel` decides how the modes of a multi-modal prediction are collapsed:
    ``None`` averages over all of the top ``top_k``, ``"min"`` and ``"max"`` take the mode
    with the smallest or largest total error, and ``"highest"`` takes the most confident
    one. ``None`` therefore reports an average over modes, not the best-of-``k`` minADE
    that some benchmarks report -- which follows the original package.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (EST_INDEX, GT_INDEX, MATCH_STATUS)
    REQUIRES_ESTIMATION: ClassVar[tuple[ComponentDescriptor, ...]] = (
        CLASS_ID,
        WAYPOINTS,
        MODE_CONFIDENCE,
    )
    REQUIRES_GROUND_TRUTH: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID, WAYPOINTS)
    METRIC_NAME: ClassVar[str] = "displacement"

    top_k: int = field(default=3, kw_only=True)
    miss_tolerance: float = field(default=2.0, kw_only=True)
    kernel: str | None = field(default=None, kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        if self.miss_tolerance <= 0.0:
            raise ValueError(f"miss_tolerance must be positive, got {self.miss_tolerance}")
        if self.kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {self.kernel!r}")

    @property
    def targets(self) -> tuple[EntityPath, ...]:
        root = as_entity_path(self.target)
        return (root / "ade", root / "fde", root / "miss_rate")

    def compute(self, join: MatchJoin, ctx: SystemContext) -> dict[EntityPath, list[MetricRow]]:
        ade_target, fde_target, miss_target = self.targets
        ade: list[MetricRow] = []
        fde: list[MetricRow] = []
        miss: list[MetricRow] = []

        classes = self.classes(ctx, join)
        if not len(join.matches) or not len(join.estimation) or not len(join.ground_truth):
            empty = [(int(c), float("nan"), float("nan"), 0) for c in classes]
            return {ade_target: empty, fde_target: list(empty), miss_target: list(empty)}

        status = join.match_component(MATCH_STATUS)
        gt_class = join.gt_component(CLASS_ID)
        scored = (status == int(MatchStatus.TP)) & join.is_label_correct()

        gt_classes_all = join.ground_truth.component(CLASS_ID).values

        for class_id in classes:
            rows = np.flatnonzero(scored & (gt_class == class_id))
            num_ground_truth = int(np.count_nonzero(gt_classes_all == class_id))

            if rows.size == 0:
                undefined = (int(class_id), float("nan"), float("nan"), num_ground_truth)
                ade.append(undefined)
                fde.append(undefined)
                miss.append(undefined)
                continue

            distances = self._distances(join, rows)
            ade.append(
                (int(class_id), float("nan"), float(distances.mean()), num_ground_truth),
            )
            fde.append(
                (
                    int(class_id),
                    float("nan"),
                    float(distances[:, :, -1].mean()),
                    num_ground_truth,
                ),
            )
            miss.append(
                (
                    int(class_id),
                    float("nan"),
                    float(np.count_nonzero(self.miss_tolerance <= distances) / distances.size),
                    num_ground_truth,
                ),
            )

        return {ade_target: ade, fde_target: fde, miss_target: miss}

    def _distances(self, join: MatchJoin, rows: np.ndarray) -> NDArrayF64:
        """Return the xy displacement per (object, kept mode, timestep)."""
        est_waypoints = join.est_component(WAYPOINTS)[rows]
        gt_waypoints = join.gt_component(WAYPOINTS)[rows]
        confidence = join.est_component(MODE_CONFIDENCE)[rows]

        # Most confident first, then only the modes that are actually being evaluated.
        order = np.argsort(confidence, axis=1, kind="stable")[:, ::-1]
        est_waypoints = np.take_along_axis(est_waypoints, order[:, :, None, None], axis=1)
        confidence = np.take_along_axis(confidence, order, axis=1)
        keep = min(self.top_k, est_waypoints.shape[1])
        est_waypoints = est_waypoints[:, :keep]
        confidence = confidence[:, :keep]

        est_waypoints = _align_timesteps(est_waypoints, gt_waypoints.shape[2])

        # The ground truth is single-mode: the one future that happened, compared against
        # every mode that was offered.
        observed = gt_waypoints[:, :1, :, :2]
        distances = np.linalg.norm(est_waypoints[:, :, :, :2] - observed, axis=-1)

        if self.kernel is None:
            return distances

        if self.kernel == "highest":
            chosen = np.zeros(len(rows), dtype=np.int64)
        else:
            totals = distances.sum(axis=2)
            chosen = totals.argmin(axis=1) if self.kernel == "min" else totals.argmax(axis=1)

        return np.take_along_axis(distances, chosen[:, None, None], axis=1)
