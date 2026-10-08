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
    MODE_VALID,
    POSITION,
    TIME_OFFSET,
    TIMESTEP_VALID,
    WAYPOINTS,
)
from t4perceval.system.metric.base import MetricRow, MetricSystem

if TYPE_CHECKING:
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath
    from t4perceval.core.view import EntityView
    from t4perceval.system.base import SystemContext
    from t4perceval.system.matching.join import MatchJoin
    from t4perceval.typing import NDArrayBool, NDArrayF64, NDArrayI64

__all__ = ("PathDisplacementSystem",)


@define(frozen=True, slots=True)
class _Scores:
    """Per-object errors of one class's scored rows.

    Rows whose ground truth has no valid future step are not scored -- nothing happened
    to compare against -- and are left out of every array here.
    """

    ade: NDArrayF64
    fde: NDArrayF64
    misses: NDArrayI64
    elements: NDArrayI64


def _gather_mask(
    join: MatchJoin,
    view: EntityView,
    descriptor: ComponentDescriptor,
    rows: NDArrayI64,
    shape: tuple[int, ...],
    *,
    estimation: bool,
) -> NDArrayBool:
    """Return a validity mask on ``rows``, all ``True`` when the entity does not carry it."""
    if view.component(descriptor) is None:
        return np.ones(shape, dtype=np.bool_)
    gathered = join.est_component(descriptor) if estimation else join.gt_component(descriptor)
    return gathered[rows] > 0.5


def _interpolate(
    knot_time: NDArrayF64,
    knot_xy: NDArrayF64,
    knot_valid: NDArrayBool,
    query: NDArrayF64,
) -> NDArrayF64:
    """Linearly interpolate each mode's valid knots at the query times.

    Shapes: knots ``(R, M, K)`` / ``(R, M, K, 2)``, queries ``(R, T)``, result
    ``(R, M, T, 2)``. Every mode must have at least one valid knot. A query past the last
    valid knot holds its value, as ``np.interp`` does -- the "penalised, not excused"
    policy for a prediction that ends before the ground truth does.
    """
    # Invalid knots sort to the end; the valid ones keep their (increasing) time order.
    time = np.where(knot_valid, knot_time, np.inf)
    order = np.argsort(time, axis=-1, kind="stable")
    time = np.take_along_axis(time, order, axis=-1)
    xy = np.take_along_axis(knot_xy, order[..., None], axis=-2)
    count = knot_valid.sum(axis=-1)  # (R, M), at least 1

    # How many valid knots lie at or before each query: the bracket is (j - 1, j).
    at_or_before = (time[:, :, :, None] <= query[:, None, None, :]).sum(axis=2)  # (R, M, T)
    last = (count - 1)[:, :, None]
    lo = np.clip(at_or_before - 1, 0, last)
    hi = np.clip(at_or_before, 0, last)

    t_lo = np.take_along_axis(time, lo, axis=-1)
    t_hi = np.take_along_axis(time, hi, axis=-1)
    span = t_hi - t_lo
    weight = np.where(span > 0, (query[:, None, :] - t_lo) / np.where(span > 0, span, 1.0), 0.0)
    weight = np.clip(weight, 0.0, 1.0)[..., None]

    xy_lo = np.take_along_axis(xy, lo[..., None], axis=-2)
    xy_hi = np.take_along_axis(xy, hi[..., None], axis=-2)
    return xy_lo + weight * (xy_hi - xy_lo)


@define(slots=True)
class PathDisplacementSystem(MetricSystem):
    """ADE, FDE and miss rate between predicted trajectories and what happened.

    Only true positives whose classes agree are scored. Each valid ground-truth step is
    compared with the prediction interpolated at the same ``time_offset`` -- held at its
    last point past its horizon -- so ``time_offset`` is required on both sides. Padded
    modes and steps, as marked by ``mode_valid`` / ``timestep_valid``, are never scored.

    ADE and FDE are averaged per object, then over objects. See the prediction evaluation
    guide for the full rules.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (EST_INDEX, GT_INDEX, MATCH_STATUS)
    REQUIRES_ESTIMATION: ClassVar[tuple[ComponentDescriptor, ...]] = (
        CLASS_ID,
        POSITION,
        WAYPOINTS,
        MODE_CONFIDENCE,
        TIME_OFFSET,
    )
    REQUIRES_GROUND_TRUTH: ClassVar[tuple[ComponentDescriptor, ...]] = (
        CLASS_ID,
        WAYPOINTS,
        TIME_OFFSET,
    )
    METRIC_NAME: ClassVar[str] = "path_displacement"

    top_k: int = field(default=3, kw_only=True)
    """How many modes to score, most confident first."""

    miss_tolerance: float = field(default=2.0, kw_only=True)
    """A displacement at or above this, in metres, counts as a miss."""

    best_of_k: bool = field(default=False, kw_only=True)
    """Report minADE_k / minFDE_k instead of the mean over the ``top_k`` modes.

    The two minima are taken independently; the miss rate is counted on the minADE mode.
    """

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        if self.miss_tolerance <= 0.0:
            raise ValueError(f"miss_tolerance must be positive, got {self.miss_tolerance}")

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
            scores = self._scores(join, rows) if rows.size else None

            if scores is None or scores.ade.size == 0:
                undefined = (int(class_id), float("nan"), float("nan"), num_ground_truth)
                ade.append(undefined)
                fde.append(undefined)
                miss.append(undefined)
                continue

            ade.append((int(class_id), float("nan"), float(scores.ade.mean()), num_ground_truth))
            fde.append((int(class_id), float("nan"), float(scores.fde.mean()), num_ground_truth))
            miss.append(
                (
                    int(class_id),
                    float("nan"),
                    float(scores.misses.sum() / scores.elements.sum()),
                    num_ground_truth,
                ),
            )

        return {ade_target: ade, fde_target: fde, miss_target: miss}

    def _scores(self, join: MatchJoin, rows: NDArrayI64) -> _Scores:
        """Return the per-object errors of ``rows``."""
        est_waypoints = join.est_component(WAYPOINTS)[rows]  # (R, M, Te, 3)
        num_rows, num_modes, est_steps = est_waypoints.shape[:3]
        gt_waypoints = join.gt_component(WAYPOINTS)[rows]  # (R, 1, Tg, 3)
        gt_steps = gt_waypoints.shape[2]

        # -- the ground truth: the one future that happened, at its valid steps ---------
        # The two masks are independently optional, so a step is valid only when its mode
        # is too -- exactly as on the estimation side.
        gt_valid = (
            _gather_mask(
                join,
                join.ground_truth,
                TIMESTEP_VALID,
                rows,
                (num_rows, gt_waypoints.shape[1], gt_steps),
                estimation=False,
            )[:, 0]
            & _gather_mask(
                join,
                join.ground_truth,
                MODE_VALID,
                rows,
                (num_rows, gt_waypoints.shape[1]),
                estimation=False,
            )[:, :1]
        )
        has_future = gt_valid.any(axis=1)
        if not has_future.any():
            empty = np.empty(0, dtype=np.float64)
            return _Scores(empty, empty, np.empty(0, np.int64), np.empty(0, np.int64))

        keep_rows = np.flatnonzero(has_future)
        rows = rows[keep_rows]
        est_waypoints = est_waypoints[keep_rows]
        gt_xy = gt_waypoints[keep_rows, 0, :, :2]
        gt_valid = gt_valid[keep_rows]
        gt_time = join.gt_component(TIME_OFFSET)[rows]  # (R, Tg)
        num_rows = len(rows)

        # -- the candidate modes, most confident first ----------------------------------
        step_valid = _gather_mask(
            join,
            join.estimation,
            TIMESTEP_VALID,
            rows,
            (num_rows, num_modes, est_steps),
            estimation=True,
        )
        mode_valid = _gather_mask(
            join,
            join.estimation,
            MODE_VALID,
            rows,
            (num_rows, num_modes),
            estimation=True,
        )
        candidate = mode_valid & step_valid.any(axis=2)
        confidence = np.where(candidate, join.est_component(MODE_CONFIDENCE)[rows], -np.inf)

        # Descending and stable as before; the -inf of a non-candidate sorts it last.
        order = np.argsort(confidence, axis=1, kind="stable")[:, ::-1]
        keep = min(self.top_k, num_modes)
        order = order[:, :keep]
        waypoints = np.take_along_axis(est_waypoints, order[:, :, None, None], axis=1)
        step_valid = np.take_along_axis(step_valid, order[:, :, None], axis=1)
        kept = np.arange(keep)[None, :] < np.minimum(candidate.sum(axis=1), keep)[:, None]

        # An estimation that offers no future at all is scored as standing still.
        silent = ~kept.any(axis=1)
        kept[silent, 0] = True
        step_valid[silent, 0] = False

        # -- interpolate each kept mode at the ground truth's times ---------------------
        est_time = join.est_component(TIME_OFFSET)[rows]  # (R, Te)
        position = join.est_component(POSITION)[rows][:, :2]  # (R, 2)

        knot_time = np.concatenate(
            (np.zeros((num_rows, 1)), est_time),
            axis=1,
        )[:, None, :].repeat(keep, axis=1)  # (R, K, 1 + Te)
        knot_xy = np.concatenate(
            (
                np.broadcast_to(position[:, None, None, :], (num_rows, keep, 1, 2)),
                waypoints[..., :2],
            ),
            axis=2,
        )
        # The current position anchors offset 0 unless a real waypoint already sits there.
        anchor = ~(step_valid & (est_time[:, None, :] == 0)).any(axis=2)
        knot_valid = np.concatenate((anchor[:, :, None], step_valid), axis=2)

        predicted = _interpolate(knot_time, knot_xy, knot_valid, gt_time)  # (R, K, Tg, 2)
        distances = np.linalg.norm(predicted - gt_xy[:, None, :, :], axis=-1)  # (R, K, Tg)

        # -- per-object errors ----------------------------------------------------------
        steps = gt_valid.sum(axis=1)  # (R,)
        valid_distances = np.where(gt_valid[:, None, :], distances, 0.0)
        per_mode_ade = valid_distances.sum(axis=2) / steps[:, None]  # (R, K)

        final = gt_steps - 1 - np.argmax(gt_valid[:, ::-1], axis=1)  # last valid GT step
        per_mode_fde = np.take_along_axis(
            distances,
            final[:, None, None].repeat(keep, axis=1),
            axis=2,
        )[:, :, 0]  # (R, K)

        if self.best_of_k:
            # minADE_k and minFDE_k each take their own best mode; the miss rate keeps its
            # per-step definition and is counted on the minADE mode.
            ade = np.where(kept, per_mode_ade, np.inf).min(axis=1)
            fde = np.where(kept, per_mode_fde, np.inf).min(axis=1)
            best = np.where(kept, per_mode_ade, np.inf).argmin(axis=1)
            kept = np.arange(keep)[None, :] == best[:, None]
        else:
            ade = np.where(kept, per_mode_ade, 0.0).sum(axis=1) / kept.sum(axis=1)
            fde = np.where(kept, per_mode_fde, 0.0).sum(axis=1) / kept.sum(axis=1)

        scored = kept[:, :, None] & gt_valid[:, None, :]
        elements = scored.sum(axis=(1, 2))
        misses = (scored & (distances >= self.miss_tolerance)).sum(axis=(1, 2))
        return _Scores(ade, fde, misses.astype(np.int64), elements.astype(np.int64))
