"""Matching systems.

Matching pairs an estimation stream against a ground-truth stream and records the verdict
as a :class:`~t4perceval.archetype.MatchResults` chunk. Because the verdict is data --
row indices, a score, a TP/FP/FN status -- it can be stored, re-read and re-analysed,
which is what ``DynamicObjectWithPerceptionResult`` could not do: it held live object
references.

Every matcher is only its score matrix: :class:`MatchingSystem` implements the frame loop,
the feasibility rules and the assignment once. The modes differ in *what* they measure and
in whether a higher score is better.

Each mode writes to its own ``/matching/<mode>`` entity, so several can run over the same
frame and be compared afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from attrs import Factory, define, field
from scipy.optimize import linear_sum_assignment

from t4perceval import geometry
from t4perceval.archetype.matching import MatchResults
from t4perceval.component import MatchStatus
from t4perceval.core.chunk import concat_chunks
from t4perceval.core.timeline import TimePoint, TimeRange
from t4perceval.descriptors import (
    CLASS_ID,
    EST_INDEX,
    GT_INDEX,
    MATCH_STATUS,
    MATCHING_SCORE,
    POSITION,
    QUATERNION,
    ROI,
    SIZE,
    THRESHOLD,
)
from t4perceval.system.base import (
    EntitySystem,
    SystemContext,
    require,
    require_same_frame,
    resolve_frame,
)
from t4perceval.system.threshold import Thresholds

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.view import EntityView
    from t4perceval.typing import NDArrayBool, NDArrayF64

__all__ = (
    "CenterDistanceBEVMatchingSystem",
    "CenterDistanceMatchingSystem",
    "IoU3DMatchingSystem",
    "IoUBEVMatchingSystem",
    "IoURoiMatchingSystem",
    "MatchingSystem",
    "PlaneDistanceMatchingSystem",
)

#: Components describing a 3D box, needed by every mode that looks at the box's extent.
_BOX_3D: tuple[ComponentDescriptor, ...] = (POSITION, QUATERNION, SIZE, CLASS_ID)


@define(slots=True)
class MatchingSystem(EntitySystem):
    """Base for a system that pairs an estimation entity against a ground-truth entity.

    A subclass declares its ``REQUIRES``, a :attr:`MATCHING_NAME` used to build the
    default target path, whether :attr:`HIGHER_IS_BETTER`, a
    :attr:`DEFAULT_THRESHOLD`, and :meth:`score_matrix`.

    A globally optimal one-to-one assignment is solved per frame, so a good pair is not
    lost to a greedy earlier choice.
    """

    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = (
        EST_INDEX,
        GT_INDEX,
        MATCHING_SCORE,
        MATCH_STATUS,
        THRESHOLD,
    )

    #: Default last segment of the target path, ``/matching/<MATCHING_NAME>``.
    MATCHING_NAME: ClassVar[str] = "matching"

    #: Whether a larger score means a better match, as for IoU rather than a distance.
    HIGHER_IS_BETTER: ClassVar[bool] = False

    #: Threshold used when the caller does not give one.
    DEFAULT_THRESHOLD: ClassVar[float] = 1.0

    threshold: Thresholds = field(
        default=Factory(lambda self: type(self).DEFAULT_THRESHOLD, takes_self=True),
        converter=Thresholds.coerce,
        kw_only=True,
    )
    class_agnostic: bool = field(default=False, kw_only=True)

    check_frames: bool = field(default=True, kw_only=True)
    """Whether to refuse inputs that state different coordinate frames.

    On by default because the alternative is silent: distances between two frames are
    numbers, not errors, so the resulting metric looks plausible. Turn it off only when
    the frames are known to coincide despite their names -- the same escape hatch
    :func:`~t4perceval.evaluation.build_evaluation_store_from` offers at assembly time.
    """

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 2:
            raise ValueError(
                f"{type(self).__name__} needs exactly two sources "
                f"(estimation, ground truth), got {len(self.sources)}",
            )
        thresholds = (self.threshold.default, *(t for _, t in self.threshold.by_class))
        if self.HIGHER_IS_BETTER:
            if not all(0.0 < value <= 1.0 for value in thresholds):
                raise ValueError(
                    f"{type(self).__name__} thresholds are overlap ratios and must lie in "
                    f"(0, 1], got {list(thresholds)}",
                )
        elif not all(value > 0.0 for value in thresholds):
            raise ValueError(
                f"{type(self).__name__} thresholds are distances and must be positive, "
                f"got {list(thresholds)}",
            )

    @classmethod
    def between(
        cls,
        estimation: EntityPathLike,
        ground_truth: EntityPathLike,
        *,
        target: EntityPathLike | None = None,
        **params: Any,
    ) -> Self:
        """Build a matcher between an estimation and a ground-truth entity.

        The target defaults to ``/matching/<MATCHING_NAME>``. ``params`` are the
        subclass's own fields.
        """
        return cls(
            (estimation, ground_truth),
            target if target is not None else f"/matching/{cls.MATCHING_NAME}",
            **params,
        )

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        """Return the score of every pair, with shape ``(len(est_view), len(gt_view))``."""
        raise NotImplementedError

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        estimation, ground_truth = self.sources
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)

        times = sorted(
            set(ctx.store.times(estimation, ctx.timeline).tolist())
            | set(ctx.store.times(ground_truth, ctx.timeline).tolist()),
        )
        selected = [time for time in times if bool(time_range.contains(time))]
        if not selected:
            return ()

        pieces: list[Chunk] = []
        for time in selected:
            single = TimeRange.single(time)
            est_view = ctx.store.range(estimation, timeline=ctx.timeline, time_range=single)
            gt_view = ctx.store.range(ground_truth, timeline=ctx.timeline, time_range=single)
            if len(est_view):
                require(est_view, *self.REQUIRES)
            if len(gt_view):
                require(gt_view, *self.REQUIRES)

            frame_id = (
                require_same_frame(est_view, gt_view)
                if self.check_frames
                else resolve_frame(est_view, gt_view)
            )

            result = self._match_frame(est_view, gt_view, ctx)
            pieces.append(
                result.to_chunk(
                    self.target,
                    at=TimePoint(((ctx.timeline, time),)),
                    frame_id=frame_id,
                ),
            )

        return (concat_chunks(pieces),)

    def _match_frame(
        self,
        est_view: EntityView,
        gt_view: EntityView,
        ctx: SystemContext,
    ) -> MatchResults:
        num_est = len(est_view)
        num_gt = len(gt_view)

        est_rows: list[int] = []
        gt_rows: list[int] = []
        scores: list[float] = []
        statuses: list[int] = []
        thresholds: list[float] = []

        matched_est: set[int] = set()
        matched_gt: set[int] = set()

        # Resolved per row so that a per-class threshold is recorded as the value that
        # actually applied. A matched or missed ground truth is scored by its own class;
        # a false positive has none, so the estimation's class decides.
        est_thresholds = (
            self.threshold.resolve(est_view.component(CLASS_ID).values, ctx.labels)
            if num_est
            else np.empty(0, dtype=np.float64)
        )
        gt_thresholds = (
            self.threshold.resolve(gt_view.component(CLASS_ID).values, ctx.labels)
            if num_gt
            else np.empty(0, dtype=np.float64)
        )

        if num_est and num_gt:
            score = self.score_matrix(est_view, gt_view)
            if score.shape != (num_est, num_gt):
                raise ValueError(
                    f"{type(self).__name__}.score_matrix() returned shape {score.shape}, "
                    f"expected {(num_est, num_gt)}",
                )

            feasible = self._feasible(score, est_view, gt_view, ctx)
            cost = -score if self.HIGHER_IS_BETTER else score

            # `linear_sum_assignment` cannot represent forbidden pairs, so infeasible
            # entries get a cost above any feasible one and are rejected afterwards.
            rejected = float(cost[feasible].max()) + 1.0 if feasible.any() else 0.0
            padded = np.where(feasible, cost, rejected)

            for est_row, gt_row in zip(*linear_sum_assignment(padded)):
                if not feasible[est_row, gt_row]:
                    continue
                est_rows.append(int(est_row))
                gt_rows.append(int(gt_row))
                scores.append(float(score[est_row, gt_row]))
                statuses.append(int(MatchStatus.TP))
                thresholds.append(float(gt_thresholds[gt_row]))
                matched_est.add(int(est_row))
                matched_gt.add(int(gt_row))

        for est_row in range(num_est):
            if est_row not in matched_est:
                est_rows.append(est_row)
                gt_rows.append(-1)
                scores.append(float("nan"))
                statuses.append(int(MatchStatus.FP))
                thresholds.append(float(est_thresholds[est_row]))

        for gt_row in range(num_gt):
            if gt_row not in matched_gt:
                est_rows.append(-1)
                gt_rows.append(gt_row)
                scores.append(float("nan"))
                statuses.append(int(MatchStatus.FN))
                thresholds.append(float(gt_thresholds[gt_row]))

        if not est_rows:
            return MatchResults.empty()

        return MatchResults(
            est_index=np.asarray(est_rows, dtype=np.int64),
            gt_index=np.asarray(gt_rows, dtype=np.int64),
            matching_score=np.asarray(scores, dtype=np.float64),
            match_status=np.asarray(statuses, dtype=np.int8),
            threshold=np.asarray(thresholds, dtype=np.float64),
        )

    def _feasible(
        self,
        score: NDArrayF64,
        est_view: EntityView,
        gt_view: EntityView,
        ctx: SystemContext,
    ) -> NDArrayBool:
        """Return which pairs may be assigned at all."""
        gt_class = gt_view.component(CLASS_ID).values
        # Per-class thresholds follow the ground truth, which is the authority on class.
        threshold = self.threshold.resolve(gt_class, ctx.labels)[None, :]

        feasible = np.isfinite(score)
        feasible &= score >= threshold if self.HIGHER_IS_BETTER else score <= threshold

        if not self.class_agnostic:
            est_class = est_view.component(CLASS_ID).values
            feasible &= est_class[:, None] == gt_class[None, :]

        return feasible


@define(slots=True)
class CenterDistanceMatchingSystem(MatchingSystem):
    """Match by the 3D distance between box centres."""

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POSITION, CLASS_ID)
    MATCHING_NAME: ClassVar[str] = "center_distance"
    DEFAULT_THRESHOLD: ClassVar[float] = 1.0

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        est_position = est_view.component(POSITION).values
        gt_position = gt_view.component(POSITION).values
        return np.linalg.norm(est_position[:, None, :] - gt_position[None, :, :], axis=-1)


@define(slots=True)
class CenterDistanceBEVMatchingSystem(MatchingSystem):
    """Match by the distance between box centres in the xy plane.

    Kept apart from :class:`CenterDistanceMatchingSystem` because a matching mode names
    the entity a metric later reads, so the two must not share a target. This is the mode
    the original package's ``center_distance_bev_thresholds`` configured.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POSITION, CLASS_ID)
    MATCHING_NAME: ClassVar[str] = "center_distance_bev"
    DEFAULT_THRESHOLD: ClassVar[float] = 1.0

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        est_xy = est_view.component(POSITION).values[:, :2]
        gt_xy = gt_view.component(POSITION).values[:, :2]
        return np.linalg.norm(est_xy[:, None, :] - gt_xy[None, :, :], axis=-1)


@define(slots=True)
class PlaneDistanceMatchingSystem(MatchingSystem):
    """Match by the distance between the boxes' nearest faces.

    Two boxes can agree closely on the face the sensor observes while disagreeing about
    the far side, which is why this mode exists alongside centre distance: it scores what
    the perception system could actually see. See
    :func:`t4perceval.geometry.pairwise_plane_distance`.

    Positions must be expressed in the frame the distance from the origin is measured in
    -- normally ``base_link``, which puts the ego at the origin.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = _BOX_3D
    MATCHING_NAME: ClassVar[str] = "plane_distance"
    DEFAULT_THRESHOLD: ClassVar[float] = 2.0

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        return geometry.pairwise_plane_distance(
            est_view.component(POSITION).values,
            est_view.component(QUATERNION).values,
            est_view.component(SIZE).values,
            gt_view.component(POSITION).values,
            gt_view.component(QUATERNION).values,
            gt_view.component(SIZE).values,
        )


@define(slots=True)
class IoUBEVMatchingSystem(MatchingSystem):
    """Match 3D boxes by the IoU of their footprints.

    This is the 2D IoU the original package computed for 3D tasks -- its
    ``iou_2d_thresholds`` -- measured on the rotated footprint rather than an
    axis-aligned box. For image-plane IoU of 2D detections, use
    :class:`IoURoiMatchingSystem`.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = _BOX_3D
    MATCHING_NAME: ClassVar[str] = "iou_bev"
    HIGHER_IS_BETTER: ClassVar[bool] = True
    DEFAULT_THRESHOLD: ClassVar[float] = 0.5

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        return geometry.pairwise_bev_iou(
            est_view.component(POSITION).values,
            est_view.component(QUATERNION).values,
            est_view.component(SIZE).values,
            gt_view.component(POSITION).values,
            gt_view.component(QUATERNION).values,
            gt_view.component(SIZE).values,
        )


@define(slots=True)
class IoU3DMatchingSystem(MatchingSystem):
    """Match 3D boxes by the IoU of their volumes."""

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = _BOX_3D
    MATCHING_NAME: ClassVar[str] = "iou_3d"
    HIGHER_IS_BETTER: ClassVar[bool] = True
    DEFAULT_THRESHOLD: ClassVar[float] = 0.5

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        return geometry.pairwise_volume_iou(
            est_view.component(POSITION).values,
            est_view.component(QUATERNION).values,
            est_view.component(SIZE).values,
            gt_view.component(POSITION).values,
            gt_view.component(QUATERNION).values,
            gt_view.component(SIZE).values,
        )


@define(slots=True)
class IoURoiMatchingSystem(MatchingSystem):
    """Match 2D detections by the IoU of their image-plane regions.

    This is the ``iou_2d_thresholds`` of the original package's 2D tasks. It requires
    :data:`~t4perceval.descriptors.ROI` rather than a 3D box, which is why it is a
    separate system from :class:`IoUBEVMatchingSystem` instead of one class that inspects
    which components happen to be present.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (ROI, CLASS_ID)
    MATCHING_NAME: ClassVar[str] = "iou_roi"
    HIGHER_IS_BETTER: ClassVar[bool] = True
    DEFAULT_THRESHOLD: ClassVar[float] = 0.5

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        return geometry.pairwise_roi_iou(
            est_view.component(ROI).values,
            gt_view.component(ROI).values,
        )
