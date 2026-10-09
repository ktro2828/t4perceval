"""The shared base of every matcher: the frame loop, the feasibility rules, the assignment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias

import numpy as np
from attrs import Factory, converters, define, field
from scipy.optimize import linear_sum_assignment

from t4perceval import geometry
from t4perceval.archetype.matching import MatchResults
from t4perceval.component import MatchStatus
from t4perceval.core.chunk import concat_chunks
from t4perceval.core.timeline import TimePoint, TimeRange
from t4perceval.descriptors import (
    CLASS_ID,
    CONFIDENCE,
    EST_INDEX,
    GT_INDEX,
    MATCH_STATUS,
    MATCHING_SCORE,
    POSITION,
    QUATERNION,
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
from t4perceval.system.matching.threshold import Thresholds

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.view import EntityView
    from t4perceval.typing import ArrayLike, NDArrayBool, NDArrayF64, NDArrayI64


#: How pairs are chosen among the feasible ones; see :attr:`MatchingSystem.assignment`.
Assignment: TypeAlias = Literal["hungarian", "greedy"]

_ASSIGNMENTS: tuple[str, ...] = ("hungarian", "greedy")


def confidence_order(confidence: ArrayLike, rows: ArrayLike) -> NDArrayI64:
    """Return the positions of ``confidence`` from most to least confident.

    Ties go to the smaller ``rows`` value -- the estimation's row, so the one logged first.
    Greedy matching visits estimations in this order and average precision ranks them in
    it, and the two must agree: were ties broken differently, a tied estimation could be
    credited with the match while its twin is ranked ahead of it as a false positive.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    return np.lexsort((np.asarray(rows, dtype=np.int64), -confidence)).astype(np.int64)


#: Components describing a 3D box, needed by every mode that looks at the box's extent.
BOX_3D: tuple[ComponentDescriptor, ...] = (POSITION, QUATERNION, SIZE, CLASS_ID)


@define(slots=True)
class MatchingSystem(EntitySystem):
    """Base for a system that pairs an estimation entity against a ground-truth entity.

    A subclass declares its ``REQUIRES``, a :attr:`MATCHING_NAME` used to build the
    default target path, whether :attr:`HIGHER_IS_BETTER`, a
    :attr:`DEFAULT_THRESHOLD`, and :meth:`score_matrix`.

    Pairs are one-to-one per frame and chosen by :attr:`assignment`: a globally optimal
    linear-sum (Hungarian) assignment by default, or nuScenes-style greedy matching in
    descending estimation confidence.
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

    max_matchable_distance: Thresholds | None = field(
        default=None,
        converter=converters.optional(Thresholds.coerce),
        kw_only=True,
    )
    """Largest 3D distance between centres, in metres, at which a pair may be assigned.

    A gate applied on top of :attr:`threshold`, whatever the mode scores: two large boxes
    can overlap well enough to pass an IoU threshold while their centres are metres apart.
    A number, or :class:`Thresholds` keyed by ground-truth class like :attr:`threshold`.
    ``None`` disables the gate. Modes without :data:`~t4perceval.descriptors.POSITION`
    reject it.
    """

    assignment: Assignment = field(default="hungarian", kw_only=True)
    """How pairs are chosen among the feasible ones.

    ``"hungarian"`` solves a globally optimal linear-sum assignment, so a good pair is not
    lost to an earlier choice. ``"greedy"`` walks the estimations in descending
    :data:`~t4perceval.descriptors.CONFIDENCE` and gives each the best-scoring ground truth
    still free, as nuScenes does for average precision: a confident estimate keeps its
    match even when a less confident one lies closer. Greedy requires ``CONFIDENCE`` on the
    estimation.
    """

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
        if self.assignment not in _ASSIGNMENTS:
            raise ValueError(
                f"{type(self).__name__} assignment must be one of {list(_ASSIGNMENTS)}, "
                f"got {self.assignment!r}",
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

        if self.max_matchable_distance is not None:
            if POSITION not in self.REQUIRES:
                raise ValueError(
                    f"{type(self).__name__} does not require {POSITION.component!r}, "
                    "so it has no centre for max_matchable_distance to gate on",
                )
            limits = (
                self.max_matchable_distance.default,
                *(t for _, t in self.max_matchable_distance.by_class),
            )
            if not all(value > 0.0 for value in limits):
                raise ValueError(
                    f"{type(self).__name__} max_matchable_distance must be positive, "
                    f"got {list(limits)}",
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

    def requires_for(self, index: int) -> tuple[ComponentDescriptor, ...]:
        """Return :attr:`REQUIRES`, plus the estimation's confidence for greedy matching."""
        if index == 0 and self.assignment == "greedy":
            return (*self.REQUIRES, CONFIDENCE)
        return self.REQUIRES

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
                require(est_view, *self.requires_for(0))
            if len(gt_view):
                require(gt_view, *self.requires_for(1))

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
            pairs = (
                self._assign_greedy(score, feasible, est_view)
                if self.assignment == "greedy"
                else self._assign_hungarian(score, feasible)
            )
            for est_row, gt_row in pairs:
                est_rows.append(est_row)
                gt_rows.append(gt_row)
                scores.append(float(score[est_row, gt_row]))
                statuses.append(int(MatchStatus.TP))
                thresholds.append(float(gt_thresholds[gt_row]))
                matched_est.add(est_row)
                matched_gt.add(gt_row)

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

    def _assign_hungarian(
        self,
        score: NDArrayF64,
        feasible: NDArrayBool,
    ) -> list[tuple[int, int]]:
        """Return the feasible pairs of a globally optimal linear-sum assignment."""
        num_est, num_gt = score.shape
        cost = -score if self.HIGHER_IS_BETTER else score

        # `linear_sum_assignment` cannot represent forbidden pairs, so infeasible
        # entries get a finite penalty and are rejected afterwards. The penalty must
        # outweigh any sum of feasible costs, not just the largest one: otherwise one
        # cheap pair plus a penalty can undercut two feasible pairs, and a match is
        # lost. With costs shifted into [0, span], every assignment uses at most
        # min(N, M) pairs, so a penalty above min(N, M) * span means more feasible
        # pairs always wins, and among equally many the cheaper total still does.
        if feasible.any():
            shifted = cost - float(cost[feasible].min())
            span = float(shifted[feasible].max())
            rejected = span * min(num_est, num_gt) + 1.0
        else:
            shifted, rejected = cost, 0.0
        padded = np.where(feasible, shifted, rejected)

        return [
            (int(est_row), int(gt_row))
            for est_row, gt_row in zip(*linear_sum_assignment(padded))
            if feasible[est_row, gt_row]
        ]

    def _assign_greedy(
        self,
        score: NDArrayF64,
        feasible: NDArrayBool,
        est_view: EntityView,
    ) -> list[tuple[int, int]]:
        """Return the pairs of nuScenes-style greedy matching in confidence order."""
        confidence = est_view.component(CONFIDENCE).values
        order = confidence_order(confidence, np.arange(len(confidence)))

        # Infeasible pairs, and later taken ground truths, get the worst possible value so
        # that argmin / argmax never prefers them; ties go to the lowest ground-truth row.
        worst = -np.inf if self.HIGHER_IS_BETTER else np.inf
        available = np.where(feasible, score, worst)
        pick = np.argmax if self.HIGHER_IS_BETTER else np.argmin

        pairs: list[tuple[int, int]] = []
        for est_row in order:
            row = available[est_row]
            gt_row = int(pick(row))
            if row[gt_row] == worst:
                continue
            pairs.append((int(est_row), gt_row))
            available[:, gt_row] = worst
        # Laid out by estimation row, as the Hungarian pairs are.
        return sorted(pairs)

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

        if self.max_matchable_distance is not None:
            distance = geometry.pairwise_center_distance(
                est_view.component(POSITION).values,
                gt_view.component(POSITION).values,
            )
            limit = self.max_matchable_distance.resolve(gt_class, ctx.labels)[None, :]
            # A non-finite distance fails the comparison, so it is never assigned either.
            feasible &= distance <= limit

        if not self.class_agnostic:
            est_class = est_view.component(CLASS_ID).values
            feasible &= est_class[:, None] == gt_class[None, :]

        return feasible
