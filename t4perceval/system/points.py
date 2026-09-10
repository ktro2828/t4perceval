"""Bringing an estimation point cloud into its ground truth's row order.

A segmentation metric compares row ``i`` with row ``i``. A model's output is not guaranteed
to enumerate the points in the order the ground truth does -- a node may re-pack the cloud,
a downsampler may drop points -- so the two are aligned by *geometry* first: each
ground-truth point is paired with the estimation point at the same location, and the
estimation is rewritten in that order. This is a stage of its own rather than something a
metric does quietly, for the same reason a coordinate transform is: the correspondence is
data worth keeping and inspecting, not hidden state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from attrs import define, evolve, field
from scipy.spatial import cKDTree

from t4perceval.component import BatchMask
from t4perceval.core.chunk import Chunk, concat_chunks
from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import TimeRange
from t4perceval.descriptors import MASK, POINT
from t4perceval.system.base import EntitySystem, Passthrough, require, require_same_frame

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.system.base import SystemContext
    from t4perceval.typing import NDArrayF64, NDArrayI64

__all__ = ("AlignPointsSystem", "FilterByCoverageSystem")


def _nearest(points: NDArrayF64, reference: NDArrayF64) -> tuple[NDArrayF64, NDArrayI64]:
    """Return, for each of ``points``, the distance to and index of its nearest ``reference``."""
    distances, indices = cKDTree(reference).query(points, k=1)
    return np.atleast_1d(distances), np.atleast_1d(indices).astype(np.int64)


@define(slots=True)
class FilterByCoverageSystem(EntitySystem):
    """Keep the points of one entity that have a counterpart in another.

    Sources are ``(source, reference)``; the mask is over ``source``, true where a
    ``reference`` point lies within :attr:`tolerance`. Its purpose is the ground truth an
    estimation did not cover -- a cropped or downsampled output -- when that gap is to be
    left out of the score rather than counted as a miss. That is an evaluation decision, so
    it is a stage you add, and the mask records exactly which points it dropped::

        covered = FilterByCoverageSystem.between(GT, EST)
        gt_kept = ApplyMaskSystem.of(GT, covered.target)
        aligned = AlignPointsSystem.between(EST, gt_kept.target)
        iou = SegmentationIoUSystem.between(aligned.target, gt_kept.target)
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POINT,)
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...] | Passthrough] = (MASK,)

    tolerance: float = field(default=1e-6, kw_only=True)
    """Largest distance at which a reference point counts as covering a source point."""

    check_frames: bool = field(default=True, kw_only=True)
    """Refuse inputs that state different coordinate frames."""

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 2:
            raise ValueError(
                f"{type(self).__name__} needs exactly two sources (source, reference), "
                f"got {len(self.sources)}",
            )
        if self.tolerance < 0.0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance}")

    @classmethod
    def between(
        cls,
        source: EntityPathLike,
        reference: EntityPathLike,
        *,
        target: EntityPathLike | None = None,
        **params: Any,
    ) -> Self:
        """Mask the points of ``source`` by whether ``reference`` covers them.

        The target defaults to ``<source>/filter/coverage``, like every other filter.
        """
        path = as_entity_path(source)
        return cls(
            (path, reference),
            target if target is not None else path / "filter" / "coverage",
            **params,
        )

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        source, reference = self.sources
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)
        times = ctx.store.times(source, ctx.timeline)

        pieces: list[Chunk] = []
        for time in (int(time) for time in times[time_range.contains(times)]):
            single = TimeRange.single(time)
            view = ctx.store.range(source, timeline=ctx.timeline, time_range=single)
            other = ctx.store.range(reference, timeline=ctx.timeline, time_range=single)
            if len(view):
                require(view, *self.REQUIRES)
            if len(other):
                require(other, *self.REQUIRES)
            if self.check_frames:
                require_same_frame(view, other)

            if not len(view) or not len(other):
                keep = np.zeros(len(view), dtype=np.bool_)
            else:
                distances, _ = _nearest(
                    view.component(POINT).values,  # type: ignore[union-attr]
                    other.component(POINT).values,  # type: ignore[union-attr]
                )
                keep = distances <= self.tolerance

            chunk = view.to_chunk()
            pieces.append(
                Chunk(
                    self.target,
                    chunk.indexes,
                    chunk.offsets,
                    {MASK: BatchMask(keep)},
                    frame_id=chunk.frame_id,
                ),
            )
        return (concat_chunks(pieces),) if pieces else ()


@define(slots=True)
class AlignPointsSystem(EntitySystem):
    """Rewrite an estimation point cloud in its ground truth's row order.

    Sources are ``(estimation, ground_truth)``. At every time, each ground-truth point is
    paired with the nearest estimation point within :attr:`tolerance`, and the estimation's
    rows -- every column it carries -- are written to the target in that order. The result
    is row-aligned with the ground truth, which is what a segmentation metric requires.

    Estimation points that no ground-truth point claims are dropped: a prediction on a point
    the ground truth does not label cannot be scored. A ground-truth point with no estimation
    point within tolerance raises -- whether such a point is a miss or is outside the
    estimation's scope is an evaluation decision, made explicit by narrowing the ground
    truth first with :class:`FilterByCoverageSystem`. An estimation point that is the
    nearest to two ground-truth points raises as well: coincident points cannot be told
    apart by geometry, so the order is undefined.

    Not to be confused with :mod:`t4perceval.align`, which pairs *frames* by timestamp.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POINT,)
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...] | Passthrough] = Passthrough(0)

    tolerance: float = field(default=1e-6, kw_only=True)
    """Largest distance, in the points' frame, at which two points are the same point."""

    check_frames: bool = field(default=True, kw_only=True)
    """Refuse inputs that state different coordinate frames."""

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 2:
            raise ValueError(
                f"{type(self).__name__} needs exactly two sources "
                f"(estimation, ground truth), got {len(self.sources)}",
            )
        if self.tolerance < 0.0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance}")

    @classmethod
    def between(
        cls,
        estimation: EntityPathLike,
        ground_truth: EntityPathLike,
        *,
        target: EntityPathLike | None = None,
        **params: Any,
    ) -> Self:
        """Align ``estimation`` to ``ground_truth``, writing to ``target``.

        The target defaults to ``<estimation>/aligned``, keeping the result beside the
        entity it came from.
        """
        path = as_entity_path(estimation)
        return cls(
            (path, ground_truth), target if target is not None else path / "aligned", **params
        )

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        estimation, ground_truth = self.sources
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)

        times = np.union1d(
            ctx.store.times(estimation, ctx.timeline),
            ctx.store.times(ground_truth, ctx.timeline),
        )
        pieces: list[Chunk] = []
        for time in (int(time) for time in times[time_range.contains(times)]):
            single = TimeRange.single(time)
            est_view = ctx.store.range(estimation, timeline=ctx.timeline, time_range=single)
            gt_view = ctx.store.range(ground_truth, timeline=ctx.timeline, time_range=single)
            if len(est_view):
                require(est_view, *self.REQUIRES)
            if len(gt_view):
                require(gt_view, *self.REQUIRES)
            if self.check_frames:
                require_same_frame(est_view, gt_view)

            where = f"{ctx.timeline.name}={time}"
            if not len(est_view):
                if len(gt_view):
                    raise ValueError(
                        f"{ground_truth} has {len(gt_view)} point(s) at {where} but {estimation} "
                        f"has none to align to them",
                    )
                continue  # nothing on either side: no frame to write

            chunk = est_view.to_chunk()
            if not len(gt_view):
                order = np.empty(
                    0, dtype=np.int64
                )  # nothing to score; keep the frame, drop the rows
            else:
                order = self._correspondence(
                    est_view.component(POINT).values,  # type: ignore[union-attr]
                    gt_view.component(POINT).values,  # type: ignore[union-attr]
                    estimation=estimation,
                    ground_truth=ground_truth,
                    where=where,
                )
            pieces.append(evolve(chunk.select(order), entity_path=self.target))

        return (concat_chunks(pieces),) if pieces else ()

    def _correspondence(
        self,
        est_points: NDArrayF64,
        gt_points: NDArrayF64,
        *,
        estimation: EntityPath,
        ground_truth: EntityPath,
        where: str,
    ) -> NDArrayI64:
        """Return, for each ground-truth row, the estimation row at the same location."""
        distances, indices = _nearest(gt_points, est_points)

        unmatched = distances > self.tolerance
        if unmatched.any():
            raise ValueError(
                f"{ground_truth} has {int(unmatched.sum())} point(s) with no {estimation} point "
                f"within {self.tolerance:g} at {where} (nearest is {distances[unmatched].min():.3g} "
                f"away); the two clouds do not describe the same points. To score only the "
                f"points the estimation covers, narrow the ground truth first with "
                f"FilterByCoverageSystem",
            )
        used, counts = np.unique(indices, return_counts=True)
        if (counts > 1).any():
            raise ValueError(
                f"{int((counts > 1).sum())} {estimation} point(s) are the nearest to more than one "
                f"{ground_truth} point at {where}; coincident points cannot be told apart, so the "
                f"order is undefined",
            )
        return indices
