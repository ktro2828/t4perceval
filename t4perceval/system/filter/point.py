"""Filtering the points of one cloud by whether the other cloud covers them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from attrs import define, field

from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import TimeRange
from t4perceval.descriptors import POINT
from t4perceval.geometry import nearest_points
from t4perceval.system.base import require, require_same_frame
from t4perceval.system.filter.base import MaskSystem

if TYPE_CHECKING:
    from typing_extensions import Self

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.view import EntityView
    from t4perceval.system.base import SystemContext
    from t4perceval.typing import NDArrayBool

__all__ = ("FilterByCoverageSystem",)


@define(slots=True)
class FilterByCoverageSystem(MaskSystem):
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
    FILTER_NAME: ClassVar[str] = "coverage"

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
            target if target is not None else path / "filter" / cls.FILTER_NAME,
            **params,
        )

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        """Pair each frame's points with the reference at that frame."""
        _, reference = self.sources
        chunk = view.to_chunk()
        index = chunk.index(ctx.timeline)
        if index is None:
            raise ValueError(
                f"{self.sources[0]} has no {ctx.timeline.name!r} index, so its frames cannot "
                f"be paired with {reference}",
            )

        keep = np.zeros(len(view), dtype=np.bool_)
        for partition in range(chunk.num_partitions):
            start, stop = int(chunk.offsets[partition]), int(chunk.offsets[partition + 1])
            if stop == start:
                continue
            other = ctx.store.range(
                reference,
                timeline=ctx.timeline,
                time_range=TimeRange.single(int(index.times[partition])),
            )
            if not len(other):
                continue
            require(other, *self.REQUIRES)
            if self.check_frames:
                require_same_frame(view, other)
            distances, _ = nearest_points(
                view.component(POINT).values[start:stop],
                other.component(POINT).values,
            )
            keep[start:stop] = distances <= self.tolerance
        return keep
