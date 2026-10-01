"""Filters over the points of a cloud rather than over objects.

:class:`FilterPointsByCoverageSystem` asks whether another cloud covers each point. The
``FilterPointsBy*`` systems apply the positional predicates of
:mod:`t4perceval.system.filter.position` -- distance, an xy box, a polar grid, a map
polygon -- to the ``POINT`` column. They are separate classes rather than the object filters
pointed at a cloud on purpose: ``POINT`` is not ``POSITION`` (see
:class:`~t4perceval.archetype.SemanticSegmentation3D`), so a filter has to say which of
the two it means.
"""

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
from t4perceval.system.filter.position import (
    FilterByDistanceSystem,
    FilterByMapSystem,
    FilterByPolarGridSystem,
    FilterByRegionSystem,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.view import EntityView
    from t4perceval.system.base import SystemContext
    from t4perceval.typing import NDArrayBool

__all__ = (
    "FilterPointsByCoverageSystem",
    "FilterPointsByDistanceSystem",
    "FilterPointsByMapSystem",
    "FilterPointsByPolarGridSystem",
    "FilterPointsByRegionSystem",
)


@define(slots=True)
class FilterPointsByCoverageSystem(MaskSystem):
    """Keep the points of one entity that have a counterpart in another.

    Sources are ``(source, reference)``; the mask is over ``source``, true where a
    ``reference`` point lies within :attr:`tolerance`. Its purpose is the ground truth an
    estimation did not cover -- a cropped or downsampled output -- when that gap is to be
    left out of the score rather than counted as a miss. That is an evaluation decision, so
    it is a stage you add, and the mask records exactly which points it dropped::

        covered = FilterPointsByCoverageSystem.between(GT, EST)
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


@define(slots=True)
class FilterPointsByDistanceSystem(FilterByDistanceSystem):
    """Keep the points whose distance from the origin is within ``[min, max]``.

    :class:`~t4perceval.system.filter.position.FilterByDistanceSystem` over the ``point``
    column of a cloud such as :class:`~t4perceval.archetype.SemanticSegmentation3D`. The
    parameters, the inclusive bounds and ``bev`` mean the same; the distance is measured in
    the frame the cloud declares, so it has to be ``base_link`` for "distance from the
    ego". The mask is written to ``<source>/filter/distance``.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POINT,)
    COLUMN: ClassVar[ComponentDescriptor] = POINT


@define(slots=True)
class FilterPointsByRegionSystem(FilterByRegionSystem):
    """Keep the points whose xy lies inside an axis-aligned region.

    :class:`~t4perceval.system.filter.position.FilterByRegionSystem` over the ``point``
    column; ``symmetric()`` builds the mirrored box as it does for objects. The mask is
    written to ``<source>/filter/region``.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POINT,)
    COLUMN: ClassVar[ComponentDescriptor] = POINT


@define(slots=True)
class FilterPointsByPolarGridSystem(FilterByPolarGridSystem):
    """Keep the points that lie in a cell of the polar grid around the origin.

    :class:`~t4perceval.system.filter.position.FilterByPolarGridSystem` over the ``point``
    column, with the same distance and angle bounds. The mask is written to
    ``<source>/filter/polar_grid``.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POINT,)
    COLUMN: ClassVar[ComponentDescriptor] = POINT


@define(slots=True)
class FilterPointsByMapSystem(FilterByMapSystem):
    """Keep the points whose xy lies inside a polygon stated in ``map``.

    :class:`~t4perceval.system.filter.position.FilterByMapSystem` over the ``point``
    column: a cloud not in ``map`` is moved there per frame through the ego pose before the
    test, and ``on_lanelet()`` builds the polygon from a Lanelet2 map. The mask is written
    to ``<source>/filter/map`` (``/filter/lanelet`` from ``on_lanelet()``).
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POINT,)
    COLUMN: ClassVar[ComponentDescriptor] = POINT
