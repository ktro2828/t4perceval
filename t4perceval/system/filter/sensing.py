"""The non-detection half of the sensing evaluation: points left where nothing should be.

A non-detection area is a region the user declares empty, such as the lane ahead. Once the
points inside any ground-truth box are discounted -- the boxes scaled as the detection half
scales them -- no point may remain there. :class:`SensingNonDetectionSystem` is a
:class:`~t4perceval.system.filter.base.MaskSystem` with two sources, like
:class:`~t4perceval.system.filter.point.FilterPointsByCoverageSystem`, but its mask has the
opposite polarity of a filter's: **true marks an offending point**. See
:mod:`t4perceval.system.sensing` for the task as a whole.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import shapely
from attrs import define, field

from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import TimeRange
from t4perceval.descriptors import (
    POINT,
    POSITION,
    QUATERNION,
    SIZE,
)
from t4perceval.geometry import box_scale_at, points_in_boxes
from t4perceval.system.base import SystemContext, require, require_same_frame
from t4perceval.system.filter.base import MaskSystem, check_range
from t4perceval.system.filter.position import _as_geometry

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry
    from typing_extensions import Self

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.view import EntityView
    from t4perceval.typing import ArrayLike, NDArrayBool, NDArrayF64

__all__ = ("NonDetectionArea", "SensingNonDetectionSystem", "check_box_scales")


def check_box_scales(scale_0m: float, scale_100m: float) -> None:
    """Raise unless both footprint scale anchors are positive."""
    if not (scale_0m > 0.0 and scale_100m > 0.0):
        raise ValueError(
            f"box_scale_0m and box_scale_100m must be positive, got {scale_0m} and {scale_100m}"
        )


@define(frozen=True, slots=True)
class NonDetectionArea:
    """A region of space declared empty: an xy polygon between two heights.

    The original package took each area as the corners of its lower and upper planes;
    :meth:`from_vertices` accepts that list. Membership is edge-inclusive in xy, like
    :class:`~t4perceval.system.FilterByRegionSystem`, and closed in z.

    Examples:
        >>> area = NonDetectionArea([(0.0, 1.5), (10.0, 1.5), (10.0, -1.5), (0.0, -1.5)],
        ...                         z_min=0.0, z_max=4.0)
        >>> area.contains([[5.0, 0.0, 1.0], [5.0, 0.0, 5.0], [20.0, 0.0, 1.0]]).tolist()
        [True, False, False]
    """

    polygon: BaseGeometry = field(converter=_as_geometry)
    """A shapely polygon, or one or more rings of ``(x, y)`` points."""

    z_min: float = field(default=-np.inf, converter=float, kw_only=True)
    """Lowest height that counts as inside."""

    z_max: float = field(default=np.inf, converter=float, kw_only=True)
    """Highest height that counts as inside."""

    def __attrs_post_init__(self) -> None:
        if self.polygon is None or self.polygon.is_empty:
            raise ValueError("a non-detection area needs a non-empty polygon")
        check_range(self.z_min, self.z_max, low_name="z_min", high_name="z_max")
        shapely.prepare(self.polygon)

    @classmethod
    def from_vertices(cls, vertices: ArrayLike) -> Self:
        """Build an area from the ``(x, y, z)`` corners of its lower and upper planes.

        This is the ``non_detection_areas`` element of the original package: the same
        ring listed twice, once per plane. The ring is read from the first half and the
        height range from the whole.
        """
        corners = np.asarray(vertices, dtype=np.float64)
        if corners.ndim != 2 or corners.shape[1] != 3 or len(corners) < 6 or len(corners) % 2:
            raise ValueError(
                "vertices must list the (x, y, z) corners of two planes with at least three "
                f"corners each, got shape {corners.shape}"
            )
        ring = corners[: len(corners) // 2, :2]
        return cls(ring, z_min=float(corners[:, 2].min()), z_max=float(corners[:, 2].max()))

    def contains(self, points: ArrayLike) -> NDArrayBool:
        """Return which of the ``(N, 3)`` points lie inside the area."""
        xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        inside = shapely.intersects_xy(self.polygon, xyz[:, 0], xyz[:, 1])
        return inside & (xyz[:, 2] >= self.z_min) & (xyz[:, 2] <= self.z_max)


def _as_areas(value: Any) -> tuple[NonDetectionArea, ...]:
    """Accept one area, or a sequence of areas and ``(polygon, z_min, z_max)`` triples."""
    if isinstance(value, NonDetectionArea):
        return (value,)
    areas: list[NonDetectionArea] = []
    for item in value:
        if isinstance(item, NonDetectionArea):
            areas.append(item)
        elif isinstance(item, (tuple, list)) and len(item) == 3 and np.ndim(item[1]) == 0:
            polygon, z_min, z_max = item
            areas.append(NonDetectionArea(polygon, z_min=z_min, z_max=z_max))
        else:
            raise TypeError(
                "areas must be NonDetectionArea instances or (polygon, z_min, z_max) triples"
            )
    return tuple(areas)


@define(slots=True)
class SensingNonDetectionSystem(MaskSystem):
    """Mark the points that remain inside a non-detection area.

    Sources are ``(points, objects)``. A point offends when it lies inside one of the
    :attr:`areas` and inside none of the ``objects`` boxes at its frame, scaled as
    :class:`SensingDetectionSystem` scales them. **The mask is true for the offending
    points** -- the opposite polarity of a filter, whose mask is true for the rows to
    keep -- so that its count is the failure count and
    :class:`~t4perceval.system.ApplyMaskSystem` materializes the failures, which is what
    the original package's ``pointcloud_failed_non_detection`` held::

        offending = SensingNonDetectionSystem.between(LIDAR, GT, areas=[lane_ahead])
        failures = ApplyMaskSystem.of(LIDAR, offending.target, name="non_detection")

    A frame of ``points`` for which ``objects`` has nothing is judged with no boxes: every
    point in an area offends. Annotations are complete by definition, so their absence
    means an empty road, unlike a missing cloud.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POINT,)
    REQUIRES_OBJECTS: ClassVar[tuple[ComponentDescriptor, ...]] = (POSITION, QUATERNION, SIZE)
    FILTER_NAME: ClassVar[str] = "non_detection"

    areas: tuple[NonDetectionArea, ...] = field(converter=_as_areas, kw_only=True)
    """The regions declared empty, as :class:`NonDetectionArea` or ``(polygon, z_min, z_max)``."""

    box_scale_0m: float = field(default=1.0, converter=float, kw_only=True)
    """Footprint scale of a box at the origin."""

    box_scale_100m: float = field(default=1.0, converter=float, kw_only=True)
    """Footprint scale of a box 100 m from the origin; the scale is linear in between."""

    check_frames: bool = field(default=True, kw_only=True)
    """Refuse inputs that state different coordinate frames."""

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 2:
            raise ValueError(
                f"{type(self).__name__} needs exactly two sources (points, objects), "
                f"got {len(self.sources)}",
            )
        if not self.areas:
            raise ValueError("areas must name at least one non-detection area")
        check_box_scales(self.box_scale_0m, self.box_scale_100m)

    def requires_for(self, index: int) -> tuple[ComponentDescriptor, ...]:
        return self.REQUIRES if index == 0 else self.REQUIRES_OBJECTS

    @classmethod
    def between(
        cls,
        points: EntityPathLike,
        objects: EntityPathLike,
        *,
        target: EntityPathLike | None = None,
        **params: Any,
    ) -> Self:
        """Mask the points of ``points`` that offend, given the boxes of ``objects``.

        The target defaults to ``<points>/filter/non_detection``.
        """
        path = as_entity_path(points)
        return cls(
            (path, objects),
            target if target is not None else path / "filter" / cls.FILTER_NAME,
            **params,
        )

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        """Return which points lie in an area and outside every box at their frame."""
        points, objects = self.sources
        chunk = view.to_chunk()
        index = chunk.index(ctx.timeline)
        if index is None:
            raise ValueError(
                f"{points} has no {ctx.timeline.name!r} index, so its frames cannot be "
                f"paired with {objects}",
            )
        xyz_all: NDArrayF64 = view.component(POINT).values  # type: ignore[union-attr]

        offending = np.zeros(len(view), dtype=np.bool_)
        for partition in range(chunk.num_partitions):
            start, stop = int(chunk.offsets[partition]), int(chunk.offsets[partition + 1])
            if stop == start:
                continue
            xyz = xyz_all[start:stop]
            in_area = np.logical_or.reduce([area.contains(xyz) for area in self.areas])
            if not in_area.any():
                continue
            boxes = ctx.store.range(
                objects,
                timeline=ctx.timeline,
                time_range=TimeRange.single(int(index.times[partition])),
            )
            if not len(boxes):
                offending[start:stop] = in_area
                continue
            require(boxes, *self.REQUIRES_OBJECTS)
            if self.check_frames:
                require_same_frame(view, boxes)
            position = boxes.component(POSITION).values  # type: ignore[union-attr]
            scale = box_scale_at(
                np.linalg.norm(position, axis=1), self.box_scale_0m, self.box_scale_100m
            )
            inside = points_in_boxes(
                xyz[in_area],
                position,
                boxes.component(QUATERNION).values,  # type: ignore[union-attr]
                boxes.component(SIZE).values,  # type: ignore[union-attr]
                scale=scale,
            ).any(axis=1)
            offending[start:stop][in_area] = ~inside
        return offending
