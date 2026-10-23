"""Sensing evaluation: did the sensor see what the annotations say is there?

The original package's ``sensing`` task asks two questions of a point cloud, both against
the ground-truth boxes and neither involving an estimation:

* **Detection area** -- the inside of every box. Enough points must fall inside each box,
  once its footprint is scaled for its distance from the ego, or the object was not
  sensed. :class:`SensingDetectionSystem` answers per box and writes a
  :class:`~t4perceval.archetype.SensingResults` aligned with the objects entity.
* **Non-detection area** -- regions the user declares empty, such as the lane ahead. No
  point may remain there once the points inside any scaled box are discounted.
  :class:`~t4perceval.system.filter.sensing.SensingNonDetectionSystem` writes a mask over
  the cloud marking the points that do remain; it lives with the filters because it is a
  mask system.

:class:`~t4perceval.system.metric.sensing.SensingDetectionRateSystem`, with the metrics,
then summarizes the verdicts per class.

Both inputs are read in the frame they are logged in, and the scale distance and the
nearest point are measured from that frame's origin, so log them in ``base_link`` -- or
move them there first with :class:`~t4perceval.system.TransformEntitySystem`. The cloud
is logged by the caller as a :class:`~t4perceval.archetype.PointCloud`; the dataset
importer does not load it. Which ground truth each system reads is the caller's choice
too: the original judged detection on the ``target_uuids``-filtered objects but cropped
the non-detection areas with every annotation, which here is a matter of pointing the two
systems at different entities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from attrs import define, field

from t4perceval.archetype.sensing import SensingResults
from t4perceval.component import SensingStatus, VisibilityLevel
from t4perceval.core.chunk import Chunk
from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import TimeRange
from t4perceval.descriptors import (
    POINT,
    POSITION,
    QUATERNION,
    SIZE,
    VISIBILITY,
)
from t4perceval.geometry import box_scale_at, points_in_boxes
from t4perceval.system.base import EntitySystem, SystemContext, require, require_same_frame
from t4perceval.system.filter.sensing import check_box_scales

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike

__all__ = ("SensingDetectionSystem",)


@define(slots=True)
class SensingDetectionSystem(EntitySystem):
    """Count the points inside each ground-truth box and judge whether it was sensed.

    Sources are ``(objects, points)``. For every box of ``objects`` at a frame, the
    footprint is scaled by :func:`~t4perceval.geometry.box_scale_at` of the box's
    distance from the origin, the points of ``points`` at that frame are tested against
    the scaled box, and the box is :attr:`~t4perceval.component.SensingStatus.DETECTED`
    when at least :attr:`min_num_points` fall inside, :attr:`~SensingStatus.FAILED`
    otherwise. A box annotated with ``visibility == NONE`` is a
    :attr:`~SensingStatus.WARNING` whatever its count, as a fully occluded object says
    nothing about the sensor. Objects without a visibility column are never warned.

    The result is one :class:`~t4perceval.archetype.SensingResults` row per box, written
    with the objects' partitioning under ``<objects>/sensing`` by default, so a row is
    found from its box by index. A frame of ``objects`` for which ``points`` has no frame
    at all is an error: a missing measurement is not a measurement of zero points. An
    empty cloud frame is, and fails every box.

    The inside count reuses the annotation's ``num_points`` descriptor, so
    ``FilterByNumPointsSystem.on(sensing.target, min_num_points=5)`` re-judges the result
    at another threshold without re-counting::

        sensing = SensingDetectionSystem.between(GT, LIDAR, box_scale_100m=1.2)
        seen = FilterByNumPointsSystem.on(sensing.target, min_num_points=5)
        seen_boxes = ApplyMaskSystem.of(GT, seen.target)
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POSITION, QUATERNION, SIZE)
    REQUIRES_POINTS: ClassVar[tuple[ComponentDescriptor, ...]] = (POINT,)
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = SensingResults.required_descriptors()

    #: Default last segment of the target path, ``<objects>/<SENSING_NAME>``.
    SENSING_NAME: ClassVar[str] = "sensing"

    box_scale_0m: float = field(default=1.0, converter=float, kw_only=True)
    """Footprint scale of a box at the origin."""

    box_scale_100m: float = field(default=1.0, converter=float, kw_only=True)
    """Footprint scale of a box 100 m from the origin; the scale is linear in between."""

    min_num_points: int = field(default=1, converter=int, kw_only=True)
    """Inside points a box needs to count as detected."""

    check_frames: bool = field(default=True, kw_only=True)
    """Refuse inputs that state different coordinate frames."""

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 2:
            raise ValueError(
                f"{type(self).__name__} needs exactly two sources (objects, points), "
                f"got {len(self.sources)}",
            )
        check_box_scales(self.box_scale_0m, self.box_scale_100m)
        if self.min_num_points < 0:
            raise ValueError(f"min_num_points must be non-negative, got {self.min_num_points}")

    def requires_for(self, index: int) -> tuple[ComponentDescriptor, ...]:
        return self.REQUIRES if index == 0 else self.REQUIRES_POINTS

    @classmethod
    def between(
        cls,
        objects: EntityPathLike,
        points: EntityPathLike,
        *,
        target: EntityPathLike | None = None,
        **params: Any,
    ) -> Self:
        """Judge the boxes of ``objects`` by the cloud ``points``.

        The target defaults to ``<objects>/sensing``: the result is a verdict about each
        box, so it is filed with the box, as a mask is.
        """
        path = as_entity_path(objects)
        return cls(
            (path, points),
            target if target is not None else path / cls.SENSING_NAME,
            **params,
        )

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        objects, points = self.sources
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)
        view = ctx.store.range(objects, timeline=ctx.timeline, time_range=time_range)
        if len(view):
            require(view, *self.REQUIRES)
        chunk = view.to_chunk()

        num_rows = len(view)
        num_points = np.zeros(num_rows, dtype=np.int32)
        status = np.full(num_rows, int(SensingStatus.FAILED), dtype=np.int8)
        nearest = np.full((num_rows, 3), np.nan, dtype=np.float64)

        if num_rows:
            index = chunk.index(ctx.timeline)
            if index is None:
                raise ValueError(
                    f"{objects} has no {ctx.timeline.name!r} index, so its frames cannot be "
                    f"paired with {points}",
                )
            cloud_times = set(ctx.store.times(points, ctx.timeline).tolist())
            position = view.component(POSITION).values  # type: ignore[union-attr]
            quaternion = view.component(QUATERNION).values  # type: ignore[union-attr]
            size = view.component(SIZE).values  # type: ignore[union-attr]
            visibility = view.component(VISIBILITY)

            for partition in range(chunk.num_partitions):
                start, stop = int(chunk.offsets[partition]), int(chunk.offsets[partition + 1])
                if stop == start:
                    continue
                time = int(index.times[partition])
                if time not in cloud_times:
                    raise ValueError(
                        f"{points} has no frame at {ctx.timeline.name}={time}, but {objects} "
                        f"has {stop - start} object(s) there; a missing cloud is not an empty "
                        "one, so align the cloud to the objects first",
                    )
                cloud = ctx.store.range(
                    points, timeline=ctx.timeline, time_range=TimeRange.single(time)
                )
                if len(cloud):
                    require(cloud, POINT)
                if self.check_frames:
                    require_same_frame(view, cloud)
                xyz = (
                    cloud.component(POINT).values  # type: ignore[union-attr]
                    if len(cloud)
                    else np.empty((0, 3), dtype=np.float64)
                )
                rows = slice(start, stop)
                scale = box_scale_at(
                    np.linalg.norm(position[rows], axis=1),
                    self.box_scale_0m,
                    self.box_scale_100m,
                )
                inside = points_in_boxes(
                    xyz, position[rows], quaternion[rows], size[rows], scale=scale
                )
                counts = inside.sum(axis=0)
                num_points[rows] = counts
                if len(xyz):
                    norms = np.linalg.norm(xyz, axis=1)
                    closest = np.argmin(np.where(inside, norms[:, None], np.inf), axis=0)
                    seen = counts > 0
                    nearest[rows][seen] = xyz[closest[seen]]
                status[rows] = np.where(
                    counts >= self.min_num_points,
                    int(SensingStatus.DETECTED),
                    int(SensingStatus.FAILED),
                )
                if visibility is not None:
                    occluded = visibility.values[rows] == int(VisibilityLevel.NONE)
                    status[rows][occluded] = int(SensingStatus.WARNING)

        result = SensingResults(num_points=num_points, sensing_status=status, nearest_point=nearest)
        return (
            Chunk(
                self.target,
                chunk.indexes,
                chunk.offsets,
                result.as_components(),
                frame_id=chunk.frame_id,
            ),
        )
