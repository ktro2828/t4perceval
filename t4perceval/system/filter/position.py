"""Filters on where an object is: distance from the ego, a box, a map region."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import shapely
from attrs import define, field
from scipy.spatial.transform import Rotation
from shapely.geometry import MultiPolygon, Polygon

from t4perceval.descriptors import POSITION
from t4perceval.system.base import SystemContext
from t4perceval.system.filter.base import MaskSystem, check_range
from t4perceval.transform.apply import pose_of
from t4perceval.transform.lookup import TransformResolver

if TYPE_CHECKING:
    from collections.abc import Collection

    from shapely.geometry.base import BaseGeometry
    from typing_extensions import Self

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.view import EntityView
    from t4perceval.lanelet import LaneletMap
    from t4perceval.typing import NDArrayBool, NDArrayF64


@define(slots=True)
class FilterByDistanceSystem(MaskSystem):
    """Keep objects whose distance from the origin is within ``[min, max]``.

    Distance is measured in the coordinate frame the source chunk declares, so positions
    must already be expressed relative to the point being measured from -- normally
    ``base_link``, which puts the ego at the origin.

    Set ``bev=True`` to measure in the xy plane only, which is what the original
    package's ``max_distance`` did; the default measures the full 3D norm, matching
    ``t4_devkit.filtering.FilterByDistance``.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POSITION,)
    FILTER_NAME: ClassVar[str] = "distance"

    min_distance: float = field(default=0.0, kw_only=True)
    max_distance: float = field(default=float("inf"), kw_only=True)
    bev: bool = field(default=False, kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.min_distance < 0.0:
            raise ValueError(f"min_distance must be non-negative, got {self.min_distance}")
        check_range(
            self.min_distance,
            self.max_distance,
            low_name="min_distance",
            high_name="max_distance",
        )

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        position = view.component(POSITION).values
        axes = position[:, :2] if self.bev else position
        distance = np.linalg.norm(axes, axis=1)
        return (distance >= self.min_distance) & (distance <= self.max_distance)


@define(slots=True)
class FilterByRegionSystem(MaskSystem):
    """Keep objects whose xy position lies inside an axis-aligned region.

    The original package's symmetric ``max_x_position`` / ``max_y_position`` is
    :meth:`symmetric`.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POSITION,)
    FILTER_NAME: ClassVar[str] = "region"

    min_xy: tuple[float, float] = field(
        default=(-float("inf"), -float("inf")),
        converter=lambda value: (float(value[0]), float(value[1])),
        kw_only=True,
    )
    max_xy: tuple[float, float] = field(
        default=(float("inf"), float("inf")),
        converter=lambda value: (float(value[0]), float(value[1])),
        kw_only=True,
    )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        for axis, name in enumerate("xy"):
            check_range(
                self.min_xy[axis],
                self.max_xy[axis],
                low_name=f"min_xy[{name}]",
                high_name=f"max_xy[{name}]",
            )

    @classmethod
    def symmetric(
        cls,
        source: EntityPathLike,
        *,
        max_xy: tuple[float, float],
        name: str | None = None,
    ) -> Self:
        """Build a region mirrored about the origin: ``[-max_xy, +max_xy]`` on each axis.

        This is the region the original ``evaluation_config_dict`` described with
        ``max_x_position`` / ``max_y_position``. Note the difference from
        ``on(max_xy=...)``, which bounds the region from above only and leaves
        :attr:`min_xy` unbounded.
        """
        return cls.on(
            source,
            name=name,
            min_xy=(-max_xy[0], -max_xy[1]),
            max_xy=max_xy,
        )

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        xy = view.component(POSITION).values[:, :2]
        lower = np.asarray(self.min_xy, dtype=np.float64)
        upper = np.asarray(self.max_xy, dtype=np.float64)
        return np.all((xy >= lower) & (xy <= upper), axis=1)


def _as_geometry(value: Any) -> BaseGeometry | None:
    """Accept a shapely polygon, or one ring / several rings of ``(x, y)`` points."""
    if value is None or isinstance(value, (Polygon, MultiPolygon)):
        return value
    if isinstance(value, shapely.geometry.base.BaseGeometry):
        raise TypeError(f"polygon must be a Polygon or MultiPolygon, got {type(value).__name__}")
    rings = list(value)
    if rings and not isinstance(rings[0], (list, tuple, np.ndarray)):
        raise TypeError("polygon must be a shapely geometry or a sequence of (x, y) rings")
    if rings and np.ndim(rings[0]) == 1:
        rings = [rings]
    return shapely.union_all([Polygon(ring) for ring in rings])


@define(slots=True)
class FilterByMapSystem(MaskSystem):
    """Keep objects whose xy position lies inside a polygon, such as a map region.

    Every other filter tests in the frame the source chunk declares, because a distance or
    a box is a claim *about that frame*. Whether an object is on the road is not: it is a
    fact about the world, the same whichever frame the object was recorded in. So the
    polygon is always in :attr:`MAP_FRAME`, and when the source declares another frame the
    filter looks the ego pose up per frame -- as
    :class:`~t4perceval.system.TransformEntitySystem` does -- and moves the *positions*
    into ``map`` before the test. The verdict is written under the source, where it
    belongs, and composes with every other mask on that entity.

    The lookup needs the ``/tf`` edges. By default they are read from the store the pipeline
    runs on; pass ``resolver`` when they live elsewhere. A source already in ``map`` needs
    no lookup at all.

    Build one from a Lanelet2 map with :meth:`on_lanelet`::

        on_road = FilterByMapSystem.on_lanelet(
            "/ground_truth/objects",
            LaneletMap.load(data_root / "map" / "lanelet2_map.osm"),
            subtypes=("road", "crosswalk"),
        )

    With no polygon the filter passes every row, like the rest of the family.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POSITION,)
    FILTER_NAME: ClassVar[str] = "map"

    #: The frame every polygon is in. A map is the world, and the world is ``map``.
    MAP_FRAME: ClassVar[str] = "map"

    polygon: BaseGeometry | None = field(default=None, kw_only=True, converter=_as_geometry)
    """A ``Polygon`` / ``MultiPolygon``, or ``(x, y)`` rings, in ``map``. ``None`` keeps
    everything."""

    resolver: TransformResolver | None = field(default=None, kw_only=True)
    """Where ``map <- source`` is looked up when the source is not in ``map``.

    ``None`` builds one over ``ctx.store`` on ``ctx.timeline`` when the system runs. Pass
    one built from the original recording when the evaluation store carries no ``/tf``,
    or when the edges live on another timeline -- a bag's ``/tf`` is on ``TIMESTAMP`` only.
    """

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.polygon is not None:
            shapely.prepare(self.polygon)

    @classmethod
    def on_lanelet(
        cls,
        source: EntityPathLike,
        lanelet_map: LaneletMap,
        *,
        subtypes: Collection[str] = ("road", "road_shoulder", "crosswalk"),
        resolver: TransformResolver | None = None,
    ) -> Self:
        """Keep objects inside the lanelets of the given ``subtypes``.

        The default is where traffic is expected: the road, its shoulder and the
        crosswalks. See :meth:`~t4perceval.lanelet.LaneletMap.region`. The mask is written
        to ``<source>/filter/lanelet``.
        """
        return cls.on(
            source,
            name="lanelet",
            polygon=lanelet_map.region(subtypes=subtypes),
            resolver=resolver,
        )

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        if self.polygon is None:
            return np.ones(len(view), dtype=np.bool_)
        position = view.component(POSITION).values
        if view.frame_id != self.MAP_FRAME:
            position = self._in_map(view, ctx, position)
        return shapely.contains_xy(self.polygon, position[:, 0], position[:, 1])

    def _in_map(self, view: EntityView, ctx: SystemContext, position: NDArrayF64) -> NDArrayF64:
        """Move ``position`` into :attr:`MAP_FRAME`, one ego pose per partition."""
        (source,) = self.sources
        source_frame = view.frame_id
        if source_frame is None:
            raise ValueError(
                f"{source} states no coordinate frame, so {type(self).__name__} cannot "
                f"bring it into {self.MAP_FRAME!r}. Pass frame_id= when logging it.",
            )

        resolver = (
            self.resolver
            if self.resolver is not None
            else TransformResolver.of(ctx.store, timeline=ctx.timeline)
        )
        chunk = view.to_chunk()
        index = chunk.index(resolver.timeline)
        if index is None:
            raise ValueError(
                f"{source} has no {resolver.timeline.name!r} index, which the resolver looks "
                f"transforms up on; build the resolver on a timeline the entity is logged on",
            )

        moved = np.array(position, dtype=np.float64)
        for partition in range(chunk.num_partitions):
            start, stop = int(chunk.offsets[partition]), int(chunk.offsets[partition + 1])
            if stop == start:
                continue
            time = int(index.times[partition])
            try:
                translation, rotation = pose_of(
                    resolver.lookup(
                        target_frame=self.MAP_FRAME, source_frame=source_frame, at=time
                    ),
                )
            except ValueError as error:
                raise ValueError(
                    f"{type(self).__name__} cannot bring {source} from {source_frame!r} "
                    f"into {self.MAP_FRAME!r} at {resolver.timeline.name}={time}: {error}",
                ) from error
            moved[start:stop] = (
                Rotation.from_quat(rotation).apply(position[start:stop]) + translation
            )
        return moved
