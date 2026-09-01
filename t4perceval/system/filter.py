"""Filter systems.

A filter emits a boolean :data:`~t4perceval.descriptors.MASK` column instead of dropping
rows. The rows stay addressable, so a later stage can ask *why* an object was excluded,
and the filter itself becomes reviewable data rather than a side effect.

Every filter is only its predicate: :class:`MaskSystem` implements the query, the
validation, the empty-frame case and the chunk construction once.

Two conventions hold across the whole family:

* **Bounds are inclusive on both ends**, so a filter constructed with its default
  parameters passes every row. The original package compared strictly (``score >
  threshold``, ``abs(x) < max_x_position``), which makes ``min_distance=0.0`` reject an
  object at the origin; the boundary itself is measure-zero in float arithmetic, and a
  default that is a guaranteed no-op is worth more than matching it exactly.
* **A missing optional column is an error, not a pass.** ``t4_devkit`` lets a box with no
  velocity through its speed filter; here, filtering on a component the entity does not
  carry means the pipeline is wired wrong, and :func:`~t4perceval.system.base.require`
  says so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from attrs import define, field

from t4perceval.component import BatchMask, VisibilityLevel
from t4perceval.core.chunk import Chunk
from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import TimeRange
from t4perceval.descriptors import (
    CLASS_ID,
    CONFIDENCE,
    INSTANCE_ID,
    MASK,
    NUM_POINTS,
    POSITION,
    VELOCITY,
    VISIBILITY,
)
from t4perceval.system.base import EntitySystem, SystemContext, require

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from typing_extensions import Self

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.store import Store
    from t4perceval.core.timeline import Timeline
    from t4perceval.core.view import EntityView
    from t4perceval.typing import NDArrayBool

__all__ = (
    "ApplyMaskSystem",
    "CombineMasksSystem",
    "FilterByConfidenceSystem",
    "FilterByDistanceSystem",
    "FilterByInstanceSystem",
    "FilterByLabelSystem",
    "FilterByNumPointsSystem",
    "FilterByRegionSystem",
    "FilterBySpeedSystem",
    "FilterByVisibilitySystem",
    "MaskSystem",
    "masked_view",
)


def _check_range(low: float, high: float, *, low_name: str, high_name: str) -> None:
    if high < low:
        raise ValueError(f"{high_name} ({high}) must not be below {low_name} ({low})")


@define(slots=True)
class MaskSystem(EntitySystem):
    """Base for a system that emits one boolean mask over one source entity.

    A subclass declares its ``REQUIRES``, a :attr:`FILTER_NAME` used to build the default
    target path, its parameters as attrs fields, and :meth:`keep`.
    """

    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = (MASK,)

    #: Default last segment of the target path, ``<source>/filter/<FILTER_NAME>``.
    FILTER_NAME: ClassVar[str] = "mask"

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 1:
            raise ValueError(
                f"{type(self).__name__} needs exactly one source, got {len(self.sources)}",
            )

    @classmethod
    def on(cls, source: EntityPathLike, *, name: str | None = None, **params: Any) -> Self:
        """Build a filter writing its mask to ``<source>/filter/<name>``.

        Keeping the mask under the source path means a prefix query finds an entity
        together with every verdict recorded about it. ``params`` are the subclass's own
        fields.
        """
        path = as_entity_path(source)
        return cls((path,), path / "filter" / (name or cls.FILTER_NAME), **params)

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        """Return which rows of ``view`` pass, as a mask of length ``len(view)``."""
        raise NotImplementedError

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        (source,) = self.sources
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)
        view = ctx.store.range(source, timeline=ctx.timeline, time_range=time_range)

        if len(view):
            require(view, *self.REQUIRES)
            keep = np.asarray(self.keep(view, ctx), dtype=np.bool_)
            if keep.shape != (len(view),):
                raise ValueError(
                    f"{type(self).__name__}.keep() returned shape {keep.shape}, "
                    f"expected {(len(view),)}",
                )
        else:
            # An entity with no rows in range is an ordinary empty frame, not a wiring
            # error, so the component check is skipped rather than failed.
            keep = np.empty(0, dtype=np.bool_)

        chunk = view.to_chunk()
        return (
            Chunk(
                self.target,
                chunk.indexes,
                chunk.offsets,
                {MASK: BatchMask(keep)},
                frame_id=chunk.frame_id,
            ),
        )


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
        _check_range(
            self.min_distance,
            self.max_distance,
            low_name="min_distance",
            high_name="max_distance",
        )

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        del ctx
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
            _check_range(
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
        del ctx
        xy = view.component(POSITION).values[:, :2]
        lower = np.asarray(self.min_xy, dtype=np.float64)
        upper = np.asarray(self.max_xy, dtype=np.float64)
        return np.all((xy >= lower) & (xy <= upper), axis=1)


def _as_class_ids(values: Sequence[str | int] | None) -> tuple[str | int, ...] | None:
    return None if values is None else tuple(values)


def _resolve_class_ids(
    values: tuple[str | int, ...],
    ctx: SystemContext,
    *,
    field_name: str,
) -> set[int]:
    """Turn a mix of class names and class ids into a set of ids."""
    names = [value for value in values if isinstance(value, str)]
    if names and ctx.labels is None:
        raise ValueError(
            f"{field_name} names {names} require a LabelRegistry; "
            "pass one as SystemContext(labels=...)",
        )
    # An unknown name raises rather than matching nothing: a typo must not silently
    # turn a filter into a no-op that skews every downstream metric.
    return {
        value if isinstance(value, int) else ctx.labels.class_id(value)  # type: ignore[union-attr]
        for value in values
    }


@define(slots=True)
class FilterByLabelSystem(MaskSystem):
    """Keep objects whose class is in ``labels`` and not in ``exclude``.

    Both accept class names, resolved through :attr:`SystemContext.labels`, or raw class
    ids. ``labels=None`` admits every class, so the two together express the original
    ``target_labels`` allowlist and ``ignore_attributes`` denylist. An unknown class name
    raises.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)
    FILTER_NAME: ClassVar[str] = "label"

    labels: tuple[str | int, ...] | None = field(
        default=None,
        converter=_as_class_ids,
        kw_only=True,
    )
    exclude: tuple[str | int, ...] | None = field(
        default=None,
        converter=_as_class_ids,
        kw_only=True,
    )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.labels is not None and len(self.labels) == 0:
            raise ValueError("labels must not be empty; pass None to admit every class")

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        class_ids = view.component(CLASS_ID).values
        keep = np.ones(len(class_ids), dtype=np.bool_)

        if self.labels is not None:
            wanted = _resolve_class_ids(self.labels, ctx, field_name="labels")
            keep &= np.isin(class_ids, sorted(wanted))

        if self.exclude:
            unwanted = _resolve_class_ids(self.exclude, ctx, field_name="exclude")
            keep &= ~np.isin(class_ids, sorted(unwanted))

        return keep


@define(slots=True)
class FilterByConfidenceSystem(MaskSystem):
    """Keep objects whose confidence is within ``[min, max]``.

    ``min_confidence`` is the original ``confidence_threshold``.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (CONFIDENCE,)
    FILTER_NAME: ClassVar[str] = "confidence"

    min_confidence: float = field(default=0.0, kw_only=True)
    max_confidence: float = field(default=1.0, kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        for value, name in ((self.min_confidence, "min"), (self.max_confidence, "max")):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}_confidence must be within [0, 1], got {value}")
        _check_range(
            self.min_confidence,
            self.max_confidence,
            low_name="min_confidence",
            high_name="max_confidence",
        )

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        del ctx
        confidence = view.component(CONFIDENCE).values
        return (confidence >= self.min_confidence) & (confidence <= self.max_confidence)


@define(slots=True)
class FilterByInstanceSystem(MaskSystem):
    """Keep objects whose instance is in ``instances`` and not in ``exclude``.

    Both accept dataset UUIDs, resolved through :attr:`SystemContext.instances`, or raw
    instance ids. This is the original ``target_uuids``. An unknown UUID raises.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (INSTANCE_ID,)
    FILTER_NAME: ClassVar[str] = "instance"

    instances: tuple[str | int, ...] | None = field(
        default=None,
        converter=_as_class_ids,
        kw_only=True,
    )
    exclude: tuple[str | int, ...] | None = field(
        default=None,
        converter=_as_class_ids,
        kw_only=True,
    )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.instances is not None and len(self.instances) == 0:
            raise ValueError("instances must not be empty; pass None to admit every instance")

    @staticmethod
    def _resolve(values: tuple[str | int, ...], ctx: SystemContext, *, field_name: str) -> set[int]:
        uuids = [value for value in values if isinstance(value, str)]
        if uuids and ctx.instances is None:
            raise ValueError(
                f"{field_name} uuids {uuids} require an InstanceRegistry; "
                "pass one as SystemContext(instances=...)",
            )
        return {
            value if isinstance(value, int) else ctx.instances.instance_id(value)  # type: ignore[union-attr]
            for value in values
        }

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        instance_ids = view.component(INSTANCE_ID).values
        keep = np.ones(len(instance_ids), dtype=np.bool_)

        if self.instances is not None:
            wanted = self._resolve(self.instances, ctx, field_name="instances")
            keep &= np.isin(instance_ids, sorted(wanted))

        if self.exclude:
            unwanted = self._resolve(self.exclude, ctx, field_name="exclude")
            keep &= ~np.isin(instance_ids, sorted(unwanted))

        return keep


@define(slots=True)
class FilterBySpeedSystem(MaskSystem):
    """Keep objects whose speed -- the L2 norm of their velocity -- is within ``[min, max]``."""

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (VELOCITY,)
    FILTER_NAME: ClassVar[str] = "speed"

    min_speed: float = field(default=0.0, kw_only=True)
    max_speed: float = field(default=float("inf"), kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.min_speed < 0.0:
            raise ValueError(f"min_speed must be non-negative, got {self.min_speed}")
        _check_range(self.min_speed, self.max_speed, low_name="min_speed", high_name="max_speed")

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        del ctx
        speed = view.component(VELOCITY).speed
        return (speed >= self.min_speed) & (speed <= self.max_speed)


@define(slots=True)
class FilterByNumPointsSystem(MaskSystem):
    """Keep objects whose box contains a number of points within ``[min, max]``.

    ``min_num_points`` is the original ``min_point_numbers``, which was a per-class list.
    Per-class thresholds are expressed here by composition -- see
    :class:`CombineMasksSystem`.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (NUM_POINTS,)
    FILTER_NAME: ClassVar[str] = "num_points"

    min_num_points: int = field(default=0, kw_only=True)
    max_num_points: int | None = field(default=None, kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.min_num_points < 0:
            raise ValueError(f"min_num_points must be non-negative, got {self.min_num_points}")
        if self.max_num_points is not None:
            _check_range(
                self.min_num_points,
                self.max_num_points,
                low_name="min_num_points",
                high_name="max_num_points",
            )

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        del ctx
        num_points = view.component(NUM_POINTS).values
        keep = num_points >= self.min_num_points
        if self.max_num_points is not None:
            keep &= num_points <= self.max_num_points
        return keep


@define(slots=True)
class FilterByVisibilitySystem(MaskSystem):
    """Keep objects at least as visible as ``min_visibility``.

    Objects annotated :attr:`VisibilityLevel.UNAVAILABLE` always pass, matching
    ``t4_devkit.filtering.FilterByVisibility``. Rejecting them instead would empty out
    every dataset that does not annotate visibility at all.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (VISIBILITY,)
    FILTER_NAME: ClassVar[str] = "visibility"

    min_visibility: VisibilityLevel = field(
        default=VisibilityLevel.NONE,
        converter=VisibilityLevel,
        kw_only=True,
    )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.min_visibility is VisibilityLevel.UNAVAILABLE:
            raise ValueError(
                "min_visibility must be a comparable level, not UNAVAILABLE; "
                "UNAVAILABLE objects always pass this filter",
            )

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        del ctx
        visibility = view.component(VISIBILITY).values
        unavailable = visibility == int(VisibilityLevel.UNAVAILABLE)
        return unavailable | (visibility >= int(self.min_visibility))


@define(slots=True)
class CombineMasksSystem(EntitySystem):
    """Combine the masks of several entities into one.

    ``mode="all"`` is the intersection and ``mode="any"`` the union. Having both is what
    makes per-class thresholds expressible without a per-class threshold parameter: AND a
    label filter with the threshold filter for each class, then OR the results.

    All sources must describe the same rows -- they normally come from filters on one
    shared source entity -- and this is checked rather than assumed.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (MASK,)
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = (MASK,)

    mode: str = field(default="all", kw_only=True)

    def __attrs_post_init__(self) -> None:
        if self.mode not in ("all", "any"):
            raise ValueError(f"mode must be 'all' or 'any', got {self.mode!r}")
        if not self.sources:
            raise ValueError(f"{type(self).__name__} needs at least one source")

    @classmethod
    def of(
        cls,
        sources: Sequence[EntityPathLike],
        target: EntityPathLike,
        *,
        mode: str = "all",
    ) -> Self:
        """Combine the masks at ``sources`` into a mask at ``target``."""
        return cls(tuple(sources), target, mode=mode)

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)
        views = [
            ctx.store.range(source, timeline=ctx.timeline, time_range=time_range)
            for source in self.sources
        ]

        lengths = {len(view) for view in views}
        if len(lengths) > 1:
            detail = ", ".join(f"{view.entity_path}: {len(view)}" for view in views)
            raise ValueError(f"Cannot combine masks describing different rows ({detail})")

        head = views[0]
        combined = (
            np.ones(len(head), dtype=np.bool_)
            if self.mode == "all"
            else np.zeros(
                len(head),
                dtype=np.bool_,
            )
        )
        for view in views:
            if len(view):
                require(view, MASK)
                values = view.component(MASK).values
                combined = combined & values if self.mode == "all" else combined | values

        chunk = head.to_chunk()
        return (
            Chunk(
                self.target,
                chunk.indexes,
                chunk.offsets,
                {MASK: BatchMask(combined)},
                frame_id=chunk.frame_id,
            ),
        )


@define(slots=True)
class ApplyMaskSystem(EntitySystem):
    """Materialize the rows a mask kept into a new entity.

    Filters mask rather than drop, which keeps the verdict inspectable -- but a metric
    that divides by the number of ground-truth objects needs an entity that *is* the
    filtered set, because recall depends on it. That is what this system produces: the
    counterpart to the lazy :func:`masked_view`.

    Point the matcher and the metric at the same materialized entity, so the row indices
    a match result stores refer to the rows both of them see.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (MASK,)

    # The columns carried over are whatever the source holds, so they cannot be declared
    # here. `Pipeline` therefore cannot link a consumer to this system by component; the
    # entity it writes is the contract.
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = ()

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 2:
            raise ValueError(
                f"{type(self).__name__} needs exactly two sources (data, mask), "
                f"got {len(self.sources)}",
            )

    @classmethod
    def of(
        cls,
        source: EntityPathLike,
        mask_source: EntityPathLike,
        *,
        target: EntityPathLike | None = None,
        name: str = "kept",
    ) -> Self:
        """Write the surviving rows of ``source`` to ``target``.

        The target defaults to ``<source>/<name>``, keeping the filtered set beside the
        entity it came from.
        """
        path = as_entity_path(source)
        return cls((path, mask_source), target if target is not None else path / name)

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        source, mask_source = self.sources
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)

        view = masked_view(
            ctx.store,
            source,
            mask_source,
            timeline=ctx.timeline,
            time_range=time_range,
        )
        chunk = view.to_chunk()
        return (
            Chunk(
                self.target,
                chunk.indexes,
                chunk.offsets,
                chunk.columns,
                frame_id=chunk.frame_id,
            ),
        )


def masked_view(
    store: Store,
    source: EntityPathLike,
    mask_source: EntityPathLike,
    *,
    timeline: Timeline,
    time_range: TimeRange,
) -> EntityView:
    """Return a view of ``source`` narrowed to the rows its mask kept.

    The lazy alternative to materializing the filtered rows: the returned view still
    refers to the original chunk, so nothing is copied until a column is asked for.
    """
    view = store.range(source, timeline=timeline, time_range=time_range)
    mask = store.range(mask_source, timeline=timeline, time_range=time_range)

    if len(mask) != len(view):
        raise ValueError(
            f"Mask at {mask.entity_path} describes {len(mask)} row(s), but "
            f"{view.entity_path} has {len(view)}",
        )
    if not len(view):
        return view

    require(mask, MASK)
    return view.select(mask.component(MASK).values)
