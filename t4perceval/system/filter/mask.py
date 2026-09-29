"""Combining masks, and materializing the rows one kept."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define, field

from t4perceval.component import BatchMask
from t4perceval.core.chunk import Chunk
from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import TimeRange
from t4perceval.descriptors import MASK
from t4perceval.system.base import EntitySystem, Passthrough, SystemContext, require

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from typing_extensions import Self

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.store import Store
    from t4perceval.core.timeline import Timeline
    from t4perceval.core.view import EntityView


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

    # The columns carried over are whatever the data source holds, which is exactly what a
    # passthrough declares: a consumer of the target is checked against the source's
    # columns when they are known, and at run time when they are not.
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...] | Passthrough] = Passthrough(0)

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 2:
            raise ValueError(
                f"{type(self).__name__} needs exactly two sources (data, mask), "
                f"got {len(self.sources)}",
            )

    def requires_for(self, index: int) -> tuple[ComponentDescriptor, ...]:
        """The mask source needs a mask; the data source needs nothing in particular."""
        return () if index == 0 else self.REQUIRES

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
