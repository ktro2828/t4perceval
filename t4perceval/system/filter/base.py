"""The shared base of every filter: one source, one boolean mask."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from attrs import define

from t4perceval.component import BatchMask
from t4perceval.core.chunk import Chunk
from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import TimeRange
from t4perceval.descriptors import MASK
from t4perceval.system.base import EntitySystem, SystemContext, require

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.view import EntityView
    from t4perceval.typing import NDArrayBool


def check_range(low: float, high: float, *, low_name: str, high_name: str) -> None:
    if high < low:
        raise ValueError(f"{high_name} ({high}) must not be below {low_name} ({low})")


@define(slots=True)
class MaskSystem(EntitySystem):
    """Base for a system that emits one boolean mask over its first source entity.

    A subclass declares its ``REQUIRES``, a :attr:`FILTER_NAME` used to build the default
    target path, its parameters as attrs fields, and :meth:`keep`. Most filters read one
    entity; one that also consults another declares it as a further source, so
    ``Pipeline`` still sees the dependency, and overrides ``__attrs_post_init__`` with its
    own source count.
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
        source = self.sources[0]
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
