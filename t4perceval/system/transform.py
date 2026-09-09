"""Expressing an entity in another coordinate frame, as a new entity."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from attrs import define, field

from t4perceval.core.chunk import concat_chunks
from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import TimeRange
from t4perceval.descriptors import MASK
from t4perceval.system.base import EntitySystem, Passthrough, resolve_times
from t4perceval.transform.apply import pose_of, transform_chunk
from t4perceval.transform.compose import identity
from t4perceval.transform.lookup import TransformResolver

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.system.base import SystemContext

__all__ = ("TransformEntitySystem",)


@define(slots=True)
class TransformEntitySystem(EntitySystem):
    """Express one entity's rows in another coordinate frame, as a new entity.

    The source is left alone, so provenance survives and the two can be compared. Writing a
    separate entity is also forced rather than stylistic: ``Store.range`` concatenates an
    entity's chunks, and refuses to when they state different frames.

    Each row's pose is looked up at that row's own time, so an object seen from a moving
    ego lands at a different ``map`` position in every frame. What moves is decided per
    component by :func:`~t4perceval.transform.apply.transform_kind`, not per archetype:

    - positions, segmentation points and waypoints are rotated and translated -- every
      waypoint of a chunk uses that chunk's own transform;
    - **velocity is rotated only, never translated.** That is the velocity a frame-fixed
      observer in the target frame would measure *if the two frames did not move relative
      to each other*; the relative motion between them is ignored;
    - orientations are composed with the frame's rotation;
    - sizes, ids, scores and everything else are carried unchanged;
    - **``MASK`` columns are dropped.** A distance or region mask is a claim about the
      source frame, and would be silently wrong in the new one. Filter the result instead.

    Static columns of the source are **not** carried: ``range`` never includes them, they
    are not frame-dependent in practice (a trajectory time axis, say), and a static frame
    claim would be wrong in the new frame. Re-log them on the target if they are needed.

    The result is a passthrough of the source minus ``MASK``, so a consumer -- a filter, a
    matcher, a metric -- can read the target in the same :class:`~t4perceval.system.Pipeline`.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = ()
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...] | Passthrough] = Passthrough(
        0,
        drops=(MASK,),
    )

    target_frame: str = field(kw_only=True)
    """The frame to express the rows in; becomes the target's ``frame_id``."""

    resolver: TransformResolver | None = field(default=None, kw_only=True)
    """Where transforms are looked up.

    ``None`` builds one over ``ctx.store`` on ``ctx.timeline`` when the system runs, which
    is enough when the evaluation store carries the ``/tf`` edges (name them in a
    ``SourceSpec``). Pass one built from the original recording when it does not, or when
    the edges live on another timeline -- a bag's ``/tf`` is on ``TIMESTAMP`` only. The
    lookup time of each row is read from the resolver's timeline, whichever it is.
    """

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 1:
            raise ValueError(
                f"{type(self).__name__} needs exactly one source, got {len(self.sources)}",
            )
        if not self.target_frame:
            raise ValueError(f"{type(self).__name__} needs a non-empty target_frame")

    @classmethod
    def of(
        cls,
        source: EntityPathLike,
        *,
        target_frame: str,
        target: EntityPathLike | None = None,
        resolver: TransformResolver | None = None,
    ) -> Self:
        """Express ``source`` in ``target_frame``, writing to ``target``.

        The target defaults to ``<source>/in/<target_frame>``, keeping the result beside
        the entity it came from. A frame name may contain ``/`` and an entity path segment
        may not, so such a frame needs an explicit ``target``.
        """
        path = as_entity_path(source)
        if target is None:
            if "/" in target_frame:
                raise ValueError(
                    f"target_frame {target_frame!r} contains '/', which an entity path "
                    f"segment cannot; pass target= explicitly",
                )
            target = path / "in" / target_frame
        return cls((path,), target, target_frame=target_frame, resolver=resolver)

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        (source,) = self.sources
        if not resolve_times(ctx, source, at):
            # Nothing in range: nothing to write, as a matcher does. (`range` would answer
            # with one empty, frame-less partition, which is not a frame of this entity.)
            return ()
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)

        chunk = ctx.store.range(source, timeline=ctx.timeline, time_range=time_range).to_chunk()
        source_frame = chunk.frame_id
        if source_frame is None:
            raise ValueError(
                f"{source} states no coordinate frame, so {type(self).__name__} cannot "
                f"bring it into {self.target_frame!r}. Pass frame_id= when logging it.",
            )

        if source_frame == self.target_frame:
            # Nothing to look up -- and nothing to look it up in: a store with no `/tf`
            # data at all is fine when the rows are already where they are wanted.
            return (
                transform_chunk(
                    chunk,
                    identity(),
                    target_frame=self.target_frame,
                    entity_path=self.target,
                ),
            )

        resolver = self.resolver
        if resolver is None:
            resolver = TransformResolver.of(ctx.store, timeline=ctx.timeline)
        index = chunk.index(resolver.timeline)
        if index is None:
            raise ValueError(
                f"{source} has no {resolver.timeline.name!r} index, which the resolver looks "
                f"transforms up on; build the resolver on a timeline the entity is logged on",
            )

        pieces = []
        for partition in range(chunk.num_partitions):
            piece = chunk.select_partitions([partition])
            if piece.num_rows == 0:
                # Nothing to move, so no pose to look up -- a frame with no objects still
                # keeps its place, and its frame claim is still rewritten.
                pose = identity()
            else:
                time = int(index.times[partition])
                try:
                    pose = pose_of(
                        resolver.lookup(
                            target_frame=self.target_frame,
                            source_frame=source_frame,
                            at=time,
                        ),
                    )
                except ValueError as error:
                    raise ValueError(
                        f"{type(self).__name__} cannot bring {source} from {source_frame!r} "
                        f"into {self.target_frame!r} at {resolver.timeline.name}={time}: "
                        f"{error}",
                    ) from error
            pieces.append(
                transform_chunk(
                    piece,
                    pose,
                    target_frame=self.target_frame,
                    entity_path=self.target,
                ),
            )
        return (concat_chunks(pieces),)
