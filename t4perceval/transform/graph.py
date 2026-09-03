"""Finding the frame graph a store recorded.

An edge is a ``Transform3D`` row: its parent is the chunk's ``frame_id`` and its child is
the ``child_frame_id`` column. Discovery therefore *reads* the data -- unlike the earlier
design, where the frame pair was encoded in the entity path and enumerating the graph
needed nothing but the list of paths. That cost is the price of frames being data: a path
is where something is filed, and filing must not decide what a frame is called.

The read is small. A chunk that has no ``child_frame_id`` is skipped on a dict lookup, and
one that has it holds a single edge per row -- so this is O(edge samples), never O(objects).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Final

from attrs import define, field

from t4perceval.core.entity import EntityPath, as_entity_path
from t4perceval.descriptors import CHILD_FRAME_ID

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.store import Store
    from t4perceval.recording import Recording

__all__ = ("DEFAULT_ROOT", "FrameGraph", "TransformEdge", "transform_edges")

#: Where transform entities are filed by convention. Only a convention: an edge is found
#: by reading a chunk, so a transform recorded anywhere is found with ``root=None``.
DEFAULT_ROOT: Final = EntityPath.parse("/tf")


@define(frozen=True, slots=True)
class TransformEdge:
    """One recorded ``parent -> child`` relationship, and where to read it."""

    parent: str
    """Frame the transform is expressed in -- the chunk's ``frame_id``."""

    child: str
    """Frame whose pose is recorded -- one value of the ``child_frame_id`` column."""

    entity_path: EntityPath = field(converter=as_entity_path)
    """Entity holding the samples."""

    is_static: bool = field(default=False, kw_only=True)
    """Whether the samples are static, i.e. not on any timeline."""

    @property
    def frames(self) -> tuple[str, str]:
        """Return ``(parent, child)``."""
        return self.parent, self.child


def _children_of(chunk: Chunk) -> Iterator[str]:
    """Yield each distinct child frame a chunk names, in row order.

    Usually one: a :class:`~t4perceval.archetype.transform.Transform3D` is one edge, so a
    chunk written through it holds a single row. A chunk assembled by hand may name several
    children of one parent, and reading them all is what keeps discovery from silently
    seeing only the first.
    """
    column = chunk.columns.get(CHILD_FRAME_ID)
    if column is None:
        return
    seen: dict[str, None] = {}
    for name in column.values.tolist():
        if name not in seen:
            seen[name] = None
            yield name


def transform_edges(
    source: Store | Recording,
    *,
    root: EntityPathLike | None = DEFAULT_ROOT,
) -> tuple[TransformEdge, ...]:
    """Return every transform edge ``source`` holds, static edges first per entity.

    Args:
        source: Store or recording to read.
        root: Only look at entities at or below this path. ``None`` scans every entity.

    Returns:
        One edge per distinct ``(parent, child)``, in discovery order -- an entity sampled
        over many frames contributes one edge, not one per sample.

    Raises:
        ValueError: When a chunk names a child frame but states no ``frame_id``, so its
            parent is unknown; or when two places record the same ``(parent, child)``.

    Note:
        A chunk without a ``child_frame_id`` column is ignored rather than rejected, so an
        unrelated entity filed under ``root`` cannot break discovery.

        An absent parent *does* raise, unlike
        :func:`~t4perceval.system.base.require_same_frame`, where an unstated frame is
        merely "no opinion". That rule is about comparing two things; this one is about
        interpreting one: a transform with no parent frame describes nothing at all.
    """
    prefix = None if root is None else as_entity_path(root)
    found: dict[tuple[str, str], TransformEdge] = {}

    for path in source.entity_paths():
        if prefix is not None and not path.starts_with(prefix):
            continue
        static = source.static_chunks(path)
        for chunk in (*static, *source.chunks(path)):
            if CHILD_FRAME_ID not in chunk.columns:
                continue
            if chunk.frame_id is None:
                raise ValueError(
                    f"{path} records transforms but states no frame_id, so the parent "
                    f"frame of its edges is unknown. Pass frame_id= when logging it.",
                )
            for child in _children_of(chunk):
                edge = TransformEdge(chunk.frame_id, child, path, is_static=chunk.is_static)
                seen = found.get(edge.frames)
                if seen is None:
                    found[edge.frames] = edge
                elif seen != edge:
                    raise ValueError(
                        f"{edge.parent!r} -> {edge.child!r} is recorded twice, at "
                        f"{seen.entity_path} and {edge.entity_path}. Nothing could choose "
                        f"between them, so one of the two has to go.",
                    )

    return tuple(found.values())


@define(frozen=True, slots=True)
class FrameGraph:
    """The frames a recording knows about, and how to get between them.

    Edges are traversed in both directions: a rigid transform inverts exactly, so
    recording ``base_link -> lidar`` also answers "where is ``base_link`` in ``lidar``".
    """

    edges: tuple[TransformEdge, ...] = field(converter=tuple)

    @classmethod
    def of(
        cls,
        source: Store | Recording,
        *,
        root: EntityPathLike | None = DEFAULT_ROOT,
    ) -> Self:
        """Build the graph a store or recording holds."""
        return cls(transform_edges(source, root=root))

    def frames(self) -> tuple[str, ...]:
        """Return every frame named by an edge, in discovery order."""
        seen: dict[str, None] = {}
        for edge in self.edges:
            seen.setdefault(edge.parent, None)
            seen.setdefault(edge.child, None)
        return tuple(seen)

    def edge(self, parent: str, child: str) -> TransformEdge | None:
        """Return the edge recording ``parent -> child``, or ``None``."""
        for edge in self.edges:
            if edge.frames == (parent, child):
                return edge
        return None

    def path(
        self, *, target_frame: str, source_frame: str
    ) -> tuple[tuple[TransformEdge, bool], ...]:
        """Return the hops that carry a point from ``source_frame`` into ``target_frame``.

        Each hop is ``(edge, inverted)``: ``inverted`` marks an edge walked against the
        direction it was recorded in, which is exact for a rigid transform.

        Returns:
            The hops in application order -- the first is applied to a point in
            ``source_frame``. Empty when the two frames are the same.

        Raises:
            ValueError: When either frame is unknown, or no chain of edges connects them.

        Note:
            Breadth-first, so the answer uses the fewest edges; every composition
            compounds floating-point error, and no edge is "cheaper" than another. Ties
            between equal-length chains are broken by discovery order, which makes the
            result deterministic but arbitrary -- a frame tree should not offer two ways
            round in the first place.
        """
        if source_frame == target_frame:
            return ()

        known = self.frames()
        for frame in (target_frame, source_frame):
            if frame not in known:
                raise ValueError(
                    f"Unknown coordinate frame {frame!r}; this recording knows {list(known)}",
                )

        # Adjacency: from -> (to, edge, inverted). An edge maps its child into its parent,
        # so walking child -> parent uses it as recorded and parent -> child inverts it.
        adjacency: dict[str, list[tuple[str, TransformEdge, bool]]] = {}
        for edge in self.edges:
            adjacency.setdefault(edge.child, []).append((edge.parent, edge, False))
            adjacency.setdefault(edge.parent, []).append((edge.child, edge, True))

        queue: deque[tuple[str, tuple[tuple[TransformEdge, bool], ...]]] = deque(
            [(source_frame, ())],
        )
        visited = {source_frame}
        while queue:
            frame, hops = queue.popleft()
            for neighbour, edge, inverted in adjacency.get(frame, ()):
                if neighbour in visited:
                    continue
                walked = (*hops, (edge, inverted))
                if neighbour == target_frame:
                    return walked
                visited.add(neighbour)
                queue.append((neighbour, walked))

        raise ValueError(
            f"No recorded transform connects {source_frame!r} to {target_frame!r}. "
            f"Known frames: {list(known)}.",
        )
