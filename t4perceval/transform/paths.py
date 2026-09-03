"""Where transform edges live, and how to find them again.

One edge is one entity: ``/transforms/<parent>/<child>``. The frame pair is the identity
of the data, so it belongs in the entity path rather than in a column -- which also avoids
a string column in a model where every component is a numeric array, and gives each edge
its own temporal series at whatever rate it was recorded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from t4perceval.core.entity import EntityPath, as_entity_path

if TYPE_CHECKING:
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.store import Store
    from t4perceval.recording import Recording

__all__ = ("DEFAULT_ROOT", "edges", "frames_of", "transform_path")

#: Where transform edges are recorded by default.
DEFAULT_ROOT: Final = EntityPath.parse("/transforms")

#: An edge path is ``<root>/<parent>/<child>``, so two segments below the root.
_DEPTH_BELOW_ROOT: Final = 2

_SEPARATOR: Final = "/"


def transform_path(
    parent: str,
    child: str,
    *,
    root: EntityPathLike = DEFAULT_ROOT,
) -> EntityPath:
    """Return the entity a ``parent -> child`` transform is recorded at.

    Args:
        parent: Frame the pose is expressed in.
        child: Frame whose pose is recorded.
        root: Where transforms live.

    Returns:
        ``<root>/<parent>/<child>``.

    Raises:
        ValueError: When either frame name is empty, or the two are the same.
    """
    for name in (parent, child):
        if not name:
            raise ValueError(f"Frame names must not be empty, got {parent!r} -> {child!r}")
        if _SEPARATOR in name:
            # `EntityPath.__truediv__` parses its argument, so a name containing a
            # separator would silently become two segments and the edge would no longer
            # be findable. ROS frame names may contain one; encode them at the boundary
            # if that ever arrives, rather than discovering it as a missing edge.
            raise ValueError(
                f"A frame name must not contain {_SEPARATOR!r}, got {name!r}",
            )
    if parent == child:
        raise ValueError(f"A transform needs two different frames, got {parent!r} twice")
    return as_entity_path(root) / parent / child


def frames_of(path: EntityPathLike, *, root: EntityPathLike = DEFAULT_ROOT) -> tuple[str, str]:
    """Return the ``(parent, child)`` an edge path names.

    Raises:
        ValueError: When the path is not an edge below ``root``.
    """
    entity = as_entity_path(path)
    prefix = as_entity_path(root)
    if not entity.starts_with(prefix) or len(entity) != len(prefix) + _DEPTH_BELOW_ROOT:
        raise ValueError(
            f"{entity} is not a transform edge; expected {prefix}/<parent>/<child>",
        )
    return entity.parts[-2], entity.parts[-1]


def edges(
    source: Store | Recording,
    *,
    root: EntityPathLike = DEFAULT_ROOT,
) -> dict[tuple[str, str], EntityPath]:
    """Return every recorded edge, keyed by ``(parent, child)``.

    This is the whole of frame-graph discovery: the store already knows which entities
    exist, and an edge path already says which frames it relates, so nothing needs to be
    read or indexed to enumerate the graph.

    Entities under ``root`` that are not exactly two segments deep are ignored rather than
    rejected, so an unrelated entity filed nearby cannot break discovery.
    """
    prefix = as_entity_path(root)
    return {
        (path.parts[-2], path.parts[-1]): path
        for path in source.entity_paths()
        if path.starts_with(prefix) and len(path) == len(prefix) + _DEPTH_BELOW_ROOT
    }
