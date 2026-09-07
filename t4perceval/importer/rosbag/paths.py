"""Where imported bag data lands in the entity hierarchy.

A topic names a message *source*; an entity path names a semantic *location*. They are
not the same concept, so the importer does not mint a path from the topic string: every
object topic lands at ``<root>/objects``, and which topic it came from is recorded in
``SourceInfo.topic`` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from t4perceval.core.entity import EntityPath, as_entity_path

if TYPE_CHECKING:
    from t4perceval.core.entity import EntityPathLike

__all__ = ("DEFAULT_ROOT", "objects3d_path", "topic_entity_path")

#: A bag holds a perception stack's output, so its objects are estimations by default.
DEFAULT_ROOT: Final = EntityPath.parse("/estimation")


def objects3d_path(root: EntityPathLike = DEFAULT_ROOT) -> EntityPath:
    """Return the path a topic's 3D objects are logged to, ``/estimation/objects``.

    One path regardless of topic: a recording holds one topic, and the coordinate frame the
    objects are expressed in is already recorded truthfully in ``Chunk.frame_id``.
    """
    return as_entity_path(root) / "objects"


def topic_entity_path(topic: str, *, root: EntityPathLike = DEFAULT_ROOT) -> EntityPath:
    """Return ``<root>/<topic>`` for callers who *do* want one entity per topic.

    Not what the importer does by default -- see the module docstring -- but the natural
    filing when several topics of one bag are imported into one store, where they must not
    collide on ``<root>/objects``.
    """
    stripped = topic.strip("/")
    if not stripped:
        raise ValueError("topic must not be empty")
    return as_entity_path(root) / stripped
