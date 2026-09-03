"""Where imported T4 data lands in the entity hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from t4perceval.core.entity import EntityPath, as_entity_path

if TYPE_CHECKING:
    from t4perceval.core.entity import EntityPathLike

__all__ = ("DEFAULT_ROOT", "objects2d_path", "objects3d_path")

#: A T4 dataset is ground truth, so that is where its objects land by default.
DEFAULT_ROOT: Final = EntityPath.parse("/ground_truth")


def objects3d_path(root: EntityPathLike = DEFAULT_ROOT) -> EntityPath:
    """Return the path 3D objects are logged to, ``/ground_truth/objects``.

    There is one 3D path per scene regardless of which sensor channel was used to fetch
    the boxes. A sample has exactly one set of 3D annotations; the channel only decides
    which coordinate frame they are expressed in, and that is already recorded truthfully
    in ``Chunk.frame_id``. Putting it in the path as well would duplicate it and make
    ``/ground_truth/objects`` unaddressable by a system that does not care.
    """
    return as_entity_path(root) / "objects"


def objects2d_path(root: EntityPathLike, channel: str) -> EntityPath:
    """Return the path one camera's 2D objects are logged to.

    Unlike 3D, 2D annotations genuinely differ per camera -- different rows, different
    counts -- so each camera gets its own entity. The channel is kept verbatim
    (``CAM_FRONT``, not ``cam_front``) so a path traces straight back to
    ``sensor.channel``.
    """
    if not channel:
        raise ValueError("channel must not be empty")
    return as_entity_path(root) / channel / "objects"
