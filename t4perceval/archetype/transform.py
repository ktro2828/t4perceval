"""The pose of one coordinate frame in another."""

from __future__ import annotations

from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import FrameId, Position3D, Quaternion
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import CHILD_FRAME_ID, ROTATION, TRANSLATION

__all__ = ("Transform3D",)


@define(frozen=True, slots=True)
class Transform3D(Archetype):
    """The pose of a child frame expressed in its parent frame.

    A point in the child frame maps into the parent as ``p_parent = R p_child + t``.

    Examples:
        >>> Transform3D(translation=[1.2, 0.0, 1.8], rotation=[0.0, 0.0, 0.0, 1.0],
        ...             child_frame_id="lidar").child_frame_id.name
        'lidar'

    The two frames are split between the chunk and a column, which is how ROS splits them:

    ==============================  =============================
    ROS ``TransformStamped``        ``t4perceval``
    ==============================  =============================
    ``header.frame_id``             ``Chunk.frame_id`` (parent)
    ``child_frame_id``              :attr:`child_frame_id`
    ``transform.translation``       :attr:`translation`
    ``transform.rotation``          :attr:`rotation`
    ==============================  =============================

    The parent belongs on the chunk because ``frame_id`` already means "the frame these
    rows are expressed in", and that is exactly true of a transform's row.

    Every component here is **mono** (:class:`~t4perceval.core.component.MonoComponent`),
    unlike every other archetype in the package. Those describe *N* objects; this describes
    one relationship, of which an entity holds exactly one per point in time. So the
    translation is a ``(3,)`` value rather than an ``(N, 3)`` column, and the frame name is
    a ``str`` -- there is no row to index into, and "what if it has three rows?" is not a
    question the type can be asked. Underneath it is still a one-row column, so the chunk,
    the Arrow schema and the store are unaffected.

    Neither frame is in the entity path. A path says where data is filed; a frame names a
    node of the transform graph, and conflating them means a frame name has to be
    path-safe and a graph cannot be re-filed without being renamed.

    ``child_frame_id`` is required rather than optional: a transform whose parent is known
    but whose child is not describes nothing, and the resolver would have to defend against
    it on every hop.

    Note:
        A fixed extrinsic is logged with ``log_static``, an ego pose with ``log`` -- the
        same archetype either way. Static is a statement about *time*, not about the kind
        of data, and a static write keeps its ``frame_id``, so nothing about the edge is
        lost. Static rows do not surface through ``latest_at`` or ``range``, which is why
        the frame graph is built from
        :meth:`~t4perceval.core.store.Store.static_chunks` and
        :meth:`~t4perceval.core.store.Store.chunks` rather than from a time query.
    """

    translation = component_field(TRANSLATION, Position3D)
    rotation = component_field(ROTATION, Quaternion)
    child_frame_id = component_field(CHILD_FRAME_ID, FrameId)
