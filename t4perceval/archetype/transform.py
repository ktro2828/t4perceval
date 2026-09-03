"""The pose of one coordinate frame in another."""

from __future__ import annotations

from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import BatchPosition3D, BatchQuaternion
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import ROTATION, TRANSLATION

__all__ = ("Transform3D",)


@define(frozen=True, slots=True)
class Transform3D(Archetype):
    """The pose of a child frame expressed in its parent frame.

    A row maps a point in the child frame into the parent: ``p_parent = R p_child + t``.

    Which two frames those are is **not** in the columns -- it is the entity path,
    ``/transforms/<parent>/<child>``. The frames are the identity of this data, which is
    what an entity path is for, and keeping them out of the columns avoids a string column
    in a data model where every component is a numeric array. It also means one edge is
    one entity, so :meth:`~t4perceval.core.store.Store.latest_at` answers "where was the
    ego at this time" exactly, per edge, at whatever rate each edge was recorded.

    Note:
        A fixed extrinsic should still be logged as a single *temporal* sample rather than
        with ``log_static``. Static data carries no ``frame_id`` and, on an entity with no
        temporal rows, reads back as zero rows -- whereas one sample at the first frame is
        returned by ``latest_at`` at every later time, which is exactly the intended
        meaning and needs no second code path.
    """

    translation = component_field(TRANSLATION, BatchPosition3D)
    rotation = component_field(ROTATION, BatchQuaternion)
