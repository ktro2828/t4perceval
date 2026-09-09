"""Applying a resolved pose to the columns of a chunk.

Which columns move, and how, is a property of the *component*, not of the archetype that
bundled it: a position is a point wherever it appears -- an object centre, a segmentation
point, the translation of a transform edge -- a velocity is a direction, a size is neither.
So the rule is keyed on the component class, and an archetype never has to say anything.

:func:`transform_chunk` is the pure step.
:class:`~t4perceval.system.transform.TransformEntitySystem` is the system that looks the
pose up and writes the result back as an entity.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from attrs import evolve
from scipy.spatial.transform import Rotation

from t4perceval.component import (
    BatchMask,
    BatchPosition3D,
    BatchQuaternion,
    BatchVelocity,
    BatchWaypoints3D,
)
from t4perceval.core.entity import as_entity_path
from t4perceval.transform.compose import _writable

if TYPE_CHECKING:
    from t4perceval.archetype.transform import Transform3D
    from t4perceval.core.chunk import Chunk
    from t4perceval.core.component import Component
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.transform.compose import Pose
    from t4perceval.typing import NDArrayF64

__all__ = ("TRANSFORM_KINDS", "TransformKind", "pose_of", "transform_chunk", "transform_kind")


class TransformKind(Enum):
    """How a column moves when its chunk is expressed in another frame."""

    POINT = auto()
    """``p' = R p + t`` -- a location in the frame."""

    DIRECTION = auto()
    """``v' = R v`` -- a vector with no origin, so the translation does not apply."""

    ROTATION = auto()
    """``q' = q_pose * q`` -- an orientation, composed with the frame's rotation."""

    INVARIANT = auto()
    """Unchanged: sizes, ids, scores -- anything that is not geometry *in* the frame."""

    DROP = auto()
    """Omitted: a claim about the source frame that does not survive the move."""


#: Keyed on the component class and resolved along the MRO, so the mono ``Position3D`` and
#: ``Quaternion`` inherit their batch class's kind. Deliberately *not* keyed on
#: ``BatchVector3D``: ``BatchSize3D`` shares that base and must not rotate. A component no
#: entry covers is INVARIANT; register one here to make a new component frame-dependent.
TRANSFORM_KINDS: dict[type[Component], TransformKind] = {
    BatchPosition3D: TransformKind.POINT,
    BatchWaypoints3D: TransformKind.POINT,
    BatchVelocity: TransformKind.DIRECTION,
    BatchQuaternion: TransformKind.ROTATION,
    BatchMask: TransformKind.DROP,
}


def transform_kind(component: Component | type[Component]) -> TransformKind:
    """Return how a column of this component moves; a type nothing knows is INVARIANT."""
    cls = component if isinstance(component, type) else type(component)
    for base in cls.__mro__:
        kind = TRANSFORM_KINDS.get(base)
        if kind is not None:
            return kind
    return TransformKind.INVARIANT


def pose_of(transform: Transform3D) -> Pose:
    """Return a resolver's answer as the ``(translation, rotation)`` pair the maths uses."""
    return (
        np.asarray(transform.translation.value, dtype=np.float64),
        np.asarray(transform.rotation.value, dtype=np.float64),
    )


def transform_chunk(
    chunk: Chunk,
    pose: Pose,
    *,
    target_frame: str,
    entity_path: EntityPathLike | None = None,
) -> Chunk:
    """Return ``chunk`` expressed in ``target_frame``.

    ``pose`` is the pose of the chunk's own frame *in* ``target_frame`` -- what
    ``TransformResolver.lookup(target_frame=..., source_frame=chunk.frame_id)`` answers.
    Every column moves by its :func:`transform_kind`: INVARIANT columns are shared rather
    than copied, DROP columns are omitted. Indexes, offsets and ``is_static`` are kept, so
    the result lines up row for row with the source, minus the dropped columns.

    An identity pose leaves every kept column untouched, bit for bit.

    Args:
        chunk: The rows to move.
        pose: Translation ``(3,)`` and ``xyzw`` rotation ``(4,)``.
        target_frame: The frame the result is expressed in; becomes its ``frame_id``.
        entity_path: Where the result is filed. Defaults to the chunk's own path.
    """
    translation, quaternion = pose
    rotation = None if _is_identity(pose) else Rotation.from_quat(_writable(quaternion))

    columns: dict = {}
    for descriptor, column in chunk.columns.items():
        kind = transform_kind(column)
        if kind is TransformKind.DROP:
            continue
        if rotation is None or kind is TransformKind.INVARIANT:
            columns[descriptor] = column
            continue
        columns[descriptor] = type(column)(_moved(column.values, kind, rotation, translation))

    return evolve(
        chunk,
        entity_path=as_entity_path(entity_path) if entity_path is not None else chunk.entity_path,
        columns=columns,
        frame_id=target_frame,
    )


def _is_identity(pose: Pose) -> bool:
    translation, rotation = pose
    return not np.any(translation) and bool(
        np.array_equal(np.abs(rotation), [0.0, 0.0, 0.0, 1.0]),
    )


def _moved(
    values: NDArrayF64,
    kind: TransformKind,
    rotation: Rotation,
    translation: NDArrayF64,
) -> NDArrayF64:
    """Apply ``rotation`` (and, for a POINT, ``translation``) to a column's values."""
    if kind is TransformKind.ROTATION:
        # The column maps object -> source and the pose maps source -> target, so the pose
        # is applied last: (p * q).apply(v) == p.apply(q.apply(v)) in SciPy.
        return (rotation * Rotation.from_quat(_writable(values))).as_quat()
    # Any trailing-3 shape -- (N, 3) positions or (N, M, T, 3) waypoints -- is a stack of
    # vectors; every one moves by the same pose.
    flat = rotation.apply(_writable(values.reshape(-1, 3)))
    if kind is TransformKind.POINT:
        flat = flat + translation
    return flat.reshape(values.shape)
