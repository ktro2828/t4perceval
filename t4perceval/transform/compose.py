"""Composing, inverting and interpolating rigid transforms.

Kept out of :class:`~t4perceval.archetype.transform.Transform3D` for the same reason
:mod:`t4perceval.geometry` is kept out of the object archetypes: an archetype describes a
column layout, and the maths that consumes it is separate and testable on its own.

A pose here is a pair ``(translation, rotation)`` of a ``(3,)`` and a ``(4,)`` array. The
quaternion is ``xyzw``, which is both :class:`~t4perceval.component.BatchQuaternion`'s
order and SciPy's native one -- so nothing in this module reorders anything. The T4
importer *does* reorder, at the dataset boundary, which is the only place it happens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

if TYPE_CHECKING:
    from collections.abc import Iterable

    from t4perceval.typing import NDArrayF64

__all__ = ("Pose", "compose", "chain", "identity", "interpolate", "invert")

Pose: TypeAlias = "tuple[NDArrayF64, NDArrayF64]"
"""A translation ``(3,)`` and an ``xyzw`` rotation ``(4,)``."""


def _writable(vector: NDArrayF64) -> NDArrayF64:
    """Return a writable float64 copy of a vector about to be passed to SciPy.

    Component arrays are read-only by design, and ``Rotation.apply`` in SciPy 1.17.x
    rejects a read-only buffer with ``ValueError: buffer source array is read-only``
    (1.15 and 1.18 accept it). SciPy 1.18 requires Python 3.12, so on 3.11 the 1.17 series
    is the newest available and the copy -- three floats -- is what keeps the lookup
    working there.
    """
    return np.array(vector, dtype=np.float64)


def identity() -> Pose:
    """Return the pose that maps every point to itself."""
    return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])


def invert(pose: Pose) -> Pose:
    """Return the pose mapping the other way.

    Exact for a rigid transform, which is what lets the frame graph walk every recorded
    edge in both directions.
    """
    translation, rotation = pose
    inverse = Rotation.from_quat(np.asarray(rotation, dtype=np.float64)).inv()
    return -inverse.apply(_writable(translation)), inverse.as_quat()


def compose(outer: Pose, inner: Pose) -> Pose:
    """Return the pose applying ``inner`` first, then ``outer``.

    ``p -> R_outer (R_inner p + t_inner) + t_outer``, i.e. matrix ``outer @ inner``.
    """
    outer_t, outer_q = outer
    inner_t, inner_q = inner
    outer_r = Rotation.from_quat(np.asarray(outer_q, dtype=np.float64))
    inner_r = Rotation.from_quat(np.asarray(inner_q, dtype=np.float64))
    translation = outer_r.apply(_writable(inner_t)) + np.asarray(outer_t, dtype=np.float64)
    return translation, (outer_r * inner_r).as_quat()


def chain(poses: Iterable[Pose]) -> Pose:
    """Return the pose applying each of ``poses`` in the order given.

    The first pose listed acts on the point first, so a chain of hops from a source frame
    to a target frame can be passed straight in.
    """
    result = identity()
    for pose in poses:
        result = compose(pose, result)
    return result


def interpolate(before: Pose, after: Pose, *, fraction: float) -> Pose:
    """Return the pose ``fraction`` of the way from ``before`` to ``after``.

    Translation is linear and rotation is a great-circle interpolation
    (:class:`scipy.spatial.transform.Slerp`), which is the shortest path on the rotation
    group -- component-wise interpolation of a quaternion is neither shortest nor unit.
    """
    before_t, before_q = before
    after_t, after_q = after
    rotations = Rotation.from_quat(
        np.stack(
            [
                np.asarray(before_q, dtype=np.float64),
                np.asarray(after_q, dtype=np.float64),
            ],
        ),
    )
    rotation = Slerp([0.0, 1.0], rotations)([fraction]).as_quat()[0]
    translation = (1.0 - fraction) * np.asarray(before_t, dtype=np.float64) + (
        fraction * np.asarray(after_t, dtype=np.float64)
    )
    return translation, rotation
