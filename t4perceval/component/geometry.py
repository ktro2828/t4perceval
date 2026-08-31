from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define
from scipy.spatial.transform import Rotation

from t4perceval.core.component import ColumnarComponent
from t4perceval.component.vector import BatchVector2D, BatchVector3D

if TYPE_CHECKING:
    from typing_extensions import Self

    from t4perceval.typing import NDArrayF64

__all__ = (
    "BatchPosition2D",
    "BatchPosition3D",
    "BatchQuaternion",
    "BatchSize2D",
    "BatchSize3D",
    "BatchVelocity",
)


@define(frozen=True, slots=True)
class BatchPosition3D(BatchVector3D):
    """Columnar 3D positions with shape ``(N, 3)``."""


@define(frozen=True, slots=True)
class BatchPosition2D(BatchVector2D):
    """Columnar 2D positions with shape ``(N, 2)``."""


@define(frozen=True, slots=True)
class BatchVelocity(BatchVector3D):
    """Columnar 3D velocities with shape ``(N, 3)``."""

    @property
    def speed(self) -> NDArrayF64:
        """Return the L2 norm of each velocity vector."""
        return np.linalg.norm(self.values, axis=1)


@define(frozen=True, slots=True)
class BatchSize3D(BatchVector3D):
    """Columnar 3D sizes with shape ``(N, 3)`` in ``(width, length, height)`` order."""


@define(frozen=True, slots=True)
class BatchSize2D(BatchVector2D):
    """Columnar 2D sizes with shape ``(N, 2)``."""


@define(frozen=True, slots=True)
class BatchQuaternion(ColumnarComponent):
    """Columnar unit quaternions in ``xyzw`` order with shape ``(N, 4)``."""

    SHAPE = (4,)

    @property
    def qx(self) -> NDArrayF64:
        return self.values[:, 0]

    @property
    def qy(self) -> NDArrayF64:
        return self.values[:, 1]

    @property
    def qz(self) -> NDArrayF64:
        return self.values[:, 2]

    @property
    def qw(self) -> NDArrayF64:
        return self.values[:, 3]

    def as_rotation(self) -> Rotation:
        """Return a SciPy rotation built from the ``xyzw`` values."""
        return Rotation.from_quat(self.values)

    def normalized(self) -> Self:
        """Return a component whose quaternions all have unit norm."""
        norms = np.linalg.norm(self.values, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("Cannot normalize a zero quaternion")
        return type(self)(self.values / norms)

    def yaw(self) -> NDArrayF64:
        """Return the yaw angle of each rotation in radians."""
        if len(self) == 0:
            return np.empty(0, dtype=np.float64)
        return self.as_rotation().as_euler("xyz")[:, 2]
