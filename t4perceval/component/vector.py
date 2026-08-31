from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define

from t4perceval.core.component import ColumnarComponent

if TYPE_CHECKING:
    from t4perceval.typing import NDArrayF64

__all__ = ("BatchVector2D", "BatchVector3D")

# NOTE: `BatchVector3D` deliberately does NOT inherit from `BatchVector2D`. A 3D vector
# is not a 2D vector, and this package treats `isinstance` as a semantic claim.


@define(frozen=True, slots=True)
class BatchVector2D(ColumnarComponent):
    """Columnar 2D vectors with shape ``(N, 2)``."""

    SHAPE = (2,)

    @property
    def x(self) -> NDArrayF64:
        return self.values[:, 0]

    @property
    def y(self) -> NDArrayF64:
        return self.values[:, 1]


@define(frozen=True, slots=True)
class BatchVector3D(ColumnarComponent):
    """Columnar 3D vectors with shape ``(N, 3)``."""

    SHAPE = (3,)

    @property
    def x(self) -> NDArrayF64:
        return self.values[:, 0]

    @property
    def y(self) -> NDArrayF64:
        return self.values[:, 1]

    @property
    def z(self) -> NDArrayF64:
        return self.values[:, 2]
