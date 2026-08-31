from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define

from t4perceval.core.component import ColumnarComponent

if TYPE_CHECKING:
    from t4perceval.typing import NDArrayI32

__all__ = ("BatchPixel", "BatchRoi")


@define(frozen=True, slots=True)
class BatchRoi(ColumnarComponent):
    """Columnar 2D regions of interest with shape ``(N, 4)``.

    The layout is ``(x_min, y_min, height, width)``, matching the original
    ``perception_eval`` convention.
    """

    SHAPE = (4,)
    DTYPE = np.int32

    @property
    def x_min(self) -> NDArrayI32:
        return self.values[:, 0]

    @property
    def y_min(self) -> NDArrayI32:
        return self.values[:, 1]

    @property
    def height(self) -> NDArrayI32:
        return self.values[:, 2]

    @property
    def width(self) -> NDArrayI32:
        return self.values[:, 3]

    @property
    def x_max(self) -> NDArrayI32:
        return self.x_min + self.width

    @property
    def y_max(self) -> NDArrayI32:
        return self.y_min + self.height

    def area(self) -> NDArrayI32:
        """Return the pixel area of each ROI."""
        return self.height * self.width


@define(frozen=True, slots=True)
class BatchPixel(ColumnarComponent):
    """Flat pixel indices with shape ``(N,)``.

    A pixel index is ``row * image_width + column``; the image shape is carried by the
    owning archetype so the column stays a plain scalar column.
    """

    DTYPE = np.int32
