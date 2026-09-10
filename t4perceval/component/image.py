from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define

from t4perceval.core.component import ColumnarComponent

if TYPE_CHECKING:
    from t4perceval.typing import NDArrayI32

__all__ = ("BatchImageSize", "BatchRoi")


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
class BatchImageSize(ColumnarComponent):
    """Image sizes as ``(height, width)`` with shape ``(N, 2)``.

    A segmentation label image is stored as one class per pixel in row-major order, so
    the rows say nothing about the image's shape. This column does: it is logged **once,
    static**, on the entity -- ``BatchImageSize([[height, width]])`` -- and
    :meth:`~t4perceval.core.view.EntityView.component` broadcasts the single row over
    every element. Read it un-broadcast with ``store.static(path)[IMAGE_SIZE]``.
    """

    SHAPE = (2,)
    DTYPE = np.int32

    def __attrs_post_init__(self) -> None:
        if self.values.size and np.any(self.values < 0):
            raise ValueError("BatchImageSize must contain only non-negative values")

    @property
    def height(self) -> NDArrayI32:
        return self.values[:, 0]

    @property
    def width(self) -> NDArrayI32:
        return self.values[:, 1]

    def num_pixels(self) -> NDArrayI32:
        """Return ``height * width`` of each image."""
        return self.height * self.width
