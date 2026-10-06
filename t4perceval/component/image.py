from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define

from t4perceval.component.scalar import BatchClassId
from t4perceval.core.component import ANY, Component, MonoComponent

if TYPE_CHECKING:
    from t4perceval.typing import NDArrayI32

__all__ = ("BatchClassIdImage", "BatchRoi", "ClassIdImage")


@define(frozen=True, slots=True)
class BatchRoi(Component):
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
class BatchClassIdImage(Component):
    """Class-id images with shape ``(N, H, W)``: one ``(H, W)`` label image per row.

    ``H`` and ``W`` are wildcard dimensions, inferred once per column like every other
    wildcard, so every image in a column -- and therefore every image on one entity -- has
    the same resolution. That is what makes the image size a property of the entity rather
    than a column to log alongside it: a range query that spans a resolution change fails
    in ``concat_chunks`` instead of silently misaligning pixels. Pixel ``(r, c)`` of image
    ``i`` is ``values[i, r, c]``; :meth:`as_class_id` is the row-major flattening the
    segmentation metrics consume.

    ``height`` and ``width`` are plain ints, not per-row arrays as on :class:`BatchRoi`,
    because they are the same for every row by construction.
    """

    SHAPE = (ANY, ANY)
    DTYPE = np.int32

    def __attrs_post_init__(self) -> None:
        # A zero-sized side is not an image, and Arrow cannot encode a fixed-size list of
        # size 0 -- so refuse it here rather than at write time.
        if 0 in self.row_shape:
            raise ValueError(
                f"{type(self).__name__} images need a non-zero height and width, "
                f"got {self.row_shape}",
            )

    @property
    def height(self) -> int:
        return self.row_shape[0]

    @property
    def width(self) -> int:
        return self.row_shape[1]

    def num_pixels(self) -> int:
        """Return ``height * width``, the pixels of one image."""
        return self.height * self.width

    def as_class_id(self) -> BatchClassId:
        """Return every pixel as a :class:`BatchClassId` of ``N * H * W`` rows.

        Images are concatenated in row order and each is flattened row-major, so pixel
        ``(r, c)`` of image ``i`` is row ``i * H * W + r * W + c``. No copy is made.
        """
        return BatchClassId(self.values.reshape(-1))


@define(frozen=True, slots=True)
class ClassIdImage(BatchClassIdImage, MonoComponent):
    """One ``(H, W)`` class-id image, written and read without a row axis.

    The mono counterpart of :class:`BatchClassIdImage`, for data that is singular by
    nature: an entity holds one label image per point in time, as it holds one transform.
    Stored as a :class:`BatchClassIdImage` of one row.
    """

    BATCH: ClassVar[type[Component]] = BatchClassIdImage
