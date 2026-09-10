"""Semantic segmentation: a class per element, where the element is the row.

Neither archetype names its elements. Estimation and ground truth correspond because they
enumerate the same elements in the same order -- the pixels of one image, the points of one
cloud -- exactly as a mask corresponds to the entity it was computed over. That is what
lets a segmentation metric compare the two without a matching stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import BatchClassId, BatchPosition3D
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import CLASS_ID, POINT

if TYPE_CHECKING:
    from typing_extensions import Self

    from t4perceval.typing import ArrayLike, NDArrayI32

__all__ = ("SemanticSegmentation2D", "SemanticSegmentation3D")


@define(frozen=True, slots=True)
class SemanticSegmentation2D(Archetype):
    """A class per pixel of an image, in row-major order.

    The row *is* the pixel: row ``i`` is pixel ``(i // width, i % width)``, and the entity
    holds ``height * width`` rows per image. The image size is not a column -- it would be
    the same value repeated per pixel -- but a one-row **static** component on the entity::

        store.log(path, SemanticSegmentation2D.from_label_map(labels), at=..., frame_id="CAM_FRONT")
        store.log_static_components(path, {IMAGE_SIZE: BatchImageSize([[height, width]])})

    :meth:`from_label_map` and :meth:`as_label_map` convert to and from the ``(H, W)``
    image. ``frame_id`` is the camera channel, as for every 2D archetype.
    """

    class_id = component_field(CLASS_ID, BatchClassId)

    @classmethod
    def from_label_map(cls, labels: ArrayLike) -> Self:
        """Build from an ``(H, W)`` label image, flattened row-major."""
        image = np.asarray(labels)
        if image.ndim != 2:
            raise ValueError(f"A label map must be 2-D (height, width), got shape {image.shape}")
        return cls(class_id=image.reshape(-1))

    def as_label_map(self, height: int, width: int) -> NDArrayI32:
        """Return the labels as an ``(H, W)`` image."""
        if len(self) != height * width:
            raise ValueError(
                f"{len(self)} label(s) do not fill a {height}x{width} image "
                f"({height * width} pixels)",
            )
        return self.class_id.values.reshape(height, width)


@define(frozen=True, slots=True)
class SemanticSegmentation3D(Archetype):
    """A class per labelled point.

    ``point`` uses the ``POINT`` descriptor rather than ``POSITION``: a labelled point is
    not an object with a pose, and the separate name stops an object filter from being
    pointed at a point cloud and appearing to work. A coordinate transform still moves it
    as a point.
    """

    point = component_field(POINT, BatchPosition3D)
    class_id = component_field(CLASS_ID, BatchClassId)
