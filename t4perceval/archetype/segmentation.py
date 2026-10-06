"""Semantic segmentation: a class per element, the pixel of one image or the point of one cloud.

Neither archetype names its elements. Estimation and ground truth correspond because they
enumerate the same elements in the same order -- the pixels of one image in row-major order,
the points of one cloud -- exactly as a mask corresponds to the entity it was computed over.
That is what lets a segmentation metric compare the two without a matching stage.
"""

from __future__ import annotations

from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import BatchClassId, BatchPosition3D, ClassIdImage
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import CLASS_ID, CLASS_ID_IMAGE, POINT

__all__ = ("SemanticSegmentation2D", "SemanticSegmentation3D")


@define(frozen=True, slots=True)
class SemanticSegmentation2D(Archetype):
    """A class per pixel of one image.

    The one component is a **mono** ``(H, W)`` class-id image: an entity holds one image
    per point in time, as it holds one transform, so ``len()`` is 1 and
    ``class_id_image.value`` is the image. The resolution is not logged separately. Every
    image on an entity shares one column, so it must have the same ``(H, W)``, and a range
    query that spans a resolution change fails in ``concat_chunks`` rather than misaligning
    pixels::

        store.log(path, SemanticSegmentation2D(class_id_image=labels), at=..., frame_id="CAM_FRONT")

    ``frame_id`` is the camera channel, as for every 2D archetype. ``latest_at`` reads one
    image back through ``materialize``; a ``range`` over several frames is a stack of
    images, read as a column with ``view.component(CLASS_ID_IMAGE)``, not as this
    archetype. The segmentation metrics flatten each image with
    :meth:`~t4perceval.component.BatchClassIdImage.as_class_id`, so pixel ``(r, c)`` is
    element ``r * W + c`` of the comparison.
    """

    class_id_image = component_field(CLASS_ID_IMAGE, ClassIdImage)


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
