"""A bare point cloud: points in a frame, nothing said about them."""

from __future__ import annotations

from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import BatchPosition3D
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import POINT

__all__ = ("PointCloud",)


@define(frozen=True, slots=True)
class PointCloud(Archetype):
    """A set of 3D points expressed in the chunk's frame.

    This is the cloud itself, before anything is said about its points: the raw sensor
    output, a cropped or down-sampled copy of it, or the ground truth a coverage filter
    is pointed at. The one column is ``point``, so a system that asks for ``POINT``
    alone -- a coordinate transform, :class:`~t4perceval.system.FilterPointsByDistanceSystem`,
    :class:`~t4perceval.system.FilterPointsByCoverageSystem` -- runs against it unchanged.

    :class:`~t4perceval.archetype.SemanticPointCloud` is this archetype plus ``class_id``, by
    composition rather than inheritance, as :class:`~t4perceval.archetype.Trackings3D` is
    :class:`~t4perceval.archetype.Detections3D` plus ``instance_id``. Both resolve
    ``point`` to the same descriptor, so a labelled cloud satisfies every requirement a bare
    one does.

    ``point`` uses the ``POINT`` descriptor rather than ``POSITION``: a point is not an
    object with a pose, and the separate name stops an object filter from being pointed at
    a cloud and appearing to work.

    Examples:
        >>> cloud = PointCloud(point=[[1.0, 2.0, 0.3], [1.1, 2.0, 0.3]])
        >>> len(cloud)
        2
    """

    point = component_field(POINT, BatchPosition3D)
