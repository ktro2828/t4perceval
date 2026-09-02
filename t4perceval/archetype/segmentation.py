from __future__ import annotations

from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import BatchClassId, BatchPixel, BatchPosition3D
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import CLASS_ID, PIXEL, POINT

__all__ = ("SemanticSegmentation2D", "SemanticSegmentation3D")


@define(frozen=True, slots=True)
class SemanticSegmentation2D(Archetype):
    """A class per labelled pixel."""

    pixel = component_field(PIXEL, BatchPixel)
    class_id = component_field(CLASS_ID, BatchClassId)


@define(frozen=True, slots=True)
class SemanticSegmentation3D(Archetype):
    """A class per labelled point."""

    point = component_field(POINT, BatchPosition3D)
    class_id = component_field(CLASS_ID, BatchClassId)
