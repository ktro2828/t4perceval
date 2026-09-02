from __future__ import annotations

from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import (
    BatchClassId,
    BatchConfidence,
    BatchNumPoints,
    BatchPosition3D,
    BatchQuaternion,
    BatchRoi,
    BatchSize3D,
    BatchVelocity,
    BatchVisibility,
)
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import (
    CLASS_ID,
    CONFIDENCE,
    NUM_POINTS,
    POSITION,
    QUATERNION,
    ROI,
    SIZE,
    VELOCITY,
    VISIBILITY,
)

__all__ = ("Detections2D", "Detections3D")


@define(frozen=True, slots=True)
class Detections3D(Archetype):
    """3D bounding boxes with a class and a confidence."""

    position = component_field(POSITION, BatchPosition3D)
    quaternion = component_field(QUATERNION, BatchQuaternion)
    size = component_field(SIZE, BatchSize3D)
    class_id = component_field(CLASS_ID, BatchClassId)
    confidence = component_field(CONFIDENCE, BatchConfidence)
    velocity = component_field(VELOCITY, BatchVelocity, optional=True, kw_only=True)
    num_points = component_field(NUM_POINTS, BatchNumPoints, optional=True, kw_only=True)
    visibility = component_field(VISIBILITY, BatchVisibility, optional=True, kw_only=True)


@define(frozen=True, slots=True)
class Detections2D(Archetype):
    """2D regions of interest with a class and a confidence."""

    roi = component_field(ROI, BatchRoi)
    class_id = component_field(CLASS_ID, BatchClassId)
    confidence = component_field(CONFIDENCE, BatchConfidence)
    visibility = component_field(VISIBILITY, BatchVisibility, optional=True, kw_only=True)
