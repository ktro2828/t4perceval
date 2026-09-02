from __future__ import annotations

from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import (
    BatchClassId,
    BatchConfidence,
    BatchInstanceId,
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
    INSTANCE_ID,
    NUM_POINTS,
    POSITION,
    QUATERNION,
    ROI,
    SIZE,
    VELOCITY,
    VISIBILITY,
)

__all__ = ("Trackings2D", "Trackings3D")

# NOTE: The box components below are re-declared, not inherited from `Detections3D`.
# They resolve to the *same* descriptors, so
# `tracking.has(*Detections3D.required_descriptors())` is True and any system that
# requires a detection's components runs unchanged against a tracking archetype.


@define(frozen=True, slots=True)
class Trackings3D(Archetype):
    """3D bounding boxes carrying a persistent instance identifier."""

    position = component_field(POSITION, BatchPosition3D)
    quaternion = component_field(QUATERNION, BatchQuaternion)
    size = component_field(SIZE, BatchSize3D)
    class_id = component_field(CLASS_ID, BatchClassId)
    confidence = component_field(CONFIDENCE, BatchConfidence)
    instance_id = component_field(INSTANCE_ID, BatchInstanceId, kw_only=True)
    velocity = component_field(VELOCITY, BatchVelocity, optional=True, kw_only=True)
    num_points = component_field(NUM_POINTS, BatchNumPoints, optional=True, kw_only=True)
    visibility = component_field(VISIBILITY, BatchVisibility, optional=True, kw_only=True)


@define(frozen=True, slots=True)
class Trackings2D(Archetype):
    """2D regions of interest carrying a persistent instance identifier."""

    roi = component_field(ROI, BatchRoi)
    class_id = component_field(CLASS_ID, BatchClassId)
    confidence = component_field(CONFIDENCE, BatchConfidence)
    instance_id = component_field(INSTANCE_ID, BatchInstanceId, kw_only=True)
    visibility = component_field(VISIBILITY, BatchVisibility, optional=True, kw_only=True)
