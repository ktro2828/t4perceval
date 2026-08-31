from __future__ import annotations

from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import BatchClassId, BatchConfidence, BatchInstanceId
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import CLASS_ID, CONFIDENCE, INSTANCE_ID

__all__ = ("BatchClassification2D",)


@define(frozen=True, slots=True)
class BatchClassification2D(Archetype):
    """A class and a confidence per object, without any geometry."""

    class_id = component_field(CLASS_ID, BatchClassId)
    confidence = component_field(CONFIDENCE, BatchConfidence)
    instance_id = component_field(INSTANCE_ID, BatchInstanceId, optional=True, kw_only=True)
