"""t4perceval: a component-oriented perception evaluation toolkit.

The data model follows Rerun's: data lives at an :class:`~t4perceval.core.EntityPath`,
is made of :class:`~t4perceval.core.Component` columns identified by a
:class:`~t4perceval.core.ComponentDescriptor`, is bundled into
:class:`~t4perceval.core.Archetype` types, stored as
:class:`~t4perceval.core.Chunk` tables, and indexed along
:class:`~t4perceval.core.Timeline` axes inside a :class:`~t4perceval.core.Store`.
The "S" of ECS lives in :mod:`t4perceval.system`.
"""

from __future__ import annotations

from t4perceval.archetype import (
    BatchClassification2D,
    BatchDetection2D,
    BatchDetection3D,
    BatchMatchResult,
    BatchMetric,
    BatchPrediction3D,
    BatchSemanticSegmentation2D,
    BatchSemanticSegmentation3D,
    BatchTracking2D,
    BatchTracking3D,
    BatchTrajectory3D,
    TrajectoryMode3D,
)
from t4perceval.core import (
    ANY,
    FRAME,
    TIMESTAMP,
    Archetype,
    Chunk,
    ColumnarComponent,
    Component,
    ComponentDescriptor,
    EntityPath,
    EntityView,
    Store,
    TimeColumn,
    TimeKind,
    TimePoint,
    TimeRange,
    Timeline,
    concat_chunks,
)
from t4perceval.label import UNKNOWN_CLASS_ID, ClassInfo, InstanceRegistry, LabelRegistry

__all__ = (
    "ANY",
    "Archetype",
    "BatchClassification2D",
    "BatchDetection2D",
    "BatchDetection3D",
    "BatchMatchResult",
    "BatchMetric",
    "BatchPrediction3D",
    "BatchSemanticSegmentation2D",
    "BatchSemanticSegmentation3D",
    "BatchTracking2D",
    "BatchTracking3D",
    "BatchTrajectory3D",
    "Chunk",
    "ClassInfo",
    "ColumnarComponent",
    "Component",
    "ComponentDescriptor",
    "EntityPath",
    "EntityView",
    "FRAME",
    "InstanceRegistry",
    "LabelRegistry",
    "Store",
    "TIMESTAMP",
    "TimeColumn",
    "TimeKind",
    "TimePoint",
    "TimeRange",
    "Timeline",
    "TrajectoryMode3D",
    "UNKNOWN_CLASS_ID",
    "concat_chunks",
)
