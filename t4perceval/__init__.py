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
    Classifications2D,
    ConfusionMatrix,
    Detections2D,
    Detections3D,
    MatchResults,
    MetricValues,
    Predictions3D,
    SemanticSegmentation2D,
    SemanticSegmentation3D,
    Trackings2D,
    Trackings3D,
    Trajectories3D,
    TrajectoryMode3D,
    Transform3D,
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
    Timeline,
    TimePoint,
    TimeRange,
    concat_chunks,
)
from t4perceval.label import (
    BACKGROUND_CLASS_ID,
    UNKNOWN_CLASS_ID,
    ClassInfo,
    InstanceRegistry,
    LabelRegistry,
)
from t4perceval.recording import Recording, RecordingMetadata, SourceInfo

__all__ = (
    "ANY",
    "BACKGROUND_CLASS_ID",
    "FRAME",
    "TIMESTAMP",
    "UNKNOWN_CLASS_ID",
    "Archetype",
    "Chunk",
    "ClassInfo",
    "Classifications2D",
    "ColumnarComponent",
    "Component",
    "ComponentDescriptor",
    "ConfusionMatrix",
    "Detections2D",
    "Detections3D",
    "EntityPath",
    "EntityView",
    "InstanceRegistry",
    "LabelRegistry",
    "MatchResults",
    "MetricValues",
    "Predictions3D",
    "Recording",
    "RecordingMetadata",
    "SemanticSegmentation2D",
    "SemanticSegmentation3D",
    "SourceInfo",
    "Store",
    "TimeColumn",
    "TimeKind",
    "TimePoint",
    "TimeRange",
    "Timeline",
    "Trackings2D",
    "Trackings3D",
    "Trajectories3D",
    "TrajectoryMode3D",
    "Transform3D",
    "concat_chunks",
)
