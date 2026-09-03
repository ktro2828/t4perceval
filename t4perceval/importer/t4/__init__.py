"""Importing T4 datasets, through ``t4-devkit``.

Requires the ``t4`` extra::

    pip install 't4perceval[t4]'

The registry is an input, never derived here -- see :meth:`T4Importer.label_registry`::

    importer = T4Importer.open("/data/t4/db_v1")
    labels = importer.label_registry()
    recording = importer.import_scene(labels=labels)
"""

from __future__ import annotations

from t4perceval.importer.t4.convert import (
    Box2DColumns,
    Box3DColumns,
    Emit,
    Kind2D,
    Kind3D,
    TrajectoryColumns,
    boxes2d_to_columns,
    boxes3d_to_columns,
    trajectory_shape_of,
)
from t4perceval.importer.t4.importer import (
    FrameRef,
    ImportOptions,
    SceneSelection,
    T4Importer,
)
from t4perceval.importer.t4.labels import (
    UnknownLabels,
    encode_class_ids,
    label_registry_from_categories,
)
from t4perceval.importer.t4.paths import DEFAULT_ROOT, objects2d_path, objects3d_path
from t4perceval.importer.t4.source import Coords, SampleFrame, T4Source

__all__ = (
    "DEFAULT_ROOT",
    "Box2DColumns",
    "Box3DColumns",
    "Coords",
    "Emit",
    "FrameRef",
    "ImportOptions",
    "Kind2D",
    "Kind3D",
    "SampleFrame",
    "SceneSelection",
    "T4Importer",
    "T4Source",
    "TrajectoryColumns",
    "UnknownLabels",
    "boxes2d_to_columns",
    "boxes3d_to_columns",
    "encode_class_ids",
    "label_registry_from_categories",
    "objects2d_path",
    "objects3d_path",
    "trajectory_shape_of",
)
