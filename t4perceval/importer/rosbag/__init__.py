"""Importing Autoware perception output from MCAP (ROS 2) bags.

Requires the ``rosbag`` extra::

    pip install 't4perceval[rosbag]'

A bag is decoded from its own embedded message definitions, so no ROS installation is
needed. The registry is an input, never derived here -- see
:meth:`RosbagImporter.label_registry`::

    importer = RosbagImporter.open("/data/bags/run_0")
    importer.topics()                 # the object topics the bag holds
    labels = importer.label_registry()
    recording = importer.import_topic(
        labels=labels,
        selection=BagSelection(topic="/perception/object_recognition/tracking/objects"),
    )

``/tf`` samples are recorded on the ``TIMESTAMP`` timeline only -- they have no frame index
-- so a lookup is ``TransformResolver.of(recording, timeline=TIMESTAMP)``.
"""

from __future__ import annotations

from t4perceval.importer.rosbag.convert import (
    SHAPE_BOUNDING_BOX,
    SHAPE_CYLINDER,
    SHAPE_POLYGON,
    Confidence,
    Emit,
    Kind,
    Object3DColumns,
    TrajectoryColumns,
    has_any_twist,
    kind_of_schema,
    objects_to_columns,
    stamp_ns,
    trajectory_shape_of,
)
from t4perceval.importer.rosbag.importer import (
    BagSelection,
    FrameRef,
    ImportOptions,
    RosbagImporter,
)
from t4perceval.importer.rosbag.labels import (
    AUTOWARE_CLASS_NAMES,
    UnknownLabels,
    class_name_of,
    classification_name,
    encode_class_ids,
    label_registry_from_autoware,
    top_classification,
)
from t4perceval.importer.rosbag.paths import DEFAULT_ROOT, objects3d_path, topic_entity_path
from t4perceval.importer.rosbag.source import BagMessage, BagSource, TopicInfo
from t4perceval.importer.rosbag.transforms import (
    TfScope,
    TransformSample,
    log_bag_transforms,
    transform_samples,
)

__all__ = (
    "AUTOWARE_CLASS_NAMES",
    "DEFAULT_ROOT",
    "SHAPE_BOUNDING_BOX",
    "SHAPE_CYLINDER",
    "SHAPE_POLYGON",
    "BagMessage",
    "BagSelection",
    "BagSource",
    "Confidence",
    "Emit",
    "FrameRef",
    "ImportOptions",
    "Kind",
    "Object3DColumns",
    "RosbagImporter",
    "TfScope",
    "TopicInfo",
    "TrajectoryColumns",
    "TransformSample",
    "UnknownLabels",
    "class_name_of",
    "classification_name",
    "encode_class_ids",
    "has_any_twist",
    "kind_of_schema",
    "label_registry_from_autoware",
    "log_bag_transforms",
    "objects3d_path",
    "objects_to_columns",
    "stamp_ns",
    "top_classification",
    "topic_entity_path",
    "trajectory_shape_of",
    "transform_samples",
)
