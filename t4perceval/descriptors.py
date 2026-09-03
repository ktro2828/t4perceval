"""Canonical component descriptors.

A descriptor names a column by *what it means*, not by which archetype declared it: both
:class:`~t4perceval.archetype.Detections3D` and
:class:`~t4perceval.archetype.Trackings3D` expose their 3D center as
:data:`POSITION`. That is what lets a system declare ``REQUIRES = (POSITION,)`` and run
against any entity carrying a 3D position, instead of branching on an evaluation-task
enum. The ``archetype`` field of a descriptor is a hint only and never affects identity.
"""

from __future__ import annotations

from t4perceval.core.descriptor import ComponentDescriptor

__all__ = (
    "CLASS_ID",
    "CONFIDENCE",
    "COUNT",
    "ESTIMATION_CLASS_ID",
    "EST_INDEX",
    "GROUND_TRUTH_CLASS_ID",
    "GT_INDEX",
    "INSTANCE_ID",
    "MASK",
    "MATCHING_SCORE",
    "MATCH_STATUS",
    "METRIC_VALUE",
    "MODE_CONFIDENCE",
    "MODE_VALID",
    "NUM_POINTS",
    "PIXEL",
    "POINT",
    "POSITION",
    "QUATERNION",
    "ROI",
    "ROTATION",
    "SIZE",
    "SIZE_2D",
    "SUPPORT",
    "THRESHOLD",
    "TIMESTEP_VALID",
    "TIME_OFFSET",
    "TRANSLATION",
    "VELOCITY",
    "VISIBILITY",
    "WAYPOINTS",
)

# --- 3D geometry ---------------------------------------------------------------------
POSITION = ComponentDescriptor("position", component_type="BatchPosition3D")
QUATERNION = ComponentDescriptor("quaternion", component_type="BatchQuaternion")
SIZE = ComponentDescriptor("size", component_type="BatchSize3D")
VELOCITY = ComponentDescriptor("velocity", component_type="BatchVelocity")

# --- 2D geometry ---------------------------------------------------------------------
ROI = ComponentDescriptor("roi", component_type="BatchRoi")
SIZE_2D = ComponentDescriptor("size_2d", component_type="BatchSize2D")
PIXEL = ComponentDescriptor("pixel", component_type="BatchPixel")

# --- semantics -----------------------------------------------------------------------
CLASS_ID = ComponentDescriptor("class_id", component_type="BatchClassId")
CONFIDENCE = ComponentDescriptor("confidence", component_type="BatchConfidence")
INSTANCE_ID = ComponentDescriptor("instance_id", component_type="BatchInstanceId")

# --- ground-truth quality ------------------------------------------------------------
NUM_POINTS = ComponentDescriptor("num_points", component_type="BatchNumPoints")
VISIBILITY = ComponentDescriptor("visibility", component_type="BatchVisibility")

# --- point clouds --------------------------------------------------------------------
POINT = ComponentDescriptor("point", component_type="BatchPosition3D")

# --- coordinate transforms -----------------------------------------------------------
# Deliberately not POSITION/QUATERNION. A transform is a relationship between frames, not
# a thing in a frame, and separate names stop a system that asks for a 3D position -- a
# distance filter, say -- from being pointed at a transform entity and appearing to work.
TRANSLATION = ComponentDescriptor("translation", component_type="BatchPosition3D")
ROTATION = ComponentDescriptor("rotation", component_type="BatchQuaternion")

# --- trajectories --------------------------------------------------------------------
WAYPOINTS = ComponentDescriptor("waypoints", component_type="BatchWaypoints3D")
MODE_CONFIDENCE = ComponentDescriptor("mode_confidence", component_type="BatchModeConfidence")
MODE_VALID = ComponentDescriptor("mode_valid", component_type="BatchModeValid")
TIMESTEP_VALID = ComponentDescriptor("timestep_valid", component_type="BatchTimestepValid")
TIME_OFFSET = ComponentDescriptor("time_offset", component_type="BatchTimeOffset")

# --- system outputs ------------------------------------------------------------------
MASK = ComponentDescriptor("mask", component_type="BatchMask")
EST_INDEX = ComponentDescriptor("est_index", component_type="BatchRowIndex")
GT_INDEX = ComponentDescriptor("gt_index", component_type="BatchRowIndex")
MATCHING_SCORE = ComponentDescriptor("matching_score", component_type="BatchMatchingScore")
MATCH_STATUS = ComponentDescriptor("match_status", component_type="BatchMatchStatus")
THRESHOLD = ComponentDescriptor("threshold", component_type="BatchThreshold")

# --- metrics -------------------------------------------------------------------------
GROUND_TRUTH_CLASS_ID = ComponentDescriptor("ground_truth_class_id", component_type="BatchClassId")
ESTIMATION_CLASS_ID = ComponentDescriptor("estimation_class_id", component_type="BatchClassId")
COUNT = ComponentDescriptor("count", component_type="BatchCount")
METRIC_VALUE = ComponentDescriptor("metric_value", component_type="BatchMetricValue")
SUPPORT = ComponentDescriptor("support", component_type="BatchSupport")
