from __future__ import annotations

from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.archetype.trajectory import validate_trajectory_shapes
from t4perceval.component import (
    BatchClassId,
    BatchConfidence,
    BatchInstanceId,
    BatchModeConfidence,
    BatchModeValid,
    BatchPosition3D,
    BatchQuaternion,
    BatchSize3D,
    BatchTimeOffset,
    BatchTimestepValid,
    BatchVelocity,
    BatchWaypoints3D,
)
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import (
    CLASS_ID,
    CONFIDENCE,
    INSTANCE_ID,
    MODE_CONFIDENCE,
    MODE_VALID,
    POSITION,
    QUATERNION,
    SIZE,
    TIME_OFFSET,
    TIMESTEP_VALID,
    VELOCITY,
    WAYPOINTS,
)

__all__ = ("BatchPrediction3D",)


@define(frozen=True, slots=True)
class BatchPrediction3D(Archetype):
    """Tracked 3D boxes together with their predicted future trajectories.

    Composed from the box, instance and trajectory components rather than inheriting from
    :class:`~t4perceval.archetype.BatchTracking3D`, so systems requiring only the box
    components, only the instance id, or only the trajectory columns all apply directly.
    """

    position = component_field(POSITION, BatchPosition3D)
    quaternion = component_field(QUATERNION, BatchQuaternion)
    size = component_field(SIZE, BatchSize3D)
    class_id = component_field(CLASS_ID, BatchClassId)
    confidence = component_field(CONFIDENCE, BatchConfidence)
    instance_id = component_field(INSTANCE_ID, BatchInstanceId, kw_only=True)
    waypoints = component_field(WAYPOINTS, BatchWaypoints3D, kw_only=True)
    mode_confidence = component_field(MODE_CONFIDENCE, BatchModeConfidence, kw_only=True)
    velocity = component_field(VELOCITY, BatchVelocity, optional=True, kw_only=True)
    mode_valid = component_field(MODE_VALID, BatchModeValid, optional=True, kw_only=True)
    timestep_valid = component_field(
        TIMESTEP_VALID, BatchTimestepValid, optional=True, kw_only=True
    )
    time_offset = component_field(TIME_OFFSET, BatchTimeOffset, optional=True, kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        validate_trajectory_shapes(
            self.waypoints,
            self.mode_confidence,
            self.mode_valid,
            self.timestep_valid,
            self.time_offset,
        )

    @property
    def num_modes(self) -> int:
        return self.waypoints.num_modes

    @property
    def num_timesteps(self) -> int:
        return self.waypoints.num_timesteps
