from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define, field

from t4perceval.archetype._fields import component_field
from t4perceval.component import (
    BatchModeConfidence,
    BatchModeValid,
    BatchPosition3D,
    BatchTimeOffset,
    BatchTimestepValid,
    BatchWaypoints3D,
)
from t4perceval.core.archetype import Archetype, as_component
from t4perceval.descriptors import (
    MODE_CONFIDENCE,
    MODE_VALID,
    TIME_OFFSET,
    TIMESTEP_VALID,
    WAYPOINTS,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from t4perceval.typing import ArrayLike, NDArrayI64

__all__ = ("BatchTrajectory3D", "TrajectoryMode3D", "validate_trajectory_shapes")


def validate_trajectory_shapes(
    waypoints: BatchWaypoints3D,
    mode_confidence: BatchModeConfidence,
    mode_valid: BatchModeValid | None,
    timestep_valid: BatchTimestepValid | None,
    time_offset: BatchTimeOffset | None,
) -> None:
    """Check that every trajectory column agrees on the mode and timestep counts."""
    num_modes = waypoints.num_modes
    num_timesteps = waypoints.num_timesteps

    if num_modes == 0:
        raise ValueError("waypoints must contain at least one trajectory mode")
    if num_timesteps == 0:
        raise ValueError("waypoints must contain at least one timestep")

    if mode_confidence.num_modes != num_modes:
        raise ValueError(
            f"mode_confidence has {mode_confidence.num_modes} modes, expected {num_modes}",
        )
    if mode_valid is not None and mode_valid.row_shape != (num_modes,):
        raise ValueError(
            f"mode_valid has row shape {mode_valid.row_shape}, expected {(num_modes,)}",
        )
    if timestep_valid is not None and timestep_valid.row_shape != (num_modes, num_timesteps):
        raise ValueError(
            f"timestep_valid has row shape {timestep_valid.row_shape}, "
            f"expected {(num_modes, num_timesteps)}",
        )
    if time_offset is not None and time_offset.row_shape != (num_timesteps,):
        raise ValueError(
            f"time_offset has row shape {time_offset.row_shape}, expected {(num_timesteps,)}",
        )


@define(frozen=True, slots=True)
class TrajectoryMode3D:
    """One trajectory mode of a single object.

    A readable, row-at-a-time constructor for building dense batches in tests and
    adapters. It is not a component: the columnar form is
    :class:`~t4perceval.component.BatchWaypoints3D` and friends.
    """

    position: BatchPosition3D = field(converter=lambda v: as_component(v, BatchPosition3D))
    confidence: float = field(converter=float)
    time_offset_ns: NDArrayI64 = field(
        converter=lambda v: np.ascontiguousarray(np.asarray(v, dtype=np.int64)),
    )

    def __attrs_post_init__(self) -> None:
        if self.time_offset_ns.ndim != 1:
            raise ValueError(
                f"time_offset_ns must have shape (T,), got {self.time_offset_ns.shape}",
            )
        num_timesteps = len(self.time_offset_ns)
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        if num_timesteps == 0:
            raise ValueError("Trajectory mode must contain at least one timestep")
        if len(self.position) != num_timesteps:
            raise ValueError(
                f"position has length {len(self.position)}, expected {num_timesteps}",
            )
        if np.any(self.time_offset_ns < 0):
            raise ValueError("time_offset_ns must be non-negative")
        if num_timesteps > 1 and np.any(np.diff(self.time_offset_ns) <= 0):
            raise ValueError("time_offset_ns must be strictly increasing")

    def __len__(self) -> int:
        return len(self.time_offset_ns)


@define(frozen=True, slots=True)
class BatchTrajectory3D(Archetype):
    """Dense multi-modal future trajectories for ``N`` objects.

    ``M`` modes and ``T`` timesteps are fixed within one instance; shorter trajectories
    are padded and masked by :attr:`mode_valid` / :attr:`timestep_valid`.
    """

    waypoints = component_field(WAYPOINTS, BatchWaypoints3D)
    mode_confidence = component_field(MODE_CONFIDENCE, BatchModeConfidence)
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

    @classmethod
    def empty(cls, *, num_modes: int, num_timesteps: int) -> Self:
        """Return a trajectory batch with zero objects but a fixed mode/time shape."""
        if num_modes <= 0 or num_timesteps <= 0:
            raise ValueError("num_modes and num_timesteps must both be positive")
        return cls(
            waypoints=np.empty((0, num_modes, num_timesteps, 3), dtype=np.float64),
            mode_confidence=np.empty((0, num_modes), dtype=np.float64),
        )

    @classmethod
    def from_modes(cls, objects: Sequence[Sequence[TrajectoryMode3D]]) -> Self:
        """Build a dense batch from equally shaped modes grouped by object.

        Every object must contribute the same number of modes, and every mode must share
        the same time axis.
        """
        if not objects:
            raise ValueError("objects must contain at least one object")

        num_modes = len(objects[0])
        if num_modes == 0:
            raise ValueError("Every object must contain at least one trajectory mode")
        if any(len(modes) != num_modes for modes in objects):
            raise ValueError("Every object must contain the same number of trajectory modes")

        reference_time = objects[0][0].time_offset_ns
        for modes in objects:
            for mode in modes:
                if not np.array_equal(mode.time_offset_ns, reference_time):
                    raise ValueError("Every trajectory mode must use the same time_offset_ns")

        num_objects = len(objects)
        return cls(
            waypoints=np.stack(
                [np.stack([mode.position.values for mode in modes]) for modes in objects],
            ),
            mode_confidence=np.asarray(
                [[mode.confidence for mode in modes] for modes in objects],
                dtype=np.float64,
            ),
            time_offset=np.tile(reference_time, (num_objects, 1)),
        )

    def to_modes(self, object_index: int) -> tuple[TrajectoryMode3D, ...]:
        """Return every mode belonging to one object."""
        if object_index < 0:
            object_index += len(self)
        if not 0 <= object_index < len(self):
            raise IndexError("trajectory object index out of range")

        if self.time_offset is not None:
            offsets: ArrayLike = self.time_offset.values[object_index]
        else:
            offsets = np.arange(self.num_timesteps, dtype=np.int64)

        return tuple(
            TrajectoryMode3D(
                position=self.waypoints.values[object_index, mode_index],
                confidence=float(self.mode_confidence.values[object_index, mode_index]),
                time_offset_ns=offsets,
            )
            for mode_index in range(self.num_modes)
        )
