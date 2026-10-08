"""Turning ``t4_devkit`` boxes into columns.

One extraction per box list, then a projection per archetype. ``Detections3D``,
``Trackings3D`` and ``Predictions3D`` are a nested superset chain over the *same*
``sample_annotation`` rows, and a descriptor is identified by its component name alone --
so logging either archetype to one entity path writes the same columns under the same
keys. The archetype is a projection over a fixed column set, not a separate conversion,
and three conversions would mean three copies of the quaternion reorder, the visibility
map and the velocity policy, where one divergence is silent geometric corruption.

Nothing here imports ``t4_devkit`` at runtime: the extraction only reads attributes off
the boxes, so it is duck-typed and testable without the devkit installed.

The conversions that are not identity, each of which would silently corrupt data if
skipped:

===================  =============================  ==============================
Value                ``t4_devkit``                  ``t4perceval``
===================  =============================  ==============================
Quaternion           ``elements`` is ``wxyz``       ``BatchQuaternion`` is ``xyzw``
Region of interest   ``(xmin, ymin, xmax, ymax)``   ``(x_min, y_min, height, width)``
Visibility           ranked ``full=4 .. none=1``    ``FULL=3 .. NONE=0``
Time                 absolute microseconds          nanosecond offsets from the frame
===================  =============================  ==============================

Box size needs no permutation: both are ``(width, length, height)``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from attrs import define, field

from t4perceval.archetype import (
    Detections2D,
    Detections3D,
    Predictions3D,
    Trackings2D,
    Trackings3D,
)
from t4perceval.component import VisibilityLevel
from t4perceval.importer._columns import (
    NAN3,
    Emit,
    TrajectoryColumns,
    column,
    encode_instance_ids,
    normalize_quaternions,
    padded_trajectory,
    select_kept,
    wxyz_to_xyzw,
)
from t4perceval.importer._labels import encode_class_ids

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from t4perceval.core.archetype import Archetype
    from t4perceval.importer._labels import UnknownLabels
    from t4perceval.label import InstanceRegistry, LabelRegistry
    from t4perceval.typing import NDArrayF64, NDArrayI8, NDArrayI32, NDArrayI64

__all__ = (
    "Box2DColumns",
    "Box3DColumns",
    "Emit",
    "Kind2D",
    "Kind3D",
    "TrajectoryColumns",
    "boxes2d_to_columns",
    "boxes3d_to_columns",
    "trajectory_shape_of",
)

Kind3D: TypeAlias = Literal["detections", "trackings", "predictions"]
Kind2D: TypeAlias = Literal["detections", "trackings"]


#: Devkit visibility is a string enum ranked ``full=4 .. none=1``, with ``None`` for
#: unavailable. Ours is ``FULL=3 .. NONE=0`` with ``UNAVAILABLE=-1``. The ranks are off by
#: one and the ``None`` would not survive ``int8``, so the mapping goes by name.
_VISIBILITY: dict[str, VisibilityLevel] = {
    "full": VisibilityLevel.FULL,
    "most": VisibilityLevel.MOST,
    "partial": VisibilityLevel.PARTIAL,
    "none": VisibilityLevel.NONE,
    "unavailable": VisibilityLevel.UNAVAILABLE,
}


@define(frozen=True, slots=True)
class Box3DColumns:
    """Every column a list of 3D boxes can yield, before an archetype is chosen."""

    position: NDArrayF64
    quaternion: NDArrayF64
    size: NDArrayF64
    class_id: NDArrayI32
    confidence: NDArrayF64
    instance_id: NDArrayI64
    velocity: NDArrayF64 | None = field(default=None, kw_only=True)
    num_points: NDArrayI32 | None = field(default=None, kw_only=True)
    visibility: NDArrayI8 | None = field(default=None, kw_only=True)
    trajectory: TrajectoryColumns | None = field(default=None, kw_only=True)
    frame_id: str | None = field(default=None, kw_only=True)

    kept: NDArrayI64 = field(factory=lambda: np.empty(0, dtype=np.int64), kw_only=True)
    """Indices of the source boxes that survived label filtering."""

    def __len__(self) -> int:
        return len(self.position)

    def _optional(self) -> dict[str, Any]:
        return {
            "velocity": self.velocity,
            "num_points": self.num_points,
            "visibility": self.visibility,
        }

    def as_detections(self) -> Detections3D:
        """Project to :class:`~t4perceval.archetype.Detections3D`."""
        return Detections3D(
            self.position,
            self.quaternion,
            self.size,
            self.class_id,
            self.confidence,
            **self._optional(),
        )

    def as_trackings(self) -> Trackings3D:
        """Project to :class:`~t4perceval.archetype.Trackings3D`."""
        return Trackings3D(
            self.position,
            self.quaternion,
            self.size,
            self.class_id,
            self.confidence,
            instance_id=self.instance_id,
            **self._optional(),
        )

    def as_predictions(self) -> Predictions3D:
        """Project to :class:`~t4perceval.archetype.Predictions3D`.

        Raises:
            ValueError: When the extraction carried no trajectory columns.
        """
        if self.trajectory is None:
            raise ValueError(
                "Box3DColumns carries no trajectory; "
                "pass trajectory=(num_modes, num_timesteps) to boxes3d_to_columns()",
            )
        return Predictions3D(
            self.position,
            self.quaternion,
            self.size,
            self.class_id,
            self.confidence,
            instance_id=self.instance_id,
            waypoints=self.trajectory.waypoints,
            mode_confidence=self.trajectory.mode_confidence,
            mode_valid=self.trajectory.mode_valid,
            timestep_valid=self.trajectory.timestep_valid,
            time_offset=self.trajectory.time_offset,
            velocity=self.velocity,
        )

    def as_archetype(self, kind: Kind3D) -> Archetype:
        """Project to the archetype named by ``kind``."""
        if kind == "detections":
            return self.as_detections()
        if kind == "trackings":
            return self.as_trackings()
        if kind == "predictions":
            return self.as_predictions()
        raise ValueError(f"Unknown 3D archetype kind {kind!r}")


@define(frozen=True, slots=True)
class Box2DColumns:
    """Every column a list of 2D boxes can yield.

    There is no visibility column: ``Box2D`` carries no such attribute, so modelling one
    would promise data this source can never supply.
    """

    roi: NDArrayI32
    class_id: NDArrayI32
    confidence: NDArrayF64
    instance_id: NDArrayI64
    frame_id: str | None = field(default=None, kw_only=True)
    kept: NDArrayI64 = field(factory=lambda: np.empty(0, dtype=np.int64), kw_only=True)

    def __len__(self) -> int:
        return len(self.roi)

    def as_detections(self) -> Detections2D:
        """Project to :class:`~t4perceval.archetype.Detections2D`."""
        return Detections2D(self.roi, self.class_id, self.confidence)

    def as_trackings(self) -> Trackings2D:
        """Project to :class:`~t4perceval.archetype.Trackings2D`."""
        return Trackings2D(
            self.roi,
            self.class_id,
            self.confidence,
            instance_id=self.instance_id,
        )

    def as_archetype(self, kind: Kind2D) -> Archetype:
        """Project to the archetype named by ``kind``."""
        if kind == "detections":
            return self.as_detections()
        if kind == "trackings":
            return self.as_trackings()
        raise ValueError(f"Unknown 2D archetype kind {kind!r}")


def trajectory_shape_of(frames: Iterable[Sequence[Any]]) -> tuple[int, int]:
    """Return the ``(num_modes, num_timesteps)`` covering every box in a scene.

    Falls back to ``(1, 1)`` when no box carries a future: a zero-length mode or timestep
    axis is rejected by the trajectory validator, so a prediction import of a scene with
    no future annotations still has to produce a well-formed, fully masked batch.
    """
    modes = 0
    timesteps = 0
    for boxes in frames:
        for box in boxes:
            future = getattr(box, "future", None)
            if future is None:
                continue
            shape = np.shape(future.waypoints)
            modes = max(modes, shape[0])
            timesteps = max(timesteps, shape[1])
    return (max(modes, 1), max(timesteps, 1))


def boxes3d_to_columns(
    boxes: Sequence[Any],
    *,
    labels: LabelRegistry,
    instances: InstanceRegistry,
    base_time_ns: int,
    instance_namespace: str = "",
    unknown_labels: UnknownLabels = "error",
    velocity: Emit = "auto",
    num_points: Emit = "auto",
    visibility: Emit = "auto",
    trajectory: tuple[int, int] | None = None,
) -> Box3DColumns:
    """Extract every column a batch of 3D boxes can yield, in one pass.

    Args:
        boxes: ``Box3D`` records for one frame.
        labels: Registry deciding class ids. Never derived here -- see
            :func:`~t4perceval.importer.t4.labels.label_registry_from_categories`.
        instances: Registry interning object identities. Mutated as new ones appear.
        base_time_ns: When the frame was observed, in nanoseconds. Trajectory offsets are
            measured from here, because ``Future.timestamps`` are *absolute* microseconds
            while ``BatchTimeOffset`` wants non-negative relative nanoseconds.
        instance_namespace: Prefix for interned identities, so identities from different
            sources cannot collide on one integer.
        unknown_labels: What to do with a category the registry lacks.
        velocity: Whether to emit the velocity column. Resolve ``"auto"`` scene-wide.
        num_points: Whether to emit the point-count column.
        visibility: Whether to emit the visibility column.
        trajectory: ``(num_modes, num_timesteps)`` to pad to, or ``None`` to skip the
            trajectory columns. Fixed by the caller rather than inferred per frame,
            because a scene whose timestep count varies cannot be concatenated.

    Returns:
        The extracted columns, ready to project to any 3D archetype.
    """
    names = [str(box.semantic_label.name) for box in boxes]
    class_id, keep = encode_class_ids(labels, names, unknown=unknown_labels)

    selected, kept = select_kept(boxes, keep)
    count = len(selected)

    position = column([box.position for box in selected], count, (3,), np.float64)
    size = column([box.shape.size for box in selected], count, (3,), np.float64)
    quaternion = _quaternion_column(selected, count)
    confidence = np.fromiter(
        (float(box.confidence) for box in selected),
        dtype=np.float64,
        count=count,
    )
    instance_id = _instance_column(selected, instances, instance_namespace)

    return Box3DColumns(
        position,
        quaternion,
        size,
        class_id,
        confidence,
        instance_id,
        velocity=_velocity_column(selected, count, velocity),
        num_points=_num_points_column(selected, count, num_points),
        visibility=_visibility_column(selected, count, visibility),
        trajectory=(
            None
            if trajectory is None
            else _trajectory_columns(
                selected,
                position,
                base_time_ns=base_time_ns,
                num_modes=trajectory[0],
                num_timesteps=trajectory[1],
            )
        ),
        frame_id=str(selected[0].frame_id) if count else None,
        kept=kept,
    )


def boxes2d_to_columns(
    boxes: Sequence[Any],
    *,
    labels: LabelRegistry,
    instances: InstanceRegistry,
    instance_namespace: str = "",
    unknown_labels: UnknownLabels = "error",
) -> Box2DColumns:
    """Extract the columns a batch of 2D boxes can yield.

    Args:
        boxes: ``Box2D`` records for one camera at one frame.
        labels: Registry deciding class ids.
        instances: Registry interning object identities.
        instance_namespace: Prefix for interned identities.
        unknown_labels: What to do with a category the registry lacks.

    Returns:
        The extracted columns, ready to project to a 2D archetype.
    """
    names = [str(box.semantic_label.name) for box in boxes]
    class_id, keep = encode_class_ids(labels, names, unknown=unknown_labels)

    selected, kept = select_kept(boxes, keep)
    count = len(selected)

    confidence = np.fromiter(
        (float(box.confidence) for box in selected),
        dtype=np.float64,
        count=count,
    )

    return Box2DColumns(
        _roi_column(selected, count),
        class_id,
        confidence,
        _instance_column(selected, instances, instance_namespace),
        frame_id=str(selected[0].frame_id) if count else None,
        kept=kept,
    )


# -- column builders --------------------------------------------------------------------


def _quaternion_column(boxes: Sequence[Any], count: int) -> NDArrayF64:
    """Return unit ``xyzw`` quaternions.

    ``pyquaternion`` stores ``wxyz``; ours is ``xyzw``. Both are ``(4,)`` float arrays, so
    getting this wrong produces a plausible rotation rather than an error.
    """
    wxyz = column([box.rotation.elements for box in boxes], count, (4,), np.float64)
    return normalize_quaternions(wxyz_to_xyzw(wxyz), what="Box")


def _roi_column(boxes: Sequence[Any], count: int) -> NDArrayI32:
    """Return regions of interest as ``(x_min, y_min, height, width)``.

    The devkit stores ``(xmin, ymin, xmax, ymax)``. Both are four ints, so a straight copy
    would put a coordinate where an extent belongs and go unnoticed.
    """
    for index, box in enumerate(boxes):
        if box.roi is None:
            raise ValueError(f"Box {index} has no region of interest")

    xyxy = column([tuple(box.roi) for box in boxes], count, (4,), np.int32)
    roi = np.empty((count, 4), dtype=np.int32)
    if count:
        roi[:, 0] = xyxy[:, 0]
        roi[:, 1] = xyxy[:, 1]
        roi[:, 2] = xyxy[:, 3] - xyxy[:, 1]
        roi[:, 3] = xyxy[:, 2] - xyxy[:, 0]
    return roi


def _instance_column(
    boxes: Sequence[Any],
    instances: InstanceRegistry,
    namespace: str,
) -> NDArrayI64:
    """Intern each box's identity, namespaced so sources cannot collide."""
    for index, box in enumerate(boxes):
        if box.uuid is None:
            raise ValueError(f"Box {index} has no uuid, so it cannot be tracked")
    return encode_instance_ids(instances, [str(box.uuid) for box in boxes], namespace=namespace)


def _velocity_column(boxes: Sequence[Any], count: int, emit: Emit) -> NDArrayF64 | None:
    """Return the velocity column, or ``None`` when it is not emitted.

    The devkit reports an inestimable velocity as a NaN vector rather than as a missing
    value, and the two can appear in one frame. NaN rows are kept as they are: zero would
    be a claim, NaN is the absence of one. Note that a speed filter rejects NaN rows,
    since every comparison against NaN is false.
    """
    if emit == "never":
        return None

    raw = column(
        [box.velocity if box.velocity is not None else NAN3 for box in boxes],
        count,
        (3,),
        np.float64,
    )
    if emit == "always":
        return raw
    return raw if count and bool(np.isfinite(raw).all(axis=1).any()) else None


def _num_points_column(boxes: Sequence[Any], count: int, emit: Emit) -> NDArrayI32 | None:
    """Return the point-count column, or ``None`` when it is not emitted."""
    if emit == "never":
        return None

    present = all(box.num_points is not None for box in boxes)
    if emit == "auto" and not (count and present):
        return None
    if not present:
        raise ValueError('Some boxes have no num_points; pass num_points="never"')
    return np.fromiter((int(box.num_points) for box in boxes), dtype=np.int32, count=count)


def _visibility_column(boxes: Sequence[Any], count: int, emit: Emit) -> NDArrayI8 | None:
    """Return the visibility column, or ``None`` when it is not emitted."""
    if emit == "never" or (emit == "auto" and count == 0):
        return None
    return np.fromiter(
        (_VISIBILITY[str(box.visibility.value)] for box in boxes),
        dtype=np.int8,
        count=count,
    )


def _trajectory_columns(
    boxes: Sequence[Any],
    positions: NDArrayF64,
    *,
    base_time_ns: int,
    num_modes: int,
    num_timesteps: int,
) -> TrajectoryColumns:
    """Fill the padded trajectory columns with each box's future, where it has one."""
    columns = padded_trajectory(positions, num_modes=num_modes, num_timesteps=num_timesteps)
    waypoints = columns.waypoints
    mode_confidence = columns.mode_confidence
    mode_valid = columns.mode_valid
    timestep_valid = columns.timestep_valid
    time_offset = columns.time_offset

    truncated = 0
    for index, box in enumerate(boxes):
        future = getattr(box, "future", None)
        if future is None:
            continue

        available = np.shape(future.waypoints)
        real_modes = min(available[0], num_modes)
        real_steps = min(available[1], num_timesteps)
        if available[0] > num_modes or available[1] > num_timesteps:
            truncated += 1

        real = np.asarray(future.waypoints, dtype=np.float64)
        waypoints[index, :real_modes, :real_steps] = real[:real_modes, :real_steps]
        waypoints[index, :real_modes, real_steps:] = real[
            :real_modes,
            real_steps - 1 : real_steps,
        ]

        mode_confidence[index, :real_modes] = np.asarray(future.confidences)[:real_modes]
        mode_valid[index, :real_modes] = True
        timestep_valid[index, :real_modes, :real_steps] = True

        offsets = np.asarray(future.timestamps[:real_steps], dtype=np.int64) * 1000 - base_time_ns
        if offsets[0] <= 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError(
                f"Box {index} has future timestamps that do not increase after the "
                f"frame: {offsets.tolist()}",
            )
        time_offset[index, :real_steps] = offsets
        time_offset[index, real_steps:] = offsets[-1] + np.arange(
            1,
            num_timesteps - real_steps + 1,
            dtype=np.int64,
        )

    if truncated:
        warnings.warn(
            f"{truncated} box(es) had a future longer than the "
            f"({num_modes}, {num_timesteps}) trajectory shape it was given; "
            f"the excess was dropped",
            stacklevel=3,
        )

    return columns
