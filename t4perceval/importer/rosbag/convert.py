"""Turning decoded Autoware object messages into columns.

``DetectedObjects``, ``TrackedObjects`` and ``PredictedObjects`` are three message schemas
with three different kinematics layouts, so unlike the T4 importer there is no single box
type to project three archetypes from. What they share is a per-object *shape*: a pose, a
``Shape``, a classification list, an existence probability, and -- for two of them -- an
``object_id``. The extraction below reads those through per-kind accessors and produces
one column set; the archetype is then a projection over it, exactly as for T4.

Nothing here imports the MCAP libraries: the extraction only reads attributes off decoded
messages, so it is duck-typed and testable with plain namespaces.

The conversions that are not identity, each of which would silently corrupt data if
skipped:

=============  ==========================================  ==================================
Value          Autoware                                    ``t4perceval``
=============  ==========================================  ==================================
Size           ``dimensions`` is ``(length, width, h)``;   ``BatchSize3D`` is ``(width,
               ``CYLINDER`` is ``(diameter, diameter, h)``  length, height)``
Confidence     ``existence_probability`` **and** a         one ``BatchConfidence`` column,
               probability per classification              chosen by ``confidence=``
Velocity       ``twist.linear`` in the object's body frame  rotated into ``header.frame_id``
Trajectory     ``path[0]`` is the current pose, ``path[i]``  waypoints are ``path[1:]``,
               is at ``i * time_step``                     offsets ``(i + 1) * time_step``
Instance       ``object_id.uuid`` is 16 raw bytes          canonical UUID string, interned
Time           ``header.stamp`` is ``(sec, nanosec)``      one ``int64`` nanosecond value
=============  ==========================================  ==================================

Quaternions need no reorder: ``geometry_msgs/Quaternion`` is ``xyzw``, as is
``BatchQuaternion``. A ``POLYGON`` footprint has no component to land in and is dropped;
its ``dimensions`` are used as given.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from attrs import define, field
from scipy.spatial.transform import Rotation

from t4perceval.archetype import Detections3D, Predictions3D, Trackings3D
from t4perceval.importer._labels import encode_class_ids
from t4perceval.importer.rosbag.labels import classification_name, top_classification

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from t4perceval.core.archetype import Archetype
    from t4perceval.importer._labels import UnknownLabels
    from t4perceval.label import InstanceRegistry, LabelRegistry
    from t4perceval.typing import NDArrayBool, NDArrayF64, NDArrayI32, NDArrayI64

__all__ = (
    "SHAPE_BOUNDING_BOX",
    "SHAPE_CYLINDER",
    "SHAPE_POLYGON",
    "Confidence",
    "Emit",
    "Kind",
    "Object3DColumns",
    "TrajectoryColumns",
    "has_any_twist",
    "kind_of_schema",
    "objects_to_columns",
    "stamp_ns",
    "trajectory_shape_of",
)

Kind: TypeAlias = Literal["detections", "trackings", "predictions"]
"""Which archetype a topic is written as; decided by its message schema."""

Emit: TypeAlias = Literal["auto", "always", "never"]
"""Whether an optional column is written.

``"auto"`` decides from the batch in hand. Callers importing a whole topic must resolve
it **topic-wide** and pass ``"always"`` or ``"never"``: ``concat_chunks`` rejects chunks
whose column sets differ, so a column present on one frame and absent on the next makes
``Store.range()`` raise over that topic.
"""

Confidence: TypeAlias = Literal["classification", "existence", "product"]
"""Which message field becomes ``BatchConfidence``.

``"classification"`` is the probability of the highest-scoring classification -- what a
detector's score usually ends up as, and what the incumbent evaluator used.
``"existence"`` is ``existence_probability``; ``"product"`` multiplies the two. An object
with no classification falls back to ``existence_probability`` in every mode.
"""

#: ``autoware_perception_msgs/msg/Shape.type`` values.
SHAPE_BOUNDING_BOX = 0
SHAPE_CYLINDER = 1
SHAPE_POLYGON = 2

_KINDS: dict[str, Kind] = {
    "DetectedObjects": "detections",
    "TrackedObjects": "trackings",
    "PredictedObjects": "predictions",
}

_NAN3 = np.full(3, np.nan, dtype=np.float64)


def kind_of_schema(name: str) -> Kind:
    """Return the archetype kind an object message schema maps to.

    Matches on the message name alone, so ``autoware_perception_msgs/msg/DetectedObjects``
    and the legacy ``autoware_auto_perception_msgs/msg/DetectedObjects`` both qualify: the
    fields this module reads are the same in both.
    """
    tail = name.rsplit("/", 1)[-1]
    try:
        return _KINDS[tail]
    except KeyError:
        raise ValueError(
            f"{name!r} is not an Autoware object message; expected one of "
            f"{sorted(_KINDS)} from autoware_perception_msgs or "
            f"autoware_auto_perception_msgs",
        ) from None


def stamp_ns(stamp: Any) -> int:
    """Fold a ``builtin_interfaces/Time`` or ``Duration`` into one nanosecond integer."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _column(
    values: list[Any],
    count: int,
    row_shape: tuple[int, ...],
    dtype: Any,
) -> Any:
    """Stack per-object values into a column.

    An empty batch is allocated rather than inferred: ``np.asarray([])`` collapses to
    ``(0,)``, which fails the component's per-row shape check.
    """
    if count == 0:
        return np.empty((0, *row_shape), dtype=dtype)
    return np.asarray(values, dtype=dtype).reshape(count, *row_shape)


@define(frozen=True, slots=True)
class TrajectoryColumns:
    """Dense trajectory columns with a fixed mode and timestep count."""

    waypoints: NDArrayF64
    """``(N, M, T, 3)``, always finite -- padding holds the last real position."""

    mode_confidence: NDArrayF64
    """``(N, M)`` in ``[0, 1]``."""

    mode_valid: NDArrayBool
    """``(N, M)``; ``False`` for a padded mode."""

    timestep_valid: NDArrayBool
    """``(N, M, T)``; ``False`` for a padded timestep."""

    time_offset: NDArrayI64
    """``(N, T)`` nanoseconds from the frame, non-negative and strictly increasing."""


@define(frozen=True, slots=True)
class Object3DColumns:
    """Every column one object message can yield, as raw arrays.

    A ``DetectedObjects`` message yields no ``instance_id`` and no ``trajectory``; a
    ``TrackedObjects`` message yields no ``trajectory``. The projections raise when asked
    for an archetype the source cannot supply, rather than inventing identities.
    """

    position: NDArrayF64
    quaternion: NDArrayF64
    """Unit ``xyzw``."""

    size: NDArrayF64
    """``(width, length, height)``."""

    class_id: NDArrayI32
    confidence: NDArrayF64

    instance_id: NDArrayI64 | None = field(default=None, kw_only=True)
    velocity: NDArrayF64 | None = field(default=None, kw_only=True)
    """Expressed in ``frame_id``, like ``position``. ``None`` when not emitted."""

    trajectory: TrajectoryColumns | None = field(default=None, kw_only=True)
    frame_id: str | None = field(default=None, kw_only=True)
    """The message's ``header.frame_id``."""

    kept: NDArrayI64 = field(factory=lambda: np.empty(0, dtype=np.int64), kw_only=True)
    """Indices into the message's ``objects`` that survived label filtering."""

    def __len__(self) -> int:
        return len(self.position)

    def as_detections(self) -> Detections3D:
        """Project onto :class:`~t4perceval.archetype.Detections3D`."""
        return Detections3D(
            self.position,
            self.quaternion,
            self.size,
            self.class_id,
            self.confidence,
            velocity=self.velocity,
        )

    def as_trackings(self) -> Trackings3D:
        """Project onto :class:`~t4perceval.archetype.Trackings3D`."""
        if self.instance_id is None:
            raise ValueError(
                "DetectedObjects carry no object_id, so they cannot be written as "
                "Trackings3D; import the tracking topic instead",
            )
        return Trackings3D(
            self.position,
            self.quaternion,
            self.size,
            self.class_id,
            self.confidence,
            instance_id=self.instance_id,
            velocity=self.velocity,
        )

    def as_predictions(self) -> Predictions3D:
        """Project onto :class:`~t4perceval.archetype.Predictions3D`."""
        if self.instance_id is None or self.trajectory is None:
            raise ValueError(
                "Only PredictedObjects carry predicted paths, so this message cannot be "
                "written as Predictions3D",
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
            velocity=self.velocity,
            mode_valid=self.trajectory.mode_valid,
            timestep_valid=self.trajectory.timestep_valid,
            time_offset=self.trajectory.time_offset,
        )

    def as_archetype(self, kind: Kind) -> Archetype:
        """Project onto the archetype named by ``kind``."""
        if kind == "detections":
            return self.as_detections()
        if kind == "trackings":
            return self.as_trackings()
        if kind == "predictions":
            return self.as_predictions()
        raise ValueError(f"Unknown archetype kind {kind!r}")


# -- topic-wide decisions -----------------------------------------------------------------


def _paths_of(obj: Any) -> list[Any]:
    """Return an object's usable predicted paths.

    A path holds the current pose first, so one with fewer than two poses predicts nothing
    and is treated as absent rather than as a zero-length mode.
    """
    paths = getattr(obj.kinematics, "predicted_paths", None)
    if paths is None:
        return []
    return [path for path in paths if len(path.path) >= 2]


def trajectory_shape_of(messages: Iterable[Any]) -> tuple[int, int]:
    """Return the ``(num_modes, num_timesteps)`` covering every object in a topic.

    Falls back to ``(1, 1)`` when nothing carries a path: a zero-length mode or timestep
    axis is rejected by the trajectory validator, so a topic with no predictions still has
    to produce a well-formed, fully masked batch.
    """
    modes = 0
    timesteps = 0
    for message in messages:
        for obj in message.objects:
            paths = _paths_of(obj)
            modes = max(modes, len(paths))
            for path in paths:
                timesteps = max(timesteps, len(path.path) - 1)
    return (max(modes, 1), max(timesteps, 1))


def has_any_twist(messages: Iterable[Any]) -> bool:
    """Return whether any object in a topic claims a velocity.

    ``DetectedObjectKinematics.has_twist`` says so per object; tracked and predicted
    kinematics have no such flag because a twist is always present.
    """
    return any(
        getattr(obj.kinematics, "has_twist", True)
        for message in messages
        for obj in message.objects
    )


# -- the extraction -----------------------------------------------------------------------


def objects_to_columns(
    message: Any,
    *,
    kind: Kind,
    labels: LabelRegistry,
    instances: InstanceRegistry,
    instance_namespace: str = "",
    unknown_labels: UnknownLabels = "error",
    confidence: Confidence = "classification",
    velocity: Emit = "auto",
    trajectory: tuple[int, int] | None = None,
) -> Object3DColumns:
    """Extract every column one object message can yield, in one pass.

    Args:
        message: A decoded ``DetectedObjects`` / ``TrackedObjects`` / ``PredictedObjects``.
        kind: Which of the three it is; decides where the pose and twist are read from.
        labels: Registry deciding class ids. Never derived here -- see
            :func:`~t4perceval.importer.rosbag.labels.label_registry_from_autoware`.
        instances: Registry interning object identities. Mutated as new ones appear.
        instance_namespace: Prefix for interned identities.
        unknown_labels: What to do with a class the registry does not know.
        confidence: Which message field becomes ``BatchConfidence``.
        velocity: Whether to emit the velocity column.
        trajectory: ``(num_modes, num_timesteps)`` to build trajectory columns with, or
            ``None`` to build none. Every object's paths must fit; excess is an error,
            because a pinned shape smaller than the data means the caller's topic-wide
            pass disagreed with this message.

    Returns:
        The columns, filtered by the label policy.
    """
    objects = list(message.objects)
    names = [classification_name(obj.classification) for obj in objects]
    class_id, keep = encode_class_ids(labels, names, unknown=unknown_labels)
    kept = np.flatnonzero(keep).astype(np.int64)
    objects = [objects[index] for index in kept]
    count = len(objects)

    poses = [_pose(obj, kind) for obj in objects]
    position = _column(
        [[pose.position.x, pose.position.y, pose.position.z] for pose in poses],
        count,
        (3,),
        np.float64,
    )
    quaternion = _quaternion_column(poses, count)

    return Object3DColumns(
        position,
        quaternion,
        _size_column(objects, count),
        class_id,
        _confidence_column(objects, count, confidence),
        instance_id=(
            _instance_column(objects, instances, instance_namespace)
            if kind != "detections"
            else None
        ),
        velocity=_velocity_column(objects, kind, quaternion, count, velocity),
        trajectory=(
            _trajectory_columns(
                objects,
                position,
                num_modes=trajectory[0],
                num_timesteps=trajectory[1],
            )
            if trajectory is not None
            else None
        ),
        frame_id=str(message.header.frame_id),
        kept=kept,
    )


# -- per-kind accessors -------------------------------------------------------------------


def _pose(obj: Any, kind: Kind) -> Any:
    """Return the object's current pose, wherever the kind keeps it."""
    if kind == "predictions":
        return obj.kinematics.initial_pose_with_covariance.pose
    return obj.kinematics.pose_with_covariance.pose


def _twist(obj: Any, kind: Kind) -> Any | None:
    """Return the object's twist, or ``None`` when the message says it has none."""
    kinematics = obj.kinematics
    if kind == "predictions":
        return kinematics.initial_twist_with_covariance.twist
    if not getattr(kinematics, "has_twist", True):
        return None
    return kinematics.twist_with_covariance.twist


# -- column builders ----------------------------------------------------------------------


def _quaternion_column(poses: Sequence[Any], count: int) -> NDArrayF64:
    """Return unit ``xyzw`` quaternions -- the message order already, only normalised."""
    xyzw = _column(
        [[p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w] for p in poses],
        count,
        (4,),
        np.float64,
    )
    if count == 0:
        return xyzw
    norms = np.linalg.norm(xyzw, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(f"Object {int(np.argmin(norms))} has a zero quaternion")
    return xyzw / norms


def _size_column(objects: Sequence[Any], count: int) -> NDArrayF64:
    """Return ``(width, length, height)``.

    ``Shape.dimensions`` is ``(x=length, y=width, z=height)`` for a bounding box, so the
    first two swap; a cylinder's ``x`` is its diameter, which is both its width and length.
    """
    rows = []
    for obj in objects:
        dims = obj.shape.dimensions
        if int(obj.shape.type) == SHAPE_CYLINDER:
            rows.append([dims.x, dims.x, dims.z])
        else:
            rows.append([dims.y, dims.x, dims.z])
    return _column(rows, count, (3,), np.float64)


def _confidence_column(objects: Sequence[Any], count: int, mode: Confidence) -> NDArrayF64:
    """Return the confidence column under ``mode``.

    Probabilities arrive as ``float32``; widening is exact, so a value the producer kept in
    ``[0, 1]`` stays there and one it did not is rejected by ``BatchConfidence`` loudly
    rather than clamped here.
    """
    values = []
    for obj in objects:
        existence = float(obj.existence_probability)
        best = top_classification(obj.classification)
        if best is None or mode == "existence":
            values.append(existence)
        elif mode == "classification":
            values.append(float(best.probability))
        elif mode == "product":
            values.append(existence * float(best.probability))
        else:
            raise ValueError(f"Unknown confidence mode {mode!r}")
    return _column(values, count, (), np.float64)


def _instance_column(
    objects: Sequence[Any],
    instances: InstanceRegistry,
    namespace: str,
) -> NDArrayI64:
    """Intern each object's identity, namespaced so sources cannot collide.

    ``unique_identifier_msgs/UUID`` is 16 raw bytes, which the decoder may hand over as
    ``bytes`` or as a list of ints depending on its version; both spell the same UUID.
    """
    uuids = []
    for obj in objects:
        raw = bytes(bytearray(obj.object_id.uuid))
        name = str(uuid.UUID(bytes=raw))
        uuids.append(f"{namespace}/{name}" if namespace else name)
    return instances.encode(uuids)


def _velocity_column(
    objects: Sequence[Any],
    kind: Kind,
    quaternion: NDArrayF64,
    count: int,
    emit: Emit,
) -> NDArrayF64 | None:
    """Return the velocity column in the message frame, or ``None`` when not emitted.

    Autoware expresses an object's twist in the object's own body frame -- ``x`` forward
    along its heading -- while every other column here is in ``header.frame_id``. Rotating
    by the object's orientation puts the velocity where ``position`` is, so a speed reads
    the same either way and a direction reads correctly. An object without a twist keeps a
    NaN row: zero would be a claim, NaN is the absence of one.
    """
    if emit == "never":
        return None

    rows = []
    for obj in objects:
        twist = _twist(obj, kind)
        rows.append(_NAN3 if twist is None else [twist.linear.x, twist.linear.y, twist.linear.z])
    body = _column(rows, count, (3,), np.float64)

    finite = np.isfinite(body).all(axis=1)
    if emit == "auto" and not finite.any():
        return None

    velocity = body.copy()
    if finite.any():
        velocity[finite] = Rotation.from_quat(quaternion[finite]).apply(body[finite])
    return velocity


def _trajectory_columns(
    objects: Sequence[Any],
    positions: NDArrayF64,
    *,
    num_modes: int,
    num_timesteps: int,
) -> TrajectoryColumns:
    """Build dense, padded, fully-masked trajectory columns.

    Every row is well formed whether or not its object has a path, so prediction rows line
    up one-to-one with the tracking rows of the same frame.
    """
    count = len(objects)
    shape = (count, num_modes, num_timesteps)

    # A row with no path holds station at its own centre. Zeros would teleport it to the
    # origin, which reads as a real -- and badly wrong -- prediction to anything that
    # forgets the mask. NaN is not an option: waypoints must be finite.
    waypoints = np.repeat(
        positions[:, None, None, :],
        num_modes * num_timesteps,
        axis=1,
    ).reshape(*shape, 3)

    mode_confidence = np.zeros((count, num_modes), dtype=np.float64)
    mode_valid = np.zeros((count, num_modes), dtype=np.bool_)
    timestep_valid = np.zeros(shape, dtype=np.bool_)

    # The default axis is already non-negative and strictly increasing, which is what the
    # time-offset column requires even of rows that carry no real path.
    steps = np.arange(1, num_timesteps + 1, dtype=np.int64)
    time_offset = np.tile(steps, (count, 1))

    for index, obj in enumerate(objects):
        paths = _paths_of(obj)
        if not paths:
            continue

        # One time axis per row, so every path of an object must be sampled alike. A
        # uniform step also makes the padded tail truthful: it continues the same axis.
        time_steps = {stamp_ns(path.time_step) for path in paths}
        if len(time_steps) > 1:
            raise ValueError(
                f"Object {index} has predicted paths with different time steps "
                f"{sorted(time_steps)} ns; BatchTimeOffset holds one time axis per object",
            )
        step = time_steps.pop()
        if step <= 0:
            raise ValueError(f"Object {index} has a non-positive predicted path time step")

        longest = max(len(path.path) - 1 for path in paths)
        if len(paths) > num_modes or longest > num_timesteps:
            raise ValueError(
                f"Object {index} has {len(paths)} path(s) of up to {longest} step(s), which "
                f"does not fit the ({num_modes}, {num_timesteps}) shape this topic was "
                f"pinned to",
            )

        for mode, path in enumerate(paths):
            xyz = np.asarray(
                [[p.position.x, p.position.y, p.position.z] for p in path.path[1:]],
                dtype=np.float64,
            )
            real = len(xyz)
            waypoints[index, mode, :real] = xyz
            waypoints[index, mode, real:] = xyz[-1]
            timestep_valid[index, mode, :real] = True
            mode_confidence[index, mode] = float(path.confidence)
        mode_valid[index, : len(paths)] = True
        time_offset[index] = steps * step

    return TrajectoryColumns(waypoints, mode_confidence, mode_valid, timestep_valid, time_offset)
