"""Importing a T4 scene into a :class:`~t4perceval.recording.Recording`.

The importer decides *when and where* data is recorded -- sample traversal, timelines,
entity paths -- and delegates *what* the data is to
:mod:`~t4perceval.importer.t4.convert`.

It runs in two passes, and that is forced by the data model rather than chosen for
tidiness. ``concat_chunks`` rejects chunks whose column sets differ, and ``Store.range()``
concatenates, so whether a scene emits a velocity column and what trajectory shape it uses
have to be settled for the whole scene before the first frame is written. A per-frame
decision produces chunks that log without complaint and then make any whole-scene query
raise.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
from attrs import define, field

from t4perceval.core.entity import as_entity_path
from t4perceval.core.store import Store
from t4perceval.core.timeline import TimePoint
from t4perceval.importer.t4.convert import (
    boxes2d_to_columns,
    boxes3d_to_columns,
    trajectory_shape_of,
)
from t4perceval.importer.t4.labels import label_registry_from_categories
from t4perceval.importer.t4.paths import DEFAULT_ROOT, objects2d_path, objects3d_path
from t4perceval.importer.t4.source import T4Source
from t4perceval.label import InstanceRegistry
from t4perceval.recording import Recording, RecordingMetadata, SourceInfo

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from typing_extensions import Self

    from t4perceval.core.entity import EntityPathLike
    from t4perceval.importer.t4.convert import Emit, Kind2D, Kind3D
    from t4perceval.importer.t4.labels import UnknownLabels
    from t4perceval.importer.t4.source import Coords, SampleFrame
    from t4perceval.label import LabelRegistry

__all__ = ("FrameRef", "ImportOptions", "SceneSelection", "T4Importer")


@define(frozen=True, slots=True)
class SceneSelection:
    """Which slice of a dataset to import."""

    scene: str | int | None = None
    """Scene token, index, or ``None`` for the first scene."""

    samples: slice | Sequence[int] | None = None
    """Positions in the scene's sample chain, or ``None`` for all of them."""

    channel_3d: str | None = "LIDAR_TOP"
    """Channel the 3D boxes are fetched through. ``None`` skips the 3D import."""

    channels_2d: tuple[str, ...] = field(default=(), converter=tuple)
    """Camera channels to import 2D boxes for. Each gets its own entity."""


@define(frozen=True, slots=True)
class ImportOptions:
    """How the import is performed."""

    kind_3d: Kind3D = "trackings"
    """Which 3D archetype to write. ``Trackings3D`` by default: it costs one extra column
    over ``Detections3D`` and is what tracking metrics need, and a detection system runs
    against it unchanged because it requires only a subset of the columns."""

    kind_2d: Kind2D = "trackings"
    coords: Coords = "base_link"
    future_seconds: float = 0.0
    """How far ahead to fetch trajectories. Only consulted for ``kind_3d="predictions"``."""

    num_modes: int | None = None
    """Pin the trajectory mode count, or ``None`` to fit the scene."""

    num_timesteps: int | None = None
    """Pin the trajectory timestep count, or ``None`` to fit the scene."""

    velocity: Emit = "auto"
    num_points: Emit = "auto"
    visibility: Emit = "auto"
    unknown_labels: UnknownLabels = "error"
    entity_root: EntityPathLike = DEFAULT_ROOT
    instance_namespace: str = "gt"
    """Prefix for interned identities, so two sources cannot collide on one integer."""

    strict: bool = True
    """Whether a sample missing a requested channel is an error."""


@define(frozen=True, slots=True)
class FrameRef:
    """What a ``FRAME`` index refers to back in the dataset."""

    frame: int
    sample_token: str
    timestamp_ns: int
    num_objects_3d: int


@define(slots=True)
class T4Importer:
    """Imports T4 scenes into recordings.

    Args:
        source: The dataset to read.
        options: How to perform the import.
    """

    source: T4Source
    options: ImportOptions = field(factory=ImportOptions, kw_only=True)

    @classmethod
    def open(
        cls,
        data_root: str | Path,
        revision: str | None = None,
        *,
        options: ImportOptions | None = None,
        verbose: bool = False,
    ) -> Self:
        """Open a dataset by path."""
        return cls(
            T4Source(data_root, revision, verbose=verbose),
            options=options if options is not None else ImportOptions(),
        )

    def label_registry(self, **kwargs: Any) -> LabelRegistry:
        """Return a registry covering every category in this dataset.

        Offered as a convenience, not as a default: :meth:`import_scene` requires a
        registry to be passed in explicitly. Deriving one silently on each side of an
        evaluation is how two sources come to disagree about what a class id means, and
        the disagreement shows up as plausible numbers rather than as an error.
        """
        return label_registry_from_categories(self.source.categories(), **kwargs)

    def scene_tokens(self) -> tuple[str, ...]:
        """Return every scene token in the dataset."""
        return self.source.scene_tokens()

    def import_scenes(
        self,
        *,
        labels: LabelRegistry,
        instances: InstanceRegistry | None = None,
        selection: SceneSelection | None = None,
    ) -> tuple[Recording, ...]:
        """Import every scene, one recording each.

        One scene is one recording: frame indices restart per scene, so mixing two into
        one store would make identical frame numbers collide.
        """
        base = selection if selection is not None else SceneSelection()
        shared = instances if instances is not None else InstanceRegistry()
        return tuple(
            self.import_scene(
                labels=labels,
                instances=shared,
                selection=SceneSelection(
                    token,
                    base.samples,
                    base.channel_3d,
                    base.channels_2d,
                ),
            )
            for token in self.scene_tokens()
        )

    def import_scene(
        self,
        *,
        labels: LabelRegistry,
        instances: InstanceRegistry | None = None,
        selection: SceneSelection | None = None,
    ) -> Recording:
        """Import one scene.

        Args:
            labels: Registry deciding class ids. Required, never derived -- see
                :meth:`label_registry`.
            instances: Registry interning object identities. Pass the same one to every
                importer whose output will be evaluated together.
            selection: Which scene, samples and channels to import.

        Returns:
            A recording holding the scene, bound to the registries that encoded it.
        """
        options = self.options
        chosen = selection if selection is not None else SceneSelection()
        registry = instances if instances is not None else InstanceRegistry()

        scene_token = self.source.resolve_scene(chosen.scene)
        frames = _select(self.source.frames(scene_token), chosen.samples)

        if chosen.channel_3d is not None and self.source.is_camera(chosen.channel_3d):
            raise ValueError(
                f"channel_3d={chosen.channel_3d!r} is a camera. Fetching 3D boxes through "
                f"a camera silently drops every box outside the image, so the row counts "
                f"would not match the annotations. Use a lidar channel.",
            )

        # -- pass 1: materialize, then settle the scene-wide shape ----------------------
        boxes_3d = (
            self.source.collect(
                frames,
                chosen.channel_3d,
                coords=options.coords,
                future_seconds=options.future_seconds,
                strict=options.strict,
            )
            if chosen.channel_3d is not None
            else {}
        )

        trajectory = (
            _trajectory_shape(options, boxes_3d.values())
            if options.kind_3d == "predictions"
            else None
        )
        emit = _resolve_emit(options, boxes_3d.values())
        frame_id = _scene_frame_id(boxes_3d)

        # -- pass 2: convert and log ----------------------------------------------------
        store = Store()
        entity_root = as_entity_path(options.entity_root)
        path_3d = objects3d_path(entity_root)
        paths_2d = {channel: objects2d_path(entity_root, channel) for channel in chosen.channels_2d}
        refs: list[FrameRef] = []

        for frame in frames:
            timestamp_ns = frame.timestamp_us * 1000
            at = TimePoint.at(frame=frame.frame, timestamp_ns=timestamp_ns)
            count = 0

            if frame.frame in boxes_3d:
                columns = boxes3d_to_columns(
                    boxes_3d[frame.frame],
                    labels=labels,
                    instances=registry,
                    base_time_ns=timestamp_ns,
                    instance_namespace=options.instance_namespace,
                    unknown_labels=options.unknown_labels,
                    trajectory=trajectory,
                    **emit,
                )
                count = len(columns)
                store.log(
                    path_3d,
                    columns.as_archetype(options.kind_3d),
                    at=at,
                    frame_id=columns.frame_id or frame_id,
                )

            for channel, path in paths_2d.items():
                token = frame.data.get(channel)
                if token is None:
                    if options.strict:
                        raise KeyError(
                            f"Sample {frame.sample_token} has no {channel!r} data; "
                            f"pass strict=False to skip such frames",
                        )
                    continue
                columns_2d = boxes2d_to_columns(
                    self.source.boxes2d(token),
                    labels=labels,
                    instances=registry,
                    instance_namespace=options.instance_namespace,
                    unknown_labels=options.unknown_labels,
                )
                store.log(
                    path,
                    columns_2d.as_archetype(options.kind_2d),
                    # A camera's own capture time differs from the sample's, which is
                    # true and worth keeping; they share a FRAME so they still join.
                    at=TimePoint.at(
                        frame=frame.frame,
                        timestamp_ns=self.source.sample_data_timestamp(token) * 1000,
                    ),
                    frame_id=channel,
                )

            refs.append(FrameRef(frame.frame, frame.sample_token, timestamp_ns, count))

        return Recording.of(
            store,
            labels=labels,
            instances=registry,
            metadata=RecordingMetadata(
                t4perceval_version=_version(),
                created_at_ns=time.time_ns(),
                sources=(
                    SourceInfo(
                        "t4",
                        self.source.data_root,
                        version=self.source.version,
                        scene=scene_token,
                        entity_path=str(path_3d),
                        extra={
                            "channel_3d": chosen.channel_3d or "",
                            "coords": options.coords,
                            "kind_3d": options.kind_3d,
                            "frames": str(len(refs)),
                        },
                    ),
                ),
                labels_fingerprint=labels.fingerprint(),
                frame_id=frame_id,
            ),
        )


def _select(
    frames: Sequence[SampleFrame],
    samples: slice | Sequence[int] | None,
) -> tuple[SampleFrame, ...]:
    """Narrow a scene's frames, keeping each one's index in the *full* chain.

    So ``samples=slice(10, 20)`` yields frames numbered 10..19, and two selections of one
    scene stay directly comparable instead of both starting at zero.
    """
    if samples is None:
        return tuple(frames)
    if isinstance(samples, slice):
        return tuple(frames[samples])
    return tuple(frames[index] for index in samples)


def _trajectory_shape(options: ImportOptions, frames: Any) -> tuple[int, int]:
    """Return the scene-wide trajectory shape, honouring any pinned dimension."""
    fitted = trajectory_shape_of(frames)
    return (
        options.num_modes if options.num_modes is not None else fitted[0],
        options.num_timesteps if options.num_timesteps is not None else fitted[1],
    )


def _resolve_emit(options: ImportOptions, frames: Any) -> dict[str, Emit]:
    """Settle each optional column for the whole scene.

    ``"auto"`` cannot be left to the per-frame converter: a column present on one frame
    and absent on the next passes ``log`` and then breaks ``Store.range()``.
    """
    boxes = [box for frame in frames for box in frame]

    def resolve(setting: Emit, present: bool) -> Emit:
        return setting if setting != "auto" else ("always" if present else "never")

    has_velocity = any(
        box.velocity is not None and bool(np.isfinite(np.asarray(box.velocity)).all())
        for box in boxes
    )
    has_num_points = bool(boxes) and all(box.num_points is not None for box in boxes)

    return {
        "velocity": resolve(options.velocity, has_velocity),
        "num_points": resolve(options.num_points, has_num_points),
        "visibility": resolve(options.visibility, bool(boxes)),
    }


def _scene_frame_id(boxes_3d: dict[int, list[Any]]) -> str | None:
    """Return the one frame the scene's boxes are in, rejecting a mixture.

    A chunk carries a single ``frame_id`` and ``concat_chunks`` refuses to join chunks
    that disagree, so picking one silently would break the scene-wide query later, far
    from the cause.
    """
    seen = {str(box.frame_id) for boxes in boxes_3d.values() for box in boxes}
    if len(seen) > 1:
        raise ValueError(f"Scene mixes coordinate frames: {sorted(seen)}")
    return seen.pop() if seen else None


def _version() -> str:
    """Return the installed package version, or an empty string when unavailable."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("t4perceval")
    except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
        return ""
