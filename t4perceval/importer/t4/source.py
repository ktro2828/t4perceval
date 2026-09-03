"""The only module that talks to ``t4_devkit``.

Everything else in ``t4perceval`` -- including the conversion in
:mod:`~t4perceval.importer.t4.convert`, which reaches boxes by attribute access alone --
stays free of the dependency. That is the rule ``t4perceval/typing.py`` states, kept here
in its strongest form: one file to audit when the devkit changes, and a conversion layer
that unit-tests without a dataset.

Also the place where two devkit behaviours are corrected, both of which return plausible
data rather than raising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from attrs import define, field

from t4perceval.importer._optional import require

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from typing_extensions import Self

__all__ = ("Coords", "SampleFrame", "T4Source")

Coords: TypeAlias = Literal["map", "base_link", "sensor"]
"""Which frame 3D boxes are expressed in.

``"map"`` is the annotation's own frame, ``"base_link"`` the ego vehicle, ``"sensor"`` the
channel the boxes were fetched through.
"""


@define(frozen=True, slots=True)
class SampleFrame:
    """One keyframe of a scene."""

    frame: int
    """Position in the scene's full sample chain, zero-based."""

    sample_token: str
    timestamp_us: int
    data: Mapping[str, str] = field(converter=dict)
    """Sensor channel to ``sample_data`` token, keyframes only."""


class T4Source:
    """Traversal and annotation access for one T4 dataset.

    Wraps ``t4_devkit.T4Devkit`` and normalizes the parts the importer depends on.
    """

    def __init__(
        self,
        data_root: str | Path,
        revision: str | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        devkit = require("t4_devkit", extra="t4")

        self._t4 = devkit.T4Devkit(str(data_root), revision, verbose=verbose)

    @classmethod
    def wrap(cls, t4: Any) -> Self:
        """Adopt an already-open ``T4Devkit``, so a caller who has one pays no second load."""
        source = cls.__new__(cls)
        source._t4 = t4
        return source

    @property
    def devkit(self) -> Any:
        """The wrapped ``T4Devkit``, for callers that need something not surfaced here."""
        return self._t4

    @property
    def data_root(self) -> str:
        """Where the dataset was loaded from."""
        return str(self.devkit.data_root)

    @property
    def version(self) -> str | None:
        """The dataset version, when the layout carries one."""
        return self.devkit.version

    # -- tables ------------------------------------------------------------------------

    def categories(self) -> tuple[Any, ...]:
        """Return the ``category`` table."""
        return tuple(self._t4.category)

    def scene_tokens(self) -> tuple[str, ...]:
        """Return every scene token, in table order."""
        return tuple(scene.token for scene in self._t4.scene)

    def channels(self) -> tuple[str, ...]:
        """Return every sensor channel, in table order."""
        return tuple(sensor.channel for sensor in self._t4.sensor)

    def resolve_scene(self, scene: str | int | None) -> str:
        """Return a scene token from a token, an index, or ``None`` for the first scene."""
        tokens = self.scene_tokens()
        if not tokens:
            raise ValueError(f"{self.data_root} contains no scenes")
        if scene is None:
            return tokens[0]
        if isinstance(scene, int):
            return tokens[scene]
        if scene not in tokens:
            raise KeyError(f"Unknown scene {scene!r}; this dataset has {list(tokens)}")
        return scene

    # -- traversal ---------------------------------------------------------------------

    def frames(self, scene_token: str) -> tuple[SampleFrame, ...]:
        """Walk a scene's samples in order.

        Follows ``Scene.first_sample_token`` through the ``next`` chain, which is the only
        ordering the schema guarantees -- ``sample.scene_token`` identifies membership but
        says nothing about sequence.
        """
        scene = self._t4.get("scene", scene_token)
        frames: list[SampleFrame] = []
        token = scene.first_sample_token
        while token:
            sample = self._t4.get("sample", token)
            frames.append(
                SampleFrame(
                    len(frames),
                    sample.token,
                    int(sample.timestamp),
                    sample.data,
                ),
            )
            token = sample.next
        return tuple(frames)

    def sample_data_timestamp(self, sample_data_token: str) -> int:
        """Return one ``sample_data`` record's own capture time, in microseconds."""
        return int(self._t4.get("sample_data", sample_data_token).timestamp)

    # -- annotations -------------------------------------------------------------------

    def boxes3d(
        self,
        sample_data_token: str,
        *,
        coords: Coords = "base_link",
        future_seconds: float = 0.0,
    ) -> list[Any]:
        """Return the 3D boxes of a sample, in the requested frame.

        Box positions and their future waypoints stay in the same frame: ``Box3D.translate``
        and ``Box3D.rotate`` propagate into ``Box3D.future``, so no re-projection is needed
        here. There is a test pinning that, because a devkit change would otherwise put
        trajectories and boxes in different frames without any error.

        Note:
            Asking for a camera channel with ``coords`` other than ``"map"`` silently drops
            boxes that fall outside the image. The importer only ever passes a lidar
            channel, and rejects a camera one.
        """
        if coords == "map":
            return list(self._t4.get_box3ds(sample_data_token, future_seconds=future_seconds))

        _, boxes, _ = self._t4.get_sample_data(
            sample_data_token,
            as_3d=True,
            as_sensor_coord=(coords == "sensor"),
            future_seconds=future_seconds,
        )
        return list(boxes)

    def boxes2d(self, sample_data_token: str) -> list[Any]:
        """Return only the 2D boxes annotated on *this* ``sample_data``.

        ``T4Devkit.get_box2ds`` returns every ``object_ann`` of the whole sample whatever
        channel it was asked for, so requesting the rear camera hands back the front
        camera's boxes. Filtering on ``ObjectAnn.sample_data_token`` is what makes a
        per-camera entity mean what its path says.
        """
        record = self._t4.get("sample_data", sample_data_token)
        sample = self._t4.get("sample", record.sample_token)
        return [
            self._t4.get_box2d(token)
            for token in sample.ann_2ds
            if self._t4.get("object_ann", token).sample_data_token == sample_data_token
        ]

    def is_camera(self, channel: str) -> bool:
        """Return whether a channel is a camera.

        Compares ``modality.value``: ``SensorModality`` is a plain ``Enum``, so ``str()``
        of a member is ``"SensorModality.CAMERA"`` rather than ``"camera"``.
        """
        for sensor in self._t4.sensor:
            if sensor.channel == channel:
                return getattr(sensor.modality, "value", sensor.modality) == "camera"
        raise KeyError(f"Unknown channel {channel!r}; this dataset has {list(self.channels())}")

    def collect(
        self,
        frames: Sequence[SampleFrame],
        channel: str,
        *,
        coords: Coords,
        future_seconds: float,
        strict: bool,
    ) -> dict[int, list[Any]]:
        """Return the 3D boxes of every frame, keyed by frame index."""
        collected: dict[int, list[Any]] = {}
        for frame in frames:
            token = frame.data.get(channel)
            if token is None:
                if strict:
                    raise KeyError(
                        f"Sample {frame.sample_token} has no {channel!r} data; "
                        f"pass strict=False to skip such frames",
                    )
                continue
            collected[frame.frame] = self.boxes3d(
                token,
                coords=coords,
                future_seconds=future_seconds,
            )
        return collected
