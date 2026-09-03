"""Helpers for building T4 input the vendored fixture cannot express.

The fixture in ``tests/data/t4dataset`` answers "does the importer read the real format".
These builders answer "does it handle the format's awkward corners" -- a future whose
timestamps go backwards, a category the registry does not know, a box with no region of
interest -- without editing a shared fixture to carry them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from t4_devkit.dataclass import Box2D, Box3D, SemanticLabel, Shape, ShapeType
from t4_devkit.dataclass.trajectory import Future
from t4_devkit.schema import VisibilityLevel

if TYPE_CHECKING:
    from collections.abc import Sequence

FRAME_TIME_US = 1_000_000
"""A frame time round enough that offsets stay readable in failure messages."""


def future_of(
    *,
    waypoints: Sequence[Sequence[Sequence[float]]],
    timestamps: Sequence[int],
    confidences: Sequence[float] | None = None,
) -> Future:
    """Build a future from ``(M, T, 3)`` waypoints and absolute microsecond timestamps."""
    array = np.asarray(waypoints, dtype=np.float64)
    return Future(
        timestamps=np.asarray(timestamps, dtype=np.int64),
        confidences=np.asarray(
            confidences if confidences is not None else [1.0] * array.shape[0],
            dtype=np.float64,
        ),
        waypoints=array,
    )


def box3d(
    name: str = "car",
    *,
    position: Sequence[float] = (1.0, 2.0, 3.0),
    rotation: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    size: Sequence[float] = (2.0, 4.5, 1.6),
    velocity: Sequence[float] | None = None,
    num_points: int | None = 10,
    visibility: VisibilityLevel = VisibilityLevel.FULL,
    future: Future | None = None,
    confidence: float = 1.0,
    uuid: str = "instance-0",
    unix_time: int = FRAME_TIME_US,
    frame_id: str = "base_link",
) -> Box3D:
    """Build a 3D box. ``rotation`` is ``wxyz``, as ``pyquaternion`` takes it."""
    return Box3D(
        unix_time,
        frame_id,
        SemanticLabel(name),
        np.asarray(position, dtype=np.float64),
        np.asarray(rotation, dtype=np.float64),
        Shape(ShapeType.BOUNDING_BOX, np.asarray(size, dtype=np.float64)),
        None if velocity is None else np.asarray(velocity, dtype=np.float64),
        num_points,
        visibility,
        future,
        confidence=confidence,
        uuid=uuid,
    )


def box2d(
    name: str = "car",
    *,
    roi: Sequence[int] | None = (100, 100, 200, 150),
    confidence: float = 1.0,
    uuid: str = "instance-0",
    unix_time: int = FRAME_TIME_US,
    frame_id: str = "CAM_FRONT",
) -> Box2D:
    """Build a 2D box. ``roi`` is ``(xmin, ymin, xmax, ymax)``, as the devkit stores it."""
    return Box2D(
        unix_time,
        frame_id,
        SemanticLabel(name),
        roi,
        confidence=confidence,
        uuid=uuid,
    )


def yawed(degrees: float) -> tuple[float, float, float, float]:
    """Return a ``wxyz`` quaternion for a rotation of ``degrees`` about z."""
    half = np.deg2rad(degrees) / 2.0
    return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


def columns_of(boxes: Sequence[Any], **kwargs: Any) -> Any:
    """Convert boxes with a fresh registry pair, for tests that only inspect columns."""
    from t4perceval.importer.t4.convert import boxes3d_to_columns
    from t4perceval.label import InstanceRegistry, LabelRegistry

    kwargs.setdefault("labels", LabelRegistry.from_names(["car", "pedestrian", "bicycle"]))
    kwargs.setdefault("instances", InstanceRegistry())
    kwargs.setdefault("base_time_ns", FRAME_TIME_US * 1000)
    return boxes3d_to_columns(boxes, **kwargs)
