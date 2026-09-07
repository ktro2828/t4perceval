"""Builders for small Autoware bags, written through ``mcap_ros2`` from plain dicts.

No binary fixture is committed. A test writes exactly the bag it needs into ``tmp_path``
from the message definitions in :mod:`tests.rosbag_msgdefs`, which are the same text a real
rosbag2 recording embeds -- so the importer's decode path is the real one, and the fixture
is reviewable as source.

Importing this module needs nothing optional; ``write_bag`` imports ``mcap_ros2`` lazily,
so callers ``pytest.importorskip("mcap_ros2")`` first.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from tests.rosbag_msgdefs import MSGDEFS

DETECTED = "autoware_perception_msgs/msg/DetectedObjects"
TRACKED = "autoware_perception_msgs/msg/TrackedObjects"
PREDICTED = "autoware_perception_msgs/msg/PredictedObjects"
TF = "tf2_msgs/msg/TFMessage"

DETECTION_TOPIC = "/perception/object_recognition/detection/objects"
TRACKING_TOPIC = "/perception/object_recognition/tracking/objects"
PREDICTION_TOPIC = "/perception/object_recognition/objects"

NS = 1_000_000_000
#: A round epoch so offsets stay readable in failure messages.
T0 = 1_700_000_000 * NS

#: `/tf` sample times relative to ``T0``. The object stamps span 0..2 s, so the default
#: selection window keeps -0.05 .. 2.5 (one bracket sample either side) and drops the rest.
TF_SECONDS = (-1.0, -0.05, 0.0, 0.5, 1.0, 2.0, 2.5, 3.0)

UUID_A = "aa" * 16
UUID_B = "bb" * 16

# ObjectClassification enum values.
UNKNOWN, CAR, TRUCK, BUS, TRAILER, MOTORCYCLE, BICYCLE, PEDESTRIAN = range(8)

Record = tuple[str, str, dict[str, Any], int]
"""``(topic, datatype, message, log_time_ns)``."""


# -- message pieces -----------------------------------------------------------------------


def stamp(ns: int) -> dict[str, int]:
    return {"sec": ns // NS, "nanosec": ns % NS}


def header(stamp_ns: int, frame_id: str) -> dict[str, Any]:
    return {"stamp": stamp(stamp_ns), "frame_id": frame_id}


def quaternion_yaw(degrees: float) -> dict[str, float]:
    half = math.radians(degrees) / 2
    return {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)}


def xyz(values: tuple[float, float, float]) -> dict[str, float]:
    return dict(zip("xyz", map(float, values)))


def pose(position: tuple[float, float, float], yaw_deg: float = 0.0) -> dict[str, Any]:
    return {"position": xyz(position), "orientation": quaternion_yaw(yaw_deg)}


def classification(label: int, probability: float) -> dict[str, Any]:
    return {"label": label, "probability": probability}


def shape(dims_xyz: tuple[float, float, float], kind: int = 0) -> dict[str, Any]:
    """``dims_xyz`` in the message's own order: ``(x=length, y=width, z=height)``."""
    return {"type": kind, "dimensions": xyz(dims_xyz)}


def uuid_msg(hex32: str) -> dict[str, Any]:
    return {"uuid": list(bytes.fromhex(hex32))}


def _twist(linear: tuple[float, float, float] | None) -> dict[str, Any]:
    return {"twist": {"linear": xyz(linear if linear is not None else (0.0, 0.0, 0.0))}}


# -- objects ------------------------------------------------------------------------------


def detected_object(
    *,
    position: tuple[float, float, float],
    dims: tuple[float, float, float],
    label: int = CAR,
    probability: float = 0.9,
    existence: float = 0.5,
    yaw_deg: float = 0.0,
    twist: tuple[float, float, float] | None = None,
    shape_type: int = 0,
    classifications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "existence_probability": existence,
        "classification": (
            classifications if classifications is not None else [classification(label, probability)]
        ),
        "kinematics": {
            "pose_with_covariance": {"pose": pose(position, yaw_deg)},
            "twist_with_covariance": _twist(twist),
            "has_twist": twist is not None,
        },
        "shape": shape(dims, shape_type),
    }


def tracked_object(
    *,
    uuid: str,
    position: tuple[float, float, float],
    dims: tuple[float, float, float],
    label: int = CAR,
    probability: float = 0.9,
    existence: float = 0.5,
    yaw_deg: float = 0.0,
    twist: tuple[float, float, float] = (0.0, 0.0, 0.0),
    shape_type: int = 0,
) -> dict[str, Any]:
    return {
        "object_id": uuid_msg(uuid),
        "existence_probability": existence,
        "classification": [classification(label, probability)],
        "kinematics": {
            "pose_with_covariance": {"pose": pose(position, yaw_deg)},
            "twist_with_covariance": _twist(twist),
        },
        "shape": shape(dims, shape_type),
    }


def predicted_path(
    positions: list[tuple[float, float, float]],
    *,
    time_step_ns: int,
    confidence: float,
) -> dict[str, Any]:
    """A path whose first pose is the current one, as Autoware writes it."""
    return {
        "path": [pose(p) for p in positions],
        "time_step": stamp(time_step_ns),
        "confidence": confidence,
    }


def predicted_object(
    *,
    uuid: str,
    position: tuple[float, float, float],
    dims: tuple[float, float, float],
    paths: list[dict[str, Any]],
    label: int = CAR,
    probability: float = 0.9,
    existence: float = 0.5,
    yaw_deg: float = 0.0,
    twist: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    return {
        "object_id": uuid_msg(uuid),
        "existence_probability": existence,
        "classification": [classification(label, probability)],
        "kinematics": {
            "initial_pose_with_covariance": {"pose": pose(position, yaw_deg)},
            "initial_twist_with_covariance": _twist(twist),
            "predicted_paths": paths,
        },
        "shape": shape(dims),
    }


def objects_message(stamp_ns: int, frame_id: str, objects: list[dict[str, Any]]) -> dict[str, Any]:
    return {"header": header(stamp_ns, frame_id), "objects": objects}


def tf_message(
    stamp_ns: int,
    parent: str,
    child: str,
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> dict[str, Any]:
    return {
        "transforms": [
            {
                "header": header(stamp_ns, parent),
                "child_frame_id": child,
                "transform": {
                    "translation": xyz(translation),
                    "rotation": dict(zip("xyzw", map(float, rotation))),
                },
            },
        ],
    }


# -- writing ------------------------------------------------------------------------------


def write_bag(path: Path, records: list[Record]) -> Path:
    """Write ``records`` to ``path`` as one MCAP file and return the path."""
    from mcap_ros2.writer import Writer

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        writer = Writer(stream)
        schemas: dict[str, Any] = {}
        for sequence, (topic, datatype, message, log_time_ns) in enumerate(records):
            schema = schemas.get(datatype)
            if schema is None:
                schema = schemas[datatype] = writer.register_msgdef(datatype, MSGDEFS[datatype])
            writer.write_message(
                topic,
                schema,
                message,
                log_time=log_time_ns,
                publish_time=log_time_ns,
                sequence=sequence,
            )
        writer.finish()
    return path


def fixture_records() -> list[Record]:
    """The standard fixture bag: three object topics over three frames, plus a frame tree.

    What it covers, and why:

    * The last frame of every topic is **empty**, so an empty chunk must still carry every
      column and its ``frame_id``.
    * Detections mix ``has_twist`` true and false in one frame, so velocity is ``auto``
      emitted with a NaN row; one object is yawed 30 degrees so a quaternion mix-up or a
      body-frame velocity left unrotated changes the answer; one is a **cylinder**.
    * Trackings are in ``map`` while detections are in ``base_link``, so the frame is read
      from the header rather than assumed.
    * Predictions hold one object with **two paths of different lengths** and one with
      **no path**, pinning the topic-wide shape and the padding.
    * ``/tf`` has samples before, between and after the object stamps at 10 m/s along
      ``x``, so the selection window and interpolation are observable; ``/tf_static`` holds
      one sensor extrinsic.
    """
    t = [T0, T0 + NS, T0 + 2 * NS]
    records: list[Record] = []

    # detections, base_link
    records += [
        (
            DETECTION_TOPIC,
            DETECTED,
            objects_message(
                t[0],
                "base_link",
                [
                    detected_object(
                        position=(10.0, 0.0, 0.0),
                        dims=(4.0, 2.0, 1.5),
                        label=CAR,
                        probability=0.9,
                        existence=0.6,
                        yaw_deg=30.0,
                        twist=(2.0, 0.0, 0.0),
                    ),
                    detected_object(
                        position=(5.0, 3.0, 0.0),
                        dims=(0.6, 0.6, 1.7),
                        label=PEDESTRIAN,
                        probability=0.8,
                        existence=0.4,
                    ),
                ],
            ),
            t[0],
        ),
        (
            DETECTION_TOPIC,
            DETECTED,
            objects_message(
                t[1],
                "base_link",
                [
                    detected_object(
                        position=(3.0, -2.0, 0.0),
                        # A cylinder's `y` is ignored: `x` is the diameter, width and length both.
                        dims=(1.0, 7.0, 2.0),
                        label=UNKNOWN,
                        probability=0.5,
                        existence=0.3,
                        shape_type=1,
                    ),
                ],
            ),
            t[1],
        ),
        (DETECTION_TOPIC, DETECTED, objects_message(t[2], "base_link", []), t[2]),
    ]

    # trackings, map
    car = dict(dims=(4.0, 2.0, 1.5), label=CAR, twist=(2.0, 0.0, 0.0))
    records += [
        (
            TRACKING_TOPIC,
            TRACKED,
            objects_message(
                t[0],
                "map",
                [
                    tracked_object(uuid=UUID_A, position=(10.0, 0.0, 0.0), **car),
                    tracked_object(
                        uuid=UUID_B,
                        position=(5.0, 3.0, 0.0),
                        dims=(0.6, 0.6, 1.7),
                        label=PEDESTRIAN,
                    ),
                ],
            ),
            t[0],
        ),
        (
            TRACKING_TOPIC,
            TRACKED,
            objects_message(
                t[1],
                "map",
                [tracked_object(uuid=UUID_A, position=(12.0, 0.0, 0.0), **car)],
            ),
            t[1],
        ),
        (TRACKING_TOPIC, TRACKED, objects_message(t[2], "map", []), t[2]),
    ]

    # predictions, map
    step = NS // 2
    records += [
        (
            PREDICTION_TOPIC,
            PREDICTED,
            objects_message(
                t[0],
                "map",
                [
                    predicted_object(
                        uuid=UUID_A,
                        position=(10.0, 0.0, 0.0),
                        dims=(4.0, 2.0, 1.5),
                        paths=[
                            predicted_path(
                                [
                                    (10.0, 0.0, 0.0),
                                    (11.0, 0.0, 0.0),
                                    (12.0, 0.0, 0.0),
                                    (13.0, 0.0, 0.0),
                                ],
                                time_step_ns=step,
                                confidence=0.7,
                            ),
                            predicted_path(
                                [(10.0, 0.0, 0.0), (10.5, 0.5, 0.0)],
                                time_step_ns=step,
                                confidence=0.3,
                            ),
                        ],
                    ),
                    predicted_object(
                        uuid=UUID_B,
                        position=(5.0, 3.0, 0.0),
                        dims=(0.6, 0.6, 1.7),
                        label=PEDESTRIAN,
                        paths=[],
                    ),
                ],
            ),
            t[0],
        ),
        (
            PREDICTION_TOPIC,
            PREDICTED,
            objects_message(
                t[1],
                "map",
                [
                    predicted_object(
                        uuid=UUID_A,
                        position=(12.0, 0.0, 0.0),
                        dims=(4.0, 2.0, 1.5),
                        paths=[
                            predicted_path(
                                [(12.0, 0.0, 0.0), (13.0, 0.0, 0.0), (14.0, 0.0, 0.0)],
                                time_step_ns=step,
                                confidence=1.0,
                            ),
                        ],
                    ),
                ],
            ),
            t[1],
        ),
        (PREDICTION_TOPIC, PREDICTED, objects_message(t[2], "map", []), t[2]),
    ]

    # frame tree
    for seconds in TF_SECONDS:
        when = T0 + int(seconds * NS)
        records.append(
            ("/tf", TF, tf_message(when, "map", "base_link", (10.0 * seconds, 0.0, 0.0)), when),
        )
    records.append(
        ("/tf_static", TF, tf_message(T0, "base_link", "lidar_top", (0.0, 0.0, 2.0)), T0 - NS),
    )
    return records
