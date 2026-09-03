"""Recording a scene's coordinate-frame tree.

A T4 scene has two kinds of edge, and both become ordinary entities in the store:

* ``map -> base_link``, one sample per frame, from the ``ego_pose`` table.
* ``base_link -> <channel>``, fixed, from the ``calibrated_sensor`` table.

Neither is logged with ``log_static``. Static data carries no ``frame_id`` and, on an
entity with no temporal rows, reads back as zero rows -- whereas a single temporal sample
is returned by ``latest_at`` at every later time, which is the meaning wanted anyway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from t4perceval.archetype.transform import Transform3D
from t4perceval.core.timeline import TimePoint
from t4perceval.transform.paths import DEFAULT_ROOT, transform_path

if TYPE_CHECKING:
    from collections.abc import Sequence

    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.core.store import Store
    from t4perceval.importer.t4.source import SampleFrame, T4Source

__all__ = ("EGO_FRAME", "MAP_FRAME", "log_scene_transforms")

#: The frame the T4 annotations are authored in.
MAP_FRAME = "map"

#: The ego-vehicle frame, matching the name `t4_devkit` stamps on a transformed box.
EGO_FRAME = "base_link"


def _pose_columns(record: Any) -> tuple[list[float], list[float]]:
    """Return one record's translation and its rotation as ``xyzw``.

    The dataset stores ``wxyz`` and this package stores ``xyzw``. Both are four floats, so
    taking them verbatim yields a plausible rotation rather than an error -- the fixture's
    rear camera would read as unrotated instead of turned through 180 degrees.
    """
    wxyz = np.asarray(record.rotation.elements, dtype=np.float64)
    return list(np.asarray(record.translation, dtype=np.float64)), list(wxyz[[1, 2, 3, 0]])


def log_scene_transforms(
    store: Store,
    source: T4Source,
    frames: Sequence[SampleFrame],
    *,
    channel: str,
    root: EntityPathLike = DEFAULT_ROOT,
) -> tuple[EntityPath, ...]:
    """Record the scene's frame tree into ``store``.

    Args:
        store: Where to log.
        source: The dataset being imported.
        frames: The selected keyframes, in order.
        channel: Channel whose ``sample_data`` supplies each frame's ego pose.
        root: Where transform edges live.

    Returns:
        The entity paths written, ego edge first.

    Note:
        One ego sample per **keyframe**, indexed on both timelines exactly as object chunks
        are, so a lookup works whether an evaluation runs on frame indices or timestamps.
        A non-keyframe ``sample_data`` has its own, finer-grained pose that this does not
        record; sub-frame ego motion is therefore not represented, which is the same
        resolution the annotations themselves have.
    """
    written: list[EntityPath] = []

    ego_path = transform_path(MAP_FRAME, EGO_FRAME, root=root)
    for frame in frames:
        token = frame.data.get(channel)
        if token is None:
            continue
        translation, rotation = _pose_columns(source.ego_pose(token))
        store.log(
            ego_path,
            Transform3D(translation=[translation], rotation=[rotation]),
            at=TimePoint.at(frame=frame.frame, timestamp_ns=frame.timestamp_us * 1000),
            frame_id=MAP_FRAME,
        )
    if written or store.chunks(ego_path):
        written.append(ego_path)

    first = frames[0] if frames else None
    for sensor_channel, record in sorted(source.extrinsics().items()):
        translation, rotation = _pose_columns(record)
        path = transform_path(EGO_FRAME, sensor_channel, root=root)
        store.log(
            path,
            Transform3D(translation=[translation], rotation=[rotation]),
            # A fixed edge needs one sample. `latest_at` returns the most recent at or
            # before a time, so one entry at the scene's first frame answers every later
            # query without re-emitting it per frame.
            at=TimePoint.at(
                frame=first.frame if first else 0,
                timestamp_ns=(first.timestamp_us * 1000) if first else 0,
            ),
            frame_id=EGO_FRAME,
        )
        written.append(path)

    return tuple(written)
