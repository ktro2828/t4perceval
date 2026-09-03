"""Recording a scene's coordinate-frame tree.

A T4 scene has two kinds of edge, and both become ordinary entities in the store:

* ``map -> base_link``, one sample per keyframe, from the ``ego_pose`` table.
* ``base_link -> <channel>``, fixed, from the ``calibrated_sensor`` table.

They differ only in *when* they hold. An ego pose is logged with ``log`` because it changes
with time; a sensor extrinsic is logged with ``log_static`` because it does not. Same
archetype, same components, same meaning of ``frame_id`` -- the parent frame of the edge.

The entity path is a filing decision, not a frame name: ``/tf/<child>`` is chosen because
it reads well next to the data it describes, and discovery does not depend on it. Both
kinds of edge state their parent through ``frame_id`` and their child through
``child_frame_id``, so :func:`~t4perceval.transform.graph.transform_edges` recovers the
tree by reading the chunks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from t4perceval.archetype.transform import Transform3D
from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import TimePoint
from t4perceval.transform.graph import DEFAULT_ROOT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.core.store import Store
    from t4perceval.importer.t4.source import SampleFrame, T4Source

__all__ = ("EGO_FRAME", "MAP_FRAME", "log_scene_transforms", "tf_path")

#: The frame the T4 annotations are authored in.
MAP_FRAME = "map"

#: The ego-vehicle frame, matching the name `t4_devkit` stamps on a transformed box.
EGO_FRAME = "base_link"


def tf_path(child: str, *, root: EntityPathLike = DEFAULT_ROOT) -> EntityPath:
    """Return the entity the transforms of one child frame are filed under.

    A convenience, not a convention anything relies on: the child frame is recorded in the
    ``child_frame_id`` component, so a reader never parses this path. That is what lets a
    frame name contain a ``/`` -- it simply produces a deeper path here.
    """
    return as_entity_path(root) / child


def _pose_values(record: Any) -> tuple[list[float], list[float]]:
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
        root: Where transform entities are filed.

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

    ego_path = tf_path(EGO_FRAME, root=root)
    for frame in frames:
        token = frame.data.get(channel)
        if token is None:
            continue
        translation, rotation = _pose_values(source.ego_pose(token))
        store.log(
            ego_path,
            Transform3D(translation=translation, rotation=rotation, child_frame_id=EGO_FRAME),
            at=TimePoint.at(frame=frame.frame, timestamp_ns=frame.timestamp_us * 1000),
            frame_id=MAP_FRAME,
        )
    if store.chunks(ego_path):
        written.append(ego_path)

    for sensor_channel, record in sorted(source.extrinsics().items()):
        translation, rotation = _pose_values(record)
        path = tf_path(sensor_channel, root=root)
        store.log_static(
            path,
            Transform3D(
                translation=translation,
                rotation=rotation,
                child_frame_id=sensor_channel,
            ),
            # Static, so there is no sample time to invent. The old design had to place a
            # fixed extrinsic at the scene's first frame and rely on `latest_at` reaching
            # forward from it, which left it invisible to a range query over any later
            # interval.
            frame_id=EGO_FRAME,
        )
        written.append(path)

    return tuple(written)
