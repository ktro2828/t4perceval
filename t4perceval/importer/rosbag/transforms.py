"""Recording a bag's coordinate-frame tree.

A ROS bag carries its frame tree on two topics, and both become ordinary entities:

* ``/tf_static`` -- latched, fixed edges such as sensor extrinsics. Logged with
  ``log_static``, so they belong to every timeline and every query interval.
* ``/tf`` -- live edges such as ``map -> base_link``, at whatever rate the stack publishes.
  Logged with ``log`` on the ``TIMESTAMP`` timeline **only**.

The second point is where a bag differs from a T4 scene. Object messages are indexed on
both ``FRAME`` and ``TIMESTAMP``, but a ``/tf`` sample between two object messages has no
frame index -- sub-frame ego motion is the whole point of keeping it -- so the samples
carry the one axis they truthfully have. Look them up with
``TransformResolver.of(recording, timeline=TIMESTAMP)``.

Every edge states its parent through ``frame_id`` and its child through
``child_frame_id``, so :func:`~t4perceval.transform.graph.transform_edges` recovers the
tree by reading the chunks. The entity path ``/tf/<child>`` is a filing decision only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from attrs import define

from t4perceval.archetype.transform import Transform3D
from t4perceval.core.timeline import TimePoint
from t4perceval.importer.rosbag.convert import stamp_ns
from t4perceval.transform.graph import DEFAULT_ROOT, tf_path

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.core.store import Store

__all__ = ("TfScope", "TransformSample", "log_bag_transforms", "transform_samples")

TfScope: TypeAlias = Literal["selection", "all"]
"""How much of ``/tf`` to keep.

``"selection"`` keeps the samples spanning the imported object messages -- everything
between the first and last stamp, plus one sample either side per child so a lookup at the
first frame has a predecessor and one at the last has a successor. ``"all"`` keeps the
whole topic.
"""


@define(frozen=True, slots=True)
class TransformSample:
    """One ``TransformStamped``, as plain values."""

    parent: str
    child: str
    stamp_ns: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    """``xyzw`` -- the message order, which is also this package's."""


def transform_samples(messages: Iterable[Any]) -> list[TransformSample]:
    """Flatten decoded ``TFMessage`` records into samples."""
    samples = []
    for message in messages:
        for record in message.transforms:
            translation = record.transform.translation
            rotation = record.transform.rotation
            samples.append(
                TransformSample(
                    str(record.header.frame_id),
                    str(record.child_frame_id),
                    stamp_ns(record.header.stamp),
                    (float(translation.x), float(translation.y), float(translation.z)),
                    (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
                ),
            )
    return samples


def log_bag_transforms(
    store: Store,
    *,
    static: Sequence[TransformSample],
    dynamic: Sequence[TransformSample],
    window: tuple[int, int] | None = None,
    root: EntityPathLike = DEFAULT_ROOT,
) -> tuple[EntityPath, ...]:
    """Record a bag's frame tree into ``store``.

    Args:
        store: Where to log.
        static: Samples from the latched topic(s). A ``(parent, child)`` repeated with the
            same values is written once; repeated with different values it is an error.
        dynamic: Samples from the live topic(s), logged on ``TIMESTAMP`` only.
        window: ``(first_ns, last_ns)`` of the imported object messages, or ``None`` to keep
            every dynamic sample.
        root: Where transform entities are filed.

    Returns:
        The entity paths written, static edges first.

    Raises:
        ValueError: When a child frame has two parents, or appears on both topics. A child
            is filed under one entity and a chunk carries one ``frame_id``, so either would
            produce chunks that log fine and make the graph unreadable later.
    """
    written: list[EntityPath] = []

    fixed: dict[str, TransformSample] = {}
    for sample in static:
        previous = fixed.get(sample.child)
        if previous is None:
            fixed[sample.child] = sample
            continue
        if previous.parent != sample.parent:
            raise ValueError(
                f"{sample.child!r} has two parents on /tf_static: {previous.parent!r} and "
                f"{sample.parent!r}",
            )
        if not _same_pose(previous, sample):
            raise ValueError(
                f"/tf_static records {sample.parent!r} -> {sample.child!r} twice with "
                f"different values: {previous.translation, previous.rotation} vs "
                f"{sample.translation, sample.rotation}",
            )

    for child in sorted(fixed):
        sample = fixed[child]
        path = tf_path(child, root=root)
        store.log_static(path, _archetype(sample), frame_id=sample.parent)
        written.append(path)

    by_child: dict[str, list[TransformSample]] = {}
    for sample in dynamic:
        by_child.setdefault(sample.child, []).append(sample)

    for child in sorted(by_child):
        samples = sorted(by_child[child], key=lambda s: s.stamp_ns)
        parents = {sample.parent for sample in samples}
        if len(parents) > 1:
            raise ValueError(
                f"{child!r} has two parents on /tf: {sorted(parents)}. A child frame is "
                f"filed under one entity, so its edges cannot disagree on the parent",
            )
        if child in fixed:
            raise ValueError(
                f"{child!r} is recorded on both /tf and /tf_static; one edge per "
                f"(parent, child) is allowed. Drop one side via tf_topics= or "
                f"tf_static_topics=",
            )

        path = tf_path(child, root=root)
        for sample in _within(samples, window):
            store.log(
                path,
                _archetype(sample),
                at=TimePoint.at(timestamp_ns=sample.stamp_ns),
                frame_id=sample.parent,
            )
        if store.chunks(path):
            written.append(path)

    return tuple(written)


def _archetype(sample: TransformSample) -> Transform3D:
    return Transform3D(
        translation=list(sample.translation),
        rotation=list(sample.rotation),
        child_frame_id=sample.child,
    )


def _same_pose(left: TransformSample, right: TransformSample) -> bool:
    return bool(
        np.allclose(left.translation, right.translation)
        and np.allclose(left.rotation, right.rotation),
    )


def _within(
    samples: Sequence[TransformSample],
    window: tuple[int, int] | None,
) -> Sequence[TransformSample]:
    """Return the samples spanning ``window``, one bracket sample either side included."""
    if window is None or not samples:
        return samples
    first, last = window
    stamps = np.fromiter(
        (sample.stamp_ns for sample in samples), dtype=np.int64, count=len(samples)
    )
    # `samples` is sorted, so the bracket is the last sample before `first` and the first
    # sample after `last`.
    lo = int(np.searchsorted(stamps, first, side="left"))
    hi = int(np.searchsorted(stamps, last, side="right"))
    lo = max(lo - 1, 0)
    hi = min(hi + 1, len(samples))
    return samples[lo:hi]
