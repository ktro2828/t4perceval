"""Importing one object topic of a bag into a :class:`~t4perceval.recording.Recording`.

The importer decides *when and where* data is recorded -- message traversal, timelines,
entity paths -- and delegates *what* the data is to
:mod:`~t4perceval.importer.rosbag.convert`.

It runs in two passes, for the same reason the T4 importer does. ``concat_chunks`` rejects
chunks whose column sets differ, and ``Store.range()`` concatenates, so whether a topic
emits a velocity column and what trajectory shape it uses have to be settled for the whole
topic before the first message is written.

One recording holds one topic. A bag usually carries detection, tracking and prediction
outputs side by side; importing each is a separate call, and evaluating them together is
what :func:`~t4perceval.evaluation.build_evaluation_store` is for.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from attrs import define, field

from t4perceval.core.entity import as_entity_path
from t4perceval.core.store import Store
from t4perceval.core.timeline import TimePoint
from t4perceval.importer.rosbag.convert import (
    has_any_twist,
    kind_of_schema,
    objects_to_columns,
    stamp_ns,
    trajectory_shape_of,
)
from t4perceval.importer.rosbag.labels import label_registry_from_autoware
from t4perceval.importer.rosbag.paths import DEFAULT_ROOT, objects3d_path
from t4perceval.importer.rosbag.source import BagSource
from t4perceval.importer.rosbag.transforms import log_bag_transforms, transform_samples
from t4perceval.label import InstanceRegistry
from t4perceval.recording import Recording, RecordingMetadata, SourceInfo

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from typing_extensions import Self

    from t4perceval.core.entity import EntityPathLike
    from t4perceval.importer._labels import UnknownLabels
    from t4perceval.importer.rosbag.convert import Confidence, Emit
    from t4perceval.importer.rosbag.source import BagMessage, TopicInfo
    from t4perceval.importer.rosbag.transforms import TfScope, TransformSample
    from t4perceval.label import LabelRegistry

__all__ = ("BagSelection", "FrameRef", "ImportOptions", "RosbagImporter")


@define(frozen=True, slots=True)
class BagSelection:
    """Which slice of a bag to import."""

    topic: str | None = None
    """The object topic, or ``None`` when the bag holds exactly one."""

    messages: slice | Sequence[int] | None = None
    """Positions in the topic's message sequence, or ``None`` for all of them."""


@define(frozen=True, slots=True)
class ImportOptions:
    """How the import is performed."""

    confidence: Confidence = "classification"
    """Which message field becomes ``BatchConfidence``."""

    velocity: Emit = "auto"
    num_modes: int | None = None
    """Pin the trajectory mode count, or ``None`` to fit the topic."""

    num_timesteps: int | None = None
    """Pin the trajectory timestep count, or ``None`` to fit the topic."""

    unknown_labels: UnknownLabels = "error"
    entity_root: EntityPathLike = DEFAULT_ROOT
    instance_namespace: str = "est"
    """Prefix for interned identities, so two sources cannot collide on one integer."""

    transforms: bool = True
    """Whether to record the bag's frame tree alongside the objects.

    Without it a recording cannot be re-expressed in another frame later, which is the
    whole reason importers preserve the source frame rather than converting on the way in.
    """

    tf_topics: tuple[str, ...] = field(default=("/tf",), converter=tuple)
    tf_static_topics: tuple[str, ...] = field(default=("/tf_static",), converter=tuple)
    tf_scope: TfScope = "selection"


@define(frozen=True, slots=True)
class FrameRef:
    """What a ``FRAME`` index refers to back in the bag."""

    frame: int
    log_time_ns: int
    timestamp_ns: int
    num_objects: int


@define(slots=True)
class RosbagImporter:
    """Imports object topics of an MCAP bag into recordings.

    Args:
        source: The bag to read.
        options: How to perform the import.
    """

    source: BagSource
    options: ImportOptions = field(factory=ImportOptions, kw_only=True)

    @classmethod
    def open(cls, path: str | Path, *, options: ImportOptions | None = None) -> Self:
        """Open a bag file or directory."""
        return cls(
            BagSource(path),
            options=options if options is not None else ImportOptions(),
        )

    def topics(self) -> tuple[TopicInfo, ...]:
        """Return the object topics the bag holds."""
        return self.source.object_topics()

    def label_registry(self, **kwargs: object) -> LabelRegistry:
        """Build a registry over the Autoware class enum.

        A convenience, not a default: ``import_topic`` still takes the registry as an
        argument, so that the ground-truth source it will be evaluated against can be handed
        the very same one.
        """
        return label_registry_from_autoware(**kwargs)  # type: ignore[arg-type]

    def import_topic(
        self,
        *,
        labels: LabelRegistry,
        instances: InstanceRegistry | None = None,
        selection: BagSelection | None = None,
    ) -> Recording:
        """Import one object topic.

        Args:
            labels: Registry deciding class ids. Required, never derived -- see
                :meth:`label_registry`.
            instances: Registry interning object identities. Pass the same one to every
                importer whose output will be evaluated together.
            selection: Which topic and messages to import.

        Returns:
            A recording holding the topic, bound to the registries that encoded it.
        """
        options = self.options
        chosen = selection if selection is not None else BagSelection()
        registry = instances if instances is not None else InstanceRegistry()

        info = _resolve_topic(self.source.object_topics(), chosen.topic)
        kind = kind_of_schema(info.schema)

        # -- pass 1: materialize, then settle the topic-wide shape ----------------------
        messages = _select(self.source.collect(info.topic), chosen.messages)
        decoded = [message.data for _, message in messages]

        frame_id = _topic_frame_id(decoded, info.topic)
        trajectory = _trajectory_shape(options, decoded) if kind == "predictions" else None
        velocity: Emit = (
            options.velocity
            if options.velocity != "auto"
            else ("always" if has_any_twist(decoded) else "never")
        )

        # -- pass 2: convert and log ----------------------------------------------------
        store = Store()
        path = objects3d_path(as_entity_path(options.entity_root))
        refs: list[FrameRef] = []

        if options.transforms:
            stamps = [stamp_ns(message.header.stamp) for message in decoded]
            window = (
                (min(stamps), max(stamps)) if stamps and options.tf_scope == "selection" else None
            )
            log_bag_transforms(
                store,
                static=self._transforms(options.tf_static_topics),
                dynamic=self._transforms(options.tf_topics),
                window=window,
            )

        for index, message in messages:
            timestamp_ns = stamp_ns(message.data.header.stamp)
            columns = objects_to_columns(
                message.data,
                kind=kind,
                labels=labels,
                instances=registry,
                instance_namespace=options.instance_namespace,
                unknown_labels=options.unknown_labels,
                confidence=options.confidence,
                velocity=velocity,
                trajectory=trajectory,
            )
            store.log(
                path,
                columns.as_archetype(kind),
                at=TimePoint.at(frame=index, timestamp_ns=timestamp_ns),
                frame_id=columns.frame_id,
            )
            refs.append(FrameRef(index, message.log_time_ns, timestamp_ns, len(columns)))

        return Recording.of(
            store,
            labels=labels,
            instances=registry,
            metadata=RecordingMetadata(
                t4perceval_version=_version(),
                created_at_ns=time.time_ns(),
                sources=(
                    SourceInfo(
                        "rosbag",
                        self.source.uri,
                        topic=info.topic,
                        entity_path=str(path),
                        extra={
                            "schema": info.schema,
                            "kind": kind,
                            "frames": str(len(refs)),
                            "confidence": options.confidence,
                        },
                    ),
                ),
                labels_fingerprint=labels.fingerprint(),
                frame_id=frame_id,
            ),
        )

    def _transforms(self, topics: Sequence[str]) -> list[TransformSample]:
        """Collect transform samples from the given topics, skipping absent ones."""
        samples: list[TransformSample] = []
        for topic in topics:
            if self.source.has_topic(topic):
                samples.extend(
                    transform_samples(message.data for message in self.source.iter_messages(topic)),
                )
        return samples


def _resolve_topic(candidates: Sequence[TopicInfo], topic: str | None) -> TopicInfo:
    """Pick the object topic to import, or explain what the bag offers."""
    listing = [(info.topic, info.schema) for info in candidates]
    if topic is None:
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"Bag has {len(candidates)} object topic(s); pass BagSelection(topic=...). "
            f"Found: {listing}",
        )
    for info in candidates:
        if info.topic == topic:
            return info
    raise ValueError(f"Topic {topic!r} is not an object topic of this bag. Found: {listing}")


def _select(
    messages: Sequence[BagMessage],
    chosen: slice | Sequence[int] | None,
) -> tuple[tuple[int, BagMessage], ...]:
    """Narrow a topic's messages, keeping each one's index in the *full* sequence.

    So ``messages=slice(10, 20)`` yields frames numbered 10..19, and two selections of one
    topic stay directly comparable instead of both starting at zero.
    """
    indexed = tuple(enumerate(messages))
    if chosen is None:
        return indexed
    if isinstance(chosen, slice):
        return indexed[chosen]
    return tuple(indexed[index] for index in chosen)


def _topic_frame_id(messages: Sequence[object], topic: str) -> str | None:
    """Return the one frame the topic's objects are in, rejecting a mixture.

    A chunk carries a single ``frame_id`` and ``concat_chunks`` refuses to join chunks
    that disagree, so picking one silently would break the topic-wide query later, far
    from the cause.
    """
    seen = {str(message.header.frame_id) for message in messages}  # type: ignore[attr-defined]
    if len(seen) > 1:
        raise ValueError(f"Topic {topic!r} mixes coordinate frames: {sorted(seen)}")
    return seen.pop() if seen else None


def _trajectory_shape(options: ImportOptions, messages: Sequence[object]) -> tuple[int, int]:
    """Return the topic-wide trajectory shape, honouring any pinned dimension.

    Unlike a T4 future, a predicted path's length is not a dataset constant, so a pin
    smaller than the data is refused rather than silently truncated.
    """
    fitted = trajectory_shape_of(messages)
    modes = options.num_modes if options.num_modes is not None else fitted[0]
    timesteps = options.num_timesteps if options.num_timesteps is not None else fitted[1]
    if modes < fitted[0] or timesteps < fitted[1]:
        raise ValueError(
            f"Topic needs a ({fitted[0]}, {fitted[1]}) trajectory shape but "
            f"({modes}, {timesteps}) was pinned",
        )
    return (modes, timesteps)


def _version() -> str:
    """Return the installed package version, or an empty string when unavailable."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("t4perceval")
    except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
        return ""
