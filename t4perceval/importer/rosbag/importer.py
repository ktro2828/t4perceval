"""Importing object topics of a bag into a :class:`~t4perceval.recording.Recording`.

The importer decides *when and where* data is recorded -- message traversal, timelines,
entity paths -- and delegates *what* the data is to
:mod:`~t4perceval.importer.rosbag.convert`.

It runs in two passes, for the same reason the T4 importer does. ``concat_chunks`` rejects
chunks whose column sets differ, and ``Store.range()`` concatenates, so whether a topic
emits a velocity column and what trajectory shape it uses have to be settled for the whole
topic before the first message is written.

A bag usually carries detection, tracking and prediction outputs side by side, all against
one frame tree. :meth:`RosbagImporter.import_topics` imports several of them into one
recording, each under its own entity root, and reads ``/tf`` once for all of them;
:meth:`RosbagImporter.import_topic` is the one-topic case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from t4perceval.core.entity import EntityPath, as_entity_path
from t4perceval.core.store import Store
from t4perceval.core.timeline import TimePoint
from t4perceval.importer._columns import resolve_emit
from t4perceval.importer._importer import (
    import_metadata,
    narrow,
    pin_trajectory_shape,
    single_frame_id,
)
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
from t4perceval.recording import Recording, SourceInfo
from t4perceval.transform.graph import DEFAULT_ROOT as TF_ROOT

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from typing_extensions import Self

    from t4perceval.core.entity import EntityPathLike
    from t4perceval.importer._labels import UnknownLabels
    from t4perceval.importer.rosbag.convert import Confidence, Emit, Kind
    from t4perceval.importer.rosbag.source import BagMessage, TopicInfo
    from t4perceval.importer.rosbag.transforms import TfScope, TransformSample
    from t4perceval.label import LabelRegistry

__all__ = ("BagSelection", "ImportOptions", "RosbagImporter")


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


@define(slots=True)
class RosbagImporter:
    """Imports object topics of an MCAP bag into recordings.

    Attributes:
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

    def label_registry(
        self,
        *,
        colors: Mapping[str, tuple[int, int, int]] | None = None,
    ) -> LabelRegistry:
        """Build a registry over the Autoware class enum.

        A convenience, not a default: ``import_topic`` still takes the registry as an
        argument, so that the ground-truth source it will be evaluated against can be handed
        the very same one.
        """
        return label_registry_from_autoware(colors=colors)

    def import_topic(
        self,
        *,
        labels: LabelRegistry,
        instances: InstanceRegistry | None = None,
        selection: BagSelection | None = None,
    ) -> Recording:
        """Import one object topic.

        The one-topic case of :meth:`import_topics`, filed under
        :attr:`ImportOptions.entity_root`. To import several topics of one bag, prefer
        :meth:`import_topics`: calling this once per topic reads the frame tree again each
        time and gives each recording its own copy of it.

        Args:
            labels: Registry deciding class ids. Required, never derived -- see
                :meth:`label_registry`.
            instances: Registry interning object identities. Pass the same one to every
                importer whose output will be evaluated together.
            selection: Which topic and messages to import.

        Returns:
            A recording holding the topic, bound to the registries that encoded it.
        """
        return self.import_topics(
            labels=labels,
            instances=instances,
            selections={
                self.options.entity_root: selection if selection is not None else BagSelection(),
            },
        )

    def import_topics(
        self,
        *,
        labels: LabelRegistry,
        selections: Mapping[EntityPathLike, BagSelection],
        instances: InstanceRegistry | None = None,
    ) -> Recording:
        """Import several object topics into one recording, sharing one frame tree.

        Each topic lands at ``<root>/objects`` for its key in ``selections``, so the topics
        cannot collide on one path. ``/tf`` and ``/tf_static`` are read once and recorded
        once; with ``tf_scope="selection"`` the kept window spans every selected message.

        Examples:
            >>> importer.import_topics(  # doctest: +SKIP
            ...     labels=labels,
            ...     selections={
            ...         "/estimation/detection": BagSelection(topic=DETECTION_TOPIC),
            ...         "/estimation/tracking": BagSelection(topic=TRACKING_TOPIC),
            ...     },
            ... )

        Args:
            labels: Registry deciding class ids. Required, never derived -- see
                :meth:`label_registry`.
            selections: Entity root -> which topic and messages to file under it.
            instances: Registry interning object identities, shared by every topic. Pass
                the same one to every importer whose output will be evaluated together.

        Returns:
            A recording holding every topic, bound to the registries that encoded it.
            Its metadata names one source per topic, in ``selections`` order, and states a
            ``frame_id`` only when every topic is in the same frame.

        Raises:
            ValueError: When ``selections`` is empty, two keys name the same root, a root
                would file objects among the transform edges, or a topic is ambiguous or
                not an object topic of the bag.
        """
        options = self.options
        registry = instances if instances is not None else InstanceRegistry()
        roots = _resolve_roots(selections, transforms=options.transforms)
        candidates = self.source.object_topics()

        # -- pass 1: materialize every topic, then settle each topic-wide shape ---------
        # All topics come first so that the transform window can span all of them.
        topics = [
            self._materialize(_resolve_topic(candidates, chosen.topic), chosen)
            for chosen in selections.values()
        ]

        # -- pass 2: convert and log ----------------------------------------------------
        store = Store()

        if options.transforms:
            stamps = [stamp for topic in topics for stamp in topic.stamps]
            window = (
                (min(stamps), max(stamps)) if stamps and options.tf_scope == "selection" else None
            )
            log_bag_transforms(
                store,
                static=self._transforms(options.tf_static_topics),
                dynamic=self._transforms(options.tf_topics),
                window=window,
            )

        sources = tuple(
            self._log_topic(store, topic, objects3d_path(root), labels=labels, instances=registry)
            for root, topic in zip(roots, topics)
        )

        frame_ids = {topic.frame_id for topic in topics}
        return Recording.of(
            store,
            labels=labels,
            instances=registry,
            metadata=import_metadata(
                *sources,
                frame_id=frame_ids.pop() if len(frame_ids) == 1 else None,
            ),
        )

    def _materialize(self, info: TopicInfo, chosen: BagSelection) -> _Topic:
        """Decode one topic's selected messages and settle what must hold topic-wide."""
        options = self.options
        kind = kind_of_schema(info.schema)
        messages = narrow(self.source.collect(info.topic), chosen.messages)
        decoded = [message.data for _, message in messages]

        return _Topic(
            info=info,
            kind=kind,
            messages=messages,
            frame_id=single_frame_id(
                (str(message.header.frame_id) for message in decoded),
                what=f"Topic {info.topic!r}",
            ),
            trajectory=_trajectory_shape(options, decoded) if kind == "predictions" else None,
            velocity=resolve_emit(options.velocity, has_any_twist(decoded)),
            stamps=tuple(stamp_ns(message.header.stamp) for message in decoded),
        )

    def _log_topic(
        self,
        store: Store,
        topic: _Topic,
        path: EntityPath,
        *,
        labels: LabelRegistry,
        instances: InstanceRegistry,
    ) -> SourceInfo:
        """Convert and log one materialized topic, returning its provenance."""
        options = self.options
        for index, message in topic.messages:
            columns = objects_to_columns(
                message.data,
                kind=topic.kind,
                labels=labels,
                instances=instances,
                instance_namespace=options.instance_namespace,
                unknown_labels=options.unknown_labels,
                confidence=options.confidence,
                velocity=topic.velocity,
                trajectory=topic.trajectory,
            )
            store.log(
                path,
                columns.as_archetype(topic.kind),
                at=TimePoint.at(frame=index, timestamp_ns=stamp_ns(message.data.header.stamp)),
                frame_id=columns.frame_id,
            )

        return SourceInfo(
            "rosbag",
            self.source.uri,
            topic=topic.info.topic,
            entity_path=str(path),
            extra={
                "schema": topic.info.schema,
                "kind": topic.kind,
                "frames": str(len(topic.messages)),
                "confidence": options.confidence,
            },
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


@define(frozen=True, slots=True)
class _Topic:
    """One topic after pass 1: its messages, and the decisions that hold topic-wide."""

    info: TopicInfo
    kind: Kind
    messages: tuple[tuple[int, BagMessage], ...]
    frame_id: str | None
    trajectory: tuple[int, int] | None
    velocity: Emit
    stamps: tuple[int, ...]


def _resolve_roots(
    selections: Mapping[EntityPathLike, BagSelection],
    *,
    transforms: bool,
) -> tuple[EntityPath, ...]:
    """Return the entity root of every selection, refusing roots that would collide."""
    if not selections:
        raise ValueError("import_topics() needs at least one selection")

    roots = tuple(as_entity_path(root) for root in selections)
    seen: dict[EntityPath, int] = {}
    for root in roots:
        seen[root] = seen.get(root, 0) + 1
    repeated = sorted(str(root) for root, count in seen.items() if count > 1)
    if repeated:
        raise ValueError(
            f"Two selections name the same entity root {repeated}; their objects would be "
            f"logged to one path",
        )

    if transforms:
        under_tf = sorted(str(root) for root in roots if root.starts_with(TF_ROOT))
        if under_tf:
            raise ValueError(
                f"Entity root(s) {under_tf} lie under {TF_ROOT}, where transform edges are "
                f"filed; their objects would be read as frames",
            )
    return roots


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


def _trajectory_shape(options: ImportOptions, messages: Sequence[object]) -> tuple[int, int]:
    """Return the topic-wide trajectory shape, honouring any pinned dimension.

    Unlike a T4 future, a predicted path's length is not a dataset constant, so a pin
    smaller than the data is refused rather than silently truncated.
    """
    fitted = trajectory_shape_of(messages)
    modes, timesteps = pin_trajectory_shape(
        fitted,
        num_modes=options.num_modes,
        num_timesteps=options.num_timesteps,
    )
    if modes < fitted[0] or timesteps < fitted[1]:
        raise ValueError(
            f"Topic needs a ({fitted[0]}, {fitted[1]}) trajectory shape but "
            f"({modes}, {timesteps}) was pinned",
        )
    return (modes, timesteps)
