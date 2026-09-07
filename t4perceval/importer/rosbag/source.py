"""The only module that talks to the MCAP libraries.

Everything else in ``t4perceval`` -- including the conversion in
:mod:`~t4perceval.importer.rosbag.convert`, which reaches messages by attribute access
alone -- stays free of the dependency. One file to audit when ``mcap`` changes, and a
conversion layer that unit-tests without a bag.

A bag is self-describing: every channel points at a schema record holding the message
definition it was written with, and ``mcap_ros2`` decodes CDR from that text. So an
Autoware bag decodes from its own embedded schemas, with no ROS installation and no
Autoware message packages -- and no version pinning against a project that releases
independently of this one.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from attrs import define

from t4perceval.importer._optional import require
from t4perceval.importer.rosbag.convert import kind_of_schema

if TYPE_CHECKING:
    from collections.abc import Iterator

    from t4perceval.importer.rosbag.convert import Kind

__all__ = ("BagMessage", "BagSource", "TopicInfo")

#: The schema encoding ``mcap_ros2`` decodes. rosbag2 writes it; ``ros2idl`` bags do not
#: carry the ``.msg`` text this decoder needs.
ROS2MSG = "ros2msg"


@define(frozen=True, slots=True)
class TopicInfo:
    """One channel of a bag, as its summary describes it."""

    topic: str
    schema: str
    """Full datatype name, e.g. ``autoware_perception_msgs/msg/DetectedObjects``."""

    encoding: str
    """Schema encoding, normally ``ros2msg``."""

    count: int | None = None
    """Messages recorded, when the bag's statistics say."""

    @property
    def kind(self) -> Kind | None:
        """The archetype kind the schema maps to, or ``None`` for a non-object topic."""
        try:
            return kind_of_schema(self.schema)
        except ValueError:
            return None


@define(frozen=True, slots=True)
class BagMessage:
    """One decoded message and where it sat in the bag."""

    topic: str
    schema: str
    log_time_ns: int
    publish_time_ns: int
    sequence: int
    data: Any
    """The decoded ROS message, an attribute-access object."""


class BagSource:
    """Topic listing and decoded message access for one bag.

    Args:
        path: An ``.mcap`` file, or a rosbag2 directory holding one or more of them. A
            split recording (``<name>_0.mcap``, ``<name>_1.mcap``, ...) is read in sorted
            file order.
    """

    def __init__(self, path: str | Path) -> None:
        # Resolve the extra before touching the filesystem, so a missing dependency is
        # reported as such rather than hidden behind a path error.
        self._reader = require("mcap.reader", extra="rosbag")
        self._decoder = require("mcap_ros2.decoder", extra="rosbag")

        location = Path(path)
        if location.is_dir():
            files = tuple(sorted(location.glob("*.mcap")))
            if not files:
                raise FileNotFoundError(f"No .mcap files under {location}")
        elif location.is_file():
            files = (location,)
        else:
            raise FileNotFoundError(f"No such bag: {location}")

        self.files: tuple[Path, ...] = files
        self.uri = str(path)
        self._topics: tuple[TopicInfo, ...] | None = None

    def _open(self, file: Path) -> tuple[Any, Any]:
        stream = open(file, "rb")  # noqa: SIM115 - closed by the caller
        reader = self._reader.make_reader(
            stream,
            decoder_factories=[self._decoder.DecoderFactory()],
        )
        return stream, reader

    def topics(self) -> tuple[TopicInfo, ...]:
        """Return every topic in the bag, merged across split files, in topic order."""
        if self._topics is not None:
            return self._topics

        merged: dict[str, TopicInfo] = {}
        for file in self.files:
            stream, reader = self._open(file)
            try:
                summary = reader.get_summary()
            finally:
                stream.close()
            if summary is None:
                raise ValueError(
                    f"{file} has no summary section, so its topics cannot be listed without "
                    f"reading the whole file. Rewrite it with `mcap recover`.",
                )
            counts = (
                summary.statistics.channel_message_counts if summary.statistics is not None else {}
            )
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id)
                count = counts.get(channel.id)
                previous = merged.get(channel.topic)
                if previous is not None and previous.count is not None and count is not None:
                    count += previous.count
                merged[channel.topic] = TopicInfo(
                    channel.topic,
                    schema.name if schema is not None else "",
                    schema.encoding if schema is not None else "",
                    count,
                )

        self._topics = tuple(merged[name] for name in sorted(merged))
        return self._topics

    def object_topics(self) -> tuple[TopicInfo, ...]:
        """Return the topics whose schema is an Autoware object message."""
        return tuple(info for info in self.topics() if info.kind is not None)

    def has_topic(self, topic: str) -> bool:
        """Return whether the bag records ``topic``."""
        return any(info.topic == topic for info in self.topics())

    def topic(self, name: str) -> TopicInfo:
        """Return one topic's description, or raise naming what the bag does hold."""
        for info in self.topics():
            if info.topic == name:
                return info
        raise KeyError(
            f"Topic {name!r} is not in the bag. It holds: {[info.topic for info in self.topics()]}",
        )

    def iter_messages(
        self,
        topic: str,
        *,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> Iterator[BagMessage]:
        """Yield one topic's messages in log-time order, decoding nothing else.

        Args:
            topic: The topic to read.
            start_ns: Skip messages logged before this time.
            end_ns: Skip messages logged at or after this time.
        """
        info = self.topic(topic)
        if info.encoding != ROS2MSG:
            raise ValueError(
                f"Topic {topic!r} has schema encoding {info.encoding!r}; only {ROS2MSG!r} "
                f"carries the message text this decoder needs",
            )

        for file in self.files:
            stream, reader = self._open(file)
            try:
                for schema, channel, message, decoded in reader.iter_decoded_messages(
                    topics=[topic],
                    start_time=start_ns,
                    end_time=end_ns,
                    log_time_order=True,
                ):
                    yield BagMessage(
                        channel.topic,
                        schema.name,
                        message.log_time,
                        message.publish_time,
                        message.sequence,
                        decoded,
                    )
            finally:
                stream.close()

    def collect(self, topic: str) -> list[BagMessage]:
        """Materialize one topic, for the pass that settles topic-wide decisions."""
        return list(self.iter_messages(topic))
