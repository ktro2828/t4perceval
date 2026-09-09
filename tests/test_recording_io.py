"""The recording contract of the ``.t4eval`` format.

``tests/test_columnar_io.py`` owns what one chunk guarantees (dtypes, shapes, nullability).
This file owns what the recording level adds on top: per-entity log order, static and
temporal data side by side, the registries, provenance, and the errors a damaged directory
must raise.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
import pytest
from conftest import make_detections, make_metric_scene, make_trackings

from t4perceval import (
    FRAME,
    TIMESTAMP,
    Chunk,
    InstanceRegistry,
    LabelRegistry,
    MatchResults,
    MetricValues,
    Recording,
    RecordingMetadata,
    SourceInfo,
    Store,
    TimeColumn,
    TimeKind,
    Timeline,
    TimePoint,
    TimeRange,
    concat_chunks,
)
from t4perceval.component import (
    ALL_CLASSES,
    BatchConfidence,
    BatchMask,
    BatchTimeOffset,
    MatchStatus,
)
from t4perceval.descriptors import (
    CLASS_ID,
    CONFIDENCE,
    INSTANCE_ID,
    MASK,
    METRIC_VALUE,
    POSITION,
    SUPPORT,
    THRESHOLD,
    TIME_OFFSET,
)
from t4perceval.evaluation import SourceSpec, build_evaluation_store_from
from t4perceval.io import (
    CHUNKS_DIRNAME,
    MANIFEST_NAME,
    METADATA_KEY,
    RECORDING_FORMAT_VERSION,
    chunk_to_table,
    read_parquet,
    read_recording,
    write_parquet,
    write_recording,
)
from t4perceval.system import (
    AveragePrecisionSystem,
    CenterDistanceMatchingSystem,
    Pipeline,
    SystemContext,
)

if TYPE_CHECKING:
    from pathlib import Path

GT = "/ground_truth/objects"
EST = "/estimation/objects"
SENSOR = "/zzz/sensor"
STEPS = "/scenario/steps"
FILTER = "/estimation/objects/filter/distance"
METRIC = "/metrics/ap"
EMPTY = "/empty/entity"
LIDAR = "/tf/lidar"
MATCHING = "/matching/center_distance"

STEP = Timeline("scenario_step", TimeKind.DURATION)


# -- helpers ---------------------------------------------------------------------------------


def written(recording: Recording, tmp_path: Path, name: str = "result.t4eval") -> Path:
    return write_recording(recording, tmp_path / name)


def _static_frame_outcome(recording: Recording, path: str) -> tuple[str, object]:
    try:
        return ("ok", recording.static_frame_id(path))
    except ValueError as error:
        return ("error", str(error))


def assert_recordings_equal(actual: Recording, expected: Recording) -> None:
    """Compare two recordings through their public read surface, ordered and positional.

    ``actual == expected`` is never meaningful: ``Recording`` compares its private store by
    identity. Everything here is a tuple or list comparison on purpose -- a set, a sort or a
    length check would let an order-losing writer pass.
    """
    assert tuple(actual.entity_paths()) == tuple(expected.entity_paths())
    assert actual.timelines() == expected.timelines()
    for path in expected.entity_paths():
        assert list(actual.chunks(path)) == list(expected.chunks(path))
        assert [c.descriptors for c in actual.chunks(path)] == [
            c.descriptors for c in expected.chunks(path)
        ]
        assert list(actual.static_chunks(path)) == list(expected.static_chunks(path))
        assert [c.descriptors for c in actual.static_chunks(path)] == [
            c.descriptors for c in expected.static_chunks(path)
        ]
        assert actual.static(path) == expected.static(path)
        assert tuple(actual.static(path)) == tuple(expected.static(path))
        assert _static_frame_outcome(actual, path) == _static_frame_outcome(expected, path)
        for timeline in expected.timelines():
            np.testing.assert_array_equal(
                actual.times(path, timeline),
                expected.times(path, timeline),
            )
            assert actual.times(path, timeline).dtype == np.int64
    assert actual.labels == expected.labels
    assert actual.labels.fingerprint() == expected.labels.fingerprint()
    assert actual.instances == expected.instances
    assert actual.metadata == expected.metadata


# -- fixtures --------------------------------------------------------------------------------


@pytest.fixture
def awkward_recording(labels: LabelRegistry) -> Recording:
    """Every case a naive writer gets wrong, logged in an order that exposes it.

    Comments number the log calls; tests refer to them.
    """
    store = Store()
    instances = InstanceRegistry()

    # 1. A static-only entity, logged first: it must still come *last* in entity_paths().
    store.log_static(LIDAR, make_detections([[0.0, 0.0, 2.0]]), frame_id="base_link")
    # 2. Alphabetically last, not the earliest time.
    store.log(
        SENSOR, make_detections([[9.0, 0.0, 0.0]]), at=TimePoint.at(frame=2, timestamp_ns=3_000)
    )
    # 3-4. Two chunks at the same frame *and* timestamp; the later one has the smaller x.
    for x in (2.0, 1.0):
        store.log(
            GT,
            make_detections([[x, 0.0, 0.0]], [0]),
            at=TimePoint.at(frame=0, timestamp_ns=1_000),
            frame_id="base_link",
        )
    # 5. One multi-partition chunk whose middle partition is empty: offsets [0, 2, 2, 3].
    pieces = [
        make_detections([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]).to_chunk(
            GT, at=TimePoint.at(frame=1, timestamp_ns=2_000), frame_id="base_link"
        ),
        make_detections([]).to_chunk(
            GT, at=TimePoint.at(frame=2, timestamp_ns=3_000), frame_id="base_link"
        ),
        make_detections([[3.0, 0.0, 0.0]]).to_chunk(
            GT, at=TimePoint.at(frame=3, timestamp_ns=4_000), frame_id="base_link"
        ),
    ]
    store.send_chunk(concat_chunks(pieces))
    # 6. A whole zero-row chunk holding its place on the timeline.
    store.log(
        GT,
        make_detections([]),
        at=TimePoint.at(frame=4, timestamp_ns=5_000),
        frame_id="base_link",
    )
    # 7-8. Instances interned in a non-alphabetical order.
    store.log(
        EST,
        make_trackings([[1.1, 0.0, 0.0]], instances.encode(["zebra"]), [2]),
        at=TimePoint.at(frame=0, timestamp_ns=1_000),
        frame_id="base_link",
    )
    store.log(
        EST,
        make_trackings(
            [[2.0, 2.0, 0.0], [3.0, 0.0, 0.0]], instances.encode(["apple", "mango"]), [0, 1]
        ),
        at=TimePoint.at(frame=1, timestamp_ns=2_000),
        frame_id="base_link",
    )
    # 9. A third timeline with a non-default kind, on its own entity.
    store.log(STEPS, make_detections([[0.0, 0.0, 0.0]]), at=TimePoint(((STEP, 500),)))
    # 10. Two static chunks on one entity with an overlapping descriptor: 0.9 must win.
    store.log_static_components(GT, {CONFIDENCE: BatchConfidence([0.1])})
    store.log_static_components(
        GT,
        {CONFIDENCE: BatchConfidence([0.9]), TIME_OFFSET: BatchTimeOffset([[0, 100]])},
    )
    # 11. A static column stating a *different* frame than the entity's temporal rows.
    store.log_static_components(EST, {CONFIDENCE: BatchConfidence([0.5])}, frame_id="map")
    # 12. A filter mask -- a bool column under a system-output path.
    store.send_chunk(
        Chunk.from_columns(
            FILTER,
            {MASK: BatchMask([True, False])},
            indexes=(TimeColumn.of(FRAME, [0]),),
        ),
    )
    # 13. A metric with NaN in two columns, the ALL_CLASSES sentinel and support == 0.
    store.log(
        METRIC,
        MetricValues.from_rows(
            [(labels.class_id("car"), 1.0, 0.75, 4), (ALL_CLASSES, np.nan, np.nan, 0)],
        ),
        at=TimePoint.at(frame=0),
    )
    # 14. A chunk with an index but no columns at all.
    store.send_chunk(
        Chunk(
            EMPTY,
            (TimeColumn.of(FRAME, [5]), TimeColumn.of(TIMESTAMP, [6_000])),
            np.array([0, 0]),
            {},
        ),
    )

    metadata = RecordingMetadata(
        t4perceval_version="0.1.0",
        created_at_ns=1_700_000_000_000_000_000,
        sources=(
            SourceInfo(
                "t4", "/data/db_x", version="v1.0", scene="s1", extra={"channel_3d": "LIDAR_TOP"}
            ),
            SourceInfo("rosbag", "/bags/run_7.mcap", topic="/perception/objects"),
        ),
        pipeline=("FilterByDistanceSystem",),
        frame_id="base_link",
        tags={"run": "nightly"},
        notes="fixture",
    )
    return Recording.of(store, labels=labels, instances=instances, metadata=metadata)


EXPECTED_ENTITY_ORDER = (SENSOR, GT, EST, STEPS, FILTER, METRIC, EMPTY, LIDAR)


@pytest.fixture
def evaluated_recording(labels: LabelRegistry) -> Recording:
    """Real matching and metric output, with a genuine false positive."""
    store = make_metric_scene(
        labels,
        [
            (0, [(0.0, "car"), (10.0, "car")], [(0.1, "car", 0.95), (500.0, "car", 0.85)]),
            (1, [(1.0, "pedestrian")], [(1.1, "pedestrian", 0.7)]),
        ],
    )
    systems = [
        CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0),
        AveragePrecisionSystem.on(MATCHING, EST, GT),
    ]
    Pipeline(systems).run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())
    return Recording.of(
        store,
        labels=labels,
        metadata=RecordingMetadata(pipeline=tuple(type(s).__name__ for s in systems)),
    )


def _metric_columns(recording: Recording, path: str) -> dict[str, np.ndarray]:
    view = recording.range(path, timeline=FRAME, time_range=TimeRange.everything())
    return {
        d.component: view.component(d).values  # type: ignore[union-attr]
        for d in (CLASS_ID, THRESHOLD, METRIC_VALUE, SUPPORT)
    }


def _metric_paths(recording: Recording) -> list[str]:
    return [str(p) for p in recording.entity_paths() if str(p).startswith("/metrics/")]


# -- tests -----------------------------------------------------------------------------------


class TestRoundTrip:
    def test_a_whole_recording_survives_a_round_trip(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        assert_recordings_equal(restored, awkward_recording)

    def test_an_ordinary_scene_survives_a_round_trip(
        self, scene_store: Store, labels: LabelRegistry, tmp_path: Path
    ) -> None:
        recording = Recording.of(scene_store, labels=labels)
        restored = read_recording(written(recording, tmp_path))
        assert_recordings_equal(restored, recording)

    def test_a_second_round_trip_changes_nothing(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        once = read_recording(written(awkward_recording, tmp_path, "one.t4eval"))
        twice = read_recording(written(once, tmp_path, "two.t4eval"))
        assert_recordings_equal(twice, awkward_recording)

    def test_a_reopened_recording_exposes_no_writers(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        assert isinstance(restored, Recording)
        for name in ("send_chunk", "log", "log_static"):
            assert not hasattr(restored, name)

    @pytest.mark.parametrize("as_str", [True, False])
    def test_accepts_a_str_or_a_path(
        self, awkward_recording: Recording, tmp_path: Path, as_str: bool
    ) -> None:
        target = tmp_path / "result.t4eval"
        location = str(target) if as_str else target
        assert write_recording(awkward_recording, location) == target
        assert_recordings_equal(read_recording(location), awkward_recording)

    def test_writing_creates_the_directory_and_its_parents(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        target = tmp_path / "deep" / "er" / "result.t4eval"
        assert write_recording(awkward_recording, target) == target
        assert (target / MANIFEST_NAME).is_file()

    def test_a_recording_with_no_entities_round_trips(
        self, labels: LabelRegistry, tmp_path: Path
    ) -> None:
        recording = Recording.of(Store(), labels=labels)
        restored = read_recording(written(recording, tmp_path))
        assert restored.entity_paths() == ()
        assert restored.timelines() == ()
        assert_recordings_equal(restored, recording)


class TestLogOrder:
    """Designed to fail on an order-losing writer, not merely to pass on a correct one."""

    @pytest.mark.parametrize(("logged", "expected"), [((1.0, 2.0), 2.0), ((2.0, 1.0), 1.0)])
    def test_the_most_recently_logged_chunk_still_wins_a_tie(
        self, labels: LabelRegistry, tmp_path: Path, logged: tuple[float, float], expected: float
    ) -> None:
        store = Store()
        for x in logged:
            store.log(
                GT, make_detections([[x, 0.0, 0.0]]), at=TimePoint.at(frame=0, timestamp_ns=7)
            )
        restored = read_recording(written(Recording.of(store, labels=labels), tmp_path))

        for timeline, at in ((FRAME, 0), (TIMESTAMP, 7)):
            view = restored.latest_at(GT, timeline=timeline, at=at)
            assert view.component(POSITION).values[0, 0] == expected  # type: ignore[union-attr]

    def test_chunk_order_survives_beyond_ten_files(
        self, labels: LabelRegistry, tmp_path: Path
    ) -> None:
        store = Store()
        for x in range(12):
            store.log(GT, make_detections([[float(x), 0.0, 0.0]]), at=TimePoint.at(frame=0))
        restored = read_recording(written(Recording.of(store, labels=labels), tmp_path))

        xs = [c.columns[POSITION].values[0, 0] for c in restored.chunks(GT)]
        assert xs == [float(x) for x in range(12)]
        view = restored.latest_at(GT, timeline=FRAME, at=0)
        assert view.component(POSITION).values[0, 0] == 11.0  # type: ignore[union-attr]

    def test_range_keeps_its_tie_order(self, awkward_recording: Recording, tmp_path: Path) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        before = awkward_recording.range(GT, timeline=FRAME, time_range=TimeRange.everything())
        after = restored.range(GT, timeline=FRAME, time_range=TimeRange.everything())

        # Row times: the two empty partitions (frames 2 and 4) contribute no rows.
        assert after.times(FRAME).tolist() == before.times(FRAME).tolist() == [0, 0, 1, 1, 3]
        assert after.partition_ids().tolist() == before.partition_ids().tolist()
        xs = after.component(POSITION).values[:, 0].tolist()  # type: ignore[union-attr]
        assert xs == [2.0, 1.0, 1.0, 2.0, 3.0]

    def test_each_entity_keeps_its_own_order_when_they_interleave(
        self, labels: LabelRegistry, tmp_path: Path
    ) -> None:
        store = Store()
        for x in (0.0, 1.0):
            store.log(GT, make_detections([[x, 0.0, 0.0]]), at=TimePoint.at(frame=0))
            store.log(EST, make_detections([[x + 10.0, 0.0, 0.0]]), at=TimePoint.at(frame=0))
        restored = read_recording(written(Recording.of(store, labels=labels), tmp_path))

        assert [c.columns[POSITION].values[0, 0] for c in restored.chunks(GT)] == [0.0, 1.0]
        assert [c.columns[POSITION].values[0, 0] for c in restored.chunks(EST)] == [10.0, 11.0]

    def test_entity_path_order_is_neither_alphabetical_nor_sorted(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        assert tuple(str(p) for p in restored.entity_paths()) == EXPECTED_ENTITY_ORDER


class TestStaticData:
    def test_static_and_temporal_data_both_survive(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        assert len(restored.chunks(GT)) == 4
        assert len(restored.static_chunks(GT)) == 2

    def test_static_chunks_stay_unfolded_with_their_own_frame_ids(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        first, second = restored.static_chunks(GT)
        assert first.descriptors == (CONFIDENCE,)
        assert second.descriptors == (CONFIDENCE, TIME_OFFSET)
        assert restored.static_chunks(EST)[0].frame_id == "map"
        assert restored.static_chunks(LIDAR)[0].frame_id == "base_link"

    @pytest.mark.parametrize(("order", "winner"), [((0.1, 0.9), 0.9), ((0.9, 0.1), 0.1)])
    def test_the_last_static_write_still_wins(
        self, labels: LabelRegistry, tmp_path: Path, order: tuple[float, float], winner: float
    ) -> None:
        store = Store()
        for value in order:
            store.log_static_components(GT, {CONFIDENCE: BatchConfidence([value])})
        restored = read_recording(written(Recording.of(store, labels=labels), tmp_path))
        assert restored.static(GT)[CONFIDENCE].values.tolist() == [winner]

    def test_a_static_only_entity_does_not_come_back_as_temporal(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        assert restored.chunks(LIDAR) == ()
        assert restored.static_chunks(LIDAR)[0].is_static
        assert restored.static_frame_id(LIDAR) == "base_link"
        assert len(restored.range(LIDAR, timeline=FRAME, time_range=TimeRange.everything())) == 0

    def test_a_static_frame_disagreement_survives(
        self, labels: LabelRegistry, tmp_path: Path
    ) -> None:
        store = Store()
        store.log_static_components(GT, {CONFIDENCE: BatchConfidence([0.1])}, frame_id="map")
        store.log_static_components(GT, {CONFIDENCE: BatchConfidence([0.2])}, frame_id="base_link")
        restored = read_recording(written(Recording.of(store, labels=labels), tmp_path))
        with pytest.raises(ValueError, match="more than one coordinate frame"):
            restored.static_frame_id(GT)

    def test_static_precedence_over_temporal_survives(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        view = restored.latest_at(EST, timeline=FRAME, at=1)
        assert np.all(view.component(CONFIDENCE).values == 0.5)  # type: ignore[union-attr]

    def test_a_single_row_static_column_still_broadcasts(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        view = restored.latest_at(GT, timeline=FRAME, at=1)
        offsets = view.component(TIME_OFFSET).values  # type: ignore[union-attr]
        assert offsets.tolist()[0] == [0, 100]


class TestEmptiness:
    def test_a_zero_row_chunk_keeps_its_place_on_the_timeline(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        assert 4 in restored.times(GT, FRAME).tolist()
        assert restored.chunks(GT)[3].num_rows == 0
        # The assertion that matters: a skipped chunk would carry frame 3 forward.
        assert len(restored.latest_at(GT, timeline=FRAME, at=4)) == 0

    def test_an_empty_partition_inside_a_chunk_survives(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        chunk = restored.chunks(GT)[2]
        assert chunk.offsets.tolist() == [0, 2, 2, 3]
        assert chunk.index(FRAME).times.tolist() == [1, 2, 3]  # type: ignore[union-attr]
        assert len(restored.latest_at(GT, timeline=FRAME, at=2)) == 0

    def test_a_chunk_with_no_columns_survives(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        (chunk,) = restored.chunks(EMPTY)
        assert chunk.descriptors == ()
        assert restored.times(EMPTY, FRAME).tolist() == [5]
        assert restored.times(EMPTY, TIMESTAMP).tolist() == [6_000]


class TestTimelines:
    def test_every_timeline_survives_in_order(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        assert restored.timelines() == (FRAME, TIMESTAMP, STEP)
        assert restored.timelines()[2].kind is TimeKind.DURATION

    def test_a_chunk_indexed_on_one_timeline_does_not_gain_another(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        (chunk,) = restored.chunks(STEPS)
        assert chunk.timelines == (STEP,)
        assert chunk.index(FRAME) is None
        assert restored.times(STEPS, STEP).tolist() == [500]
        assert restored.times(STEPS, FRAME).tolist() == []

    def test_a_static_chunk_contributes_no_timeline(
        self, labels: LabelRegistry, tmp_path: Path
    ) -> None:
        store = Store()
        store.log_static(LIDAR, make_detections([[0.0, 0.0, 2.0]]), frame_id="base_link")
        restored = read_recording(written(Recording.of(store, labels=labels), tmp_path))
        assert restored.timelines() == ()


class TestRegistries:
    def test_the_label_registry_survives_with_its_aliases(self, tmp_path: Path) -> None:
        labels = LabelRegistry.from_names(
            ["car", "truck", "pedestrian"], colors={"car": (255, 0, 0), "truck": (0, 255, 0)}
        ).merged({"vehicle": ["car", "truck"]})
        recording = Recording.of(Store(), labels=labels)
        restored = read_recording(written(recording, tmp_path))
        assert restored.labels == labels
        assert restored.labels.fingerprint() == labels.fingerprint()
        assert restored.labels.class_id("car") == restored.labels.class_id("vehicle")

    def test_the_metadata_fingerprint_still_matches_the_restored_registry(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        assert restored.metadata.labels_fingerprint == restored.labels.fingerprint()
        assert restored.agrees_with(awkward_recording)

    def test_instance_ids_still_decode_to_their_uuids(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        second = restored.chunks(EST)[1]
        ids = second.columns[INSTANCE_ID].values
        assert restored.instances.decode(ids) == ("apple", "mango")
        assert restored.instances.uuid(0) == "zebra"

    def test_an_empty_instance_registry_round_trips_and_stays_usable(
        self, labels: LabelRegistry, tmp_path: Path
    ) -> None:
        restored = read_recording(written(Recording.of(Store(), labels=labels), tmp_path))
        assert len(restored.instances) == 0
        assert restored.instances.intern("a") == 0
        with pytest.raises(KeyError):
            restored.instances.uuid(99)


class TestManifestMetadata:
    def test_the_metadata_survives_intact(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        assert restored.metadata == awkward_recording.metadata
        assert restored.metadata.sources[0].extra == (("channel_3d", "LIDAR_TOP"),)
        assert restored.metadata.sources[1].topic == "/perception/objects"

    def test_writing_stamps_only_the_format_version(
        self, labels: LabelRegistry, tmp_path: Path
    ) -> None:
        metadata = RecordingMetadata(format_version=7, t4perceval_version="0.0.1", created_at_ns=42)
        recording = Recording.of(Store(), labels=labels, metadata=metadata)
        path = written(recording, tmp_path)

        assert recording.metadata.format_version == 7, "the caller's object is untouched"
        manifest = json.loads((path / MANIFEST_NAME).read_text())
        assert manifest["format_version"] == RECORDING_FORMAT_VERSION
        restored = read_recording(path)
        assert restored.metadata.format_version == RECORDING_FORMAT_VERSION
        assert restored.metadata.t4perceval_version == "0.0.1"
        assert restored.metadata.created_at_ns == 42

    def test_the_pipeline_recorded_by_into_recording_survives(
        self, evaluated_recording: Recording, tmp_path: Path
    ) -> None:
        setup = build_evaluation_store_from(
            [SourceSpec.of(evaluated_recording, GT), SourceSpec.of(evaluated_recording, EST)],
        )
        systems = [CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)]
        Pipeline(systems).run(setup.context(), TimeRange.everything())
        recording = setup.into_recording(pipeline=systems)

        restored = read_recording(written(recording, tmp_path))
        assert restored.metadata.pipeline == ("CenterDistanceMatchingSystem",)


class TestMetrics:
    def test_metric_values_are_identical_after_a_round_trip(
        self, evaluated_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(evaluated_recording, tmp_path))
        paths = _metric_paths(evaluated_recording)
        assert paths and _metric_paths(restored) == paths
        for path in paths:
            before, after = (
                _metric_columns(evaluated_recording, path),
                _metric_columns(restored, path),
            )
            for name in before:
                np.testing.assert_array_equal(after[name], before[name])
                assert after[name].dtype == before[name].dtype

    def test_an_undefined_metric_value_stays_nan(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        values = restored.range(METRIC, timeline=FRAME, time_range=TimeRange.everything())
        class_ids = values.component(CLASS_ID).values.tolist()  # type: ignore[union-attr]
        assert class_ids == [restored.labels.class_id("car"), ALL_CLASSES]
        assert np.isnan(values.component(METRIC_VALUE).values[1])  # type: ignore[index]
        assert np.isnan(values.component(THRESHOLD).values[1])  # type: ignore[index]
        assert values.component(SUPPORT).values.tolist() == [4, 0]  # type: ignore[union-attr]

    def test_match_verdict_counts_are_unchanged(
        self, evaluated_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(evaluated_recording, tmp_path))
        before = evaluated_recording.range(
            MATCHING, timeline=FRAME, time_range=TimeRange.everything()
        ).materialize(MatchResults)
        after = restored.range(
            MATCHING, timeline=FRAME, time_range=TimeRange.everything()
        ).materialize(MatchResults)
        assert len(after) == len(before)
        for status in MatchStatus:
            assert after.count(status) == before.count(status)
        assert after.count(MatchStatus(int(after.match_status.values[0]))) > 0

    def test_metrics_recomputed_from_a_reopened_recording_match(
        self, evaluated_recording: Recording, labels: LabelRegistry, tmp_path: Path
    ) -> None:
        restored = read_recording(written(evaluated_recording, tmp_path))
        setup = build_evaluation_store_from(
            [SourceSpec.of(restored, GT), SourceSpec.of(restored, EST)],
        )
        Pipeline(
            [
                CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0),
                AveragePrecisionSystem.on(MATCHING, EST, GT),
            ],
        ).run(setup.context(), TimeRange.everything())
        recomputed = setup.into_recording()

        for path in _metric_paths(evaluated_recording):
            before, after = (
                _metric_columns(evaluated_recording, path),
                _metric_columns(recomputed, path),
            )
            for name in before:
                np.testing.assert_array_equal(after[name], before[name])


class TestFormatLayout:
    def test_it_writes_a_manifest_and_one_parquet_file_per_chunk(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        path = written(awkward_recording, tmp_path)
        expected = sum(
            len(awkward_recording.chunks(p)) + len(awkward_recording.static_chunks(p))
            for p in awkward_recording.entity_paths()
        )
        files = sorted(f.name for f in (path / CHUNKS_DIRNAME).glob("*.parquet"))
        assert files == [f"{i:06d}.parquet" for i in range(expected)]
        manifest = json.loads((path / MANIFEST_NAME).read_text())
        assert manifest["format"] == "t4eval"
        assert [entry["id"] for entry in manifest["chunks"]] == list(range(expected))

    def test_each_chunk_file_is_readable_on_its_own(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        path = written(awkward_recording, tmp_path)
        chunk, _ = read_parquet(path / CHUNKS_DIRNAME / "000000.parquet")
        assert chunk == awkward_recording.chunks(SENSOR)[0]

    def test_the_registries_live_only_in_the_manifest(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        path = written(awkward_recording, tmp_path)
        for file in (path / CHUNKS_DIRNAME).glob("*.parquet"):
            assert read_parquet(file)[1] is None
        manifest = json.loads((path / MANIFEST_NAME).read_text())
        assert manifest["labels"] == awkward_recording.labels.to_metadata()
        assert manifest["instances"] == {"uuids": ["zebra", "apple", "mango"]}

    def test_entity_paths_are_not_used_as_filenames(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        path = written(awkward_recording, tmp_path)
        manifest = json.loads((path / MANIFEST_NAME).read_text())
        for entry in manifest["chunks"]:
            assert "/" not in entry["file"].removeprefix(f"{CHUNKS_DIRNAME}/")
            assert entry["entity_path"].startswith("/")
        assert (path / CHUNKS_DIRNAME / "000000.parquet").is_file()

    def test_temporal_chunks_precede_static_ones_in_the_manifest(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        path = written(awkward_recording, tmp_path)
        flags = [e["is_static"] for e in json.loads((path / MANIFEST_NAME).read_text())["chunks"]]
        assert flags == sorted(flags)
        assert flags.count(True) == 4

    def test_an_existing_directory_is_refused_unless_exist_ok(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        path = written(awkward_recording, tmp_path)
        with pytest.raises(FileExistsError, match="already exists; pass exist_ok=True"):
            write_recording(awkward_recording, path)
        assert write_recording(awkward_recording, path, exist_ok=True) == path
        assert_recordings_equal(read_recording(path), awkward_recording)

    def test_a_file_at_the_destination_is_refused(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        target = tmp_path / "result.t4eval"
        target.write_text("")
        with pytest.raises(ValueError, match="exists and is not a directory"):
            write_recording(awkward_recording, target, exist_ok=True)

    def test_rewriting_ignores_files_it_did_not_list(
        self, labels: LabelRegistry, tmp_path: Path
    ) -> None:
        big, small = Store(), Store()
        for x in range(12):
            big.log(GT, make_detections([[float(x), 0.0, 0.0]]), at=TimePoint.at(frame=x))
        for x in range(2):
            small.log(GT, make_detections([[float(x), 0.0, 0.0]]), at=TimePoint.at(frame=x))

        path = written(Recording.of(big, labels=labels), tmp_path)
        write_recording(Recording.of(small, labels=labels), path, exist_ok=True)

        restored = read_recording(path)
        assert len(restored.chunks(GT)) == 2
        assert len(list((path / CHUNKS_DIRNAME).glob("*.parquet"))) == 12, "stale files stay put"


class TestQueriesAfterReload:
    def test_latest_at_carries_the_last_frame_forward(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        view = restored.latest_at(EST, timeline=FRAME, at=99)
        assert len(view) == 2
        assert view.times(FRAME).tolist() == [1, 1]

    def test_latest_at_before_the_first_frame_is_empty(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        assert len(restored.latest_at(GT, timeline=FRAME, at=-1)) == 0

    def test_a_view_still_does_not_borrow_the_static_frame(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        assert restored.static_frame_id(EST) == "map"
        assert restored.latest_at(EST, timeline=FRAME, at=0).frame_id == "base_link"

    def test_column_restriction_still_works(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        view = restored.latest_at(GT, timeline=FRAME, at=1, components=[POSITION])
        assert view.has(POSITION)
        assert not view.has(CLASS_ID)

    def test_descriptor_order_inside_a_chunk_is_preserved(
        self, awkward_recording: Recording, tmp_path: Path
    ) -> None:
        restored = read_recording(written(awkward_recording, tmp_path))
        assert restored.chunks(GT)[0].descriptors == awkward_recording.chunks(GT)[0].descriptors
        assert restored.chunks(EST)[0].descriptors == awkward_recording.chunks(EST)[0].descriptors


class TestErrors:
    @pytest.fixture
    def path(self, awkward_recording: Recording, tmp_path: Path) -> Path:
        return written(awkward_recording, tmp_path)

    @staticmethod
    def _edit_manifest(path: Path, edit) -> None:  # noqa: ANN001
        manifest_path = path / MANIFEST_NAME
        data = json.loads(manifest_path.read_text())
        edit(data)
        manifest_path.write_text(json.dumps(data))

    def test_a_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No such recording"):
            read_recording(tmp_path / "nope.t4eval")

    def test_a_directory_without_a_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="is not a t4perceval recording: no manifest.json"):
            read_recording(tmp_path / "empty")

    def test_a_file_instead_of_a_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "x.t4eval"
        target.write_text("")
        with pytest.raises(ValueError, match="no manifest.json"):
            read_recording(target)

    def test_a_malformed_manifest(self, path: Path) -> None:
        (path / MANIFEST_NAME).write_text("{not json")
        with pytest.raises(ValueError, match="is not valid JSON"):
            read_recording(path)

    def test_a_manifest_of_some_other_format(self, path: Path) -> None:
        self._edit_manifest(path, lambda d: d.update(format="rrd"))
        with pytest.raises(ValueError, match="format is 'rrd', expected 't4eval'"):
            read_recording(path)

    @pytest.mark.parametrize(
        ("version", "message"),
        [
            (0, "Unsupported recording format version 0, expected"),
            ("1", "Unsupported recording format version '1', expected"),
            (2, "is newer than this build supports"),
            (999, "is newer than this build supports"),
        ],
    )
    def test_an_unsupported_format_version(self, path: Path, version: object, message: str) -> None:
        self._edit_manifest(path, lambda d: d.update(format_version=version))
        # The version check must run before any chunk is opened.
        for file in (path / CHUNKS_DIRNAME).glob("*.parquet"):
            file.unlink()
        with pytest.raises(ValueError, match=message):
            read_recording(path)

    @pytest.mark.parametrize("key", ["labels", "instances", "metadata", "chunks"])
    def test_a_missing_required_key(self, path: Path, key: str) -> None:
        self._edit_manifest(path, lambda d: d.pop(key))
        with pytest.raises(ValueError, match=f"missing required key '{key}'"):
            read_recording(path)

    def test_a_missing_chunk_file(self, path: Path) -> None:
        (path / CHUNKS_DIRNAME / "000000.parquet").unlink()
        with pytest.raises(ValueError, match="missing chunk file 'chunks/000000.parquet'"):
            read_recording(path)

    @pytest.mark.parametrize(
        "damage", [lambda raw: raw[: len(raw) // 2], lambda raw: b"not parquet"]
    )
    def test_a_corrupt_chunk_file_is_named(self, path: Path, damage) -> None:  # noqa: ANN001
        file = path / CHUNKS_DIRNAME / "000001.parquet"
        file.write_bytes(damage(file.read_bytes()))
        with pytest.raises(ValueError, match="Chunk file 'chunks/000001.parquet': "):
            read_recording(path)

    def test_an_unsupported_chunk_schema_version_names_the_file(
        self, path: Path, awkward_recording: Recording
    ) -> None:
        chunk = awkward_recording.chunks(SENSOR)[0]
        table = chunk_to_table(chunk).replace_schema_metadata({METADATA_KEY: b'{"version": 999}'})
        pq.write_table(table, path / CHUNKS_DIRNAME / "000000.parquet")
        with pytest.raises(
            ValueError,
            match="Chunk file 'chunks/000000.parquet': Unsupported chunk schema version 999",
        ):
            read_recording(path)

    def test_an_entity_path_disagreement(self, path: Path) -> None:
        self._edit_manifest(path, lambda d: d["chunks"][0].update(entity_path="/somewhere/else"))
        with pytest.raises(
            ValueError, match="holds entity path '/zzz/sensor' but manifest.json lists"
        ):
            read_recording(path)

    def test_a_static_flag_disagreement(self, path: Path) -> None:
        self._edit_manifest(path, lambda d: d["chunks"][0].update(is_static=True))
        with pytest.raises(
            ValueError, match="listed as static in manifest.json but the chunk declares"
        ):
            read_recording(path)

    def test_a_reordered_manifest(self, path: Path) -> None:
        def swap(d: dict) -> None:
            d["chunks"][0], d["chunks"][1] = d["chunks"][1], d["chunks"][0]

        self._edit_manifest(path, swap)
        with pytest.raises(
            ValueError, match="lists chunk id 1 at position 0; chunk order is load-bearing"
        ):
            read_recording(path)

    def test_a_chunk_file_outside_the_recording(self, path: Path) -> None:
        self._edit_manifest(path, lambda d: d["chunks"][0].update(file="../escape.parquet"))
        with pytest.raises(ValueError, match="outside the recording directory"):
            read_recording(path)

    def test_a_chunk_carrying_its_own_label_registry(
        self, path: Path, awkward_recording: Recording
    ) -> None:
        chunk = awkward_recording.chunks(SENSOR)[0]
        write_parquet(
            chunk, path / CHUNKS_DIRNAME / "000000.parquet", labels=awkward_recording.labels
        )
        with pytest.raises(ValueError, match="carries an embedded label registry"):
            read_recording(path)

    def test_an_unlisted_file_is_ignored(self, path: Path, awkward_recording: Recording) -> None:
        (path / CHUNKS_DIRNAME / "999999.parquet").write_bytes(b"junk")
        assert_recordings_equal(read_recording(path), awkward_recording)
