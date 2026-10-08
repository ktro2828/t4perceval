"""Importing a generated Autoware bag, end to end."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from t4perceval import (
    FRAME,
    TIMESTAMP,
    MatchResults,
    MetricValues,
    Predictions3D,
    TimeRange,
    Transform3D,
)
from t4perceval.descriptors import CLASS_ID, CONFIDENCE, INSTANCE_ID, SIZE, TRANSLATION, VELOCITY
from t4perceval.align import AlignOptions
from t4perceval.evaluation import build_evaluation_store
from t4perceval.system import CenterDistanceMatchingSystem, Pipeline
from t4perceval.system.preset import average_precision_sweep
from t4perceval.transform import TransformResolver, transform_edges

pytest.importorskip("mcap_ros2")

from t4perceval.importer.rosbag import (  # noqa: E402
    BagSelection,
    ImportOptions,
    RosbagImporter,
)
from tests.rosbag_builder import (  # noqa: E402
    DETECTED,
    DETECTION_TOPIC,
    NS,
    PREDICTION_TOPIC,
    T0,
    TF,
    TF_SECONDS,
    TRACKING_TOPIC,
    UUID_A,
    Record,
    detected_object,
    objects_message,
    tf_message,
    write_bag,
)

if TYPE_CHECKING:
    from pathlib import Path

    from t4perceval.recording import Recording

EVERYTHING = TimeRange.everything()
OBJECTS = "/estimation/objects"
FRAME_TIMES_NS = [T0, T0 + NS, T0 + 2 * NS]


@pytest.fixture
def importer(rosbag_importer: object) -> RosbagImporter:
    assert isinstance(rosbag_importer, RosbagImporter)
    return rosbag_importer


@pytest.fixture
def detections(importer: RosbagImporter) -> Recording:
    return importer.import_topic(
        labels=importer.label_registry(),
        selection=BagSelection(topic=DETECTION_TOPIC),
    )


@pytest.fixture
def trackings(importer: RosbagImporter) -> Recording:
    return importer.import_topic(
        labels=importer.label_registry(),
        selection=BagSelection(topic=TRACKING_TOPIC),
    )


@pytest.fixture
def predictions(importer: RosbagImporter) -> Recording:
    return importer.import_topic(
        labels=importer.label_registry(),
        selection=BagSelection(topic=PREDICTION_TOPIC),
    )


def everything(recording: Recording, path: str = OBJECTS):  # noqa: ANN201
    return recording.range(path, timeline=FRAME, time_range=EVERYTHING)


class TestTopics:
    def test_object_topics_are_listed_with_their_schemas(self, importer: RosbagImporter) -> None:
        assert {(info.topic, info.kind, info.count) for info in importer.topics()} == {
            (DETECTION_TOPIC, "detections", 3),
            (TRACKING_TOPIC, "trackings", 3),
            (PREDICTION_TOPIC, "predictions", 3),
        }

    def test_the_topic_must_be_named_when_there_are_several(self, importer: RosbagImporter) -> None:
        with pytest.raises(ValueError, match="3 object topic"):
            importer.import_topic(labels=importer.label_registry())

    def test_an_unknown_topic_lists_the_candidates(self, importer: RosbagImporter) -> None:
        with pytest.raises(ValueError, match=DETECTION_TOPIC):
            importer.import_topic(
                labels=importer.label_registry(),
                selection=BagSelection(topic="/nope"),
            )

    def test_the_registry_is_required_rather_than_derived(self, importer: RosbagImporter) -> None:
        with pytest.raises(TypeError, match="labels"):
            importer.import_topic()  # type: ignore[call-arg]


class TestTimelines:
    def test_frames_count_the_messages(self, detections: Recording) -> None:
        assert detections.times(OBJECTS, FRAME).tolist() == [0, 1, 2]

    def test_header_stamps_become_nanoseconds(self, detections: Recording) -> None:
        assert detections.times(OBJECTS, TIMESTAMP).tolist() == FRAME_TIMES_NS

    def test_a_selection_keeps_its_place_in_the_topic(self, importer: RosbagImporter) -> None:
        recording = importer.import_topic(
            labels=importer.label_registry(),
            selection=BagSelection(topic=DETECTION_TOPIC, messages=slice(1, 3)),
        )

        assert recording.times(OBJECTS, FRAME).tolist() == [1, 2]
        assert dict(recording.metadata.sources[0].extra)["frames"] == "2"


class TestWholeTopic:
    @pytest.mark.parametrize(
        ("fixture", "frame_id", "rows"),
        [("detections", "base_link", 3), ("trackings", "map", 3), ("predictions", "map", 3)],
    )
    def test_a_range_query_spans_the_topic(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
        frame_id: str,
        rows: int,
    ) -> None:
        recording: Recording = request.getfixturevalue(fixture)

        # The last frame is empty; it still concatenates, so its column set matched.
        view = everything(recording)
        assert len(view) == rows
        assert recording.times(OBJECTS, FRAME).tolist() == [0, 1, 2]
        assert recording.metadata.frame_id == frame_id


class TestDetections:
    def test_sizes_are_width_length_height(self, detections: Recording) -> None:
        size = everything(detections).component(SIZE).values

        # (x=length, y=width, z=height) swaps; a cylinder's x is its diameter both ways.
        assert size.tolist() == [[2.0, 4.0, 1.5], [0.6, 0.6, 1.7], [1.0, 1.0, 2.0]]

    def test_velocity_is_emitted_in_the_message_frame_with_nan_for_none(
        self,
        detections: Recording,
    ) -> None:
        velocity = everything(detections).component(VELOCITY).values

        # 2 m/s along a heading of 30 degrees.
        np.testing.assert_allclose(velocity[0], [2 * np.cos(np.pi / 6), 1.0, 0.0])
        assert np.isnan(velocity[1:]).all()

    def test_confidence_defaults_to_the_classification_probability(
        self,
        detections: Recording,
    ) -> None:
        confidence = everything(detections).component(CONFIDENCE).values

        np.testing.assert_allclose(confidence, [0.9, 0.8, 0.5], rtol=1e-6)

    def test_confidence_can_be_the_existence_probability(self, rosbag_path: Path) -> None:
        importer = RosbagImporter.open(rosbag_path, options=ImportOptions(confidence="existence"))
        recording = importer.import_topic(
            labels=importer.label_registry(),
            selection=BagSelection(topic=DETECTION_TOPIC),
        )

        confidence = everything(recording).component(CONFIDENCE).values
        np.testing.assert_allclose(confidence, [0.6, 0.4, 0.3], rtol=1e-6)
        assert dict(recording.metadata.sources[0].extra)["confidence"] == "existence"

    def test_class_ids_come_from_the_registry(self, detections: Recording) -> None:
        class_id = everything(detections).component(CLASS_ID).values

        assert detections.labels.decode(class_id) == ("car", "pedestrian", "unknown")

    def test_detections_carry_no_identity(self, detections: Recording) -> None:
        assert INSTANCE_ID not in everything(detections).descriptors


class TestTrackings:
    def test_identities_are_stable_across_frames(self, trackings: Recording) -> None:
        ids = everything(trackings).component(INSTANCE_ID).values.tolist()

        assert ids[0] == ids[2]  # the car in frames 0 and 1
        assert ids[0] != ids[1]

    def test_identities_are_namespaced_uuids(self, trackings: Recording) -> None:
        ids = everything(trackings).component(INSTANCE_ID).values

        assert trackings.instances.uuid(int(ids[0])) == "est/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        assert UUID_A == "aa" * 16


class TestPredictions:
    def test_the_shape_is_settled_topic_wide(self, predictions: Recording) -> None:
        batch = everything(predictions).materialize(Predictions3D)

        assert (batch.num_modes, batch.num_timesteps) == (2, 3)
        assert len(batch.position) == 3

    def test_paths_are_padded_and_masked(self, predictions: Recording) -> None:
        batch = everything(predictions).materialize(Predictions3D)
        assert batch.mode_valid is not None and batch.time_offset is not None

        assert batch.mode_valid.values.tolist() == [[True, True], [False, False], [True, False]]
        assert batch.time_offset.values[0].tolist() == [NS // 2, NS, 3 * NS // 2]
        assert np.isfinite(batch.waypoints.values).all()

    def test_a_pin_smaller_than_the_data_is_refused(self, rosbag_path: Path) -> None:
        importer = RosbagImporter.open(rosbag_path, options=ImportOptions(num_timesteps=2))

        with pytest.raises(ValueError, match="pinned"):
            importer.import_topic(
                labels=importer.label_registry(),
                selection=BagSelection(topic=PREDICTION_TOPIC),
            )

    def test_a_larger_pin_pads(self, rosbag_path: Path) -> None:
        importer = RosbagImporter.open(rosbag_path, options=ImportOptions(num_timesteps=5))
        recording = importer.import_topic(
            labels=importer.label_registry(),
            selection=BagSelection(topic=PREDICTION_TOPIC),
        )

        assert everything(recording).materialize(Predictions3D).num_timesteps == 5


class TestProvenance:
    def test_the_source_is_recorded(self, trackings: Recording, rosbag_path: Path) -> None:
        (source,) = trackings.metadata.sources

        assert source.kind == "rosbag"
        assert source.uri == str(rosbag_path)
        assert source.topic == TRACKING_TOPIC
        assert source.entity_path == OBJECTS
        assert dict(source.extra)["schema"] == "autoware_perception_msgs/msg/TrackedObjects"
        assert dict(source.extra)["kind"] == "trackings"

    def test_the_registry_is_fingerprinted(self, trackings: Recording) -> None:
        assert trackings.metadata.labels_fingerprint == trackings.labels.fingerprint()


class TestEvaluation:
    def test_a_self_match_scores_perfectly(self, trackings: Recording) -> None:
        setup = build_evaluation_store(
            trackings,
            trackings,
            reference_path=OBJECTS,
            reference_target="/ground_truth/objects",
        )
        systems = average_precision_sweep(OBJECTS, "/ground_truth/objects", thresholds=[1.0])
        Pipeline(systems).run(setup.context(), EVERYTHING)

        matches = setup.store.range(
            "/matching/center_distance/0",
            timeline=FRAME,
            time_range=EVERYTHING,
        ).materialize(MatchResults)
        metrics = setup.store.range(
            "/metrics/map",
            timeline=FRAME,
            time_range=EVERYTHING,
        ).materialize(MetricValues)

        assert (matches.num_tp, matches.num_fp, matches.num_fn) == (3, 0, 0)
        assert metrics.aggregate == pytest.approx(1.0)

    def test_a_self_match_survives_alignment(self, trackings: Recording) -> None:
        # Pairing a recording with itself by timestamp is the identity on its frames.
        setup = build_evaluation_store(
            trackings,
            trackings,
            reference_path=OBJECTS,
            reference_target="/ground_truth/objects",
            align=AlignOptions(),
        )
        Pipeline(
            [CenterDistanceMatchingSystem.between(OBJECTS, "/ground_truth/objects", threshold=1.0)],
        ).run(setup.context(), EVERYTHING)

        matches = setup.store.range(
            "/matching/center_distance",
            timeline=FRAME,
            time_range=EVERYTHING,
        ).materialize(MatchResults)

        assert (matches.num_tp, matches.num_fp, matches.num_fn) == (3, 0, 0)
        assert dict(setup.metadata.tags)["align.pairs"] == "3"


class TestTransforms:
    def test_the_frame_tree_is_recorded(self, trackings: Recording) -> None:
        by_child = {edge.child: edge for edge in transform_edges(trackings)}

        assert {edge.frames for edge in by_child.values()} == {
            ("map", "base_link"),
            ("base_link", "lidar_top"),
        }
        assert by_child["base_link"].is_static is False
        assert by_child["lidar_top"].is_static is True

    def test_live_edges_sit_on_the_timestamp_timeline_only(self, trackings: Recording) -> None:
        # A /tf sample between two messages has no frame index, so none is invented.
        assert trackings.times("/tf/base_link", FRAME).tolist() == []
        kept = (trackings.times("/tf/base_link", TIMESTAMP) - T0) / NS

        # The object stamps span 0..2 s; one bracket sample either side is kept.
        assert kept.tolist() == [-0.05, 0.0, 0.5, 1.0, 2.0, 2.5]

    def test_the_window_follows_the_selection(self, importer: RosbagImporter) -> None:
        recording = importer.import_topic(
            labels=importer.label_registry(),
            selection=BagSelection(topic=TRACKING_TOPIC, messages=slice(0, 2)),
        )
        kept = (recording.times("/tf/base_link", TIMESTAMP) - T0) / NS

        assert kept.tolist() == [-0.05, 0.0, 0.5, 1.0, 2.0]

    def test_the_whole_topic_can_be_kept(self, rosbag_path: Path) -> None:
        importer = RosbagImporter.open(rosbag_path, options=ImportOptions(tf_scope="all"))
        recording = importer.import_topic(
            labels=importer.label_registry(),
            selection=BagSelection(topic=TRACKING_TOPIC),
        )
        kept = (recording.times("/tf/base_link", TIMESTAMP) - T0) / NS

        assert kept.tolist() == list(TF_SECONDS)

    def test_ego_poses_are_read_as_a_column(self, trackings: Recording) -> None:
        view = trackings.range("/tf/base_link", timeline=TIMESTAMP, time_range=EVERYTHING)

        # 10 m/s along x.
        np.testing.assert_allclose(
            view.component(TRANSLATION).values[:, 0],
            [-0.5, 0.0, 5.0, 10.0, 20.0, 25.0],
        )

    def test_a_static_edge_needs_no_time(self, trackings: Recording) -> None:
        chunk = trackings.static_chunks("/tf/lidar_top")[0]

        assert trackings.static_frame_id("/tf/lidar_top") == "base_link"
        assert trackings.times("/tf/lidar_top", TIMESTAMP).tolist() == []
        assert Transform3D.from_chunk(chunk).translation.value.tolist() == [0.0, 0.0, 2.0]

    def test_a_lookup_composes_live_and_static_edges(self, trackings: Recording) -> None:
        resolver = TransformResolver.of(trackings, timeline=TIMESTAMP)

        pose = resolver.lookup(target_frame="map", source_frame="lidar_top", at=T0 + NS)
        assert pose.translation.value.tolist() == [10.0, 0.0, 2.0]

    def test_transforms_can_be_left_out(self, rosbag_path: Path) -> None:
        importer = RosbagImporter.open(rosbag_path, options=ImportOptions(transforms=False))
        recording = importer.import_topic(
            labels=importer.label_registry(),
            selection=BagSelection(topic=TRACKING_TOPIC),
        )

        assert transform_edges(recording) == ()
        assert [str(path) for path in recording.entity_paths()] == [OBJECTS]


class TestSeveralTopics:
    """``import_topics`` files each topic under its own root, against one frame tree."""

    DETECTION_ROOT = "/estimation/detection"
    TRACKING_ROOT = "/estimation/tracking"

    def import_both(self, importer: RosbagImporter, **kwargs: object) -> Recording:
        return importer.import_topics(
            labels=importer.label_registry(),
            selections={
                self.DETECTION_ROOT: BagSelection(topic=DETECTION_TOPIC),
                self.TRACKING_ROOT: BagSelection(topic=TRACKING_TOPIC),
            },
            **kwargs,  # type: ignore[arg-type]
        )

    def test_each_topic_lands_under_its_own_root(self, importer: RosbagImporter) -> None:
        recording = self.import_both(importer)

        assert sorted(str(path) for path in recording.entity_paths()) == [
            "/estimation/detection/objects",
            "/estimation/tracking/objects",
            "/tf/base_link",
            "/tf/lidar_top",
        ]

    def test_each_topic_matches_its_own_import(
        self,
        importer: RosbagImporter,
        detections: Recording,
        trackings: Recording,
    ) -> None:
        recording = self.import_both(importer)

        for root, alone in ((self.DETECTION_ROOT, detections), (self.TRACKING_ROOT, trackings)):
            together = everything(recording, f"{root}/objects")
            assert len(together) == len(everything(alone))
            assert together.times(FRAME).tolist() == everything(alone).times(FRAME).tolist()
            assert (
                together.component(CLASS_ID).values.tolist()
                == everything(alone).component(CLASS_ID).values.tolist()
            )

    def test_the_frame_tree_is_read_and_recorded_once(
        self,
        importer: RosbagImporter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reads: list[str] = []
        original = type(importer.source).iter_messages

        def counting(self: object, topic: str, **kwargs: object):  # noqa: ANN202
            reads.append(topic)
            return original(self, topic, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(type(importer.source), "iter_messages", counting)
        recording = self.import_both(importer)

        assert reads.count("/tf") == 1
        assert reads.count("/tf_static") == 1
        assert len(recording.static_chunks("/tf/lidar_top")) == 1

    def test_the_window_spans_every_selection(self, importer: RosbagImporter) -> None:
        # Detections from the last frame only (2 s), trackings from the first only (0 s).
        recording = importer.import_topics(
            labels=importer.label_registry(),
            selections={
                self.DETECTION_ROOT: BagSelection(topic=DETECTION_TOPIC, messages=[2]),
                self.TRACKING_ROOT: BagSelection(topic=TRACKING_TOPIC, messages=[0]),
            },
        )
        kept = (recording.times("/tf/base_link", TIMESTAMP) - T0) / NS

        assert kept.tolist() == [-0.05, 0.0, 0.5, 1.0, 2.0, 2.5]

    def test_one_identity_registry_serves_every_topic(self, importer: RosbagImporter) -> None:
        recording = importer.import_topics(
            labels=importer.label_registry(),
            selections={
                self.TRACKING_ROOT: BagSelection(topic=TRACKING_TOPIC),
                "/estimation/prediction": BagSelection(topic=PREDICTION_TOPIC),
            },
        )
        tracked = everything(recording, f"{self.TRACKING_ROOT}/objects")
        predicted = everything(recording, "/estimation/prediction/objects")

        # Both topics carry UUID_A first in frame 0; it is one object, so one id.
        assert (
            tracked.component(INSTANCE_ID).values[0] == predicted.component(INSTANCE_ID).values[0]
        )

    def test_every_topic_is_a_recorded_source(self, importer: RosbagImporter) -> None:
        sources = self.import_both(importer).metadata.sources

        assert [(source.topic, source.entity_path) for source in sources] == [
            (DETECTION_TOPIC, "/estimation/detection/objects"),
            (TRACKING_TOPIC, "/estimation/tracking/objects"),
        ]

    def test_a_frame_is_stated_only_when_every_topic_agrees(
        self,
        importer: RosbagImporter,
    ) -> None:
        mixed = self.import_both(importer)  # detections in base_link, trackings in map
        agreeing = importer.import_topics(
            labels=importer.label_registry(),
            selections={
                self.TRACKING_ROOT: BagSelection(topic=TRACKING_TOPIC),
                "/estimation/prediction": BagSelection(topic=PREDICTION_TOPIC),
            },
        )

        assert mixed.metadata.frame_id is None
        assert agreeing.metadata.frame_id == "map"
        # The chunks still say where their own rows live.
        assert {chunk.frame_id for chunk in mixed.chunks(f"{self.DETECTION_ROOT}/objects")} == {
            "base_link"
        }

    def test_the_same_topic_may_be_filed_twice(self, importer: RosbagImporter) -> None:
        recording = importer.import_topics(
            labels=importer.label_registry(),
            selections={
                "/estimation/early": BagSelection(topic=TRACKING_TOPIC, messages=slice(0, 1)),
                "/estimation/late": BagSelection(topic=TRACKING_TOPIC, messages=slice(1, 3)),
            },
        )

        assert recording.times("/estimation/early/objects", FRAME).tolist() == [0]
        assert recording.times("/estimation/late/objects", FRAME).tolist() == [1, 2]

    def test_it_needs_a_selection(self, importer: RosbagImporter) -> None:
        with pytest.raises(ValueError, match="at least one selection"):
            importer.import_topics(labels=importer.label_registry(), selections={})

    def test_two_keys_for_one_root_are_refused(self, importer: RosbagImporter) -> None:
        from t4perceval.core.entity import EntityPath

        with pytest.raises(ValueError, match="same entity root"):
            importer.import_topics(
                labels=importer.label_registry(),
                selections={
                    "/estimation/a": BagSelection(topic=DETECTION_TOPIC),
                    EntityPath.parse("/estimation/a"): BagSelection(topic=TRACKING_TOPIC),
                },
            )

    def test_a_root_among_the_transforms_is_refused(self, importer: RosbagImporter) -> None:
        with pytest.raises(ValueError, match="where transform edges are filed"):
            importer.import_topics(
                labels=importer.label_registry(),
                selections={"/tf/objects": BagSelection(topic=DETECTION_TOPIC)},
            )

    def test_without_transforms_any_root_is_allowed(self, rosbag_path: Path) -> None:
        importer = RosbagImporter.open(rosbag_path, options=ImportOptions(transforms=False))
        recording = importer.import_topics(
            labels=importer.label_registry(),
            selections={"/tf/objects": BagSelection(topic=DETECTION_TOPIC)},
        )

        assert [str(path) for path in recording.entity_paths()] == ["/tf/objects/objects"]

    def test_an_unknown_topic_is_reported(self, importer: RosbagImporter) -> None:
        with pytest.raises(ValueError, match="not an object topic"):
            importer.import_topics(
                labels=importer.label_registry(),
                selections={self.DETECTION_ROOT: BagSelection(topic="/nope")},
            )

    def test_the_combined_recording_evaluates(self, importer: RosbagImporter) -> None:
        recording = self.import_both(importer)
        setup = build_evaluation_store(
            recording,
            recording,
            reference_path=f"{self.TRACKING_ROOT}/objects",
            query_path=f"{self.TRACKING_ROOT}/objects",
            reference_target="/ground_truth/objects",
            query_target="/estimation/objects",
        )
        matcher = CenterDistanceMatchingSystem.between(
            "/estimation/objects",
            "/ground_truth/objects",
            threshold=1.0,
        )
        Pipeline([matcher]).run(setup.context(), EVERYTHING)
        result = setup.store.range(
            matcher.target,
            timeline=FRAME,
            time_range=EVERYTHING,
        ).materialize(MatchResults)

        assert (result.num_fp, result.num_fn) == (0, 0)
        assert result.num_tp > 0


class TestTransformConflicts:
    """Bags whose frame tree cannot be filed as one edge per child."""

    def _import(self, tmp_path: Path, tf_records: list[Record]) -> Recording:
        records: list[Record] = [
            (
                DETECTION_TOPIC,
                DETECTED,
                objects_message(
                    T0,
                    "base_link",
                    [detected_object(position=(1.0, 0.0, 0.0), dims=(1.0, 1.0, 1.0))],
                ),
                T0,
            ),
            *tf_records,
        ]
        importer = RosbagImporter.open(write_bag(tmp_path / "bag.mcap", records))
        return importer.import_topic(labels=importer.label_registry())

    def test_a_repeated_identical_static_edge_is_written_once(self, tmp_path: Path) -> None:
        recording = self._import(
            tmp_path,
            [
                ("/tf_static", TF, tf_message(T0, "base_link", "lidar", (0.0, 0.0, 2.0)), T0),
                (
                    "/tf_static",
                    TF,
                    tf_message(T0 + 1, "base_link", "lidar", (0.0, 0.0, 2.0)),
                    T0 + 1,
                ),
            ],
        )

        assert len(recording.static_chunks("/tf/lidar")) == 1

    def test_a_static_edge_that_changes_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="twice with different values"):
            self._import(
                tmp_path,
                [
                    ("/tf_static", TF, tf_message(T0, "base_link", "lidar", (0.0, 0.0, 2.0)), T0),
                    ("/tf_static", TF, tf_message(T0, "base_link", "lidar", (0.0, 0.0, 3.0)), T0),
                ],
            )

    def test_a_child_on_both_topics_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="both /tf and /tf_static"):
            self._import(
                tmp_path,
                [
                    ("/tf_static", TF, tf_message(T0, "base_link", "lidar", (0.0, 0.0, 2.0)), T0),
                    ("/tf", TF, tf_message(T0, "base_link", "lidar", (0.0, 0.0, 2.0)), T0),
                ],
            )

    def test_a_child_with_two_parents_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="two parents"):
            self._import(
                tmp_path,
                [
                    ("/tf", TF, tf_message(T0, "map", "base_link", (0.0, 0.0, 0.0)), T0),
                    ("/tf", TF, tf_message(T0 + 1, "odom", "base_link", (0.0, 0.0, 0.0)), T0 + 1),
                ],
            )

    def test_a_bag_without_transforms_imports_its_objects(self, tmp_path: Path) -> None:
        recording = self._import(tmp_path, [])

        assert transform_edges(recording) == ()
        assert len(everything(recording)) == 1
