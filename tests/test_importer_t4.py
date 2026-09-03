"""Importing the vendored minimal T4 dataset, end to end."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from attrs import evolve

from t4perceval import (
    FRAME,
    TIMESTAMP,
    MatchResults,
    MetricValues,
    Predictions3D,
    TimeRange,
    Transform3D,
)
from t4perceval.descriptors import INSTANCE_ID, TRANSLATION
from t4perceval.evaluation import build_evaluation_store
from t4perceval.system import Pipeline
from t4perceval.transform import transform_edges
from t4perceval.system.preset import average_precision_sweep

pytest.importorskip("t4_devkit")

from t4perceval.importer.t4 import (
    ImportOptions,
    SceneSelection,
    T4Importer,
    T4Source,
)

if TYPE_CHECKING:
    from pathlib import Path

    from t4perceval.recording import Recording

EVERYTHING = TimeRange.everything()

#: Sample timestamps of the fixture scene, in nanoseconds.
FRAME_TIMES_NS = [1_704_067_200_000_000_000, 1_704_067_201_000_000_000, 1_704_067_202_000_000_000]


@pytest.fixture
def scene(t4_importer: T4Importer) -> Recording:
    return t4_importer.import_scene(labels=t4_importer.label_registry())


def rows(recording: Recording, path: str) -> int:
    return len(recording.range(path, timeline=FRAME, time_range=EVERYTHING))


class TestLabelRegistry:
    def test_is_built_from_the_dataset_categories(self, t4_importer: T4Importer) -> None:
        assert t4_importer.label_registry().names == (
            "car",
            "pedestrian",
            "bicycle",
            "traffic_cone",
        )

    def test_is_required_rather_than_derived(self, t4_importer: T4Importer) -> None:
        # Deriving a registry on each side of an evaluation is how two sources come to
        # disagree about what a class id means, so the caller has to say.
        with pytest.raises(TypeError, match="labels"):
            t4_importer.import_scene()  # type: ignore[call-arg]


class TestTimelines:
    def test_frames_follow_the_sample_chain(self, scene: Recording) -> None:
        assert scene.times("/ground_truth/objects", FRAME).tolist() == [0, 1, 2]

    def test_microseconds_become_nanoseconds(self, scene: Recording) -> None:
        assert scene.times("/ground_truth/objects", TIMESTAMP).tolist() == FRAME_TIMES_NS

    def test_a_selection_keeps_its_place_in_the_full_chain(
        self,
        t4_importer: T4Importer,
    ) -> None:
        # So two selections of one scene stay comparable instead of both starting at zero.
        scene = t4_importer.import_scene(
            labels=t4_importer.label_registry(),
            selection=SceneSelection(samples=slice(1, 3)),
        )

        assert scene.times("/ground_truth/objects", FRAME).tolist() == [1, 2]


class TestWholeScene:
    def test_the_scene_concatenates(self, scene: Recording) -> None:
        # The regression test for the constraint that shapes the importer: chunks with
        # different column sets cannot be concatenated, and `range` concatenates. A
        # per-frame decision about an optional column logs fine and breaks here.
        assert rows(scene, "/ground_truth/objects") == 4

    def test_the_last_frame_is_empty(self, scene: Recording) -> None:
        assert len(scene.latest_at("/ground_truth/objects", timeline=FRAME, at=2)) == 0

    def test_an_empty_frame_still_carries_every_column(self, scene: Recording) -> None:
        populated = scene.latest_at("/ground_truth/objects", timeline=FRAME, at=0)
        empty = scene.latest_at("/ground_truth/objects", timeline=FRAME, at=2)

        assert set(empty.descriptors) == set(populated.descriptors)

    def test_optional_columns_are_decided_scene_wide(self, scene: Recording) -> None:
        # The fixture has a finite velocity on one box and NaN on another in the same
        # frame, so "some row somewhere is estimable" is the whole scene's answer.
        view = scene.range("/ground_truth/objects", timeline=FRAME, time_range=EVERYTHING)
        names = {descriptor.component for descriptor in view.descriptors}

        assert {"velocity", "num_points", "visibility"} <= names


class TestCameras:
    def test_each_camera_gets_its_own_entity(self, t4_importer: T4Importer) -> None:
        scene = t4_importer.import_scene(
            labels=t4_importer.label_registry(),
            selection=SceneSelection(channels_2d=("CAM_FRONT", "CAM_BACK")),
        )

        assert rows(scene, "/ground_truth/CAM_FRONT/objects") == 4

    def test_a_camera_without_annotations_is_empty(self, t4_importer: T4Importer) -> None:
        # `get_box2ds` returns every 2D annotation of the sample whatever channel it is
        # asked for, so without filtering the rear camera reports the front camera's boxes.
        scene = t4_importer.import_scene(
            labels=t4_importer.label_registry(),
            selection=SceneSelection(channels_2d=("CAM_FRONT", "CAM_BACK")),
        )

        assert rows(scene, "/ground_truth/CAM_BACK/objects") == 0

    def test_a_camera_keeps_its_own_capture_time(self, t4_importer: T4Importer) -> None:
        scene = t4_importer.import_scene(
            labels=t4_importer.label_registry(),
            selection=SceneSelection(channels_2d=("CAM_FRONT",)),
        )
        camera = scene.times("/ground_truth/CAM_FRONT/objects", FRAME)

        # A camera and the lidar share a FRAME so they join, while TIMESTAMP stays truthful.
        assert camera.tolist() == [0, 1, 2]

    def test_fetching_3d_boxes_through_a_camera_is_rejected(
        self,
        t4_importer: T4Importer,
    ) -> None:
        with pytest.raises(ValueError, match="is a camera"):
            t4_importer.import_scene(
                labels=t4_importer.label_registry(),
                selection=SceneSelection(channel_3d="CAM_FRONT"),
            )


class TestCoordinateFrames:
    @pytest.mark.parametrize(
        ("coords", "expected"),
        [("map", "map"), ("base_link", "base_link"), ("sensor", "LIDAR_TOP")],
    )
    def test_the_frame_is_recorded_on_the_chunk(
        self,
        t4_dataset_root: Path,
        coords: str,
        expected: str,
    ) -> None:
        importer = T4Importer.open(t4_dataset_root, options=ImportOptions(coords=coords))  # type: ignore[arg-type]
        scene = importer.import_scene(labels=importer.label_registry())

        assert scene.metadata.frame_id == expected

    def test_sensor_coordinates_move_the_boxes(self, t4_dataset_root: Path) -> None:
        source = T4Source(t4_dataset_root)
        frames = source.frames(source.resolve_scene(None))
        token = frames[0].data["LIDAR_TOP"]

        in_map = source.boxes3d(token, coords="map")[0].position
        in_sensor = source.boxes3d(token, coords="sensor")[0].position

        assert float(in_map[2] - in_sensor[2]) == pytest.approx(2.0)

    def test_a_future_stays_in_its_box_frame(self, t4_dataset_root: Path) -> None:
        # `Box3D.translate`/`rotate` propagate into `Box3D.future`, so the converter needs
        # no re-projection. If a devkit upgrade stopped doing that, trajectories and boxes
        # would end up in different frames with nothing to say so.
        source = T4Source(t4_dataset_root)
        frames = source.frames(source.resolve_scene(None))
        token = frames[0].data["LIDAR_TOP"]

        box = next(
            candidate
            for candidate in source.boxes3d(token, coords="sensor", future_seconds=2.0)
            if candidate.future is not None
        )

        assert float(box.future.waypoints[0, 0, 2]) == pytest.approx(float(box.position[2]))


class TestPredictions:
    def test_trajectories_are_padded_and_masked(self, t4_dataset_root: Path) -> None:
        importer = T4Importer.open(
            t4_dataset_root,
            options=ImportOptions(kind_3d="predictions", future_seconds=2.0),
        )
        scene = importer.import_scene(labels=importer.label_registry())
        view = scene.range("/ground_truth/objects", timeline=FRAME, time_range=EVERYTHING)
        predictions = view.materialize(Predictions3D)

        # Exactly one box in the fixture has a future; the rest must still be well formed.
        assert predictions.mode_valid.values.sum() == 1
        assert np.isfinite(predictions.waypoints.values).all()
        assert (predictions.time_offset.values > 0).all()

    def test_a_real_future_offset_is_relative_to_its_frame(
        self,
        t4_dataset_root: Path,
    ) -> None:
        importer = T4Importer.open(
            t4_dataset_root,
            options=ImportOptions(kind_3d="predictions", future_seconds=2.0),
        )
        scene = importer.import_scene(labels=importer.label_registry())
        view = scene.range("/ground_truth/objects", timeline=FRAME, time_range=EVERYTHING)
        predictions = view.materialize(Predictions3D)

        valid = predictions.mode_valid.values.ravel()
        offsets = predictions.time_offset.values[valid]

        # The fixture's futures are one sample ahead, and samples are one second apart.
        assert offsets.tolist() == [[1_000_000_000]]


class TestInstances:
    def test_an_identity_is_stable_across_frames(self, scene: Recording) -> None:
        first = scene.latest_at("/ground_truth/objects", timeline=FRAME, at=0)
        second = scene.latest_at("/ground_truth/objects", timeline=FRAME, at=1)

        shared = set(first.component(INSTANCE_ID).values.tolist()) & set(  # type: ignore[union-attr]
            second.component(INSTANCE_ID).values.tolist(),  # type: ignore[union-attr]
        )

        # The car is annotated in both frames; tracking metrics depend on it staying one
        # identity.
        assert len(shared) == 1

    def test_identities_are_namespaced(self, scene: Recording) -> None:
        assert all(uuid.startswith("gt/") for uuid in scene.instances.decode([0]))


class TestProvenance:
    def test_the_source_is_recorded(self, scene: Recording) -> None:
        source = scene.metadata.sources[0]

        assert source.kind == "t4"
        assert source.entity_path == "/ground_truth/objects"
        assert dict(source.extra)["channel_3d"] == "LIDAR_TOP"

    def test_the_registry_is_fingerprinted(self, scene: Recording) -> None:
        assert scene.metadata.labels_fingerprint == scene.labels.fingerprint()


class TestEvaluation:
    def test_imported_data_runs_a_pipeline(self, scene: Recording) -> None:
        setup = build_evaluation_store(
            scene,
            scene,
            query_path="/ground_truth/objects",
            query_target="/estimation/objects",
        )
        systems = average_precision_sweep(
            "/estimation/objects",
            "/ground_truth/objects",
            thresholds=[1.0],
        )
        Pipeline(systems).run(setup.context(), EVERYTHING)

        matches = setup.store.range(
            "/matching/center_distance/0",
            timeline=FRAME,
            time_range=EVERYTHING,
        ).materialize(MatchResults)

        assert (matches.num_tp, matches.num_fp, matches.num_fn) == (4, 0, 0)

    def test_a_self_match_scores_perfectly(self, scene: Recording) -> None:
        setup = build_evaluation_store(
            scene,
            scene,
            query_path="/ground_truth/objects",
            query_target="/estimation/objects",
        )
        systems = average_precision_sweep(
            "/estimation/objects",
            "/ground_truth/objects",
            thresholds=[1.0],
        )
        Pipeline(systems).run(setup.context(), EVERYTHING)

        metrics = setup.store.range(
            "/metrics/map",
            timeline=FRAME,
            time_range=EVERYTHING,
        ).materialize(MetricValues)

        assert metrics.aggregate == pytest.approx(1.0)


class TestTransforms:
    def test_the_frame_tree_is_recorded(self, scene: Recording) -> None:
        # Read from the data, not from the entity paths: each chunk states its parent
        # through `frame_id` and its child through the `child_frame_id` column.
        assert {edge.frames for edge in transform_edges(scene)} == {
            ("map", "base_link"),
            ("base_link", "LIDAR_TOP"),
            ("base_link", "CAM_FRONT"),
            ("base_link", "CAM_BACK"),
        }

    def test_an_extrinsic_is_static_and_an_ego_pose_is_not(self, scene: Recording) -> None:
        by_child = {edge.child: edge for edge in transform_edges(scene)}

        assert by_child["base_link"].is_static is False
        assert by_child["LIDAR_TOP"].is_static is True

    def test_ego_poses_are_recorded_per_frame(self, scene: Recording) -> None:
        # A `Transform3D` is one edge, so a scene of poses is read as a column rather than
        # materialized into the archetype.
        view = scene.range("/tf/base_link", timeline=FRAME, time_range=EVERYTHING)
        translation = view.component(TRANSLATION).values

        # The fixture's ego travels 10 m along x per second.
        assert translation[:, 0].tolist() == [0.0, 10.0, 20.0]

    def test_microseconds_become_nanoseconds(self, scene: Recording) -> None:
        assert scene.times("/tf/base_link", TIMESTAMP).tolist() == FRAME_TIMES_NS

    def test_an_extrinsic_needs_no_time_at_all(self, scene: Recording) -> None:
        # It is not on a timeline, so there is no sample time to invent and no window a
        # range query could start after.
        chunk = scene.static_chunks("/tf/LIDAR_TOP")[0]

        assert scene.static_frame_id("/tf/LIDAR_TOP") == "base_link"
        assert scene.times("/tf/LIDAR_TOP", FRAME).tolist() == []
        # One row in, one value out: the archetype narrows the stored columns back.
        assert Transform3D.from_chunk(chunk).translation.value.tolist() == [0.0, 0.0, 2.0]

    def test_extrinsics_match_the_calibration(self, scene: Recording) -> None:
        offsets = {
            channel: Transform3D.from_chunk(
                scene.static_chunks(f"/tf/{channel}")[0],
            ).translation.value.tolist()
            for channel in ("LIDAR_TOP", "CAM_FRONT", "CAM_BACK")
        }

        assert offsets == {
            "LIDAR_TOP": [0.0, 0.0, 2.0],
            "CAM_FRONT": [1.5, 0.0, 1.8],
            "CAM_BACK": [-1.5, 0.0, 1.8],
        }

    def test_the_rear_camera_keeps_its_half_turn(self, scene: Recording) -> None:
        # The dataset stores wxyz and this package stores xyzw. Taken verbatim the rear
        # camera would read as unrotated -- a plausible answer, and completely wrong.

        rotation = Transform3D.from_chunk(scene.static_chunks("/tf/CAM_BACK")[0]).rotation

        yaw = rotation.as_rotation().as_euler("xyz", degrees=True)[2]

        assert abs(yaw) == pytest.approx(180.0)

    def test_an_ego_pose_is_queryable_on_either_timeline(self, scene: Recording) -> None:
        for timeline, at in ((FRAME, 1), (TIMESTAMP, FRAME_TIMES_NS[1])):
            view = scene.latest_at("/tf/base_link", timeline=timeline, at=at)
            pose = view.materialize(Transform3D)
            assert pose.translation.value[0] == pytest.approx(10.0)

    def test_they_can_be_switched_off(self, t4_dataset_root: Path) -> None:
        importer = T4Importer.open(t4_dataset_root, options=ImportOptions(transforms=False))
        scene = importer.import_scene(labels=importer.label_registry())

        assert transform_edges(scene) == ()


class TestExtrinsics:
    """Where a sensor's fixed pose comes from."""

    def test_one_entry_per_sensor(self, t4_dataset_root: Path) -> None:
        source = T4Source(t4_dataset_root)
        devkit = source.devkit

        # `calibrated_sensor` holds one row per sensor, so that is what is read -- rather
        # than gathering the same value again from every frame of every channel.
        assert len(source.extrinsics()) == len(devkit.sensor)
        assert set(source.extrinsics()) == {sensor.channel for sensor in devkit.sensor}

    def test_it_does_not_depend_on_the_frames_imported(
        self,
        t4_importer: T4Importer,
    ) -> None:
        one_frame = t4_importer.import_scene(
            labels=t4_importer.label_registry(),
            selection=SceneSelection(samples=[0]),
        )
        whole_scene = t4_importer.import_scene(labels=t4_importer.label_registry())

        def fixed(scene: Recording) -> set[tuple[str, str]]:
            return {edge.frames for edge in transform_edges(scene) if edge.is_static}

        assert fixed(one_frame) == fixed(whole_scene)

    def test_disagreeing_calibrations_are_rejected(self, t4_dataset_root: Path) -> None:
        # Two calibrations for one channel is ambiguous. Settling on whichever came first
        # would put a silently wrong extrinsic into the frame tree.
        source = T4Source(t4_dataset_root)
        devkit = source.devkit
        original = devkit.calibrated_sensor[0]
        moved = evolve(original, translation=np.asarray(original.translation) + 1.0)
        devkit.calibrated_sensor = [*devkit.calibrated_sensor, moved]

        with pytest.raises(ValueError, match="more than one calibration"):
            source.extrinsics()

    def test_an_identical_duplicate_is_accepted(self, t4_dataset_root: Path) -> None:
        source = T4Source(t4_dataset_root)
        devkit = source.devkit
        devkit.calibrated_sensor = [
            *devkit.calibrated_sensor,
            evolve(devkit.calibrated_sensor[0]),
        ]

        assert len(source.extrinsics()) == len(devkit.sensor)

    def test_a_calibration_for_an_unknown_sensor_is_rejected(
        self,
        t4_dataset_root: Path,
    ) -> None:
        source = T4Source(t4_dataset_root)
        devkit = source.devkit
        orphan = evolve(devkit.calibrated_sensor[0], sensor_token="0" * 32)
        devkit.calibrated_sensor = [*devkit.calibrated_sensor, orphan]

        with pytest.raises(KeyError, match="unknown sensor"):
            source.extrinsics()
