"""Materializing an entity in another frame, and composing recordings across frames."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_detections, make_ego_scene, make_predictions, tf_edge

from t4perceval import (
    FRAME,
    TIMESTAMP,
    Chunk,
    EntityPath,
    LabelRegistry,
    MatchResults,
    MetricValues,
    Recording,
    Store,
    TimeColumn,
    TimePoint,
    TimeRange,
)
from t4perceval.component import BatchMask, BatchNumPoints, BatchPosition3D
from t4perceval.descriptors import (
    MASK,
    MODE_CONFIDENCE,
    NUM_POINTS,
    POSITION,
    TIME_OFFSET,
    VELOCITY,
    WAYPOINTS,
)
from t4perceval.evaluation import SourceSpec, build_evaluation_store_from
from t4perceval.system import (
    ApplyMaskSystem,
    AveragePrecisionSystem,
    CenterDistanceMatchingSystem,
    FilterByDistanceSystem,
    Passthrough,
    Pipeline,
    SystemContext,
    TransformEntitySystem,
)
from t4perceval.transform import LookupPolicy, TransformResolver

EST = "/estimation/objects"
GT = "/ground_truth/objects"
EVERYTHING = TimeRange.everything()


def at(frame: int) -> TimePoint:
    """The time points `make_ego_scene` stamps: frame ``i`` at ``(i + 1) * 1_000`` ns."""
    return TimePoint.at(frame=frame, timestamp_ns=(frame + 1) * 1_000)


def log_objects(
    store: Store, path: str, frame: int, positions, *, frame_id="base_link", confidences=None
):  # noqa: ANN001
    count = len(positions)
    store.log(
        path,
        make_detections(
            positions,
            confidences=confidences,
            velocity=np.tile([[1.0, 0.0, 0.0]], (count, 1)) if count else np.zeros((0, 3)),
        ),
        at=at(frame),
        frame_id=frame_id,
    )


@pytest.fixture
def ego_store() -> Store:
    """Ego at x = 0, 10, 20 over frames 0-2; one object 1 m ahead, two at frame 1, none at 2."""
    store = make_ego_scene(Store())
    log_objects(store, EST, 0, [[1.0, 0.0, 0.0]])
    log_objects(store, EST, 1, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    log_objects(store, EST, 2, [])
    return store


def run(store: Store, *systems, at=EVERYTHING, labels=None):  # noqa: ANN001
    return Pipeline(list(systems)).run(SystemContext(store, FRAME, labels=labels), at)


class TestWiring:
    def test_defaults_to_source_in_frame(self) -> None:
        moved = TransformEntitySystem.of(EST, target_frame="map")
        assert str(moved.target) == "/estimation/objects/in/map"
        assert moved.sources == (EntityPath.parse(EST),)
        assert moved.target_frame == "map"
        assert moved.resolver is None

    def test_the_target_can_be_named(self) -> None:
        moved = TransformEntitySystem.of(EST, target_frame="map", target="/estimation/objects_map")
        assert str(moved.target) == "/estimation/objects_map"

    def test_a_frame_with_a_slash_needs_an_explicit_target(self) -> None:
        with pytest.raises(ValueError, match="contains '/', which an entity path segment cannot"):
            TransformEntitySystem.of(EST, target_frame="sensor/lidar")
        moved = TransformEntitySystem.of(EST, target_frame="sensor/lidar", target="/x")
        assert moved.target_frame == "sensor/lidar"

    def test_needs_exactly_one_source_and_a_frame(self) -> None:
        with pytest.raises(ValueError, match="needs exactly one source, got 2"):
            TransformEntitySystem((EST, GT), "/x", target_frame="map")
        with pytest.raises(ValueError, match="non-empty target_frame"):
            TransformEntitySystem.of(EST, target_frame="")

    def test_requires_nothing_and_carries_the_source_minus_masks(self) -> None:
        assert TransformEntitySystem.REQUIRES == ()
        assert TransformEntitySystem.PROVIDES == Passthrough(0, drops=(MASK,))


class TestPerTimeTransform:
    def test_each_frame_uses_its_own_ego_pose(self, ego_store: Store) -> None:
        moved = TransformEntitySystem.of(EST, target_frame="map")
        run(ego_store, moved)
        view = ego_store.range(moved.target, timeline=FRAME, time_range=EVERYTHING)
        assert view.component(POSITION).values[:, 0].tolist() == [1.0, 11.0, 12.0]  # type: ignore[union-attr]
        assert view.times(FRAME).tolist() == [0, 1, 1]

    def test_partitions_and_zero_row_frames_are_preserved(self, ego_store: Store) -> None:
        moved = TransformEntitySystem.of(EST, target_frame="map")
        run(ego_store, moved)
        chunk = ego_store.range(moved.target, timeline=FRAME, time_range=EVERYTHING).to_chunk()
        assert chunk.num_partitions == 3
        assert chunk.partition_sizes().tolist() == [1, 2, 0]
        assert chunk.index(FRAME).times.tolist() == [0, 1, 2]  # type: ignore[union-attr]
        assert chunk.index(TIMESTAMP).times.tolist() == [1_000, 2_000, 3_000]  # type: ignore[union-attr]

    def test_writes_the_target_frame_and_leaves_the_source_alone(self, ego_store: Store) -> None:
        moved = TransformEntitySystem.of(EST, target_frame="map")
        before = ego_store.range(EST, timeline=FRAME, time_range=EVERYTHING).to_chunk()
        run(ego_store, moved)
        assert all(c.frame_id == "map" for c in ego_store.chunks(moved.target))
        assert ego_store.range(EST, timeline=FRAME, time_range=EVERYTHING).to_chunk() == before
        assert len(ego_store.chunks(EST)) == 3

    def test_a_single_time_query(self, ego_store: Store) -> None:
        moved = TransformEntitySystem.of(EST, target_frame="map")
        run(ego_store, moved, at=1)
        view = ego_store.range(moved.target, timeline=FRAME, time_range=EVERYTHING)
        assert view.component(POSITION).values[:, 0].tolist() == [11.0, 12.0]  # type: ignore[union-attr]

    def test_velocity_follows_the_ego_heading(self) -> None:
        store = make_ego_scene(Store(), xs=(10.0,), yaws=(90.0,))
        log_objects(store, EST, 0, [[1.0, 0.0, 0.0]])
        moved = TransformEntitySystem.of(EST, target_frame="map")
        run(store, moved)
        view = store.range(moved.target, timeline=FRAME, time_range=EVERYTHING)
        np.testing.assert_allclose(view.component(POSITION).values, [[10.0, 1.0, 0.0]], atol=1e-12)  # type: ignore[union-attr]
        np.testing.assert_allclose(view.component(VELOCITY).values, [[0.0, 1.0, 0.0]], atol=1e-12)  # type: ignore[union-attr]

    def test_predictions_move_with_the_ego(self) -> None:
        store = make_ego_scene(Store())
        store.log(EST, make_predictions([[0.0, 0.0, 0.0]], [1]), at=at(1), frame_id="base_link")
        moved = TransformEntitySystem.of(EST, target_frame="map")
        run(store, moved)
        view = store.range(moved.target, timeline=FRAME, time_range=EVERYTHING)
        source = store.range(EST, timeline=FRAME, time_range=EVERYTHING)
        waypoints = view.component(WAYPOINTS).values  # type: ignore[union-attr]
        np.testing.assert_allclose(
            waypoints, np.broadcast_to([10.0, 0.0, 0.0], waypoints.shape), atol=1e-12
        )
        for descriptor in (TIME_OFFSET, MODE_CONFIDENCE):
            assert np.array_equal(
                view.component(descriptor).values, source.component(descriptor).values
            )  # type: ignore[union-attr]

    def test_the_lookup_policy_travels_through_the_resolver(self, ego_store: Store) -> None:
        log_objects(ego_store, EST, 3, [[1.0, 0.0, 0.0]])  # no ego pose at frame 3
        latest = TransformEntitySystem.of(EST, target_frame="map")
        run(ego_store, latest)
        view = ego_store.range(latest.target, timeline=FRAME, time_range=TimeRange.single(3))
        assert view.component(POSITION).values[:, 0].tolist() == [21.0]  # type: ignore[union-attr]

        exact = TransformEntitySystem.of(
            EST,
            target_frame="map",
            target="/exact",
            resolver=TransformResolver.of(ego_store, policy=LookupPolicy.EXACT),
        )
        with pytest.raises(
            ValueError,
            match="cannot bring /estimation/objects from 'base_link' into 'map' at frame=3: .*has no sample at frame=3",
        ):
            run(ego_store, exact)


class TestFramesAndResolvers:
    def test_same_frame_is_a_re_path_without_any_transform_data(self) -> None:
        store = Store()
        log_objects(store, EST, 0, [[1.0, 2.0, 3.0]])
        moved = TransformEntitySystem.of(EST, target_frame="base_link")
        run(store, moved)
        assert str(moved.target) == "/estimation/objects/in/base_link"
        view = store.range(moved.target, timeline=FRAME, time_range=EVERYTHING)
        assert view.frame_id == "base_link"
        assert np.array_equal(view.component(POSITION).values, [[1.0, 2.0, 3.0]])  # type: ignore[union-attr]

    def test_a_source_without_a_frame_is_rejected(self) -> None:
        store = Store()
        log_objects(store, EST, 0, [[1.0, 0.0, 0.0]], frame_id=None)
        with pytest.raises(ValueError, match="/estimation/objects states no coordinate frame"):
            run(store, TransformEntitySystem.of(EST, target_frame="map"))

    def test_an_unknown_target_frame_names_the_entity(self, ego_store: Store) -> None:
        with pytest.raises(
            ValueError,
            match="cannot bring /estimation/objects from 'base_link' into 'radar'.*Unknown coordinate frame 'radar'",
        ):
            run(ego_store, TransformEntitySystem.of(EST, target_frame="radar"))

    def test_a_disconnected_frame_is_reported(self, ego_store: Store) -> None:
        ego_store.log_static("/tf/cam", tf_edge("cam", [0.0, 0.0, 1.0]), frame_id="rig")
        with pytest.raises(ValueError, match="No recorded transform connects"):
            run(ego_store, TransformEntitySystem.of(EST, target_frame="cam"))

    def test_an_explicit_resolver_may_come_from_another_store(self, ego_store: Store) -> None:
        objects_only = Store()
        for chunk in ego_store.chunks(EST):
            objects_only.send_chunk(chunk)

        with pytest.raises(ValueError, match="Unknown coordinate frame"):
            run(objects_only, TransformEntitySystem.of(EST, target_frame="map"))

        moved = TransformEntitySystem.of(
            EST, target_frame="map", resolver=TransformResolver.of(ego_store)
        )
        run(objects_only, moved)
        view = objects_only.range(moved.target, timeline=FRAME, time_range=EVERYTHING)
        assert view.component(POSITION).values[:, 0].tolist() == [1.0, 11.0, 12.0]  # type: ignore[union-attr]

    def test_the_resolver_may_walk_another_timeline_than_the_pipeline(
        self, ego_store: Store
    ) -> None:
        moved = TransformEntitySystem.of(
            EST, target_frame="map", resolver=TransformResolver.of(ego_store, timeline=TIMESTAMP)
        )
        run(ego_store, moved)  # the pipeline runs on FRAME
        view = ego_store.range(moved.target, timeline=FRAME, time_range=EVERYTHING)
        assert view.component(POSITION).values[:, 0].tolist() == [1.0, 11.0, 12.0]  # type: ignore[union-attr]

    def test_a_source_missing_the_resolver_timeline_is_reported(self) -> None:
        store = make_ego_scene(Store())
        store.log(
            EST, make_detections([[1.0, 0.0, 0.0]]), at=TimePoint.at(frame=0), frame_id="base_link"
        )
        moved = TransformEntitySystem.of(
            EST, target_frame="map", resolver=TransformResolver.of(store, timeline=TIMESTAMP)
        )
        with pytest.raises(ValueError, match="has no 'timestamp_ns' index"):
            run(store, moved)

    def test_static_source_columns_are_not_carried(self, ego_store: Store) -> None:
        ego_store.log_static_components(EST, {NUM_POINTS: BatchNumPoints([7])})
        moved = TransformEntitySystem.of(EST, target_frame="map")
        run(ego_store, moved)
        assert ego_store.static(moved.target) == {}
        assert ego_store.static(EST) != {}

    def test_masks_are_dropped(self) -> None:
        store = make_ego_scene(Store(), xs=(0.0,))
        store.send_chunk(
            Chunk.from_columns(
                EST,
                {POSITION: BatchPosition3D([[1.0, 0.0, 0.0]]), MASK: BatchMask([True])},
                indexes=(TimeColumn.of(FRAME, [0]),),
                frame_id="base_link",
            ),
        )
        moved = TransformEntitySystem.of(EST, target_frame="map")
        run(store, moved)
        view = store.range(moved.target, timeline=FRAME, time_range=EVERYTHING)
        assert view.has(POSITION) and not view.has(MASK)


class TestEmptyRange:
    def test_an_empty_range_produces_nothing(self, ego_store: Store) -> None:
        moved = TransformEntitySystem.of(EST, target_frame="map")
        assert tuple(moved(SystemContext(ego_store, FRAME), 99)) == ()
        assert run(ego_store, moved, at=99) == ()
        assert moved.target not in ego_store.entity_paths()


class TestPipelineIntegration:
    def test_downstream_consumers_validate(self) -> None:
        moved = TransformEntitySystem.of(EST, target_frame="map")
        matcher = CenterDistanceMatchingSystem.between(moved.target, GT)
        assert len(Pipeline([moved, matcher])) == 2
        with pytest.raises(ValueError, match="before a later system writes it"):
            Pipeline([matcher, moved])

    def test_transform_filter_mask_match_metric_in_one_pipeline(
        self, ego_store: Store, labels: LabelRegistry
    ) -> None:
        log_objects(ego_store, GT, 0, [[1.0, 0.0, 0.0]], frame_id="map")
        log_objects(ego_store, GT, 1, [[11.0, 0.0, 0.0], [12.0, 0.0, 0.0]], frame_id="map")

        moved = TransformEntitySystem.of(EST, target_frame="map")
        near = FilterByDistanceSystem.on(moved.target, max_distance=100.0)
        kept = ApplyMaskSystem.of(moved.target, near.target)
        matcher = CenterDistanceMatchingSystem.between(kept.target, GT, threshold=0.5)
        metric = AveragePrecisionSystem.on(matcher.target, kept.target, GT)

        produced = run(ego_store, moved, near, kept, matcher, metric, labels=labels)

        assert len(produced) == 5
        matches = ego_store.range(
            matcher.target, timeline=FRAME, time_range=EVERYTHING
        ).materialize(MatchResults)
        assert (matches.num_tp, matches.num_fp, matches.num_fn) == (3, 0, 0)
        ap = ego_store.range(metric.target, timeline=FRAME, time_range=EVERYTHING).materialize(
            MetricValues
        )
        assert ap.of_class(labels.class_id("car")) == pytest.approx(1.0)


class TestCrossRecordingComposition:
    """Recording A is in base_link with its ego poses; Recording B is in map."""

    @pytest.fixture
    def recording_a(self, labels: LabelRegistry) -> Recording:
        store = make_ego_scene(Store(), xs=(10.0, 20.0))
        log_objects(store, EST, 0, [[0.1, 0.0, 0.0], [5.0, 0.0, 0.0]], confidences=[0.9, 0.8])
        log_objects(store, EST, 1, [[1.1, 0.0, 0.0]], confidences=[0.7])
        return Recording.of(store, labels=labels)

    @pytest.fixture
    def recording_a_in_map(self, labels: LabelRegistry) -> Recording:
        store = Store()
        log_objects(
            store,
            EST,
            0,
            [[10.1, 0.0, 0.0], [15.0, 0.0, 0.0]],
            frame_id="map",
            confidences=[0.9, 0.8],
        )
        log_objects(store, EST, 1, [[21.1, 0.0, 0.0]], frame_id="map", confidences=[0.7])
        return Recording.of(store, labels=labels)

    @pytest.fixture
    def recording_b(self, labels: LabelRegistry) -> Recording:
        store = Store()
        log_objects(
            store,
            GT,
            0,
            [[10.0, 0.0, 0.0], [15.0, 0.0, 0.0]],
            frame_id="map",
            confidences=[1.0, 1.0],
        )
        log_objects(store, GT, 1, [[21.0, 0.0, 0.0]], frame_id="map", confidences=[1.0])
        return Recording.of(store, labels=labels)

    @staticmethod
    def evaluate(setup, estimation: str, systems, labels: LabelRegistry):  # noqa: ANN001
        matcher = CenterDistanceMatchingSystem.between(estimation, GT, threshold=1.0)
        metric = AveragePrecisionSystem.on(matcher.target, estimation, GT)
        Pipeline([*systems, matcher, metric]).run(setup.context(), EVERYTHING)
        matches = setup.store.range(
            matcher.target, timeline=FRAME, time_range=EVERYTHING
        ).materialize(MatchResults)
        ap = setup.store.range(metric.target, timeline=FRAME, time_range=EVERYTHING).materialize(
            MetricValues
        )
        return matches, ap

    def test_matching_across_frames_is_refused_without_a_transform(
        self, recording_a, recording_b, labels
    ) -> None:  # noqa: ANN001
        setup = build_evaluation_store_from(
            [SourceSpec.of(recording_b, GT), SourceSpec.of(recording_a, EST)],
            require_same_frame_id=False,
        )
        with pytest.raises(ValueError, match="across coordinate frames"):
            self.evaluate(setup, EST, [], labels)

    def test_a_transformed_estimation_matches_ground_truth_in_map(
        self, recording_a, recording_b, labels
    ) -> None:  # noqa: ANN001
        setup = build_evaluation_store_from(
            [SourceSpec.of(recording_b, GT), SourceSpec.of(recording_a, EST)],
            require_same_frame_id=False,
        )
        moved = TransformEntitySystem.of(
            EST, target_frame="map", resolver=TransformResolver.of(recording_a)
        )
        matches, ap = self.evaluate(setup, str(moved.target), [moved], labels)

        assert (matches.num_tp, matches.num_fp, matches.num_fn) == (3, 0, 0)
        assert ap.of_class(labels.class_id("car")) == pytest.approx(1.0)
        assert all(c.frame_id == "map" for c in setup.store.chunks(moved.target))

    def test_equals_the_run_authored_in_map(
        self, recording_a, recording_a_in_map, recording_b, labels
    ) -> None:  # noqa: ANN001
        transformed = build_evaluation_store_from(
            [SourceSpec.of(recording_b, GT), SourceSpec.of(recording_a, EST)],
            require_same_frame_id=False,
        )
        moved = TransformEntitySystem.of(
            EST, target_frame="map", resolver=TransformResolver.of(recording_a)
        )
        matches, ap = self.evaluate(transformed, str(moved.target), [moved], labels)

        reference = build_evaluation_store_from(
            [SourceSpec.of(recording_b, GT), SourceSpec.of(recording_a_in_map, EST)]
        )
        expected_matches, expected_ap = self.evaluate(reference, EST, [], labels)

        for column in ("est_index", "gt_index", "match_status", "threshold"):
            assert np.array_equal(
                getattr(matches, column).values, getattr(expected_matches, column).values
            )
        np.testing.assert_allclose(
            matches.matching_score.values, expected_matches.matching_score.values, atol=1e-12
        )
        assert ap.of_class(labels.class_id("car")) == expected_ap.of_class(labels.class_id("car"))
        for column in ("class_id", "threshold", "value", "support"):
            np.testing.assert_array_equal(
                getattr(ap, column).values, getattr(expected_ap, column).values
            )

    def test_the_tf_may_travel_into_the_evaluation_store_instead(
        self, recording_a, recording_b, labels
    ) -> None:  # noqa: ANN001
        setup = build_evaluation_store_from(
            [
                SourceSpec.of(recording_b, GT),
                SourceSpec.of(recording_a, EST),
                SourceSpec.of(recording_a, "/tf/base_link"),
            ],
            require_same_frame_id=False,
        )
        moved = TransformEntitySystem.of(EST, target_frame="map")  # resolver built from the store
        matches, ap = self.evaluate(setup, str(moved.target), [moved], labels)
        assert (matches.num_tp, matches.num_fp, matches.num_fn) == (3, 0, 0)
        assert ap.of_class(labels.class_id("car")) == pytest.approx(1.0)

    def test_the_whole_chain_is_one_pipeline(self, recording_a, recording_b, labels) -> None:  # noqa: ANN001
        setup = build_evaluation_store_from(
            [SourceSpec.of(recording_b, GT), SourceSpec.of(recording_a, EST)],
            require_same_frame_id=False,
        )
        moved = TransformEntitySystem.of(
            EST, target_frame="map", resolver=TransformResolver.of(recording_a)
        )
        near = FilterByDistanceSystem.on(moved.target, max_distance=100.0)
        kept = ApplyMaskSystem.of(moved.target, near.target)
        matcher = CenterDistanceMatchingSystem.between(kept.target, GT, threshold=1.0)
        metric = AveragePrecisionSystem.on(matcher.target, kept.target, GT)

        produced = Pipeline([moved, near, kept, matcher, metric]).run(setup.context(), EVERYTHING)

        assert len(produced) == 5
        assert {str(c.entity_path) for c in produced} == {
            str(moved.target),
            str(near.target),
            str(kept.target),
            str(matcher.target),
            str(metric.target),
        }
