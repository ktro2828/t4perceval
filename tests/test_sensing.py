"""Sensing evaluation: points inside ground-truth boxes, and points left in empty areas."""

from __future__ import annotations

import numpy as np
import pytest
import pyarrow as pa

from t4perceval import (
    FRAME,
    Detections3D,
    LabelRegistry,
    MetricValues,
    PointCloud,
    SensingResults,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.component import SensingStatus, VisibilityLevel
from t4perceval.descriptors import CLASS_ID, MASK, NEAREST_POINT, NUM_POINTS, POINT, SENSING_STATUS
from t4perceval.io import chunk_from_table, chunk_to_table
from t4perceval.system import (
    ApplyMaskSystem,
    FilterByNumPointsSystem,
    MaskSystem,
    NonDetectionArea,
    Pipeline,
    SensingDetectionRateSystem,
    SensingDetectionSystem,
    SensingNonDetectionSystem,
    System,
    SystemContext,
)

GT = "/ground_truth/objects"
CLOUD = "/sensor/lidar"
EVERYTHING = TimeRange.everything()
IDENTITY = [0.0, 0.0, 0.0, 1.0]
#: width 2, length 4, height 2: a box at x spans [x - 2, x + 2] along x.
SIZE = [2.0, 4.0, 2.0]

#: The non-detection area of the original package's ``sensing_lsim.py``: the lane ahead.
LANE_VERTICES = [
    (10.0, 1.5, 0.0),
    (10.0, -1.5, 0.0),
    (0.0, -1.5, 0.0),
    (0.0, 1.5, 0.0),
    (10.0, 1.5, 4.0),
    (10.0, -1.5, 4.0),
    (0.0, -1.5, 4.0),
    (0.0, 1.5, 4.0),
]


@pytest.fixture
def labels() -> LabelRegistry:
    return LabelRegistry.from_names(["car", "truck", "pedestrian"])


def log_boxes(
    store: Store,
    xs: list[float],
    *,
    frame: int = 0,
    classes: list[str] | None = None,
    visibility: list[int] | None = None,
    labels: LabelRegistry | None = None,
    frame_id: str | None = "base_link",
) -> None:
    """Log one box per ``x``, centred on the x axis, facing forward."""
    class_ids = labels.encode(classes) if classes and labels else [0] * len(xs)
    store.log(
        GT,
        Detections3D(
            position=[[x, 0.0, 0.0] for x in xs],
            quaternion=[IDENTITY] * len(xs),
            size=[SIZE] * len(xs),
            class_id=class_ids,
            confidence=[1.0] * len(xs),
            visibility=visibility,
        ),
        at=TimePoint.at(frame=frame),
        frame_id=frame_id,
    )


def log_cloud(
    store: Store, points: list[list[float]], *, frame: int = 0, frame_id: str | None = "base_link"
) -> None:
    store.log(CLOUD, PointCloud(point=points), at=TimePoint.at(frame=frame), frame_id=frame_id)


def run(store: Store, *systems: object, labels: LabelRegistry | None = None) -> None:
    Pipeline(list(systems)).run(SystemContext(store, FRAME, labels=labels), EVERYTHING)  # type: ignore[arg-type]


def results_of(store: Store, system: SensingDetectionSystem) -> SensingResults:
    return store.range(system.target, timeline=FRAME, time_range=EVERYTHING).materialize(
        SensingResults
    )


def mask_of(store: Store, system: SensingNonDetectionSystem) -> list[bool]:
    view = store.range(system.target, timeline=FRAME, time_range=EVERYTHING)
    return view.component(MASK).values.tolist()  # type: ignore[union-attr]


class TestSensingResults:
    def test_holds_a_verdict_per_box(self) -> None:
        results = SensingResults(
            num_points=[3, 0, 1],
            sensing_status=[SensingStatus.DETECTED, SensingStatus.FAILED, SensingStatus.WARNING],
            nearest_point=[[1.0, 0.0, 0.0], [np.nan] * 3, [2.0, 0.0, 0.0]],
        )

        assert len(results) == 3
        assert (results.num_detected, results.num_failed, results.num_warned) == (1, 1, 1)
        assert results.count(SensingStatus.WARNING) == 1
        assert results.is_detected.tolist() == [True, False, False]
        assert SensingResults.required_descriptors() == (NUM_POINTS, SENSING_STATUS, NEAREST_POINT)
        assert SensingResults.optional_descriptors() == ()

    def test_empty_has_no_rows(self) -> None:
        assert len(SensingResults.empty()) == 0

    def test_rejects_an_unknown_status(self) -> None:
        with pytest.raises(ValueError, match="unknown values: \\[7\\]"):
            SensingResults(num_points=[1], sensing_status=[7], nearest_point=[[0.0, 0.0, 0.0]])

    def test_rejects_a_negative_count(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SensingResults(num_points=[-1], sensing_status=[0], nearest_point=[[np.nan] * 3])

    @pytest.mark.parametrize(
        ("count", "point"),
        [(0, [0.0, 0.0, 0.0]), (2, [np.nan] * 3)],
        ids=["a point without a count", "a count without a point"],
    )
    def test_the_nearest_point_is_missing_exactly_when_nothing_is_inside(
        self, count: int, point: list[float]
    ) -> None:
        with pytest.raises(ValueError, match="NaN exactly where num_points is 0"):
            SensingResults(num_points=[count], sensing_status=[0], nearest_point=[point])

    def test_round_trips_with_a_missing_point(self) -> None:
        results = SensingResults(
            num_points=[2, 0],
            sensing_status=[0, 1],
            nearest_point=[[1.0, 2.0, 3.0], [np.nan] * 3],
        )
        chunk = results.to_chunk(GT + "/sensing", at=TimePoint.at(frame=0), frame_id="base_link")

        assert SensingResults.from_chunk(chunk) == results
        table = chunk_to_table(chunk)
        assert isinstance(table, pa.Table)
        assert chunk_from_table(table)[0] == chunk


class TestDetectionWiring:
    def test_defaults(self) -> None:
        system = SensingDetectionSystem.between(GT, CLOUD)

        assert str(system.target) == "/ground_truth/objects/sensing"
        assert system.requires_for(1) == (POINT,)
        assert system.PROVIDES == SensingResults.required_descriptors()
        assert isinstance(system, System)
        assert (system.box_scale_0m, system.box_scale_100m, system.min_num_points) == (1.0, 1.0, 1)

    def test_target_override(self) -> None:
        system = SensingDetectionSystem.between(GT, CLOUD, target="/x")
        assert str(system.target) == "/x"

    def test_needs_two_sources(self) -> None:
        with pytest.raises(ValueError, match=r"exactly two sources \(objects, points\)"):
            SensingDetectionSystem((GT,), "/x")

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            ({"box_scale_0m": 0.0}, "must be positive"),
            ({"box_scale_100m": -1.0}, "must be positive"),
            ({"min_num_points": -1}, "non-negative"),
        ],
    )
    def test_rejects_bad_parameters(self, params: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SensingDetectionSystem.between(GT, CLOUD, **params)

    def test_parameters_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            SensingDetectionSystem((GT, CLOUD), "/x", 1.0)  # type: ignore[misc]


class TestDetection:
    def test_counts_the_points_inside_each_box(self) -> None:
        store = Store()
        log_boxes(store, [5.0, 20.0])
        log_cloud(store, [[5.0, 0.0, 0.0], [6.5, 0.5, 0.5], [3.0, 0.0, 1.0], [20.0, 0.0, 0.0]])
        system = SensingDetectionSystem.between(GT, CLOUD)

        run(store, system)
        results = results_of(store, system)

        assert results.num_points.values.tolist() == [3, 1]
        assert results.sensing_status.values.tolist() == [0, 0]
        # The nearest inside point is the one closest to the origin, not to the box centre.
        assert results.nearest_point.values.tolist() == [[3.0, 0.0, 1.0], [20.0, 0.0, 0.0]]

    def test_the_result_is_filed_with_the_boxes(self) -> None:
        store = Store()
        log_boxes(store, [5.0], frame=0)
        log_boxes(store, [5.0, 8.0], frame=1)
        log_cloud(store, [[5.0, 0.0, 0.0]], frame=0)
        log_cloud(store, [[8.0, 0.0, 0.0]], frame=1)
        system = SensingDetectionSystem.between(GT, CLOUD)

        run(store, system)
        view = store.range(system.target, timeline=FRAME, time_range=EVERYTHING)

        assert view.to_chunk().partition_sizes().tolist() == [1, 2]
        assert view.times(FRAME).tolist() == [0, 1, 1]
        assert view.frame_id == "base_link"
        assert view.component(NUM_POINTS).values.tolist() == [1, 0, 1]  # type: ignore[union-attr]

    def test_the_threshold_decides_the_verdict(self) -> None:
        store = Store()
        log_boxes(store, [5.0])
        log_cloud(store, [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        system = SensingDetectionSystem.between(GT, CLOUD, min_num_points=3)

        run(store, system)

        results = results_of(store, system)
        assert results.num_points.values.tolist() == [2]
        assert results.sensing_status.values.tolist() == [int(SensingStatus.FAILED)]

    def test_a_threshold_of_zero_detects_an_empty_box(self) -> None:
        store = Store()
        log_boxes(store, [5.0])
        log_cloud(store, [[50.0, 0.0, 0.0]])
        system = SensingDetectionSystem.between(GT, CLOUD, min_num_points=0)

        run(store, system)

        results = results_of(store, system)
        assert results.is_detected.tolist() == [True]
        assert np.isnan(results.nearest_point.values).all()

    @pytest.mark.parametrize(
        ("scale_100m", "detected"),
        [(1.0, False), (1.2, True)],
        ids=["unscaled misses the point beyond the face", "scaled at range reaches it"],
    )
    def test_the_box_grows_with_distance(self, scale_100m: float, detected: bool) -> None:
        # A box at 100 m spans x in [98, 102]; the point sits 0.3 m beyond the front face.
        # At scale 1.2 the half length is 2.4, which covers it.
        store = Store()
        log_boxes(store, [100.0])
        log_cloud(store, [[102.3, 0.0, 0.0]])
        system = SensingDetectionSystem.between(GT, CLOUD, box_scale_100m=scale_100m)

        run(store, system)

        assert results_of(store, system).is_detected.tolist() == [detected]

    def test_the_scale_does_not_grow_the_height(self) -> None:
        store = Store()
        log_boxes(store, [100.0])
        log_cloud(store, [[100.0, 0.0, 1.5]])
        system = SensingDetectionSystem.between(GT, CLOUD, box_scale_100m=2.0)

        run(store, system)

        assert results_of(store, system).num_points.values.tolist() == [0]

    def test_a_fully_occluded_box_is_a_warning_whatever_its_count(self) -> None:
        store = Store()
        log_boxes(
            store,
            [5.0, 20.0, 40.0],
            visibility=[VisibilityLevel.NONE, VisibilityLevel.NONE, VisibilityLevel.UNAVAILABLE],
        )
        log_cloud(store, [[5.0, 0.0, 0.0]])
        system = SensingDetectionSystem.between(GT, CLOUD)

        run(store, system)

        results = results_of(store, system)
        assert results.num_points.values.tolist() == [1, 0, 0]
        assert results.sensing_status.values.tolist() == [
            int(SensingStatus.WARNING),
            int(SensingStatus.WARNING),
            int(SensingStatus.FAILED),
        ], "NONE warns even with a point inside; UNAVAILABLE is judged normally"

    def test_boxes_without_a_visibility_column_are_never_warned(self) -> None:
        store = Store()
        log_boxes(store, [5.0])
        log_cloud(store, [])
        system = SensingDetectionSystem.between(GT, CLOUD)

        run(store, system)

        assert results_of(store, system).num_warned == 0

    def test_an_empty_cloud_fails_every_box(self) -> None:
        store = Store()
        log_boxes(store, [5.0, 20.0])
        log_cloud(store, [])
        system = SensingDetectionSystem.between(GT, CLOUD)

        run(store, system)

        results = results_of(store, system)
        assert results.num_failed == 2
        assert np.isnan(results.nearest_point.values).all()

    def test_a_missing_cloud_frame_is_an_error(self) -> None:
        store = Store()
        log_boxes(store, [5.0], frame=0)
        log_boxes(store, [5.0], frame=1)
        log_cloud(store, [[5.0, 0.0, 0.0]], frame=0)
        system = SensingDetectionSystem.between(GT, CLOUD)

        with pytest.raises(ValueError, match="has no frame at frame=1"):
            run(store, system)

    def test_an_empty_objects_frame_keeps_its_partition(self) -> None:
        store = Store()
        log_boxes(store, [5.0], frame=0)
        log_boxes(store, [], frame=1)
        log_cloud(store, [[5.0, 0.0, 0.0]], frame=0)
        system = SensingDetectionSystem.between(GT, CLOUD)

        run(store, system)
        view = store.range(system.target, timeline=FRAME, time_range=EVERYTHING)

        assert view.to_chunk().partition_sizes().tolist() == [1, 0]

    def test_an_empty_range_writes_an_empty_chunk(self) -> None:
        store = Store()
        log_boxes(store, [5.0])
        log_cloud(store, [[5.0, 0.0, 0.0]])
        system = SensingDetectionSystem.between(GT, CLOUD)

        (chunk,) = system(SystemContext(store, FRAME), TimeRange.single(7))

        assert chunk.entity_path == system.target
        assert chunk.num_rows == 0

    def test_refuses_mismatched_frames(self) -> None:
        store = Store()
        log_boxes(store, [5.0], frame_id="map")
        log_cloud(store, [[5.0, 0.0, 0.0]], frame_id="base_link")

        with pytest.raises(ValueError, match="across coordinate frames"):
            run(store, SensingDetectionSystem.between(GT, CLOUD))
        run(store, SensingDetectionSystem.between(GT, CLOUD, check_frames=False))

    def test_refuses_an_object_entity_without_boxes(self) -> None:
        store = Store()
        log_cloud(store, [[5.0, 0.0, 0.0]])
        store.log("/labels", PointCloud(point=[[5.0, 0.0, 0.0]]), at=TimePoint.at(frame=0))

        with pytest.raises(ValueError, match="missing required component"):
            run(store, SensingDetectionSystem.between("/labels", CLOUD))

    def test_the_count_feeds_the_num_points_filter(self) -> None:
        store = Store()
        log_boxes(store, [5.0, 20.0, 40.0])
        log_cloud(store, [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        sensing = SensingDetectionSystem.between(GT, CLOUD)
        seen = FilterByNumPointsSystem.on(sensing.target, min_num_points=2)
        kept = ApplyMaskSystem.of(GT, seen.target)

        run(store, sensing, seen, kept)
        boxes = store.range(kept.target, timeline=FRAME, time_range=EVERYTHING)

        assert boxes.materialize(Detections3D).position.values[:, 0].tolist() == [5.0]

    def test_the_filter_cannot_run_before_the_count_exists(self) -> None:
        sensing = SensingDetectionSystem.between(GT, CLOUD)
        seen = FilterByNumPointsSystem.on(sensing.target, min_num_points=2)

        with pytest.raises(ValueError, match="before a later system writes it"):
            Pipeline([seen, sensing])


class TestNonDetectionArea:
    def test_from_the_original_vertex_list(self) -> None:
        area = NonDetectionArea.from_vertices(LANE_VERTICES)

        assert (area.z_min, area.z_max) == (0.0, 4.0)
        assert area.contains(
            [[5.0, 0.0, 1.0], [5.0, 1.5, 4.0], [5.0, 1.6, 1.0], [5.0, 0.0, 4.1], [11.0, 0.0, 1.0]]
        ).tolist() == [True, True, False, False, False]

    def test_heights_default_to_unbounded(self) -> None:
        area = NonDetectionArea([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        assert area.contains([[0.5, 0.5, -100.0], [0.5, 0.5, 100.0]]).tolist() == [True, True]

    @pytest.mark.parametrize(
        ("vertices", "match"),
        [
            (LANE_VERTICES[:4], "at least three corners each"),
            (LANE_VERTICES[:5], "at least three corners each"),
            ([(0.0, 0.0)] * 6, "shape"),
        ],
    )
    def test_rejects_malformed_vertices(self, vertices: list, match: str) -> None:  # type: ignore[type-arg]
        with pytest.raises(ValueError, match=match):
            NonDetectionArea.from_vertices(vertices)

    def test_rejects_an_empty_polygon_and_an_inverted_height_range(self) -> None:
        ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        with pytest.raises(ValueError, match="non-empty polygon"):
            NonDetectionArea([])
        with pytest.raises(ValueError, match="z_max .* must not be below z_min"):
            NonDetectionArea(ring, z_min=1.0, z_max=0.0)


class TestNonDetection:
    LANE = NonDetectionArea.from_vertices(LANE_VERTICES)

    def test_defaults(self) -> None:
        system = SensingNonDetectionSystem.between(CLOUD, GT, areas=[self.LANE])

        assert str(system.target) == "/sensor/lidar/filter/non_detection"
        assert system.REQUIRES == (POINT,) and system.PROVIDES == (MASK,)
        assert system.requires_for(1) == SensingDetectionSystem.REQUIRES
        assert isinstance(system, System) and isinstance(system, MaskSystem)

    def test_accepts_one_area_or_triples(self) -> None:
        one = SensingNonDetectionSystem.between(CLOUD, GT, areas=self.LANE)
        triple = SensingNonDetectionSystem.between(
            CLOUD, GT, areas=[([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], 0.0, 1.0)]
        )

        assert len(one.areas) == 1
        assert (triple.areas[0].z_min, triple.areas[0].z_max) == (0.0, 1.0)
        with pytest.raises(TypeError, match="NonDetectionArea instances or"):
            SensingNonDetectionSystem.between(CLOUD, GT, areas=["lane"])

    def test_rejects_no_areas_and_wrong_source_counts(self) -> None:
        with pytest.raises(ValueError, match="at least one non-detection area"):
            SensingNonDetectionSystem.between(CLOUD, GT, areas=[])
        with pytest.raises(ValueError, match=r"exactly two sources \(points, objects\)"):
            SensingNonDetectionSystem((CLOUD,), "/x", areas=[self.LANE])
        with pytest.raises(ValueError, match="must be positive"):
            SensingNonDetectionSystem.between(CLOUD, GT, areas=[self.LANE], box_scale_0m=0.0)

    def test_marks_the_points_left_in_an_area(self) -> None:
        store = Store()
        log_boxes(store, [5.0])
        log_cloud(
            store,
            [
                [5.0, 0.0, 1.0],  # in the lane, inside the box
                [8.0, 0.0, 1.0],  # in the lane, outside the box: offends
                [8.0, 0.0, 5.0],  # above the lane
                [20.0, 0.0, 1.0],  # beyond the lane
            ],
        )
        system = SensingNonDetectionSystem.between(CLOUD, GT, areas=[self.LANE])

        run(store, system)

        assert mask_of(store, system) == [False, True, False, False]

    def test_the_boxes_are_scaled_as_for_detection(self) -> None:
        # The box at x=5 ends at 7; at scale 1.5 it reaches 8.
        store = Store()
        log_boxes(store, [5.0])
        log_cloud(store, [[7.5, 0.0, 1.0]])
        plain = SensingNonDetectionSystem.between(CLOUD, GT, areas=[self.LANE])
        scaled = SensingNonDetectionSystem.between(
            CLOUD, GT, areas=[self.LANE], box_scale_0m=1.5, box_scale_100m=1.5, target="/scaled"
        )

        run(store, plain, scaled)

        assert mask_of(store, plain) == [True]
        assert mask_of(store, scaled) == [False]

    def test_several_areas_are_a_union(self) -> None:
        store = Store()
        log_boxes(store, [])
        log_cloud(store, [[5.0, 0.0, 1.0], [50.0, 0.0, 1.0], [30.0, 0.0, 1.0]])
        ahead = ([(45.0, 2.0), (55.0, 2.0), (55.0, -2.0), (45.0, -2.0)], 0.0, 4.0)
        system = SensingNonDetectionSystem.between(CLOUD, GT, areas=[self.LANE, ahead])

        run(store, system)

        assert mask_of(store, system) == [True, True, False]

    def test_a_frame_without_boxes_offends_everywhere_in_the_area(self) -> None:
        store = Store()
        log_cloud(store, [[5.0, 0.0, 1.0], [20.0, 0.0, 1.0]])
        system = SensingNonDetectionSystem.between(CLOUD, GT, areas=[self.LANE])

        run(store, system)

        assert mask_of(store, system) == [True, False]

    def test_apply_mask_materializes_the_failures(self) -> None:
        store = Store()
        log_boxes(store, [5.0])
        log_cloud(store, [[5.0, 0.0, 1.0], [8.0, 0.0, 1.0], [20.0, 0.0, 1.0]])
        system = SensingNonDetectionSystem.between(CLOUD, GT, areas=[self.LANE])
        failures = ApplyMaskSystem.of(CLOUD, system.target, name="non_detection")

        run(store, system, failures)
        cloud = store.range(failures.target, timeline=FRAME, time_range=EVERYTHING)

        assert cloud.materialize(PointCloud).point.values.tolist() == [[8.0, 0.0, 1.0]]

    def test_refuses_mismatched_frames(self) -> None:
        store = Store()
        log_boxes(store, [5.0], frame_id="map")
        log_cloud(store, [[8.0, 0.0, 1.0]], frame_id="base_link")

        with pytest.raises(ValueError, match="across coordinate frames"):
            run(store, SensingNonDetectionSystem.between(CLOUD, GT, areas=[self.LANE]))
        run(
            store,
            SensingNonDetectionSystem.between(CLOUD, GT, areas=[self.LANE], check_frames=False),
        )


class TestDetectionRate:
    def test_defaults(self) -> None:
        system = SensingDetectionRateSystem.between(GT + "/sensing", GT)

        assert str(system.target) == "/metrics/sensing/detection_rate"
        assert system.REQUIRES == (SENSING_STATUS,) and system.requires_for(1) == (CLASS_ID,)
        assert system.PROVIDES == MetricValues.required_descriptors()
        assert isinstance(system, System)
        with pytest.raises(ValueError, match=r"exactly two sources \(sensing, objects\)"):
            SensingDetectionRateSystem((GT,), "/x")

    def test_counts_detected_over_judged_per_class(self, labels: LabelRegistry) -> None:
        store = Store()
        log_boxes(
            store,
            [5.0, 20.0, 40.0, 60.0],
            classes=["car", "car", "car", "pedestrian"],
            visibility=[3, 3, 3, VisibilityLevel.NONE],
            labels=labels,
        )
        log_cloud(store, [[5.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        sensing = SensingDetectionSystem.between(GT, CLOUD)
        rate = SensingDetectionRateSystem.between(sensing.target, GT)

        run(store, sensing, rate, labels=labels)
        metrics = store.range(rate.target, timeline=FRAME, time_range=EVERYTHING).materialize(
            MetricValues
        )

        car, truck, pedestrian = (labels.class_id(n) for n in ("car", "truck", "pedestrian"))
        assert metrics.of_class(car) == pytest.approx(2 / 3)
        assert metrics.support.values[metrics.class_id.values == car].tolist() == [3]
        assert np.isnan(metrics.of_class(truck)), "a registered class with no boxes"
        assert np.isnan(metrics.of_class(pedestrian)), "a warning is not judged"
        assert metrics.aggregate == pytest.approx(2 / 3)
        assert len(metrics) == 4

    def test_spans_the_scene_and_reports_at_its_last_frame(self) -> None:
        store = Store()
        log_boxes(store, [5.0], frame=0)
        log_boxes(store, [5.0], frame=3)
        log_cloud(store, [[5.0, 0.0, 0.0]], frame=0)
        log_cloud(store, [], frame=3)
        sensing = SensingDetectionSystem.between(GT, CLOUD)
        rate = SensingDetectionRateSystem.between(sensing.target, GT)

        run(store, sensing, rate)
        view = store.range(rate.target, timeline=FRAME, time_range=EVERYTHING)

        assert view.times(FRAME).tolist() == [3, 3]
        assert view.materialize(MetricValues).aggregate == pytest.approx(0.5)

    def test_an_empty_scene_is_undefined(self) -> None:
        store = Store()
        rate = SensingDetectionRateSystem.between(GT + "/sensing", GT)

        (chunk,) = rate(SystemContext(store, FRAME), EVERYTHING)
        metrics = MetricValues.from_chunk(chunk)

        assert np.isnan(metrics.aggregate)
        assert metrics.support.values.tolist() == [0]

    def test_refuses_misaligned_inputs(self) -> None:
        store = Store()
        log_boxes(store, [5.0, 20.0])
        store.log(
            GT + "/sensing",
            SensingResults(num_points=[1], sensing_status=[0], nearest_point=[[5.0, 0.0, 0.0]]),
            at=TimePoint.at(frame=0),
        )

        with pytest.raises(ValueError, match="aligned row for row"):
            run(store, SensingDetectionRateSystem.between(GT + "/sensing", GT))

    def test_refuses_verdicts_filed_at_other_frames(self) -> None:
        # Same row count, different partitioning: a join by row would pair the wrong boxes.
        store = Store()
        log_boxes(store, [5.0, 20.0], frame=0)
        log_boxes(store, [5.0], frame=1)
        for frame, count in ((0, 1), (1, 2)):
            store.log(
                GT + "/sensing",
                SensingResults(
                    num_points=[1] * count,
                    sensing_status=[0] * count,
                    nearest_point=[[5.0, 0.0, 0.0]] * count,
                ),
                at=TimePoint.at(frame=frame),
            )

        with pytest.raises(ValueError, match="at different 'frame' times"):
            run(store, SensingDetectionRateSystem.between(GT + "/sensing", GT))
