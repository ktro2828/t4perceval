"""Segmentation metrics: element-wise comparison with no matching stage."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import SEG_EST, SEG_GT, label_scene, make_ego_scene, make_segmentation

from t4perceval import (
    FRAME,
    TIMESTAMP,
    Chunk,
    ConfusionMatrix,
    LabelRegistry,
    MetricValues,
    Recording,
    SemanticSegmentation2D,
    SemanticSegmentation3D,
    Store,
    TimeColumn,
    TimePoint,
    TimeRange,
)
from t4perceval.component import ALL_CLASSES, BACKGROUND_CLASS_ID, BatchImageSize, BatchPosition3D
from t4perceval.descriptors import CLASS_ID, IMAGE_SIZE, POINT
from t4perceval.io import chunk_from_table, chunk_to_table, read_recording, write_recording
from t4perceval.label import UNKNOWN_CLASS_ID
from t4perceval.system import (
    ApplyMaskSystem,
    AveragePrecisionSystem,
    CenterDistanceMatchingSystem,
    FilterByLabelSystem,
    Pipeline,
    SegmentationConfusionMatrixSystem,
    SegmentationIoUSystem,
    SegmentationMetricSystem,
    System,
    SystemContext,
    TransformEntitySystem,
)
from t4perceval.system.metric import latest_time, registered_classes, reporting_time

EVERYTHING = TimeRange.everything()
ROAD, CAR, PEDESTRIAN = 0, 1, 2

#: The documented example: three points, the third one mislabelled.
DOC_EXAMPLE = [(0, ["road", "car", "car"], ["road", "car", "road"])]


@pytest.fixture
def seg_labels() -> LabelRegistry:
    return LabelRegistry.from_names(["road", "car", "pedestrian"])


def run(store: Store, *systems, labels=None, at=EVERYTHING, timeline=FRAME):  # noqa: ANN001
    return Pipeline(list(systems)).run(SystemContext(store, timeline, labels=labels), at)


def iou_tables(store: Store, labels: LabelRegistry | None, **params) -> dict[str, MetricValues]:  # noqa: ANN003
    system = SegmentationIoUSystem.between(SEG_EST, SEG_GT, **params)
    run(store, system, labels=labels)
    return {
        str(target).rsplit("/", 1)[1]: store.range(
            target, timeline=FRAME, time_range=EVERYTHING
        ).materialize(MetricValues)
        for target in system.targets
    }


def confusion(store: Store, labels: LabelRegistry | None, **params) -> ConfusionMatrix:  # noqa: ANN003
    system = SegmentationConfusionMatrixSystem.between(SEG_EST, SEG_GT, **params)
    run(store, system, labels=labels)
    return store.range(system.target, timeline=FRAME, time_range=EVERYTHING).materialize(
        ConfusionMatrix
    )


def per_class(table: MetricValues) -> dict[int, float]:
    return {
        int(c): float(v)
        for c, v in zip(table.class_id.values, table.value.values, strict=True)
        if int(c) != ALL_CLASSES
    }


class TestWiring:
    def test_default_targets(self) -> None:
        matrix = SegmentationConfusionMatrixSystem.between(SEG_EST, SEG_GT)
        assert str(matrix.target) == "/metrics/segmentation/confusion_matrix"
        iou = SegmentationIoUSystem.between(SEG_EST, SEG_GT)
        assert [str(t) for t in iou.targets] == [
            "/metrics/segmentation/iou",
            "/metrics/segmentation/accuracy",
            "/metrics/segmentation/pixel_accuracy",
        ]
        assert iou.sources == (matrix.sources[0], matrix.sources[1])
        assert str(iou.sources[0]) == SEG_EST and str(iou.sources[1]) == SEG_GT

    def test_a_custom_target_roots_every_table(self) -> None:
        iou = SegmentationIoUSystem.between(SEG_EST, SEG_GT, target="/m")
        assert [str(t) for t in iou.targets] == ["/m/iou", "/m/accuracy", "/m/pixel_accuracy"]

    def test_contracts(self) -> None:
        assert SegmentationIoUSystem.PROVIDES == MetricValues.required_descriptors()
        assert SegmentationConfusionMatrixSystem.PROVIDES == ConfusionMatrix.required_descriptors()
        system = SegmentationIoUSystem.between(SEG_EST, SEG_GT)
        assert system.REQUIRES == (CLASS_ID,)
        assert system.requires_for(0) == system.requires_for(1) == (CLASS_ID,)
        assert isinstance(system, System)
        assert isinstance(SegmentationConfusionMatrixSystem.between(SEG_EST, SEG_GT), System)

    @pytest.mark.parametrize("cls", [SegmentationIoUSystem, SegmentationConfusionMatrixSystem])
    def test_needs_exactly_two_sources(self, cls: type[SegmentationMetricSystem]) -> None:
        with pytest.raises(
            ValueError, match=r"needs exactly two sources \(estimation, ground truth\), got 1"
        ):
            cls((SEG_EST,), "/x")

    def test_parameters_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            SegmentationIoUSystem((SEG_EST, SEG_GT), "/x", ("road",))  # type: ignore[misc]
        system = SegmentationIoUSystem.between(SEG_EST, SEG_GT, ignore=["road"], check_frames=False)
        assert system.ignore == ("road",)
        assert system.check_frames is False

    def test_a_missing_class_id_is_reported(self, seg_labels: LabelRegistry) -> None:
        store = Store()
        for path in (SEG_EST, SEG_GT):
            store.send_chunk(
                Chunk.from_columns(
                    path,
                    {POINT: BatchPosition3D([[0.0, 0.0, 0.0]])},
                    indexes=(TimeColumn.of(FRAME, [0]),),
                    frame_id="LIDAR_CONCAT",
                ),
            )
        with pytest.raises(ValueError, match=r"missing required component\(s\): class_id"):
            run(store, SegmentationIoUSystem.between(SEG_EST, SEG_GT), labels=seg_labels)

    def test_no_matcher_is_involved(self, seg_labels: LabelRegistry) -> None:
        store = label_scene(seg_labels, DOC_EXAMPLE)
        before = set(store.entity_paths())
        iou = SegmentationIoUSystem.between(SEG_EST, SEG_GT)
        run(store, iou, labels=seg_labels)
        added = {str(p) for p in set(store.entity_paths()) - before}
        assert added == {str(t) for t in iou.targets}
        assert not any(path.startswith("/matching") for path in added)


class TestConfusionMatrix:
    def test_the_documented_example(self, seg_labels: LabelRegistry) -> None:
        matrix = confusion(label_scene(seg_labels, DOC_EXAMPLE), seg_labels)
        assert matrix.at(ROAD, ROAD) == 1
        assert matrix.at(CAR, CAR) == 1
        assert matrix.at(CAR, ROAD) == 1
        assert int(matrix.count.values.sum()) == 3
        assert len(matrix) == 16, "complete (K+1)^2 square"
        assert matrix.as_matrix([ROAD, CAR, PEDESTRIAN]).shape == (4, 4)

    def test_rows_are_ground_truth_and_columns_are_estimation(self, seg_labels) -> None:  # noqa: ANN001
        store = label_scene(seg_labels, [(0, ["car", "car"], ["car", "road"])])
        matrix = confusion(store, seg_labels)
        assert matrix.at(CAR, ROAD) == 1
        assert matrix.at(ROAD, CAR) == 0

        swapped = SegmentationConfusionMatrixSystem.between(SEG_GT, SEG_EST, target="/swapped")
        run(store, swapped, labels=seg_labels)
        result = store.range("/swapped", timeline=FRAME, time_range=EVERYTHING).materialize(
            ConfusionMatrix
        )
        assert result.at(ROAD, CAR) == 1

    def test_the_background_row_is_always_zero(self, seg_labels: LabelRegistry) -> None:
        store = label_scene(seg_labels, [(0, ["car", "car", "car"], ["car", -1, 7])])
        matrix = confusion(store, seg_labels)
        dense = matrix.as_matrix([ROAD, CAR, PEDESTRIAN])
        assert dense[-1].sum() == 0
        assert matrix.at(CAR, BACKGROUND_CLASS_ID) == 2, "no known class predicted"
        assert matrix.at(CAR, CAR) == 1

    def test_counts_accumulate_across_frames(self, seg_labels: LabelRegistry) -> None:
        store = label_scene(seg_labels, [(0, ["car"], ["car"]), (1, ["car"], ["road"])])
        matrix = confusion(store, seg_labels)
        assert matrix.at(CAR, CAR) == 1
        assert matrix.at(CAR, ROAD) == 1

        system = SegmentationConfusionMatrixSystem.between(SEG_EST, SEG_GT)
        (chunk,) = system(SystemContext(store, FRAME, labels=seg_labels), 1)
        single = ConfusionMatrix.from_chunk(chunk)
        assert (single.at(CAR, CAR), single.at(CAR, ROAD)) == (0, 1)

    def test_an_unregistered_ground_truth_is_malformed(self, seg_labels: LabelRegistry) -> None:
        store = label_scene(seg_labels, [(0, ["car", 7], ["car", "car"])])
        with pytest.raises(
            ValueError,
            match=r"holds ground-truth class id\(s\) \[7\] absent from the label registry at frame=0",
        ):
            confusion(store, seg_labels)

    def test_unknown_ground_truth_rows_are_dropped(self, seg_labels: LabelRegistry) -> None:
        store = label_scene(seg_labels, [(0, ["car", -1, -1], ["car", "road", "car"])])
        matrix = confusion(store, seg_labels)
        assert int(matrix.count.values.sum()) == 1
        assert matrix.at(CAR, CAR) == 1
        assert matrix.as_matrix([ROAD, CAR, PEDESTRIAN])[:, ROAD].sum() == 0

    def test_an_ignored_class_leaves_the_axis(self, seg_labels: LabelRegistry) -> None:
        scene = [(0, ["road", "road", "car", "car"], ["road", "car", "car", "road"])]
        matrix = confusion(label_scene(seg_labels, scene), seg_labels, ignore=("road",))
        assert len(matrix) == 9, "car, pedestrian and background"
        assert matrix.at(CAR, CAR) == 1
        assert matrix.at(CAR, BACKGROUND_CLASS_ID) == 1, "a road prediction is no known class"
        with pytest.raises(KeyError):
            matrix.at(ROAD, ROAD)
        by_id = confusion(label_scene(seg_labels, scene), seg_labels, ignore=(ROAD,))
        assert by_id == matrix

    def test_without_a_registry_classes_come_from_the_data(self, seg_labels) -> None:  # noqa: ANN001
        store = label_scene(seg_labels, [(0, [0, 1], [0, 0]), (1, [5], [5])])
        matrix = confusion(store, None)
        assert len(matrix) == 16, "0, 1, 5 and background"
        assert matrix.at(5, 5) == 1
        assert matrix.at(1, 0) == 1


class TestIoU:
    def test_the_documented_example(self, seg_labels: LabelRegistry) -> None:
        tables = iou_tables(label_scene(seg_labels, DOC_EXAMPLE), seg_labels)
        iou, accuracy, pixel = tables["iou"], tables["accuracy"], tables["pixel_accuracy"]

        assert iou.of_class(ROAD) == 0.5
        assert iou.of_class(CAR) == 0.5
        assert np.isnan(iou.of_class(PEDESTRIAN))
        assert iou.support.values.tolist() == [1, 2, 0, 3]
        assert iou.aggregate == 0.5, "mIoU skips the undefined class"

        assert accuracy.of_class(ROAD) == 1.0
        assert accuracy.of_class(CAR) == 0.5
        assert np.isnan(accuracy.of_class(PEDESTRIAN))
        assert accuracy.aggregate == 0.75

        assert len(pixel) == 1
        assert pixel.class_id.values.tolist() == [ALL_CLASSES]
        assert pixel.aggregate == pytest.approx(2 / 3)
        assert pixel.support.values.tolist() == [3]

        for table in tables.values():
            assert np.isnan(table.threshold.values).all()

    def test_counts_are_pooled_across_frames(self, seg_labels: LabelRegistry) -> None:
        store = label_scene(seg_labels, [(0, ["car"], ["car"]), (1, ["car"], ["road"])])
        tables = iou_tables(store, seg_labels)
        assert tables["iou"].of_class(CAR) == 0.5
        assert tables["iou"].of_class(ROAD) == 0.0, "a false-positive-only class scores 0"
        assert np.isnan(tables["accuracy"].of_class(ROAD)), "but its accuracy is undefined"
        assert tables["pixel_accuracy"].aggregate == 0.5

        system = SegmentationIoUSystem.between(SEG_EST, SEG_GT)
        chunks = system(SystemContext(store, FRAME, labels=seg_labels), 0)
        iou_at_0 = MetricValues.from_chunk(next(iter(chunks)))
        assert iou_at_0.of_class(CAR) == 1.0

    def test_an_unknown_estimation_is_a_false_negative(self, seg_labels: LabelRegistry) -> None:
        tables = iou_tables(label_scene(seg_labels, [(0, ["car", "car"], ["car", -1])]), seg_labels)
        assert tables["iou"].of_class(CAR) == 0.5
        assert tables["accuracy"].of_class(CAR) == 0.5
        assert tables["pixel_accuracy"].aggregate == 0.5
        assert ALL_CLASSES not in per_class(tables["iou"])

    def test_unknown_ground_truth_rows_count_nowhere(self, seg_labels: LabelRegistry) -> None:
        tables = iou_tables(
            label_scene(seg_labels, [(0, ["car", -1, -1], ["car", "road", "car"])]), seg_labels
        )
        assert tables["iou"].of_class(CAR) == 1.0
        assert np.isnan(tables["iou"].of_class(ROAD)), (
            "a road prediction on a dropped row is not an FP"
        )
        assert tables["iou"].support.values.tolist() == [0, 1, 0, 1]
        assert tables["pixel_accuracy"].aggregate == 1.0

    def test_an_ignored_class_is_not_evaluated(self, seg_labels: LabelRegistry) -> None:
        scene = [(0, ["road", "road", "car", "car"], ["road", "car", "car", "road"])]
        plain = iou_tables(label_scene(seg_labels, scene), seg_labels)
        assert plain["iou"].of_class(ROAD) == pytest.approx(1 / 3)
        assert plain["iou"].of_class(CAR) == pytest.approx(1 / 3)

        ignored = iou_tables(label_scene(seg_labels, scene), seg_labels, ignore=("road",))
        assert sorted(per_class(ignored["iou"])) == [CAR, PEDESTRIAN], "no road row"
        assert ignored["iou"].of_class(CAR) == 0.5, "TP 1, FN 1 (predicted as road), FP 0"
        assert ignored["iou"].support.values.tolist() == [2, 0, 2]
        assert ignored["iou"].aggregate == 0.5
        assert ignored["pixel_accuracy"].aggregate == 0.5

    def test_every_registered_class_gets_a_row_once(self, seg_labels: LabelRegistry) -> None:
        tables = iou_tables(label_scene(seg_labels, [(0, ["car"], ["car"])]), seg_labels)
        for name in ("iou", "accuracy"):
            ids = tables[name].class_id.values.tolist()
            assert ids == [ROAD, CAR, PEDESTRIAN, ALL_CLASSES]
            assert ids.count(ALL_CLASSES) == 1

    def test_rows_are_compared_as_rows(self, seg_labels: LabelRegistry) -> None:
        # Same points, different labels in the second row: a genuine disagreement.
        aligned = iou_tables(
            label_scene(seg_labels, [(0, ["road", "car"], ["road", "car"])]), seg_labels
        )
        relabelled = iou_tables(
            label_scene(seg_labels, [(0, ["road", "car"], ["car", "road"])]), seg_labels
        )
        assert aligned["iou"].of_class(CAR) == 1.0
        assert relabelled["iou"].of_class(CAR) == 0.0


class TestLabelHandling:
    def test_an_unknown_ignore_name_is_reported(self, seg_labels: LabelRegistry) -> None:
        store = label_scene(seg_labels, DOC_EXAMPLE)
        with pytest.raises(KeyError, match="Unknown class name 'nope'"):
            iou_tables(store, seg_labels, ignore=("nope",))

    def test_ignore_names_need_a_registry(self, seg_labels: LabelRegistry) -> None:
        store = label_scene(seg_labels, DOC_EXAMPLE)
        with pytest.raises(ValueError, match="ignore names .* require a LabelRegistry"):
            iou_tables(store, None, ignore=("road",))

    def test_ignore_ids_do_not(self, seg_labels: LabelRegistry) -> None:
        tables = iou_tables(label_scene(seg_labels, DOC_EXAMPLE), None, ignore=(ROAD,))
        assert sorted(per_class(tables["iou"])) == [CAR]


class TestAlignment:
    def test_unequal_row_counts_name_both_entities_and_the_time(self, seg_labels) -> None:  # noqa: ANN001
        store = label_scene(
            seg_labels, [(0, ["car"] * 3, ["car"] * 3), (1, ["car"] * 3, ["car"] * 2)]
        )
        with pytest.raises(ValueError) as info:
            iou_tables(store, seg_labels)
        message = str(info.value)
        assert SEG_EST in message and SEG_GT in message
        assert "hold 2 and 3 row(s) at frame=1" in message
        assert not any(str(p).startswith("/metrics") for p in store.entity_paths()), (
            "nothing written"
        )

    def test_a_frame_with_data_on_one_side_only_is_misaligned(self, seg_labels) -> None:  # noqa: ANN001
        store = Store()
        make_segmentation(store, SEG_GT, 0, ["car"], labels=seg_labels)
        make_segmentation(store, SEG_EST, 0, ["car"], labels=seg_labels)
        make_segmentation(store, SEG_GT, 1, ["car", "car"], labels=seg_labels)
        with pytest.raises(ValueError, match=r"hold 0 and 2 row\(s\) at frame=1"):
            iou_tables(store, seg_labels)

    def test_alignment_is_checked_per_frame_not_over_the_range(self, seg_labels) -> None:  # noqa: ANN001
        store = label_scene(
            seg_labels, [(0, ["car"] * 2, ["car"] * 3), (1, ["car"] * 3, ["car"] * 2)]
        )
        with pytest.raises(ValueError, match="at frame=0"):
            confusion(store, seg_labels)


class TestFrames:
    def test_refuses_to_compare_across_coordinate_frames(self, seg_labels) -> None:  # noqa: ANN001
        store = Store()
        make_segmentation(store, SEG_GT, 0, ["car"], labels=seg_labels, frame_id="LIDAR_CONCAT")
        make_segmentation(store, SEG_EST, 0, ["car"], labels=seg_labels, frame_id="LIDAR_TOP")
        with pytest.raises(ValueError, match="across coordinate frames"):
            iou_tables(store, seg_labels)
        assert iou_tables(store, seg_labels, check_frames=False)["iou"].of_class(CAR) == 1.0

    def test_an_unstated_frame_is_not_a_disagreement(self, seg_labels) -> None:  # noqa: ANN001
        store = Store()
        make_segmentation(store, SEG_GT, 0, ["car"], labels=seg_labels, frame_id="LIDAR_CONCAT")
        make_segmentation(store, SEG_EST, 0, ["car"], labels=seg_labels, frame_id=None)
        assert iou_tables(store, seg_labels)["iou"].of_class(CAR) == 1.0

    def test_the_check_is_per_time(self, seg_labels: LabelRegistry) -> None:
        store = label_scene(seg_labels, [(0, ["car"], ["car"])])
        make_segmentation(store, SEG_GT, 1, ["car"], labels=seg_labels)
        make_segmentation(store, SEG_EST, 1, ["car"], labels=seg_labels, frame_id="LIDAR_TOP")
        with pytest.raises(ValueError, match="across coordinate frames"):
            confusion(store, seg_labels)


class TestEmptiness:
    def test_both_entities_empty_still_report_every_class(self, seg_labels) -> None:  # noqa: ANN001
        store = label_scene(seg_labels, [(0, [], [])])
        tables = iou_tables(store, seg_labels)
        assert tables["iou"].class_id.values.tolist() == [ROAD, CAR, PEDESTRIAN, ALL_CLASSES]
        assert np.isnan(tables["iou"].value.values).all()
        assert tables["iou"].support.values.tolist() == [0, 0, 0, 0]
        assert np.isnan(tables["pixel_accuracy"].aggregate)
        matrix = confusion(store, seg_labels)
        assert len(matrix) == 16 and int(matrix.count.values.sum()) == 0

    def test_an_empty_store_still_reports_via_the_registry(self, seg_labels) -> None:  # noqa: ANN001
        store = Store()
        tables = iou_tables(store, seg_labels)
        assert len(tables["iou"]) == 4
        chunk = store.range(
            "/metrics/segmentation/iou", timeline=FRAME, time_range=EVERYTHING
        ).to_chunk()
        assert chunk.index(FRAME).times.tolist() == [0]  # type: ignore[union-attr]

    def test_an_empty_frame_between_populated_frames_is_fine(self, seg_labels) -> None:  # noqa: ANN001
        store = label_scene(
            seg_labels, [(0, ["car"], ["car"]), (1, [], []), (2, ["car"], ["road"])]
        )
        assert iou_tables(store, seg_labels)["iou"].of_class(CAR) == 0.5


class TestReportingTime:
    def test_a_scene_score_is_reported_at_the_last_frame(self, seg_labels) -> None:  # noqa: ANN001
        store = label_scene(seg_labels, [(0, ["car"], ["car"]), (1, ["car"], ["car"])])
        iou = SegmentationIoUSystem.between(SEG_EST, SEG_GT)
        matrix = SegmentationConfusionMatrixSystem.between(SEG_EST, SEG_GT)
        run(store, iou, matrix, labels=seg_labels)
        for target in (*iou.targets, matrix.target):
            assert store.latest_at(target, timeline=FRAME, at=99).times(FRAME).tolist()[0] == 1

    def test_a_single_frame_query_scores_that_frame(self, seg_labels) -> None:  # noqa: ANN001
        store = label_scene(seg_labels, [(0, ["car"], ["car"]), (1, ["car"], ["car"])])
        system = SegmentationIoUSystem.between(SEG_EST, SEG_GT)
        for chunk in system(SystemContext(store, FRAME, labels=seg_labels), 0):
            assert chunk.index(FRAME).times.tolist() == [0]  # type: ignore[union-attr]

    def test_reports_on_the_context_timeline(self, seg_labels: LabelRegistry) -> None:
        store = label_scene(seg_labels, [(0, ["car"], ["car"]), (1, ["car"], ["car"])])
        iou = SegmentationIoUSystem.between(SEG_EST, SEG_GT)
        run(store, iou, labels=seg_labels, timeline=TIMESTAMP)
        chunk = store.range(iou.targets[0], timeline=TIMESTAMP, time_range=EVERYTHING).to_chunk()
        assert chunk.index(TIMESTAMP).times.tolist() == [2_000]  # type: ignore[union-attr]
        assert chunk.index(FRAME) is None


class TestMetricHelpers:
    def test_latest_and_reporting_time(self, seg_labels: LabelRegistry) -> None:
        store = label_scene(seg_labels, [(0, ["car"], ["car"]), (2, ["car"], ["car"])])
        views = [
            store.range(SEG_GT, timeline=FRAME, time_range=EVERYTHING),
            store.range(SEG_EST, timeline=FRAME, time_range=TimeRange.single(0)),
        ]
        assert latest_time(views, FRAME) == 2
        assert latest_time([], FRAME) is None
        assert reporting_time(2, EVERYTHING) == 2
        assert reporting_time(None, EVERYTHING) == 0
        assert reporting_time(None, TimeRange(5, 9)) == 5

    def test_registered_classes(self, seg_labels: LabelRegistry) -> None:
        store = label_scene(seg_labels, [(0, [5, -1], [5, 5])])
        view = store.range(SEG_GT, timeline=FRAME, time_range=EVERYTHING)
        assert registered_classes(
            SystemContext(store, FRAME, labels=seg_labels), view
        ).tolist() == [0, 1, 2]
        assert registered_classes(SystemContext(store, FRAME), view).tolist() == [5]


class TestPersistence:
    def test_every_target_survives_an_arrow_round_trip(self, seg_labels) -> None:  # noqa: ANN001
        store = label_scene(seg_labels, DOC_EXAMPLE)
        iou = SegmentationIoUSystem.between(SEG_EST, SEG_GT)
        matrix = SegmentationConfusionMatrixSystem.between(SEG_EST, SEG_GT)
        run(store, iou, matrix, labels=seg_labels)
        for target in (*iou.targets, matrix.target):
            chunk = store.range(target, timeline=FRAME, time_range=EVERYTHING).to_chunk()
            assert chunk_from_table(chunk_to_table(chunk))[0] == chunk, "NaN included"

    def test_a_segmentation_run_reopens_with_identical_metrics(self, seg_labels, tmp_path) -> None:  # noqa: ANN001
        store = Store()
        for path, image in (
            ("/ground_truth/pixels", [[ROAD, CAR, CAR], [CAR, CAR, ROAD]]),
            ("/estimation/pixels", [[ROAD, CAR, CAR], [CAR, ROAD, ROAD]]),
        ):
            store.log(
                path,
                SemanticSegmentation2D.from_label_map(image),
                at=TimePoint.at(frame=0),
                frame_id="CAM_FRONT",
            )
            store.log_static_components(path, {IMAGE_SIZE: BatchImageSize([[2, 3]])})
        iou = SegmentationIoUSystem.between("/estimation/pixels", "/ground_truth/pixels")
        run(store, iou, labels=seg_labels)
        recording = Recording.of(store, labels=seg_labels)

        restored = read_recording(write_recording(recording, tmp_path / "seg.t4eval"))

        for target in iou.targets:
            before = recording.range(target, timeline=FRAME, time_range=EVERYTHING).to_chunk()
            after = restored.range(target, timeline=FRAME, time_range=EVERYTHING).to_chunk()
            assert after == before
        assert restored.static("/ground_truth/pixels")[IMAGE_SIZE] == BatchImageSize([[2, 3]])
        assert MetricValues.from_chunk(
            restored.range(iou.targets[2], timeline=FRAME, time_range=EVERYTHING).to_chunk()
        ).aggregate == pytest.approx(5 / 6)


class TestComposition:
    def test_after_a_transform_in_one_pipeline(self, seg_labels: LabelRegistry) -> None:
        store = make_ego_scene(Store(), xs=(10.0,))
        make_segmentation(
            store, SEG_EST, 0, ["road", "car", "car"], labels=seg_labels, frame_id="base_link"
        )
        make_segmentation(
            store,
            SEG_GT,
            0,
            ["road", "car", "road"],
            labels=seg_labels,
            points=[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
            frame_id="map",
        )
        with pytest.raises(ValueError, match="across coordinate frames"):
            run(store, SegmentationIoUSystem.between(SEG_EST, SEG_GT), labels=seg_labels)

        moved = TransformEntitySystem.of(SEG_EST, target_frame="map")
        iou = SegmentationIoUSystem.between(moved.target, SEG_GT)
        with pytest.raises(ValueError, match="before a later system writes it"):
            Pipeline([iou, moved])
        run(store, moved, iou, labels=seg_labels)

        points = store.range(moved.target, timeline=FRAME, time_range=EVERYTHING).component(POINT)
        assert points.values[:, 0].tolist() == [10.0, 11.0, 12.0]  # type: ignore[union-attr]
        result = store.range(iou.targets[0], timeline=FRAME, time_range=EVERYTHING).materialize(
            MetricValues
        )
        assert result.of_class(CAR) == 0.5

    def test_a_shared_mask_is_an_alternative_to_ignore(self, seg_labels: LabelRegistry) -> None:
        scene = [(0, ["road", "road", "car", "car"], ["road", "car", "car", "road"])]
        store = label_scene(seg_labels, scene)
        drop = FilterByLabelSystem.on(SEG_GT, exclude=["road"])
        gt_kept = ApplyMaskSystem.of(SEG_GT, drop.target)
        est_kept = ApplyMaskSystem.of(SEG_EST, drop.target, target="/estimation/points/kept")
        iou = SegmentationIoUSystem.between(est_kept.target, gt_kept.target)
        run(store, drop, gt_kept, est_kept, iou, labels=seg_labels)
        masked = store.range(iou.targets[0], timeline=FRAME, time_range=EVERYTHING).materialize(
            MetricValues
        )

        ignored = iou_tables(label_scene(seg_labels, scene), seg_labels, ignore=("road",))["iou"]
        # Masking keeps road on the axis (its GT rows are gone but the class is registered);
        # the evaluated classes agree exactly.
        assert masked.of_class(CAR) == ignored.of_class(CAR) == 0.5
        assert masked.support.values.tolist()[1] == 2

    def test_metrics_share_a_namespace_with_the_object_metrics(self) -> None:
        targets = {
            str(t)
            for system in (
                SegmentationIoUSystem.between(SEG_EST, SEG_GT),
                SegmentationConfusionMatrixSystem.between(SEG_EST, SEG_GT),
                AveragePrecisionSystem.on("/matching/x", SEG_EST, SEG_GT),
                CenterDistanceMatchingSystem.between(SEG_EST, SEG_GT),
            )
            for t in system.targets
        }
        assert len(targets) == 6, "no two systems share a default target"


class TestDimensionalityParity:
    def test_2d_and_3d_entities_score_identically(self, seg_labels: LabelRegistry) -> None:
        gt, est = ["road", "car", "car"], ["road", "car", "road"]
        store = label_scene(seg_labels, [(0, gt, est)])
        for path, names in (("/ground_truth/pixels", gt), ("/estimation/pixels", est)):
            store.log(
                path,
                SemanticSegmentation2D.from_label_map(seg_labels.encode(names).reshape(1, 3)),
                at=TimePoint.at(frame=0),
                frame_id="CAM_FRONT",
            )
            store.log_static_components(path, {IMAGE_SIZE: BatchImageSize([[1, 3]])})

        flat = SegmentationIoUSystem.between(
            "/estimation/pixels", "/ground_truth/pixels", target="/2d"
        )
        cloud = SegmentationIoUSystem.between(SEG_EST, SEG_GT, target="/3d")
        run(store, flat, cloud, labels=seg_labels)

        for a, b in zip(flat.targets, cloud.targets, strict=True):
            two = store.range(a, timeline=FRAME, time_range=EVERYTHING).materialize(MetricValues)
            three = store.range(b, timeline=FRAME, time_range=EVERYTHING).materialize(MetricValues)
            for column in ("class_id", "value", "support"):
                np.testing.assert_array_equal(
                    getattr(two, column).values, getattr(three, column).values
                )

    def test_mismatched_image_sizes_are_a_row_count_error(self, seg_labels: LabelRegistry) -> None:
        store = Store()
        store.log(
            "/gt",
            SemanticSegmentation2D.from_label_map(np.zeros((2, 3), dtype=np.int32)),
            at=TimePoint.at(frame=0),
        )
        store.log(
            "/est",
            SemanticSegmentation2D.from_label_map(np.zeros((3, 3), dtype=np.int32)),
            at=TimePoint.at(frame=0),
        )
        with pytest.raises(ValueError, match=r"hold 9 and 6 row\(s\) at frame=0"):
            run(store, SegmentationIoUSystem.between("/est", "/gt"), labels=seg_labels)

    def test_unknown_class_id_is_the_default_ignore(self, seg_labels: LabelRegistry) -> None:
        assert UNKNOWN_CLASS_ID == -1
        store = label_scene(seg_labels, [(0, ["car", UNKNOWN_CLASS_ID], ["car", "car"])])
        assert iou_tables(store, seg_labels)["pixel_accuracy"].support.values.tolist() == [1]

    def test_a_3d_entity_is_untouched_by_the_2d_change(self) -> None:
        segmentation = SemanticSegmentation3D(point=[[0.0, 0.0, 0.0]], class_id=[1])
        assert SemanticSegmentation3D.required_descriptors() == (POINT, CLASS_ID)
        assert len(segmentation) == 1
