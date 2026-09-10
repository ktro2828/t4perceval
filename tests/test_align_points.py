"""Rewriting an estimation point cloud in its ground truth's row order."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import SEG_EST, SEG_GT, make_segmentation

from t4perceval import FRAME, LabelRegistry, MetricValues, Store, TimeRange
from t4perceval.descriptors import CLASS_ID, POINT
from t4perceval.descriptors import MASK
from t4perceval.system import (
    AlignPointsSystem,
    ApplyMaskSystem,
    FilterByCoverageSystem,
    Passthrough,
    Pipeline,
    SegmentationIoUSystem,
    System,
    SystemContext,
)

EVERYTHING = TimeRange.everything()
POINTS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


@pytest.fixture
def seg_labels() -> LabelRegistry:
    return LabelRegistry.from_names(["road", "car", "pedestrian"])


def scene(labels: LabelRegistry, *, est_points=POINTS[::-1], est_classes=("car", "car", "road")):  # noqa: ANN001
    """Ground truth in order; the estimation listing the same points back to front."""
    store = Store()
    make_segmentation(store, SEG_GT, 0, ["road", "car", "car"], labels=labels, points=POINTS)
    make_segmentation(store, SEG_EST, 0, list(est_classes), labels=labels, points=list(est_points))
    return store


def aligned(store: Store, labels: LabelRegistry, **params):  # noqa: ANN003
    system = AlignPointsSystem.between(SEG_EST, SEG_GT, **params)
    Pipeline([system]).run(SystemContext(store, FRAME, labels=labels), EVERYTHING)
    return store.range(system.target, timeline=FRAME, time_range=EVERYTHING)


class TestWiring:
    def test_defaults(self) -> None:
        system = AlignPointsSystem.between(SEG_EST, SEG_GT)
        assert str(system.target) == "/estimation/points/aligned"
        assert [str(s) for s in system.sources] == [SEG_EST, SEG_GT]
        assert system.REQUIRES == (POINT,)
        assert system.PROVIDES == Passthrough(0)
        assert system.tolerance == 1e-6
        assert isinstance(system, System)

    def test_needs_two_sources_and_a_sane_tolerance(self) -> None:
        with pytest.raises(
            ValueError, match=r"needs exactly two sources \(estimation, ground truth\), got 1"
        ):
            AlignPointsSystem((SEG_EST,), "/x")
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            AlignPointsSystem.between(SEG_EST, SEG_GT, tolerance=-1.0)


class TestReordering:
    def test_rows_follow_the_ground_truth_order(self, seg_labels: LabelRegistry) -> None:
        view = aligned(scene(seg_labels), seg_labels)
        np.testing.assert_array_equal(view.component(POINT).values, POINTS)  # type: ignore[union-attr]
        assert (
            view.component(CLASS_ID).values.tolist()
            == seg_labels.encode(["road", "car", "car"]).tolist()
        )  # type: ignore[union-attr]
        assert view.frame_id == "LIDAR_CONCAT"

    def test_the_source_is_untouched(self, seg_labels: LabelRegistry) -> None:
        store = scene(seg_labels)
        aligned(store, seg_labels)
        source = store.range(SEG_EST, timeline=FRAME, time_range=EVERYTHING)
        np.testing.assert_array_equal(source.component(POINT).values, POINTS[::-1])  # type: ignore[union-attr]

    def test_every_column_is_carried(self, seg_labels: LabelRegistry) -> None:
        store = scene(seg_labels)
        view = aligned(store, seg_labels)
        assert set(view.descriptors) == set(
            store.range(SEG_EST, timeline=FRAME, time_range=EVERYTHING).descriptors
        )

    def test_extra_estimation_points_are_dropped(self, seg_labels: LabelRegistry) -> None:
        store = scene(
            seg_labels,
            est_points=[*POINTS[::-1], [9.0, 9.0, 9.0]],
            est_classes=("car", "car", "road", "pedestrian"),
        )
        view = aligned(store, seg_labels)
        assert len(view) == 3
        np.testing.assert_array_equal(view.component(POINT).values, POINTS)  # type: ignore[union-attr]

    def test_a_missing_ground_truth_point_is_reported(self, seg_labels: LabelRegistry) -> None:
        store = scene(seg_labels, est_points=POINTS[:2], est_classes=("road", "car"))
        with pytest.raises(
            ValueError,
            match=r"has 1 point\(s\) with no /estimation/points point within 1e-06 at frame=0",
        ):
            aligned(store, seg_labels)

    def test_the_tolerance_decides_what_is_the_same_point(self, seg_labels: LabelRegistry) -> None:
        jittered = [[x + 0.0005, y, z] for x, y, z in POINTS[::-1]]
        with pytest.raises(ValueError, match="do not describe the same points"):
            aligned(scene(seg_labels, est_points=jittered), seg_labels)
        view = aligned(scene(seg_labels, est_points=jittered), seg_labels, tolerance=1e-3)
        np.testing.assert_allclose(view.component(POINT).values, POINTS, atol=1e-3)  # type: ignore[union-attr]

    def test_coincident_ground_truth_points_are_undefined(self, seg_labels: LabelRegistry) -> None:
        store = Store()
        make_segmentation(
            store, SEG_GT, 0, ["road", "car"], labels=seg_labels, points=[[0.0, 0.0, 0.0]] * 2
        )
        make_segmentation(
            store, SEG_EST, 0, ["road", "car"], labels=seg_labels, points=[[0.0, 0.0, 0.0]] * 2
        )
        with pytest.raises(
            ValueError, match="nearest to more than one .* point at frame=0; coincident points"
        ):
            aligned(store, seg_labels)


class TestFramesAndEmptiness:
    def test_refuses_to_align_across_coordinate_frames(self, seg_labels: LabelRegistry) -> None:
        store = Store()
        make_segmentation(store, SEG_GT, 0, ["car"], labels=seg_labels, frame_id="LIDAR_CONCAT")
        make_segmentation(store, SEG_EST, 0, ["car"], labels=seg_labels, frame_id="LIDAR_TOP")
        with pytest.raises(ValueError, match="across coordinate frames"):
            aligned(store, seg_labels)
        assert len(aligned(store, seg_labels, check_frames=False)) == 1

    def test_frames_are_aligned_independently(self, seg_labels: LabelRegistry) -> None:
        store = scene(seg_labels)
        make_segmentation(store, SEG_GT, 1, ["car"], labels=seg_labels, points=[[5.0, 0.0, 0.0]])
        make_segmentation(store, SEG_EST, 1, ["car"], labels=seg_labels, points=[[5.0, 0.0, 0.0]])
        view = aligned(store, seg_labels)
        assert view.times(FRAME).tolist() == [0, 0, 0, 1]
        assert view.to_chunk().partition_sizes().tolist() == [3, 1]

    def test_an_empty_ground_truth_frame_keeps_its_place_with_no_rows(
        self, seg_labels: LabelRegistry
    ) -> None:
        store = scene(seg_labels)
        make_segmentation(store, SEG_GT, 1, [], labels=seg_labels)
        make_segmentation(store, SEG_EST, 1, ["car"], labels=seg_labels, points=[[5.0, 0.0, 0.0]])
        chunk = aligned(store, seg_labels).to_chunk()
        assert chunk.partition_sizes().tolist() == [3, 0]
        assert chunk.index(FRAME).times.tolist() == [0, 1]  # type: ignore[union-attr]

    def test_ground_truth_without_any_estimation_is_reported(
        self, seg_labels: LabelRegistry
    ) -> None:
        store = Store()
        make_segmentation(store, SEG_GT, 0, ["car"], labels=seg_labels)
        with pytest.raises(
            ValueError, match=r"has 1 point\(s\) at frame=0 but /estimation/points has none"
        ):
            aligned(store, seg_labels)

    def test_an_empty_range_produces_nothing(self, seg_labels: LabelRegistry) -> None:
        store = scene(seg_labels)
        system = AlignPointsSystem.between(SEG_EST, SEG_GT)
        assert tuple(system(SystemContext(store, FRAME), 99)) == ()


class TestCoverage:
    """Dropping ground truth the estimation did not cover is a stage you add, and it leaves a mask."""

    @staticmethod
    def cropped(labels: LabelRegistry) -> Store:
        """Three ground-truth points; the estimation covers only the first two, back to front."""
        store = Store()
        make_segmentation(store, SEG_GT, 0, ["road", "car", "car"], labels=labels, points=POINTS)
        make_segmentation(store, SEG_EST, 0, ["car", "road"], labels=labels, points=POINTS[1::-1])
        return store

    def test_defaults(self) -> None:
        system = FilterByCoverageSystem.between(SEG_GT, SEG_EST)
        assert str(system.target) == "/ground_truth/points/filter/coverage"
        assert system.REQUIRES == (POINT,) and system.PROVIDES == (MASK,)
        assert isinstance(system, System)
        with pytest.raises(ValueError, match=r"needs exactly two sources \(source, reference\)"):
            FilterByCoverageSystem((SEG_GT,), "/x")

    def test_the_mask_says_which_points_have_a_counterpart(self, seg_labels: LabelRegistry) -> None:
        store = self.cropped(seg_labels)
        covered = FilterByCoverageSystem.between(SEG_GT, SEG_EST)
        Pipeline([covered]).run(SystemContext(store, FRAME), EVERYTHING)
        mask = store.range(covered.target, timeline=FRAME, time_range=EVERYTHING)
        assert mask.component(MASK).values.tolist() == [True, True, False]  # type: ignore[union-attr]
        assert mask.frame_id == "LIDAR_CONCAT"

    def test_aligning_alone_refuses_the_uncovered_point(self, seg_labels: LabelRegistry) -> None:
        with pytest.raises(ValueError, match="FilterByCoverageSystem"):
            aligned(self.cropped(seg_labels), seg_labels)

    def test_the_composition_scores_only_the_covered_points(
        self, seg_labels: LabelRegistry
    ) -> None:
        store = self.cropped(seg_labels)
        covered = FilterByCoverageSystem.between(SEG_GT, SEG_EST)
        gt_kept = ApplyMaskSystem.of(SEG_GT, covered.target)
        align = AlignPointsSystem.between(SEG_EST, gt_kept.target)
        iou = SegmentationIoUSystem.between(align.target, gt_kept.target)
        Pipeline([covered, gt_kept, align, iou]).run(
            SystemContext(store, FRAME, labels=seg_labels), EVERYTHING
        )

        result = store.range(iou.targets[0], timeline=FRAME, time_range=EVERYTHING).materialize(
            MetricValues
        )
        assert result.support.values.tolist() == [1, 1, 0, 2], (
            "the uncovered car point is out of the score"
        )
        assert result.aggregate == 1.0

    def test_a_tolerance_widens_coverage(self, seg_labels: LabelRegistry) -> None:
        store = Store()
        make_segmentation(
            store,
            SEG_GT,
            0,
            ["car", "car"],
            labels=seg_labels,
            points=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        )
        make_segmentation(store, SEG_EST, 0, ["car"], labels=seg_labels, points=[[0.002, 0.0, 0.0]])
        for tolerance, expected in ((1e-6, [False, False]), (1e-2, [True, False])):
            covered = FilterByCoverageSystem.between(
                SEG_GT, SEG_EST, tolerance=tolerance, target=f"/m/{tolerance}"
            )
            Pipeline([covered]).run(SystemContext(store, FRAME), EVERYTHING)
            mask = store.range(covered.target, timeline=FRAME, time_range=EVERYTHING).component(
                MASK
            )
            assert mask.values.tolist() == expected  # type: ignore[union-attr]

    def test_no_reference_covers_nothing_and_keeps_the_frame(
        self, seg_labels: LabelRegistry
    ) -> None:
        store = Store()
        make_segmentation(store, SEG_GT, 0, ["car", "car"], labels=seg_labels)
        make_segmentation(store, SEG_GT, 1, [], labels=seg_labels)
        covered = FilterByCoverageSystem.between(SEG_GT, SEG_EST)
        Pipeline([covered]).run(SystemContext(store, FRAME), EVERYTHING)
        chunk = store.range(covered.target, timeline=FRAME, time_range=EVERYTHING).to_chunk()
        assert chunk.columns[MASK].values.tolist() == [False, False]
        assert chunk.partition_sizes().tolist() == [2, 0]

    def test_refuses_to_compare_across_frames(self, seg_labels: LabelRegistry) -> None:
        store = Store()
        make_segmentation(store, SEG_GT, 0, ["car"], labels=seg_labels, frame_id="LIDAR_CONCAT")
        make_segmentation(store, SEG_EST, 0, ["car"], labels=seg_labels, frame_id="LIDAR_TOP")
        with pytest.raises(ValueError, match="across coordinate frames"):
            Pipeline([FilterByCoverageSystem.between(SEG_GT, SEG_EST)]).run(
                SystemContext(store, FRAME), EVERYTHING
            )


class TestPipeline:
    def test_align_then_score_in_one_pipeline(self, seg_labels: LabelRegistry) -> None:
        store = scene(seg_labels)
        align = AlignPointsSystem.between(SEG_EST, SEG_GT)
        iou = SegmentationIoUSystem.between(align.target, SEG_GT)
        with pytest.raises(ValueError, match="before a later system writes it"):
            Pipeline([iou, align])
        Pipeline([align, iou]).run(SystemContext(store, FRAME, labels=seg_labels), EVERYTHING)
        result = store.range(iou.targets[0], timeline=FRAME, time_range=EVERYTHING).materialize(
            MetricValues
        )
        assert result.aggregate == 1.0, "a permutation, once undone, is a perfect score"
