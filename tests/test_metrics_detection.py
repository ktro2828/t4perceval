"""Detection metrics: average precision, heading-weighted AP, and their means."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import make_detections, make_metric_scene

from t4perceval import FRAME, MetricValues, LabelRegistry, Store, TimePoint, TimeRange
from t4perceval.component import ALL_CLASSES
from t4perceval.system import (
    AveragePrecisionHeadingSystem,
    AveragePrecisionSystem,
    CenterDistanceMatchingSystem,
    MeanAveragePrecisionSystem,
    Pipeline,
    SystemContext,
)

if TYPE_CHECKING:
    from t4perceval.system.base import System

EST = "/estimation/objects"
GT = "/ground_truth/objects"


def yaw(angle: float) -> list[float]:
    return [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]


def run(store: Store, systems: list[System], labels: LabelRegistry, target: str) -> MetricValues:
    ctx = SystemContext(store, FRAME, labels=labels)
    Pipeline(systems).run(ctx, TimeRange.everything())
    return store.range(target, timeline=FRAME, time_range=TimeRange.everything()).materialize(
        MetricValues,
    )


def detection_pipeline(
    *,
    threshold: float = 1.0,
    heading: bool = False,
) -> tuple[list[System], str]:
    matcher = CenterDistanceMatchingSystem.between(EST, GT, threshold=threshold)
    metric_class = AveragePrecisionHeadingSystem if heading else AveragePrecisionSystem
    metric = metric_class.on(matcher.target, EST, GT)
    return [matcher, metric], str(metric.target)


class TestAveragePrecision:
    def test_a_perfect_detector_scores_one(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(
            labels,
            [(0, [(0.0, "car"), (10.0, "car")], [(0.1, "car", 0.95), (10.1, "car", 0.85)])],
        )
        systems, target = detection_pipeline()

        result = run(store, systems, labels, target)

        assert result.of_class(labels.class_id("car")) == pytest.approx(1.0)

    def test_a_false_positive_costs_precision(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(
            labels,
            [(0, [(0.0, "car"), (10.0, "car")], [(0.1, "car", 0.95), (500.0, "car", 0.85)])],
        )
        systems, target = detection_pipeline()

        result = run(store, systems, labels, target)

        # One of two ground truths found, and one claim that was wrong.
        assert result.of_class(labels.class_id("car")) == pytest.approx(0.438272, abs=1e-6)

    def test_confidence_ranking_matters(self, labels: LabelRegistry) -> None:
        """The same two detections score worse when the wrong one is the confident one."""
        confident_hit = make_metric_scene(
            labels,
            [(0, [(0.0, "car"), (10.0, "car")], [(0.1, "car", 0.95), (500.0, "car", 0.85)])],
        )
        confident_miss = make_metric_scene(
            labels,
            [(0, [(0.0, "car"), (10.0, "car")], [(500.0, "car", 0.95), (0.1, "car", 0.85)])],
        )
        systems, target = detection_pipeline()

        good = run(confident_hit, systems, labels, target)
        bad = run(confident_miss, list(detection_pipeline()[0]), labels, target)

        assert bad.of_class(0) == pytest.approx(0.197531, abs=1e-6)
        assert bad.of_class(0) < good.of_class(0)

    def test_a_class_mismatch_is_not_a_true_positive(self, labels: LabelRegistry) -> None:
        """A pair the matcher accepted class-agnostically still must agree on the class."""
        store = make_metric_scene(labels, [(0, [(0.0, "car")], [(0.1, "truck", 0.95)])])
        matcher = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0, class_agnostic=True)
        metric = AveragePrecisionSystem.on(matcher.target, EST, GT)

        result = run(store, [matcher, metric], labels, str(metric.target))

        assert result.of_class(labels.class_id("truck")) == pytest.approx(0.0)

    def test_nothing_to_find_and_nothing_claimed_is_undefined(
        self,
        labels: LabelRegistry,
    ) -> None:
        store = make_metric_scene(labels, [(0, [], [])])
        systems, target = detection_pipeline()

        result = run(store, systems, labels, target)

        assert np.isnan(result.value.values).all()
        assert result.support.values.tolist() == [0] * len(labels)

    def test_missing_everything_scores_zero_not_undefined(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(labels, [(0, [(0.0, "car")], [])])
        systems, target = detection_pipeline()

        result = run(store, systems, labels, target)

        assert result.of_class(labels.class_id("car")) == pytest.approx(0.0)
        assert result.support.values[0] == 1

    def test_reports_a_row_for_every_registered_class(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(labels, [(0, [(0.0, "car")], [(0.1, "car", 0.9)])])
        systems, target = detection_pipeline()

        result = run(store, systems, labels, target)

        assert result.class_id.values.tolist() == [0, 1, 2]
        assert np.isnan(result.value.values[1:]).all(), "absent classes are undefined, not zero"

    def test_records_the_threshold_the_matching_used(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(labels, [(0, [(0.0, "car")], [(0.1, "car", 0.9)])])
        systems, target = detection_pipeline(threshold=0.5)

        result = run(store, systems, labels, target)

        assert result.threshold.values[0] == pytest.approx(0.5)

    def test_a_class_seen_only_as_a_miss_still_names_its_threshold(
        self,
        labels: LabelRegistry,
    ) -> None:
        store = make_metric_scene(
            labels,
            [(0, [(0.0, "car"), (100.0, "truck")], [(0.1, "car", 0.9)])],
        )
        systems, target = detection_pipeline()

        result = run(store, systems, labels, target)

        truck = labels.class_id("truck")
        row = int(np.flatnonzero(result.class_id.values == truck)[0])
        assert result.threshold.values[row] == pytest.approx(1.0)
        assert result.value.values[row] == pytest.approx(0.0)

    def test_spans_several_frames(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(
            labels,
            [
                (0, [(0.0, "car")], [(0.1, "car", 0.95)]),
                (1, [(5.0, "car")], [(5.1, "car", 0.90)]),
            ],
        )
        systems, target = detection_pipeline()

        result = run(store, systems, labels, target)

        assert result.of_class(0) == pytest.approx(1.0)
        assert result.support.values[0] == 2, "both frames' ground truths count"

    @pytest.mark.parametrize(("min_recall", "min_precision"), [(0.0, 0.0), (0.2, 0.3)])
    def test_the_trimmed_corners_are_configurable(
        self,
        labels: LabelRegistry,
        min_recall: float,
        min_precision: float,
    ) -> None:
        store = make_metric_scene(
            labels,
            [(0, [(0.0, "car"), (10.0, "car")], [(0.1, "car", 0.95), (500.0, "car", 0.85)])],
        )
        matcher = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        metric = AveragePrecisionSystem.on(
            matcher.target,
            EST,
            GT,
            min_recall=min_recall,
            min_precision=min_precision,
        )

        result = run(store, [matcher, metric], labels, str(metric.target))

        assert 0.0 <= result.of_class(0) <= 1.0

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_rejects_an_out_of_range_corner(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"must be within \[0, 1\)"):
            AveragePrecisionSystem.on("/m", EST, GT, min_recall=bad)

    def test_rejects_too_few_recall_points(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            AveragePrecisionSystem.on("/m", EST, GT, num_recall_points=1)

    def test_needs_three_sources(self) -> None:
        with pytest.raises(ValueError, match="needs exactly three sources"):
            AveragePrecisionSystem(("/m", EST), "/metrics/ap")


class TestAveragePrecisionHeading:
    def scene(self, labels: LabelRegistry, angle: float) -> Store:
        """Two perfect detections whose heading is off by ``angle``."""
        store = Store()
        store.log(
            GT,
            make_detections(
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                labels.encode(["car", "car"]),
                confidences=[1.0, 1.0],
            ),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        rotated = make_detections(
            [[0.1, 0.0, 0.0], [10.1, 0.0, 0.0]],
            labels.encode(["car", "car"]),
            confidences=[0.95, 0.85],
        )
        store.log(
            EST,
            type(rotated)(
                position=rotated.position,
                quaternion=[yaw(angle), yaw(angle)],
                size=rotated.size,
                class_id=rotated.class_id,
                confidence=rotated.confidence,
            ),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        return store

    def test_agreeing_headings_score_like_plain_ap(self, labels: LabelRegistry) -> None:
        store = self.scene(labels, 0.0)
        systems, target = detection_pipeline(heading=True)

        result = run(store, systems, labels, target)

        assert result.of_class(0) == pytest.approx(1.0)

    def test_a_quarter_turn_halves_the_true_positive_weight(
        self,
        labels: LabelRegistry,
    ) -> None:
        store = self.scene(labels, np.pi / 2)
        systems, target = detection_pipeline(heading=True)

        result = run(store, systems, labels, target)

        # A weight of 0.5 on both hits, put through the same envelope and rescaling.
        assert result.of_class(0) == pytest.approx(0.444444, abs=1e-6)

    def test_a_reversed_heading_earns_nothing(self, labels: LabelRegistry) -> None:
        store = self.scene(labels, np.pi)
        systems, target = detection_pipeline(heading=True)

        result = run(store, systems, labels, target)

        assert result.of_class(0) == pytest.approx(0.0)

    def test_the_error_wraps_the_short_way_round(self, labels: LabelRegistry) -> None:
        clockwise = self.scene(labels, np.pi / 2)
        anticlockwise = self.scene(labels, -np.pi / 2)
        systems, target = detection_pipeline(heading=True)

        one = run(clockwise, systems, labels, target)
        other = run(anticlockwise, list(detection_pipeline(heading=True)[0]), labels, target)

        assert one.of_class(0) == pytest.approx(other.of_class(0))

    def test_writes_to_its_own_entity(self) -> None:
        assert str(AveragePrecisionHeadingSystem.on("/m", EST, GT).target) == "/metrics/aph"


class TestMeanAveragePrecision:
    def sweep(self, labels: LabelRegistry, store: Store) -> tuple[MetricValues, list[str]]:
        systems: list[System] = []
        ap_targets: list[str] = []
        for index, threshold in enumerate((0.5, 1.0)):
            matcher = CenterDistanceMatchingSystem.between(
                EST,
                GT,
                threshold=threshold,
                target=f"/matching/center_distance/{index}",
            )
            metric = AveragePrecisionSystem.on(
                matcher.target,
                EST,
                GT,
                target=f"/metrics/ap/{index}",
            )
            systems.extend((matcher, metric))
            ap_targets.append(str(metric.target))

        mean = MeanAveragePrecisionSystem.of(ap_targets)
        systems.append(mean)
        return run(store, systems, labels, str(mean.target)), ap_targets

    def test_writes_the_whole_cross_tab(self, labels: LabelRegistry) -> None:
        # The car is found at both thresholds; the truck is 0.8 m out, so only at 1.0.
        store = make_metric_scene(
            labels,
            [(0, [(0.0, "car"), (100.0, "truck")], [(0.05, "car", 0.95), (100.8, "truck", 0.9)])],
        )

        result, _ = self.sweep(labels, store)

        rows = {
            (int(c), None if np.isnan(t) else round(float(t), 3)): round(float(v), 4)
            for c, t, v in zip(
                result.class_id.values,
                result.threshold.values,
                result.value.values,
            )
        }
        assert rows[(0, None)] == pytest.approx(1.0), "car, averaged over thresholds"
        assert rows[(1, None)] == pytest.approx(0.5), "truck, averaged over thresholds"
        assert rows[(ALL_CLASSES, 0.5)] == pytest.approx(0.5), "classes, at 0.5"
        assert rows[(ALL_CLASSES, 1.0)] == pytest.approx(1.0), "classes, at 1.0"
        assert rows[(ALL_CLASSES, None)] == pytest.approx(0.75), "the mAP"

    def test_the_aggregate_row_is_reachable(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(
            labels,
            [(0, [(0.0, "car"), (100.0, "truck")], [(0.05, "car", 0.95), (100.8, "truck", 0.9)])],
        )

        result, _ = self.sweep(labels, store)

        assert result.aggregate == pytest.approx(0.75)

    def test_an_absent_class_does_not_erase_the_score(self, labels: LabelRegistry) -> None:
        """Undefined values are skipped, not propagated."""
        store = make_metric_scene(labels, [(0, [(0.0, "car")], [(0.05, "car", 0.95)])])

        result, _ = self.sweep(labels, store)

        assert result.aggregate == pytest.approx(1.0)

    def test_all_undefined_stays_undefined(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(labels, [(0, [], [])])

        result, _ = self.sweep(labels, store)

        assert np.isnan(result.aggregate)

    def test_a_single_threshold_still_produces_a_mean(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(labels, [(0, [(0.0, "car")], [(0.05, "car", 0.95)])])
        matcher = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        metric = AveragePrecisionSystem.on(matcher.target, EST, GT)
        mean = MeanAveragePrecisionSystem.of([str(metric.target)])

        result = run(store, [matcher, metric, mean], labels, str(mean.target))

        assert result.aggregate == pytest.approx(1.0)

    def test_reads_heading_entities_too(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(labels, [(0, [(0.0, "car")], [(0.05, "car", 0.95)])])
        matcher = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        metric = AveragePrecisionHeadingSystem.on(matcher.target, EST, GT)
        mean = MeanAveragePrecisionSystem.of([str(metric.target)], target="/metrics/maph")

        result = run(store, [matcher, metric, mean], labels, "/metrics/maph")

        assert result.aggregate == pytest.approx(1.0)

    def test_an_empty_store_yields_only_the_aggregate(self, labels: LabelRegistry) -> None:
        mean = MeanAveragePrecisionSystem.of(["/metrics/ap/0"])

        result = run(Store(), [mean], labels, str(mean.target))

        assert len(result) == 1
        assert np.isnan(result.aggregate)

    def test_needs_at_least_one_source(self) -> None:
        with pytest.raises(ValueError, match="at least one source"):
            MeanAveragePrecisionSystem((), "/metrics/map")
