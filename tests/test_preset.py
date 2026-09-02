"""The threshold sweep helper."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_metric_scene

from t4perceval import FRAME, MetricValues, LabelRegistry, TimeRange
from t4perceval.system import (
    CenterDistanceMatchingSystem,
    IoUBEVMatchingSystem,
    Pipeline,
    SystemContext,
    average_precision_sweep,
)

EST = "/estimation/objects"
GT = "/ground_truth/objects"


class TestComposition:
    def test_returns_a_matcher_and_a_metric_per_threshold(self) -> None:
        systems = average_precision_sweep(EST, GT, thresholds=[0.5, 1.0])

        assert [type(system).__name__ for system in systems] == [
            "CenterDistanceMatchingSystem",
            "AveragePrecisionSystem",
            "CenterDistanceMatchingSystem",
            "AveragePrecisionSystem",
            "MeanAveragePrecisionSystem",
        ]

    def test_targets_are_indexed_by_declaration_order(self) -> None:
        """A per-class threshold has no single value to put in a path, so an index is used."""
        systems = average_precision_sweep(EST, GT, thresholds=[0.5, 1.0])

        assert [str(system.target) for system in systems] == [
            "/matching/center_distance/0",
            "/metrics/ap/0",
            "/matching/center_distance/1",
            "/metrics/ap/1",
            "/metrics/map",
        ]

    def test_each_metric_reads_its_own_matcher(self) -> None:
        systems = average_precision_sweep(EST, GT, thresholds=[0.5, 1.0])

        for matcher, metric in ((systems[0], systems[1]), (systems[2], systems[3])):
            assert metric.sources[0] == matcher.target

    def test_the_mean_reads_every_metric(self) -> None:
        systems = average_precision_sweep(EST, GT, thresholds=[0.5, 1.0])

        assert systems[-1].sources == (systems[1].target, systems[3].target)

    def test_the_matching_mode_is_swappable(self) -> None:
        systems = average_precision_sweep(
            EST,
            GT,
            matcher=IoUBEVMatchingSystem,
            thresholds=[0.5],
        )

        assert isinstance(systems[0], IoUBEVMatchingSystem)
        assert str(systems[0].target) == "/matching/iou_bev/0"

    def test_heading_adds_aph_and_maph(self) -> None:
        systems = average_precision_sweep(EST, GT, thresholds=[1.0], heading=True)

        assert [type(system).__name__ for system in systems] == [
            "CenterDistanceMatchingSystem",
            "AveragePrecisionSystem",
            "AveragePrecisionHeadingSystem",
            "MeanAveragePrecisionSystem",
            "MeanAveragePrecisionSystem",
        ]
        assert str(systems[-1].target) == "/metrics/maph"

    def test_the_thresholds_reach_the_matchers(self) -> None:
        systems = average_precision_sweep(EST, GT, thresholds=[0.5, 2.0])

        assert systems[0].threshold.default == pytest.approx(0.5)
        assert systems[2].threshold.default == pytest.approx(2.0)

    def test_class_agnostic_reaches_the_matchers(self) -> None:
        systems = average_precision_sweep(EST, GT, thresholds=[1.0], class_agnostic=True)

        assert systems[0].class_agnostic is True

    def test_a_per_class_threshold_is_accepted(self) -> None:
        from t4perceval.system import Thresholds

        systems = average_precision_sweep(
            EST,
            GT,
            thresholds=[Thresholds(1.0, by_class={"car": 2.0})],
        )

        assert dict(systems[0].threshold.by_class) == {"car": 2.0}

    def test_the_result_is_editable_before_it_runs(self) -> None:
        """It returns a list, not a configured task -- you can inspect and change it."""
        systems = average_precision_sweep(EST, GT, thresholds=[1.0])

        systems.insert(0, CenterDistanceMatchingSystem.between(EST, GT, target="/matching/extra"))

        assert len(Pipeline(systems)) == 4

    def test_rejects_an_empty_threshold_list(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            average_precision_sweep(EST, GT, thresholds=[])


class TestExecution:
    def test_the_pipeline_it_builds_is_valid(self) -> None:
        assert len(Pipeline(average_precision_sweep(EST, GT, thresholds=[0.5, 1.0]))) == 5

    def test_runs_end_to_end(self, labels: LabelRegistry) -> None:
        # The car is found at both thresholds; the truck is 0.8 m out, so only at 1.0.
        store = make_metric_scene(
            labels,
            [(0, [(0.0, "car"), (100.0, "truck")], [(0.05, "car", 0.95), (100.8, "truck", 0.9)])],
        )
        systems = average_precision_sweep(EST, GT, thresholds=[0.5, 1.0])

        Pipeline(systems).run(
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        result = store.range(
            "/metrics/map",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MetricValues)
        assert result.aggregate == pytest.approx(0.75)

    def test_heading_produces_both_means(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(labels, [(0, [(0.0, "car")], [(0.05, "car", 0.95)])])
        systems = average_precision_sweep(EST, GT, thresholds=[1.0], heading=True)

        Pipeline(systems).run(
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        for target in ("/metrics/map", "/metrics/maph"):
            result = store.range(
                target,
                timeline=FRAME,
                time_range=TimeRange.everything(),
            ).materialize(MetricValues)
            assert result.aggregate == pytest.approx(1.0)

    def test_each_threshold_gets_its_own_assignment(self, labels: LabelRegistry) -> None:
        """Re-thresholding one loose run would not re-pair; separate runs do."""
        store = make_metric_scene(
            labels,
            [(0, [(0.0, "car"), (1.0, "car")], [(0.4, "car", 0.95), (0.9, "car", 0.9)])],
        )
        systems = average_precision_sweep(EST, GT, thresholds=[0.5, 2.0])

        Pipeline(systems).run(
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        tight = store.range(
            "/metrics/ap/0",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MetricValues)
        loose = store.range(
            "/metrics/ap/1",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MetricValues)

        assert tight.of_class(0) == pytest.approx(1.0)
        assert loose.of_class(0) == pytest.approx(1.0)
        assert not np.isnan(tight.threshold.values[0])
        assert tight.threshold.values[0] != loose.threshold.values[0]
