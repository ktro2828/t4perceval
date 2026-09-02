"""Classification metrics: accuracy, precision, recall and F1."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_metric_scene

from t4perceval import FRAME, MetricValues, LabelRegistry, Store, TimeRange
from t4perceval.system import (
    CenterDistanceMatchingSystem,
    ClassificationSystem,
    Pipeline,
    SystemContext,
)

EST = "/estimation/objects"
GT = "/ground_truth/objects"


def classification(
    labels: LabelRegistry,
    ground_truth: list[tuple[float, str]],
    estimation: list[tuple[float, str, float]],
) -> dict[str, MetricValues]:
    store = make_metric_scene(labels, [(0, ground_truth, estimation)])
    match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
    metric = ClassificationSystem.on(match.target, EST, GT)
    Pipeline([match, metric]).run(
        SystemContext(store, FRAME, labels=labels), TimeRange.everything()
    )

    return {
        target.name: store.range(
            target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MetricValues)
        for target in metric.targets
    }


class TestTargets:
    def test_writes_one_entity_per_metric(self) -> None:
        metric = ClassificationSystem.on("/m", EST, GT)

        assert [str(target) for target in metric.targets] == [
            "/metrics/classification/accuracy",
            "/metrics/classification/precision",
            "/metrics/classification/recall",
            "/metrics/classification/f1",
        ]


class TestValues:
    def test_a_perfect_classifier_scores_one(self, labels: LabelRegistry) -> None:
        metrics = classification(
            labels,
            [(0.0, "car"), (10.0, "car")],
            [(0.1, "car", 0.9), (10.1, "car", 0.9)],
        )

        for name, row in metrics.items():
            assert row.of_class(0) == pytest.approx(1.0), name

    def test_a_false_positive_costs_precision_but_not_recall(
        self,
        labels: LabelRegistry,
    ) -> None:
        metrics = classification(
            labels,
            [(0.0, "car"), (10.0, "car")],
            [(0.1, "car", 0.9), (10.1, "car", 0.9), (500.0, "car", 0.5)],
        )

        # Two hits out of three claims, and both objects found.
        assert metrics["precision"].of_class(0) == pytest.approx(2 / 3)
        assert metrics["recall"].of_class(0) == pytest.approx(1.0)
        assert metrics["accuracy"].of_class(0) == pytest.approx(2 / (3 + 2 - 2))
        assert metrics["f1"].of_class(0) == pytest.approx(0.8)

    def test_a_missed_object_costs_recall_but_not_precision(
        self,
        labels: LabelRegistry,
    ) -> None:
        metrics = classification(
            labels,
            [(0.0, "car"), (10.0, "car")],
            [(0.1, "car", 0.9)],
        )

        assert metrics["precision"].of_class(0) == pytest.approx(1.0)
        assert metrics["recall"].of_class(0) == pytest.approx(0.5)
        assert metrics["f1"].of_class(0) == pytest.approx(2 / 3)

    def test_a_class_mismatch_is_not_a_hit(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(labels, [(0, [(0.0, "car")], [(0.1, "truck", 0.9)])])
        match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0, class_agnostic=True)
        metric = ClassificationSystem.on(match.target, EST, GT)
        Pipeline([match, metric]).run(
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        recall = store.range(
            metric.targets[2],
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MetricValues)
        assert recall.of_class(labels.class_id("car")) == pytest.approx(0.0)

    def test_classes_are_scored_independently(self, labels: LabelRegistry) -> None:
        metrics = classification(
            labels,
            [(0.0, "car"), (50.0, "truck")],
            [(0.1, "car", 0.9), (500.0, "truck", 0.9)],
        )

        assert metrics["recall"].of_class(labels.class_id("car")) == pytest.approx(1.0)
        assert metrics["recall"].of_class(labels.class_id("truck")) == pytest.approx(0.0)

    def test_spans_several_frames(self, labels: LabelRegistry) -> None:
        store = make_metric_scene(
            labels,
            [
                (0, [(0.0, "car")], [(0.1, "car", 0.9)]),
                (1, [(5.0, "car")], [(500.0, "car", 0.9)]),
            ],
        )
        match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        metric = ClassificationSystem.on(match.target, EST, GT)
        Pipeline([match, metric]).run(
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        recall = store.range(
            metric.targets[2],
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MetricValues)
        assert recall.of_class(0) == pytest.approx(0.5), "one of two frames found its object"


class TestUndefined:
    def test_no_estimation_leaves_precision_undefined(self, labels: LabelRegistry) -> None:
        """Where the original returned ``inf``, this says undefined and shows the support."""
        metrics = classification(labels, [(0.0, "car")], [])

        assert np.isnan(metrics["precision"].of_class(0))
        assert metrics["recall"].of_class(0) == pytest.approx(0.0)
        assert np.isnan(metrics["f1"].of_class(0))
        assert metrics["precision"].support.values[0] == 1

    def test_no_ground_truth_leaves_recall_undefined(self, labels: LabelRegistry) -> None:
        metrics = classification(labels, [], [(0.1, "car", 0.9)])

        assert np.isnan(metrics["recall"].of_class(0))
        assert metrics["precision"].of_class(0) == pytest.approx(0.0)
        assert metrics["recall"].support.values[0] == 0

    def test_a_class_with_nothing_at_all_is_undefined(self, labels: LabelRegistry) -> None:
        metrics = classification(labels, [(0.0, "car")], [(0.1, "car", 0.9)])

        truck = labels.class_id("truck")
        for name, row in metrics.items():
            assert np.isnan(row.of_class(truck)), name
            assert row.support.values[truck] == 0

    def test_an_empty_store_yields_undefined_rows(self, labels: LabelRegistry) -> None:
        metric = ClassificationSystem.on("/matching/center_distance", EST, GT)
        store = Store()
        Pipeline([metric]).run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())

        for target in metric.targets:
            row = store.range(
                target,
                timeline=FRAME,
                time_range=TimeRange.everything(),
            ).materialize(MetricValues)
            assert np.isnan(row.value.values).all()
