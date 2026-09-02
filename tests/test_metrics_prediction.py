"""Prediction metrics: ADE, FDE and miss rate."""

from __future__ import annotations

import numpy as np
import pytest

from t4perceval import (
    FRAME,
    MetricValues,
    Predictions3D,
    LabelRegistry,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.system import (
    CenterDistanceMatchingSystem,
    PathDisplacementSystem,
    Pipeline,
    SystemContext,
)

EST = "/estimation/objects"
GT = "/ground_truth/objects"

#: A ground truth that carries straight on: x = 1, 2, 3 with y = 0.
STRAIGHT_AHEAD = [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]


def prediction(
    x: float,
    waypoints: list[list[list[float]]],
    confidences: list[float],
    labels: LabelRegistry,
    name: str = "car",
) -> Predictions3D:
    """One object whose predicted futures are ``waypoints``, shaped ``(M, T, 3)``."""
    return Predictions3D(
        position=[[x, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]],
        size=[[2.0, 4.0, 2.0]],
        class_id=labels.encode([name]),
        confidence=[0.9],
        instance_id=[1],
        waypoints=np.asarray(waypoints, dtype=np.float64)[None, ...],
        mode_confidence=np.asarray([confidences], dtype=np.float64),
    )


def displacement(
    labels: LabelRegistry,
    *,
    est_waypoints: list[list[list[float]]],
    est_confidences: list[float],
    gt_waypoints: list[list[list[float]]] | None = None,
    est_x: float = 0.05,
    **params: object,
) -> dict[str, MetricValues]:
    store = Store()
    store.log(
        GT,
        prediction(0.0, gt_waypoints or STRAIGHT_AHEAD, [1.0], labels),
        at=TimePoint.at(frame=0),
        frame_id="base_link",
    )
    store.log(
        EST,
        prediction(est_x, est_waypoints, est_confidences, labels),
        at=TimePoint.at(frame=0),
        frame_id="base_link",
    )

    match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
    metric = PathDisplacementSystem.on(match.target, EST, GT, **params)
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


#: Mode 0 stays 1 m off; mode 1 stays 3 m off. Mode 1 is the more confident.
TWO_MODES = [
    [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
    [[1.0, 3.0, 0.0], [2.0, 3.0, 0.0], [3.0, 3.0, 0.0]],
]
TWO_MODE_CONFIDENCES = [0.3, 0.7]


class TestTargets:
    def test_writes_one_entity_per_metric(self) -> None:
        metric = PathDisplacementSystem.on("/m", EST, GT)

        assert [str(target) for target in metric.targets] == [
            "/metrics/displacement/ade",
            "/metrics/displacement/fde",
            "/metrics/displacement/miss_rate",
        ]


class TestKernels:
    def test_by_default_every_kept_mode_counts(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
        )

        # The mean over both modes: (1 + 3) / 2. This is an average over modes, not the
        # best-of-k that some benchmarks report.
        assert metrics["ade"].of_class(0) == pytest.approx(2.0)

    def test_min_takes_the_best_mode(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            kernel="min",
        )

        assert metrics["ade"].of_class(0) == pytest.approx(1.0)

    def test_max_takes_the_worst_mode(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            kernel="max",
        )

        assert metrics["ade"].of_class(0) == pytest.approx(3.0)

    def test_highest_takes_the_most_confident_mode(self, labels: LabelRegistry) -> None:
        """Mode 1 is the confident one, and it is also the worse one."""
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            kernel="highest",
        )

        assert metrics["ade"].of_class(0) == pytest.approx(3.0)

    def test_rejects_an_unknown_kernel(self) -> None:
        with pytest.raises(ValueError, match="kernel must be one of"):
            PathDisplacementSystem.on("/m", EST, GT, kernel="mean")


class TestTopK:
    def test_keeps_only_the_most_confident_modes(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            top_k=1,
        )

        assert metrics["ade"].of_class(0) == pytest.approx(3.0), "the confident mode alone"

    def test_asking_for_more_modes_than_exist_is_harmless(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            top_k=9,
        )

        assert metrics["ade"].of_class(0) == pytest.approx(2.0)

    def test_rejects_a_non_positive_top_k(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            PathDisplacementSystem.on("/m", EST, GT, top_k=0)


class TestHorizonAlignment:
    def test_a_short_prediction_is_held_at_its_last_state(self, labels: LabelRegistry) -> None:
        """Predicting less far ahead is penalised for the gap, not excused from it."""
        short = [[[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]]

        metrics = displacement(labels, est_waypoints=short, est_confidences=[1.0])

        # Steps 1 and 2 are 1 m out; the held step 3 sits at (2, 1) against (3, 0).
        assert metrics["ade"].of_class(0) == pytest.approx((1.0 + 1.0 + np.sqrt(2.0)) / 3.0)
        assert metrics["fde"].of_class(0) == pytest.approx(np.sqrt(2.0))

    def test_a_long_prediction_is_truncated(self, labels: LabelRegistry) -> None:
        long = [
            [
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [4.0, 99.0, 0.0],
            ],
        ]

        metrics = displacement(labels, est_waypoints=long, est_confidences=[1.0])

        assert metrics["ade"].of_class(0) == pytest.approx(1.0), "the fourth step is ignored"


class TestValues:
    def test_a_perfect_prediction_scores_zero(self, labels: LabelRegistry) -> None:
        metrics = displacement(labels, est_waypoints=STRAIGHT_AHEAD, est_confidences=[1.0])

        assert metrics["ade"].of_class(0) == pytest.approx(0.0)
        assert metrics["fde"].of_class(0) == pytest.approx(0.0)
        assert metrics["miss_rate"].of_class(0) == pytest.approx(0.0)

    def test_fde_looks_only_at_the_final_step(self, labels: LabelRegistry) -> None:
        drifting = [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 5.0, 0.0]]]

        metrics = displacement(labels, est_waypoints=drifting, est_confidences=[1.0])

        assert metrics["ade"].of_class(0) == pytest.approx(5.0 / 3.0)
        assert metrics["fde"].of_class(0) == pytest.approx(5.0)

    def test_the_error_ignores_the_z_axis(self, labels: LabelRegistry) -> None:
        lifted = [[[1.0, 0.0, 9.0], [2.0, 0.0, 9.0], [3.0, 0.0, 9.0]]]

        metrics = displacement(labels, est_waypoints=lifted, est_confidences=[1.0])

        assert metrics["ade"].of_class(0) == pytest.approx(0.0)

    def test_miss_rate_counts_the_steps_beyond_tolerance(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
        )

        # Mode 0 is 1 m out and within tolerance; mode 1 is 3 m out and beyond it.
        assert metrics["miss_rate"].of_class(0) == pytest.approx(0.5)

    def test_the_tolerance_is_configurable(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            miss_tolerance=0.5,
        )

        assert metrics["miss_rate"].of_class(0) == pytest.approx(1.0)

    def test_rejects_a_non_positive_tolerance(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            PathDisplacementSystem.on("/m", EST, GT, miss_tolerance=0.0)


class TestEdges:
    def test_an_unmatched_object_contributes_nothing(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            est_x=500.0,
        )

        assert np.isnan(metrics["ade"].of_class(0))
        assert metrics["ade"].support.values[0] == 1, "the ground truth still counted"

    def test_a_class_mismatch_contributes_nothing(self, labels: LabelRegistry) -> None:
        store = Store()
        store.log(
            GT,
            prediction(0.0, STRAIGHT_AHEAD, [1.0], labels, "car"),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        store.log(
            EST,
            prediction(0.05, STRAIGHT_AHEAD, [1.0], labels, "truck"),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0, class_agnostic=True)
        metric = PathDisplacementSystem.on(match.target, EST, GT)
        Pipeline([match, metric]).run(
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        ade = store.range(
            metric.targets[0],
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MetricValues)
        assert np.isnan(ade.of_class(labels.class_id("car")))

    def test_an_empty_store_yields_undefined_rows(self, labels: LabelRegistry) -> None:
        metric = PathDisplacementSystem.on("/matching/center_distance", EST, GT)
        store = Store()
        Pipeline([metric]).run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())

        for target in metric.targets:
            row = store.range(
                target,
                timeline=FRAME,
                time_range=TimeRange.everything(),
            ).materialize(MetricValues)
            assert np.isnan(row.value.values).all()

    def test_reports_a_row_for_every_registered_class(self, labels: LabelRegistry) -> None:
        metrics = displacement(labels, est_waypoints=STRAIGHT_AHEAD, est_confidences=[1.0])

        assert metrics["ade"].class_id.values.tolist() == [0, 1, 2]
