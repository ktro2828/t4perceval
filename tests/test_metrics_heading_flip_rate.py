"""Heading flip rate: the fraction of true positives facing the wrong way."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import yaw

from t4perceval import (
    FRAME,
    Detections3D,
    LabelRegistry,
    MetricValues,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.component import ALL_CLASSES
from t4perceval.system import (
    CenterDistanceMatchingSystem,
    HeadingFlipRateSystem,
    Pipeline,
    SystemContext,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

EST = "/estimation/objects"
GT = "/ground_truth/objects"

#: A car-sized footprint: width 2, length 4, height 1.5.
CAR = (2.0, 4.0, 1.5)


def boxes(
    store: Store,
    frame: int,
    path: str,
    objects: Sequence[tuple[float, float, float, str]],
    labels: LabelRegistry,
) -> None:
    """Log ``(x, y, yaw_degrees, class_name)`` car-sized boxes at one frame."""
    count = len(objects)
    store.log(
        path,
        Detections3D(
            position=[[x, y, 0.0] for x, y, _, _ in objects],
            quaternion=[yaw(degrees) for _, _, degrees, _ in objects],
            size=[list(CAR)] * count,
            class_id=labels.encode([name for _, _, _, name in objects]),
            confidence=[0.9] * count,
        ),
        at=TimePoint.at(frame=frame),
        frame_id="base_link",
    )


def heading_flip_of(
    store: Store,
    labels: LabelRegistry,
    *,
    threshold: float = 2.0,
    flip_threshold: float = np.pi / 2.0,
) -> MetricValues:
    match = CenterDistanceMatchingSystem.between(EST, GT, threshold=threshold)
    metric = HeadingFlipRateSystem.on(match.target, EST, GT, flip_threshold=flip_threshold)
    ctx = SystemContext(store, FRAME, labels=labels)
    Pipeline([match, metric]).run(ctx, TimeRange.everything())

    return store.range(
        metric.target,
        timeline=FRAME,
        time_range=TimeRange.everything(),
    ).materialize(MetricValues)


class TestTargets:
    def test_writes_one_entity_named_by_the_metric(self) -> None:
        metric = HeadingFlipRateSystem.on("/matching", EST, GT)

        assert str(metric.target) == "/metrics/heading_flip_rate"
        assert [str(t) for t in metric.targets] == ["/metrics/heading_flip_rate"]

    def test_the_target_can_be_moved(self) -> None:
        metric = HeadingFlipRateSystem.on("/matching", EST, GT, target="/metrics/flips")

        assert str(metric.target) == "/metrics/flips"

    def test_defaults_to_a_quarter_turn(self) -> None:
        metric = HeadingFlipRateSystem.on("/matching", EST, GT)

        assert metric.flip_threshold == pytest.approx(np.pi / 2.0)

    @pytest.mark.parametrize("flip_threshold", [0.0, -0.1, np.pi + 1e-6])
    def test_rejects_a_threshold_outside_the_half_turn(self, flip_threshold: float) -> None:
        with pytest.raises(ValueError, match=r"\(0, pi\]"):
            HeadingFlipRateSystem.on("/matching", EST, GT, flip_threshold=flip_threshold)


class TestValues:
    def test_a_small_error_is_not_a_flip(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, 0.0, "car")], labels)
        boxes(store, 0, EST, [(10.0, 0.0, 10.0, "car")], labels)

        metrics = heading_flip_of(store, labels)

        assert metrics.of_class(0) == pytest.approx(0.0)

    def test_a_half_turn_is_a_flip(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, 0.0, "car")], labels)
        boxes(store, 0, EST, [(10.0, 0.0, 180.0, "car")], labels)

        metrics = heading_flip_of(store, labels)

        assert metrics.of_class(0) == pytest.approx(1.0)

    def test_the_threshold_is_configurable(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, 0.0, "car")], labels)
        boxes(store, 0, EST, [(10.0, 0.0, 10.0, "car")], labels)

        metrics = heading_flip_of(store, labels, flip_threshold=np.radians(5.0))

        assert metrics.of_class(0) == pytest.approx(1.0)

    # The quaternion round trip lands a few ULPs either side of pi/2 depending on the
    # headings: the first pair comes back exactly on it, the other three a hair above.
    @pytest.mark.parametrize(
        ("gt_degrees", "est_degrees"),
        [(0.0, 90.0), (68.3, 158.3), (-77.8, -167.8), (-176.1, -266.1)],
    )
    def test_an_error_exactly_at_the_threshold_is_not_a_flip(
        self, labels: LabelRegistry, gt_degrees: float, est_degrees: float
    ) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, gt_degrees, "car")], labels)
        boxes(store, 0, EST, [(10.0, 0.0, est_degrees, "car")], labels)

        metrics = heading_flip_of(store, labels, flip_threshold=np.pi / 2.0)

        assert metrics.of_class(0) == pytest.approx(0.0)

    def test_an_error_just_above_the_threshold_is_a_flip(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, 0.0, "car")], labels)
        boxes(store, 0, EST, [(10.0, 0.0, 90.0 + 1e-4, "car")], labels)

        metrics = heading_flip_of(store, labels, flip_threshold=np.pi / 2.0)

        assert metrics.of_class(0) == pytest.approx(1.0)

    def test_the_rate_is_the_flipped_fraction(self, labels: LabelRegistry) -> None:
        store = Store()
        xs = (0.0, 20.0, 40.0, 60.0)
        boxes(store, 0, GT, [(x, 0.0, 0.0, "car") for x in xs], labels)
        boxes(
            store,
            0,
            EST,
            [(0.0, 0.0, 180.0, "car"), *((x, 0.0, 5.0, "car") for x in xs[1:])],
            labels,
        )

        metrics = heading_flip_of(store, labels)

        assert metrics.of_class(0) == pytest.approx(0.25)

    def test_wrapping_across_pi_is_not_a_flip(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, 179.0, "car")], labels)
        boxes(store, 0, EST, [(10.0, 0.0, -179.0, "car")], labels)

        metrics = heading_flip_of(store, labels)

        assert metrics.of_class(0) == pytest.approx(0.0)

    def test_reports_the_matching_threshold(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, 0.0, "car")], labels)
        boxes(store, 0, EST, [(10.5, 0.0, 0.0, "car")], labels)

        metrics = heading_flip_of(store, labels, threshold=1.5)

        assert metrics.threshold.values[0] == pytest.approx(1.5)


class TestClassesAndEdges:
    def test_classes_are_scored_independently(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, 0.0, "car"), (50.0, 0.0, 0.0, "truck")], labels)
        boxes(store, 0, EST, [(0.5, 0.0, 180.0, "car"), (51.0, 0.0, 0.0, "truck")], labels)

        metrics = heading_flip_of(store, labels)

        assert metrics.of_class(labels.class_id("car")) == pytest.approx(1.0)
        assert metrics.of_class(labels.class_id("truck")) == pytest.approx(0.0)

    def test_a_class_without_hits_is_undefined_with_its_support(
        self, labels: LabelRegistry
    ) -> None:
        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, 0.0, "car")], labels)
        boxes(store, 0, EST, [(500.0, 0.0, 0.0, "car")], labels)

        metrics = heading_flip_of(store, labels)

        assert np.isnan(metrics.of_class(0))
        assert metrics.support.values[0] == 1

    def test_a_misclassified_hit_does_not_count(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, 0.0, "car")], labels)
        boxes(store, 0, EST, [(0.5, 0.0, 180.0, "truck")], labels)

        metrics = heading_flip_of(store, labels)

        assert np.isnan(metrics.of_class(labels.class_id("car")))
        assert np.isnan(metrics.of_class(labels.class_id("truck")))

    def test_reports_a_row_for_every_registered_class(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, 0.0, "car")], labels)
        boxes(store, 0, EST, [(0.1, 0.0, 0.0, "car")], labels)

        metrics = heading_flip_of(store, labels)

        assert metrics.class_id.values.tolist() == [0, 1, 2, ALL_CLASSES]

    def test_an_empty_store_yields_undefined_rows(self, labels: LabelRegistry) -> None:
        metrics = heading_flip_of(Store(), labels)

        assert np.isnan(metrics.value.values).all()
        assert metrics.support.values.tolist() == [0] * (len(labels) + 1)

    def test_a_frame_with_no_objects_is_harmless(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [], labels)
        boxes(store, 0, EST, [], labels)
        boxes(store, 1, GT, [(1.0, 0.0, 0.0, "car")], labels)
        boxes(store, 1, EST, [(2.0, 0.0, 180.0, "car")], labels)

        metrics = heading_flip_of(store, labels)

        assert metrics.of_class(0) == pytest.approx(1.0)


class TestAllClasses:
    def test_pools_every_true_positive_rather_than_averaging_classes(
        self,
        labels: LabelRegistry,
    ) -> None:
        """Three cars facing the right way and one reversed truck: 1/4, not (0 + 1) / 2."""
        store = Store()
        ground_truth = [(0.0, 0.0, 0.0, "car"), (10.0, 0.0, 0.0, "car"), (20.0, 0.0, 0.0, "car")]
        boxes(store, 0, GT, [*ground_truth, (50.0, 0.0, 0.0, "truck")], labels)
        boxes(store, 0, EST, [*ground_truth, (50.0, 0.0, 180.0, "truck")], labels)

        metrics = heading_flip_of(store, labels)

        assert metrics.of_class(labels.class_id("truck")) == pytest.approx(1.0)
        assert metrics.aggregate == pytest.approx(0.25)
        assert metrics.support.values[-1] == 4
