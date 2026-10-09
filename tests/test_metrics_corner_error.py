"""Corner displacement error: mean, percentiles and max over true positives."""

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
    CornerErrorSystem,
    Pipeline,
    SystemContext,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from t4perceval.core.entity import EntityPath

EST = "/estimation/objects"
GT = "/ground_truth/objects"

#: A car-sized footprint: width 2, length 4, height 1.5.
CAR = (2.0, 4.0, 1.5)


def boxes(
    store: Store,
    frame: int,
    path: str,
    objects: Sequence[tuple[float, float, float, Sequence[float], str]],
    labels: LabelRegistry,
) -> None:
    """Log ``(x, y, yaw_degrees, (width, length, height), class_name)`` boxes at one frame."""
    count = len(objects)
    store.log(
        path,
        Detections3D(
            position=[[x, y, 0.0] for x, y, _, _, _ in objects],
            quaternion=[yaw(degrees) for _, _, degrees, _, _ in objects],
            size=[list(size) for _, _, _, size, _ in objects],
            class_id=labels.encode([name for _, _, _, _, name in objects]),
            confidence=[0.9] * count,
        ),
        at=TimePoint.at(frame=frame),
        frame_id="base_link",
    )


def corner_error_of(
    store: Store,
    labels: LabelRegistry,
    *,
    threshold: float = 2.0,
    percentiles: tuple[float, ...] = (95.0,),
) -> dict[str, MetricValues]:
    match = CenterDistanceMatchingSystem.between(EST, GT, threshold=threshold)
    metric = CornerErrorSystem.on(match.target, EST, GT, percentiles=percentiles)
    ctx = SystemContext(store, FRAME, labels=labels)
    Pipeline([match, metric]).run(ctx, TimeRange.everything())

    def read(target: EntityPath) -> MetricValues:
        return store.range(
            target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MetricValues)

    return {target.name: read(target) for target in metric.targets}


class TestTargets:
    def test_writes_one_entity_per_statistic(self) -> None:
        metric = CornerErrorSystem.on("/matching", EST, GT)

        assert [str(t) for t in metric.targets] == [
            "/metrics/corner_error/mean",
            "/metrics/corner_error/max",
            "/metrics/corner_error/p95",
        ]

    def test_one_entity_per_requested_percentile(self) -> None:
        metric = CornerErrorSystem.on("/matching", EST, GT, percentiles=(50.0, 97.5))

        assert [t.name for t in metric.targets] == ["mean", "max", "p50", "p97.5"]

    def test_no_percentiles_leaves_mean_and_max(self) -> None:
        metric = CornerErrorSystem.on("/matching", EST, GT, percentiles=())

        assert [t.name for t in metric.targets] == ["mean", "max"]

    def test_the_root_moves_with_target(self) -> None:
        metric = CornerErrorSystem.on("/matching", EST, GT, target="/metrics/corners_bev")

        assert str(metric.targets[0]) == "/metrics/corners_bev/mean"

    def test_the_pipeline_sees_every_target(self) -> None:
        metric = CornerErrorSystem.on("/m", EST, GT)
        reader = CornerErrorSystem.on(str(metric.targets[-1]), EST, GT, target="/metrics/second")

        with pytest.raises(ValueError, match="before a later system writes it"):
            Pipeline([reader, metric])

    @pytest.mark.parametrize("percentiles", [(101.0,), (-1.0,), (50.0, 100.5)])
    def test_rejects_percentiles_outside_the_unit_range(
        self, percentiles: tuple[float, ...]
    ) -> None:
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            CornerErrorSystem.on("/matching", EST, GT, percentiles=percentiles)

    @pytest.mark.parametrize("percentiles", [(95.0, 95.0), (95.0, 95.0000001)])
    def test_rejects_percentiles_that_share_a_path(self, percentiles: tuple[float, ...]) -> None:
        with pytest.raises(ValueError, match="distinct paths"):
            CornerErrorSystem.on("/matching", EST, GT, percentiles=percentiles)


class TestValues:
    def test_a_pure_translation_is_exact(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, 0.0, CAR, "car")], labels)
        boxes(store, 0, EST, [(11.0, 0.0, 0.0, CAR, "car")], labels)

        metrics = corner_error_of(store, labels, percentiles=(50.0, 95.0))

        for name in ("mean", "max", "p50", "p95"):
            assert metrics[name].of_class(0) == pytest.approx(1.0), name

    def test_a_yaw_error_moves_the_corners(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, 0.0, CAR, "car")], labels)
        boxes(store, 0, EST, [(10.0, 0.0, 10.0, CAR, "car")], labels)

        metrics = corner_error_of(store, labels)

        mean = metrics["mean"].of_class(0)
        assert mean > 0.0
        assert metrics["max"].of_class(0) >= mean
        assert np.isfinite(metrics["p95"].of_class(0))

    def test_a_half_turn_is_the_same_footprint(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, 0.0, CAR, "car")], labels)
        boxes(store, 0, EST, [(10.0, 0.0, 180.0, CAR, "car")], labels)

        metrics = corner_error_of(store, labels)

        assert metrics["mean"].of_class(0) == pytest.approx(0.0, abs=1e-9)

    def test_a_quarter_turn_of_a_square_is_the_same_footprint(self, labels: LabelRegistry) -> None:
        square = (2.0, 2.0, 1.5)
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, 0.0, square, "car")], labels)
        boxes(store, 0, EST, [(10.0, 0.0, 90.0, square, "car")], labels)

        metrics = corner_error_of(store, labels)

        assert metrics["mean"].of_class(0) == pytest.approx(0.0, abs=1e-9)

    def test_size_inflation_is_seen_where_centre_distance_is_blind(
        self, labels: LabelRegistry
    ) -> None:
        # Two metres longer: every corner moves one metre along x.
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, 0.0, CAR, "car")], labels)
        boxes(store, 0, EST, [(10.0, 0.0, 0.0, (2.0, 6.0, 1.5), "car")], labels)

        metrics = corner_error_of(store, labels)

        assert metrics["mean"].of_class(0) == pytest.approx(1.0)
        assert metrics["max"].of_class(0) == pytest.approx(1.0)

    def test_statistics_summarise_all_true_positives(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, 0.0, CAR, "car"), (20.0, 0.0, 0.0, CAR, "car")], labels)
        boxes(
            store,
            0,
            EST,
            [(0.5, 0.0, 0.0, CAR, "car"), (21.5, 0.0, 0.0, CAR, "car")],
            labels,
        )

        metrics = corner_error_of(store, labels, percentiles=(50.0,))

        assert metrics["mean"].of_class(0) == pytest.approx(1.0)
        assert metrics["p50"].of_class(0) == pytest.approx(1.0)
        assert metrics["max"].of_class(0) == pytest.approx(1.5)

    def test_reports_the_matching_threshold(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, 0.0, CAR, "car")], labels)
        boxes(store, 0, EST, [(10.5, 0.0, 0.0, CAR, "car")], labels)

        metrics = corner_error_of(store, labels, threshold=1.5)

        assert metrics["mean"].threshold.values[0] == pytest.approx(1.5)


class TestClassesAndEdges:
    def test_classes_are_scored_independently(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(
            store,
            0,
            GT,
            [(0.0, 0.0, 0.0, CAR, "car"), (50.0, 0.0, 0.0, CAR, "truck")],
            labels,
        )
        boxes(
            store,
            0,
            EST,
            [(0.5, 0.0, 0.0, CAR, "car"), (51.0, 0.0, 0.0, CAR, "truck")],
            labels,
        )

        metrics = corner_error_of(store, labels)

        assert metrics["mean"].of_class(labels.class_id("car")) == pytest.approx(0.5)
        assert metrics["mean"].of_class(labels.class_id("truck")) == pytest.approx(1.0)

    def test_a_class_without_hits_is_undefined_with_its_support(
        self, labels: LabelRegistry
    ) -> None:
        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, 0.0, CAR, "car")], labels)
        boxes(store, 0, EST, [(500.0, 0.0, 0.0, CAR, "car")], labels)

        metrics = corner_error_of(store, labels)

        for name, row in metrics.items():
            assert np.isnan(row.of_class(0)), name
        assert metrics["mean"].support.values[0] == 1

    def test_a_misclassified_hit_does_not_count(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, 0.0, CAR, "car")], labels)
        boxes(store, 0, EST, [(0.5, 0.0, 0.0, CAR, "truck")], labels)

        metrics = corner_error_of(store, labels)

        assert np.isnan(metrics["mean"].of_class(labels.class_id("car")))
        assert np.isnan(metrics["mean"].of_class(labels.class_id("truck")))

    def test_reports_a_row_for_every_registered_class(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, 0.0, CAR, "car")], labels)
        boxes(store, 0, EST, [(0.1, 0.0, 0.0, CAR, "car")], labels)

        metrics = corner_error_of(store, labels)

        for row in metrics.values():
            assert row.class_id.values.tolist() == [0, 1, 2, ALL_CLASSES]

    def test_an_empty_store_yields_undefined_rows(self, labels: LabelRegistry) -> None:
        metrics = corner_error_of(Store(), labels)

        for name, row in metrics.items():
            assert np.isnan(row.value.values).all(), name
            assert row.support.values.tolist() == [0] * (len(labels) + 1)

    def test_a_frame_with_no_objects_is_harmless(self, labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [], labels)
        boxes(store, 0, EST, [], labels)
        boxes(store, 1, GT, [(1.0, 0.0, 0.0, CAR, "car")], labels)
        boxes(store, 1, EST, [(2.0, 0.0, 0.0, CAR, "car")], labels)

        metrics = corner_error_of(store, labels)

        assert metrics["mean"].of_class(0) == pytest.approx(1.0)


class TestAllClasses:
    def test_pools_every_true_positive_rather_than_averaging_classes(
        self,
        labels: LabelRegistry,
    ) -> None:
        """Two cars 0.2 m off and a truck 1.0 m off: a mean of 1.4 / 3, not 0.6."""
        store = Store()
        boxes(
            store,
            0,
            GT,
            [
                (0.0, 0.0, 0.0, CAR, "car"),
                (10.0, 0.0, 0.0, CAR, "car"),
                (50.0, 0.0, 0.0, CAR, "truck"),
            ],
            labels,
        )
        boxes(
            store,
            0,
            EST,
            [
                (0.2, 0.0, 0.0, CAR, "car"),
                (10.2, 0.0, 0.0, CAR, "car"),
                (51.0, 0.0, 0.0, CAR, "truck"),
            ],
            labels,
        )

        metrics = corner_error_of(store, labels)

        assert metrics["mean"].aggregate == pytest.approx(1.4 / 3)
        assert metrics["max"].aggregate == pytest.approx(1.0)
        assert metrics["mean"].support.values[-1] == 3
