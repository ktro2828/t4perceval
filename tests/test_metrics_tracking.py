"""Tracking metrics: MOTA, MOTP and ID switches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import make_tracking

from t4perceval import FRAME, BatchMetric, LabelRegistry, Store, TimePoint, TimeRange
from t4perceval.system import (
    CenterDistanceMatchingSystem,
    ClearSystem,
    IoUBEVMatchingSystem,
    Pipeline,
    SystemContext,
)

if TYPE_CHECKING:
    from t4perceval.core.entity import EntityPath

EST = "/estimation/objects"
GT = "/ground_truth/objects"


def tracks(
    store: Store,
    frame: int,
    path: str,
    objects: list[tuple[float, int, str]],
    labels: LabelRegistry,
) -> None:
    """Log ``(x, instance_id, class_name)`` triples at one frame."""
    store.log(
        path,
        make_tracking(
            [[x, 0.0, 0.0] for x, _, _ in objects],
            [instance for _, instance, _ in objects],
            labels.encode([name for _, _, name in objects]),
        ),
        at=TimePoint.at(frame=frame),
        frame_id="base_link",
    )


def clear_of(
    store: Store,
    labels: LabelRegistry,
    *,
    threshold: float = 1.0,
    matcher: type = CenterDistanceMatchingSystem,
) -> dict[str, BatchMetric]:
    match = matcher.between(EST, GT, threshold=threshold)
    clear = ClearSystem.on(match.target, EST, GT)
    ctx = SystemContext(store, FRAME, labels=labels)
    Pipeline([match, clear]).run(ctx, TimeRange.everything())

    def read(target: EntityPath) -> BatchMetric:
        return store.range(
            target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(BatchMetric)

    return {target.name: read(target) for target in clear.targets}


class TestTargets:
    def test_writes_one_entity_per_metric(self) -> None:
        clear = ClearSystem.on("/matching/center_distance", EST, GT)

        assert [str(target) for target in clear.targets] == [
            "/metrics/clear/mota",
            "/metrics/clear/motp",
            "/metrics/clear/id_switch",
        ]

    def test_the_family_root_can_be_moved(self) -> None:
        clear = ClearSystem.on("/m", EST, GT, target="/metrics/clear_bev")

        assert str(clear.targets[0]) == "/metrics/clear_bev/mota"

    def test_the_pipeline_sees_every_target(self) -> None:
        clear = ClearSystem.on("/m", EST, GT)
        reader = ClearSystem.on(str(clear.targets[0]), EST, GT, target="/metrics/second")

        with pytest.raises(ValueError, match="before a later system writes it"):
            Pipeline([reader, clear])


class TestIdSwitch:
    def test_a_handover_between_frames_is_a_switch(self, labels: LabelRegistry) -> None:
        """The same ground truth is held by estimation 1, then by estimation 2."""
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car")], labels)
        tracks(store, 0, EST, [(0.1, 1, "car")], labels)
        tracks(store, 1, GT, [(1.0, 100, "car")], labels)
        tracks(store, 1, EST, [(1.1, 2, "car")], labels)

        metrics = clear_of(store, labels)

        assert metrics["id_switch"].of_class(0) == pytest.approx(1.0)

    def test_a_stable_identity_is_not_a_switch(self, labels: LabelRegistry) -> None:
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car")], labels)
        tracks(store, 0, EST, [(0.1, 1, "car")], labels)
        tracks(store, 1, GT, [(1.0, 100, "car")], labels)
        tracks(store, 1, EST, [(1.1, 1, "car")], labels)

        metrics = clear_of(store, labels)

        assert metrics["id_switch"].of_class(0) == pytest.approx(0.0)

    def test_only_the_previous_frame_is_compared(self, labels: LabelRegistry) -> None:
        """An identity lost and later recovered by its original holder is not a switch."""
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car")], labels)
        tracks(store, 0, EST, [(0.1, 1, "car")], labels)
        tracks(store, 1, GT, [(1.0, 100, "car")], labels)
        tracks(store, 1, EST, [(500.0, 1, "car")], labels)  # missed
        tracks(store, 2, GT, [(2.0, 100, "car")], labels)
        tracks(store, 2, EST, [(2.1, 1, "car")], labels)  # recovered by the same one

        metrics = clear_of(store, labels)

        assert metrics["id_switch"].of_class(0) == pytest.approx(0.0)

    def test_each_ground_truth_is_tracked_separately(self, labels: LabelRegistry) -> None:
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car"), (10.0, 200, "car")], labels)
        tracks(store, 0, EST, [(0.1, 1, "car"), (10.1, 2, "car")], labels)
        tracks(store, 1, GT, [(1.0, 100, "car"), (11.0, 200, "car")], labels)
        # Only the second identity changes hands.
        tracks(store, 1, EST, [(1.1, 1, "car"), (11.1, 3, "car")], labels)

        metrics = clear_of(store, labels)

        assert metrics["id_switch"].of_class(0) == pytest.approx(1.0)

    def test_a_single_frame_scene_has_no_switches(self, labels: LabelRegistry) -> None:
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car")], labels)
        tracks(store, 0, EST, [(0.1, 1, "car")], labels)

        metrics = clear_of(store, labels)

        assert metrics["id_switch"].of_class(0) == pytest.approx(0.0)


class TestMota:
    def test_counts_hits_against_misses_and_switches(self, labels: LabelRegistry) -> None:
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car")], labels)
        tracks(store, 0, EST, [(0.1, 1, "car")], labels)
        tracks(store, 1, GT, [(1.0, 100, "car")], labels)
        tracks(store, 1, EST, [(1.1, 2, "car")], labels)

        metrics = clear_of(store, labels)

        # Two hits, no false positives, one switch, two ground truths.
        assert metrics["mota"].of_class(0) == pytest.approx((2 - 0 - 1) / 2)

    def test_a_perfect_tracker_scores_one(self, labels: LabelRegistry) -> None:
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car")], labels)
        tracks(store, 0, EST, [(0.1, 1, "car")], labels)
        tracks(store, 1, GT, [(1.0, 100, "car")], labels)
        tracks(store, 1, EST, [(1.1, 1, "car")], labels)

        metrics = clear_of(store, labels)

        assert metrics["mota"].of_class(0) == pytest.approx(1.0)

    def test_false_positives_subtract(self, labels: LabelRegistry) -> None:
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car")], labels)
        tracks(store, 0, EST, [(0.1, 1, "car"), (500.0, 2, "car")], labels)

        metrics = clear_of(store, labels)

        assert metrics["mota"].of_class(0) == pytest.approx(0.0), "(1 - 1 - 0) / 1"

    def test_it_never_goes_negative(self, labels: LabelRegistry) -> None:
        """More wrong claims than there are objects would drive the raw score below zero."""
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car")], labels)
        tracks(
            store,
            0,
            EST,
            [(0.1, 1, "car"), (500.0, 2, "car"), (600.0, 3, "car"), (700.0, 4, "car")],
            labels,
        )

        metrics = clear_of(store, labels)

        assert metrics["mota"].of_class(0) == pytest.approx(0.0)

    def test_no_ground_truth_leaves_it_undefined(self, labels: LabelRegistry) -> None:
        store = Store()
        tracks(store, 0, GT, [], labels)
        tracks(store, 0, EST, [(0.1, 1, "car")], labels)

        metrics = clear_of(store, labels)

        row = metrics["mota"]
        assert np.isnan(row.of_class(0))
        assert row.support.values[0] == 0


class TestMotp:
    def test_is_the_mean_matching_score_of_the_hits(self, labels: LabelRegistry) -> None:
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car"), (10.0, 200, "car")], labels)
        # Distances of 0.2 and 0.4.
        tracks(store, 0, EST, [(0.2, 1, "car"), (10.4, 2, "car")], labels)

        metrics = clear_of(store, labels)

        assert metrics["motp"].of_class(0) == pytest.approx(0.3)

    def test_no_hits_leaves_it_undefined(self, labels: LabelRegistry) -> None:
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car")], labels)
        tracks(store, 0, EST, [(500.0, 1, "car")], labels)

        metrics = clear_of(store, labels)

        assert np.isnan(metrics["motp"].of_class(0))

    def test_it_inherits_the_direction_of_its_matching_mode(self, labels: LabelRegistry) -> None:
        """MOTP is whatever the matching measured, so its direction is not intrinsic.

        The same pair of boxes gives 0.2 under a distance mode -- where smaller is better --
        and 0.667 under an overlap mode, where larger is. Only the matching entity the
        metric read says which way to read the number.
        """
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car")], labels)
        tracks(store, 0, EST, [(0.2, 1, "car")], labels)

        by_distance = clear_of(store, labels)["motp"].of_class(0)

        overlap_store = Store()
        tracks(overlap_store, 0, GT, [(0.0, 100, "car")], labels)
        tracks(overlap_store, 0, EST, [(0.2, 1, "car")], labels)
        by_overlap = clear_of(
            overlap_store,
            labels,
            threshold=0.5,
            matcher=IoUBEVMatchingSystem,
        )["motp"].of_class(0)

        assert by_distance == pytest.approx(0.2), "metres, lower is better"
        # 1x1 footprints offset by 0.2: intersection 0.8, union 1 + 1 - 0.8 = 1.2.
        assert by_overlap == pytest.approx(0.8 / 1.2), "a ratio, higher is better"


class TestClassesAndEdges:
    def test_classes_are_scored_independently(self, labels: LabelRegistry) -> None:
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car"), (50.0, 200, "truck")], labels)
        tracks(store, 0, EST, [(0.1, 1, "car"), (500.0, 2, "truck")], labels)

        metrics = clear_of(store, labels)

        assert metrics["mota"].of_class(labels.class_id("car")) == pytest.approx(1.0)
        assert metrics["mota"].of_class(labels.class_id("truck")) == pytest.approx(0.0)

    def test_reports_a_row_for_every_registered_class(self, labels: LabelRegistry) -> None:
        store = Store()
        tracks(store, 0, GT, [(0.0, 100, "car")], labels)
        tracks(store, 0, EST, [(0.1, 1, "car")], labels)

        metrics = clear_of(store, labels)

        assert metrics["mota"].class_id.values.tolist() == [0, 1, 2]

    def test_an_empty_store_yields_undefined_rows(self, labels: LabelRegistry) -> None:
        metrics = clear_of(Store(), labels)

        for name, row in metrics.items():
            assert np.isnan(row.value.values).all(), name
            assert row.support.values.tolist() == [0] * len(labels)

    def test_a_frame_with_no_objects_is_harmless(self, labels: LabelRegistry) -> None:
        store = Store()
        tracks(store, 0, GT, [], labels)
        tracks(store, 0, EST, [], labels)
        tracks(store, 1, GT, [(1.0, 100, "car")], labels)
        tracks(store, 1, EST, [(1.1, 1, "car")], labels)

        metrics = clear_of(store, labels)

        assert metrics["mota"].of_class(0) == pytest.approx(1.0)
