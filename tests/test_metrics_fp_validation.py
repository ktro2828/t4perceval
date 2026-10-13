"""FP validation: an annotated false positive passes when nothing matches it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from t4perceval import (
    FRAME,
    Detections3D,
    LabelRegistry,
    MetricValues,
    Store,
    TimePoint,
    TimeRange,
    Trackings3D,
)
from t4perceval.component import ALL_CLASSES
from t4perceval.system import (
    FpValidationSystem,
    Pipeline,
    PlaneDistanceMatchingSystem,
    SystemContext,
    fp_validation_by_instance,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

EST = "/estimation/objects"
GT = "/ground_truth/objects"

#: A car-sized footprint: width 2, length 4, height 1.5.
CAR = (2.0, 4.0, 1.5)

FP = "false_positive"


@pytest.fixture
def fp_labels() -> LabelRegistry:
    """A registry in which the false-positive annotation is an ordinary class."""
    return LabelRegistry.from_names(["car", "pedestrian", FP])


def boxes(
    store: Store,
    frame: int,
    path: str,
    objects: Sequence[tuple[float, float, str]],
    labels: LabelRegistry,
    *,
    instance_ids: Sequence[int] | None = None,
) -> None:
    """Log ``(x, y, class_name)`` car-sized boxes at one frame, tracked when ids are given."""
    count = len(objects)
    fields = {
        "position": [[x, y, 0.0] for x, y, _ in objects],
        "quaternion": [[0.0, 0.0, 0.0, 1.0]] * count,
        "size": [list(CAR)] * count,
        "class_id": labels.encode([name for _, _, name in objects]),
        "confidence": [0.9] * count,
    }
    archetype = (
        Detections3D(**fields)
        if instance_ids is None
        else Trackings3D(**fields, instance_id=list(instance_ids))
    )
    store.log(path, archetype, at=TimePoint.at(frame=frame), frame_id="base_link")


def fp_validation_of(
    store: Store,
    labels: LabelRegistry,
    *,
    threshold: float = 2.0,
    fp_label: str = FP,
    exempt: bool = True,
) -> MetricValues:
    """Score with the annotation class exempted from the class rule, or strictly."""
    match = PlaneDistanceMatchingSystem.between(
        EST,
        GT,
        threshold=threshold,
        agnostic_classes=[fp_label] if exempt else None,
    )
    metric = FpValidationSystem.on(match.target, EST, GT, fp_label=fp_label)
    ctx = SystemContext(store, FRAME, labels=labels)
    Pipeline([match, metric]).run(ctx, TimeRange.everything())

    return store.range(
        metric.target,
        timeline=FRAME,
        time_range=TimeRange.everything(),
    ).materialize(MetricValues)


class TestTargets:
    def test_writes_one_entity_named_by_the_metric(self) -> None:
        metric = FpValidationSystem.on("/matching", EST, GT)

        assert str(metric.target) == "/metrics/fp_validation"
        assert [str(t) for t in metric.targets] == ["/metrics/fp_validation"]

    def test_the_target_can_be_moved(self) -> None:
        metric = FpValidationSystem.on("/matching", EST, GT, target="/metrics/fp")

        assert str(metric.target) == "/metrics/fp"

    def test_defaults_to_the_conventional_class_name(self) -> None:
        metric = FpValidationSystem.on("/matching", EST, GT)

        assert metric.fp_label == "false_positive"

    def test_rejects_an_empty_class_name(self) -> None:
        with pytest.raises(ValueError, match="fp_label"):
            FpValidationSystem.on("/matching", EST, GT, fp_label="")


class TestValues:
    def test_an_estimation_on_the_annotation_is_a_fail(self, fp_labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, FP)], fp_labels)
        boxes(store, 0, EST, [(10.0, 0.0, "car")], fp_labels)

        metrics = fp_validation_of(store, fp_labels)

        assert metrics.of_class(fp_labels.class_id(FP)) == pytest.approx(0.0)
        assert metrics.support.values[fp_labels.class_id(FP)] == 1

    def test_an_annotation_nothing_claims_is_a_pass(self, fp_labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, FP)], fp_labels)
        boxes(store, 0, EST, [], fp_labels)

        metrics = fp_validation_of(store, fp_labels)

        assert metrics.of_class(fp_labels.class_id(FP)) == pytest.approx(1.0)

    def test_an_estimation_beyond_the_threshold_is_a_pass(self, fp_labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, FP)], fp_labels)
        boxes(store, 0, EST, [(16.0, 0.0, "car")], fp_labels)

        metrics = fp_validation_of(store, fp_labels, threshold=2.0)

        assert metrics.of_class(fp_labels.class_id(FP)) == pytest.approx(1.0)

    def test_any_estimation_class_can_fail_an_annotation(self, fp_labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, FP)], fp_labels)
        boxes(store, 0, EST, [(10.0, 0.0, "pedestrian")], fp_labels)

        metrics = fp_validation_of(store, fp_labels)

        assert metrics.of_class(fp_labels.class_id(FP)) == pytest.approx(0.0)

    def test_an_estimation_matching_nothing_is_ignored(self, fp_labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, FP)], fp_labels)
        boxes(store, 0, EST, [(80.0, 0.0, "car")], fp_labels)

        metrics = fp_validation_of(store, fp_labels)

        assert metrics.of_class(fp_labels.class_id(FP)) == pytest.approx(1.0)
        assert metrics.support.values[fp_labels.class_id(FP)] == 1

    def test_the_rate_is_the_passed_fraction(self, fp_labels: LabelRegistry) -> None:
        store = Store()
        xs = (0.0, 20.0, 40.0, 60.0)
        boxes(store, 0, GT, [(x, 0.0, FP) for x in xs], fp_labels)
        boxes(store, 0, EST, [(0.0, 0.0, "car")], fp_labels)

        metrics = fp_validation_of(store, fp_labels)

        assert metrics.of_class(fp_labels.class_id(FP)) == pytest.approx(0.75)
        assert metrics.support.values[fp_labels.class_id(FP)] == 4

    def test_every_frame_of_an_annotation_is_one_verdict(self, fp_labels: LabelRegistry) -> None:
        store = Store()
        for frame in range(3):
            boxes(store, frame, GT, [(10.0, 0.0, FP)], fp_labels)
        boxes(store, 0, EST, [(10.0, 0.0, "car")], fp_labels)
        boxes(store, 1, EST, [], fp_labels)
        boxes(store, 2, EST, [], fp_labels)

        metrics = fp_validation_of(store, fp_labels)

        assert metrics.of_class(fp_labels.class_id(FP)) == pytest.approx(2.0 / 3.0)
        assert metrics.support.values[fp_labels.class_id(FP)] == 3

    def test_reports_the_matching_threshold(self, fp_labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, FP)], fp_labels)
        boxes(store, 0, EST, [(10.5, 0.0, "car")], fp_labels)

        metrics = fp_validation_of(store, fp_labels, threshold=1.5)

        assert metrics.threshold.values[fp_labels.class_id(FP)] == pytest.approx(1.5)


class TestClassesAndEdges:
    def test_other_classes_carry_their_support_but_no_value(self, fp_labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, "car"), (50.0, 0.0, FP)], fp_labels)
        boxes(store, 0, EST, [(0.0, 0.0, "car")], fp_labels)

        metrics = fp_validation_of(store, fp_labels)

        car = fp_labels.class_id("car")
        assert np.isnan(metrics.of_class(car))
        assert metrics.support.values[car] == 1
        assert metrics.of_class(fp_labels.class_id(FP)) == pytest.approx(1.0)

    def test_an_ordinary_ground_truth_of_another_class_cannot_take_the_estimation(
        self, fp_labels: LabelRegistry
    ) -> None:
        """A car near a pedestrian and an annotation fails the annotation, not the pedestrian."""
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, "pedestrian"), (11.0, 0.0, FP)], fp_labels)
        boxes(store, 0, EST, [(10.2, 0.0, "car")], fp_labels)

        metrics = fp_validation_of(store, fp_labels)

        assert metrics.of_class(fp_labels.class_id(FP)) == pytest.approx(0.0)

    def test_a_normal_ground_truth_does_not_enter_the_pooled_row(
        self, fp_labels: LabelRegistry
    ) -> None:
        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, "car"), (50.0, 0.0, FP)], fp_labels)
        boxes(store, 0, EST, [(0.0, 0.0, "car"), (50.0, 0.0, "car")], fp_labels)

        metrics = fp_validation_of(store, fp_labels)

        assert metrics.aggregate == pytest.approx(0.0)
        assert metrics.support.values[-1] == 2

    def test_the_class_name_is_configurable(self) -> None:
        labels = LabelRegistry.from_names(["car", "ghost"])
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, "ghost")], labels)
        boxes(store, 0, EST, [(10.0, 0.0, "car")], labels)

        metrics = fp_validation_of(store, labels, fp_label="ghost")

        assert metrics.of_class(labels.class_id("ghost")) == pytest.approx(0.0)

    def test_a_registry_without_the_class_is_undefined(self, labels: LabelRegistry) -> None:
        """The matcher would refuse the unknown name; the metric alone reports NaN."""
        store = Store()
        boxes(store, 0, GT, [(10.0, 0.0, "car")], labels)
        boxes(store, 0, EST, [(10.0, 0.0, "car")], labels)

        metrics = fp_validation_of(store, labels, exempt=False)

        assert np.isnan(metrics.value.values).all()
        assert metrics.support.values.tolist() == [1, 0, 0, 1]

    def test_reports_a_row_for_every_registered_class(self, fp_labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, FP)], fp_labels)
        boxes(store, 0, EST, [(0.1, 0.0, "car")], fp_labels)

        metrics = fp_validation_of(store, fp_labels)

        assert metrics.class_id.values.tolist() == [0, 1, 2, ALL_CLASSES]

    def test_an_empty_store_yields_undefined_rows(self, fp_labels: LabelRegistry) -> None:
        metrics = fp_validation_of(Store(), fp_labels)

        assert np.isnan(metrics.value.values).all()
        assert metrics.support.values.tolist() == [0] * (len(fp_labels) + 1)

    def test_a_frame_with_no_objects_is_harmless(self, fp_labels: LabelRegistry) -> None:
        store = Store()
        boxes(store, 0, GT, [], fp_labels)
        boxes(store, 0, EST, [], fp_labels)
        boxes(store, 1, GT, [(1.0, 0.0, FP)], fp_labels)
        boxes(store, 1, EST, [(1.5, 0.0, "car")], fp_labels)

        metrics = fp_validation_of(store, fp_labels)

        assert metrics.of_class(fp_labels.class_id(FP)) == pytest.approx(0.0)


class TestAlongsideDetection:
    def test_detection_metrics_read_the_ground_truth_without_annotations(
        self, fp_labels: LabelRegistry
    ) -> None:
        """The recipe's mixed-dataset pipeline: AP skips the annotation class entirely."""
        from t4perceval.system import (
            ApplyMaskSystem,
            FilterByLabelSystem,
            average_precision_sweep,
        )

        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, "car"), (50.0, 0.0, FP)], fp_labels)
        boxes(store, 0, EST, [(0.2, 0.0, "car"), (50.0, 0.0, "car")], fp_labels)
        real = f"{GT}/real"
        systems = [
            FilterByLabelSystem.on(GT, name="real", exclude=[FP]),
            ApplyMaskSystem.of(GT, f"{GT}/filter/real", name="real"),
            *average_precision_sweep(EST, real, thresholds=[1.0]),
        ]
        Pipeline(systems).run(SystemContext(store, FRAME, labels=fp_labels), TimeRange.everything())

        ap = store.range(
            "/metrics/ap/0", timeline=FRAME, time_range=TimeRange.everything()
        ).materialize(MetricValues)
        m_ap = store.range(
            "/metrics/map", timeline=FRAME, time_range=TimeRange.everything()
        ).materialize(MetricValues)

        fp = fp_labels.class_id(FP)
        assert np.isnan(ap.of_class(fp))
        assert ap.support.values[fp] == 0
        assert m_ap.aggregate == pytest.approx(ap.of_class(fp_labels.class_id("car")))


class TestByInstance:
    @staticmethod
    def scene(labels: LabelRegistry) -> Store:
        """Two annotations: id 7 in three frames, id 8 in the last two only."""
        store = Store()
        boxes(store, 0, GT, [(0.0, 0.0, FP)], labels, instance_ids=[7])
        boxes(store, 1, GT, [(0.0, 0.0, FP), (50.0, 0.0, FP)], labels, instance_ids=[7, 8])
        boxes(store, 2, GT, [(0.0, 0.0, FP), (50.0, 0.0, FP)], labels, instance_ids=[7, 8])
        boxes(store, 0, EST, [(0.0, 0.0, "car")], labels)
        boxes(store, 1, EST, [(50.0, 0.0, "car")], labels)
        boxes(store, 2, EST, [(50.0, 0.0, "pedestrian")], labels)
        return store

    def test_counts_frames_passes_and_fails_per_instance(self, fp_labels: LabelRegistry) -> None:
        store = self.scene(fp_labels)
        fp_validation_of(store, fp_labels)

        verdicts = fp_validation_by_instance(
            store, "/matching/plane_distance", EST, GT, timeline=FRAME, labels=fp_labels
        )

        assert verdicts[7] == (3, 2, 1)
        assert verdicts[8] == (2, 0, 2)
        assert verdicts[7].pass_rate == pytest.approx(2.0 / 3.0)
        assert verdicts[8].pass_rate == pytest.approx(0.0)

    def test_agrees_with_the_pooled_metric(self, fp_labels: LabelRegistry) -> None:
        store = self.scene(fp_labels)
        metrics = fp_validation_of(store, fp_labels)

        verdicts = fp_validation_by_instance(
            store, "/matching/plane_distance", EST, GT, timeline=FRAME, labels=fp_labels
        )

        passes = sum(v.passes for v in verdicts.values())
        frames = sum(v.frames for v in verdicts.values())
        assert metrics.aggregate == pytest.approx(passes / frames)

    def test_a_range_narrows_the_frames(self, fp_labels: LabelRegistry) -> None:
        store = self.scene(fp_labels)
        fp_validation_of(store, fp_labels)

        verdicts = fp_validation_by_instance(
            store,
            "/matching/plane_distance",
            EST,
            GT,
            timeline=FRAME,
            labels=fp_labels,
            time_range=TimeRange(1, 2),
        )

        assert verdicts[7] == (2, 2, 0)
        assert verdicts[8] == (2, 0, 2)

    def test_an_unknown_class_is_rejected(self, fp_labels: LabelRegistry) -> None:
        store = self.scene(fp_labels)
        fp_validation_of(store, fp_labels)

        with pytest.raises(ValueError, match="fp_label"):
            fp_validation_by_instance(
                store,
                "/matching/plane_distance",
                EST,
                GT,
                timeline=FRAME,
                labels=fp_labels,
                fp_label="ghost",
            )

    def test_a_range_without_annotations_gives_no_instances(self, fp_labels: LabelRegistry) -> None:
        """Estimations alone leave FP rows in the matching, which the task ignores."""
        store = Store()
        boxes(store, 0, EST, [(10.0, 0.0, "car")], fp_labels)
        fp_validation_of(store, fp_labels)

        assert (
            fp_validation_by_instance(
                store, "/matching/plane_distance", EST, GT, timeline=FRAME, labels=fp_labels
            )
            == {}
        )

    def test_an_empty_matching_gives_no_instances(self, fp_labels: LabelRegistry) -> None:
        store = Store()
        fp_validation_of(store, fp_labels)

        assert (
            fp_validation_by_instance(
                store, "/matching/plane_distance", EST, GT, timeline=FRAME, labels=fp_labels
            )
            == {}
        )
