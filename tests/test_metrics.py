"""Properties every metric shares, so a new one cannot drift from the family."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from t4perceval import (
    FRAME,
    BatchMetric,
    Chunk,
    LabelRegistry,
    Store,
    TimeColumn,
    TimePoint,
    TimeRange,
)
from t4perceval.archetype import BatchMatchResult
from t4perceval.archetype import BatchMetric as ArchetypeBatchMetric
from t4perceval.component import ALL_CLASSES, BatchMetricValue, MatchStatus
from t4perceval.descriptors import CLASS_ID, METRIC_VALUE, SUPPORT, THRESHOLD
from t4perceval.io import chunk_from_table, chunk_to_table, read_parquet, write_parquet
from t4perceval.label import UNKNOWN_CLASS_ID
from t4perceval.system import (
    AveragePrecisionHeadingSystem,
    AveragePrecisionSystem,
    CenterDistanceMatchingSystem,
    ClassificationSystem,
    ClearSystem,
    MetricSystem,
    PathDisplacementSystem,
    Pipeline,
    SystemContext,
)
from t4perceval.system.metric import nan_mean

if TYPE_CHECKING:
    from pathlib import Path

EST = "/estimation/objects"
GT = "/ground_truth/objects"
MATCHING = "/matching/center_distance"

#: Every metric system that reads a matching result and reports per class.
ALL_METRICS: tuple[type[MetricSystem], ...] = (
    AveragePrecisionSystem,
    AveragePrecisionHeadingSystem,
    ClearSystem,
    PathDisplacementSystem,
    ClassificationSystem,
)


@pytest.fixture
def tracked_scene(labels: LabelRegistry) -> Store:
    """A two-frame scene whose entities carry every column any metric needs."""
    from conftest import make_prediction

    store = Store()
    for frame, gt_x, est_x in ((0, 0.0, 0.1), (1, 5.0, 5.1)):
        store.log(
            GT,
            make_prediction([[gt_x, 0.0, 0.0]], [100]),
            at=TimePoint.at(frame=frame),
            frame_id="base_link",
        )
        store.log(
            EST,
            make_prediction([[est_x, 0.0, 0.0]], [1]),
            at=TimePoint.at(frame=frame),
            frame_id="base_link",
        )
    return store


def outputs(
    metric: MetricSystem,
    store: Store,
    labels: LabelRegistry,
) -> list[BatchMetric]:
    match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
    Pipeline([match, metric]).run(
        SystemContext(store, FRAME, labels=labels),
        TimeRange.everything(),
    )
    return [
        store.range(target, timeline=FRAME, time_range=TimeRange.everything()).materialize(
            BatchMetric,
        )
        for target in metric.targets
    ]


class TestUniformSchema:
    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_every_metric_writes_the_same_columns(self, metric: type[MetricSystem]) -> None:
        assert metric.PROVIDES == (CLASS_ID, THRESHOLD, METRIC_VALUE, SUPPORT)

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_every_metric_names_itself_by_path(self, metric: type[MetricSystem]) -> None:
        system = metric.on(MATCHING, EST, GT)

        for target in system.targets:
            assert str(target).startswith("/metrics/")

    def test_no_two_metrics_share_a_target(self) -> None:
        targets = [
            str(target) for metric in ALL_METRICS for target in metric.on(MATCHING, EST, GT).targets
        ]

        assert len(set(targets)) == len(targets)

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_every_metric_reports_on_every_registered_class(
        self,
        metric: type[MetricSystem],
        tracked_scene: Store,
        labels: LabelRegistry,
    ) -> None:
        for result in outputs(metric.on(MATCHING, EST, GT), tracked_scene, labels):
            assert result.class_id.values.tolist() == [0, 1, 2]

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_support_counts_the_ground_truths_behind_the_value(
        self,
        metric: type[MetricSystem],
        tracked_scene: Store,
        labels: LabelRegistry,
    ) -> None:
        for result in outputs(metric.on(MATCHING, EST, GT), tracked_scene, labels):
            assert result.support.values.tolist() == [2, 0, 0], "two frames of one car"

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_a_class_that_was_never_seen_reports_nothing(
        self,
        metric: type[MetricSystem],
        tracked_scene: Store,
        labels: LabelRegistry,
    ) -> None:
        """A class absent from both entities says so, rather than scoring zero.

        ``support`` is the ground-truth count, so a metric that divides by it is undefined
        once it reaches 0. A *count* metric such as ID switches keeps reporting while its
        class appears at all -- but not when the class was never there.
        """
        for result in outputs(metric.on(MATCHING, EST, GT), tracked_scene, labels):
            absent = result.support.values == 0
            assert np.isnan(result.value.values[absent]).all()

    def test_a_count_metric_survives_a_zero_support(self, labels: LabelRegistry) -> None:
        """A class seen only as false positives still gets its switch count."""
        from conftest import make_tracking

        store = Store()
        store.log(
            GT,
            make_tracking([[0.0, 0.0, 0.0]], [100], labels.encode(["car"])),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        store.log(
            EST,
            make_tracking(
                [[0.1, 0.0, 0.0], [500.0, 0.0, 0.0]],
                [1, 2],
                labels.encode(["car", "truck"]),
            ),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )

        _, _, switches = outputs(ClearSystem.on(MATCHING, EST, GT), store, labels)

        truck = labels.class_id("truck")
        assert switches.support.values[truck] == 0, "no truck ground truth"
        assert switches.of_class(truck) == pytest.approx(0.0), "but the class was seen"

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_per_class_rows_never_use_the_aggregate_sentinel(
        self,
        metric: type[MetricSystem],
        tracked_scene: Store,
        labels: LabelRegistry,
    ) -> None:
        """``-1`` is unambiguous in a metrics table because no class is reported as -1."""
        for result in outputs(metric.on(MATCHING, EST, GT), tracked_scene, labels):
            assert ALL_CLASSES not in result.class_id.values.tolist()

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_needs_three_sources(self, metric: type[MetricSystem]) -> None:
        with pytest.raises(ValueError, match="needs exactly three sources"):
            metric((MATCHING, EST), "/metrics/out")

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_an_empty_store_still_reports(
        self,
        metric: type[MetricSystem],
        labels: LabelRegistry,
    ) -> None:
        store = Store()
        system = metric.on(MATCHING, EST, GT)
        Pipeline([system]).run(
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        for target in system.targets:
            result = store.range(
                target,
                timeline=FRAME,
                time_range=TimeRange.everything(),
            ).materialize(BatchMetric)
            assert len(result) == len(labels)
            assert np.isnan(result.value.values).all()

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_a_missing_component_is_reported(
        self,
        metric: type[MetricSystem],
        labels: LabelRegistry,
    ) -> None:
        # Entities carrying one unrelated column are missing whatever any metric needs.
        store = Store()
        for path in (EST, GT):
            store.send_chunk(
                Chunk.from_columns(
                    path,
                    {METRIC_VALUE: BatchMetricValue([1.0])},
                    indexes=(TimeColumn.of(FRAME, [0]),),
                ),
            )
        store.log(
            MATCHING,
            BatchMatchResult(
                est_index=[0],
                gt_index=[0],
                matching_score=[0.1],
                match_status=[MatchStatus.TP],
                threshold=[1.0],
            ),
            at=TimePoint.at(frame=0),
        )
        system = metric.on(MATCHING, EST, GT)

        with pytest.raises(ValueError, match="missing required component"):
            list(system(SystemContext(store, FRAME, labels=labels), TimeRange.everything()))


class TestReportingTime:
    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_a_scene_score_is_reported_at_the_last_frame(
        self,
        metric: type[MetricSystem],
        tracked_scene: Store,
        labels: LabelRegistry,
    ) -> None:
        system = metric.on(MATCHING, EST, GT)
        match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        Pipeline([match, system]).run(
            SystemContext(tracked_scene, FRAME, labels=labels),
            TimeRange.everything(),
        )

        view = tracked_scene.latest_at(system.targets[0], timeline=FRAME, at=99)

        assert view.times(FRAME).tolist()[0] == 1, "the newest score is the current one"

    def test_a_single_frame_query_scores_that_frame(
        self,
        tracked_scene: Store,
        labels: LabelRegistry,
    ) -> None:
        match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        metric = AveragePrecisionSystem.on(MATCHING, EST, GT)
        ctx = SystemContext(tracked_scene, FRAME, labels=labels)
        Pipeline([match]).run(ctx, TimeRange.everything())

        (chunk,) = metric(ctx, 0)

        assert chunk.index(FRAME).times.tolist() == [0]


class TestPersistence:
    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_a_metric_survives_an_arrow_round_trip(
        self,
        metric: type[MetricSystem],
        tracked_scene: Store,
        labels: LabelRegistry,
    ) -> None:
        system = metric.on(MATCHING, EST, GT)
        match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        Pipeline([match, system]).run(
            SystemContext(tracked_scene, FRAME, labels=labels),
            TimeRange.everything(),
        )

        chunk = tracked_scene.range(
            system.targets[0],
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).to_chunk()
        restored, _ = chunk_from_table(chunk_to_table(chunk))

        assert restored == chunk, "NaN values included"

    def test_a_metric_survives_a_parquet_file(
        self,
        tracked_scene: Store,
        labels: LabelRegistry,
        tmp_path: Path,
    ) -> None:
        match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        metric = AveragePrecisionSystem.on(MATCHING, EST, GT)
        Pipeline([match, metric]).run(
            SystemContext(tracked_scene, FRAME, labels=labels),
            TimeRange.everything(),
        )

        chunk = tracked_scene.range(
            metric.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).to_chunk()
        path = tmp_path / "ap.parquet"
        write_parquet(chunk, path, labels=labels)
        restored, restored_labels = read_parquet(path)

        assert restored == chunk
        assert restored_labels == labels
        assert BatchMetric.from_chunk(restored).of_class(0) == pytest.approx(1.0)


class TestBatchMetric:
    def test_the_archetype_module_exposes_the_canonical_type(self) -> None:
        assert ArchetypeBatchMetric is BatchMetric

    def test_builds_from_rows(self) -> None:
        metric = BatchMetric.from_rows([(0, 1.0, 0.5, 10), (ALL_CLASSES, float("nan"), 0.5, 10)])

        assert len(metric) == 2
        assert metric.of_class(0) == pytest.approx(0.5)
        assert metric.aggregate == pytest.approx(0.5)

    def test_no_rows_is_allowed(self) -> None:
        assert len(BatchMetric.empty()) == 0

    def test_reports_an_absent_class(self) -> None:
        metric = BatchMetric.from_rows([(0, 1.0, 0.5, 10)])

        with pytest.raises(KeyError, match="found 0"):
            metric.of_class(9)

    def test_reports_a_class_with_several_thresholds(self) -> None:
        metric = BatchMetric.from_rows([(0, 0.5, 0.4, 10), (0, 1.0, 0.6, 10)])

        with pytest.raises(KeyError, match="found 2"):
            metric.of_class(0)

    def test_reports_a_missing_aggregate(self) -> None:
        metric = BatchMetric.from_rows([(0, 1.0, 0.5, 10)])

        with pytest.raises(KeyError, match="aggregate row"):
            metric.aggregate

    def test_a_value_may_be_undefined(self) -> None:
        assert np.isnan(BatchMetricValue([np.nan]).values[0])

    def test_the_aggregate_sentinel_matches_the_unknown_class(self) -> None:
        """They coincide on purpose; the docs explain why that stays unambiguous."""
        assert ALL_CLASSES == UNKNOWN_CLASS_ID


class TestNanMean:
    def test_skips_undefined_values(self) -> None:
        assert nan_mean([1.0, np.nan, 3.0]) == pytest.approx(2.0)

    def test_all_undefined_stays_undefined(self) -> None:
        assert np.isnan(nan_mean([np.nan, np.nan]))

    def test_empty_is_undefined(self) -> None:
        assert np.isnan(nan_mean([]))
