"""Detection confusion-matrix metric."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_metric_scene

from t4perceval import (
    BACKGROUND_CLASS_ID,
    FRAME,
    ClassInfo,
    ConfusionMatrix,
    LabelRegistry,
    Store,
    TimeRange,
)
from t4perceval.component import BatchCount
from t4perceval.io import chunk_from_table, chunk_to_table
from t4perceval.system import (
    CenterDistanceMatchingSystem,
    ConfusionMatrixSystem,
    Pipeline,
    SystemContext,
)

EST = "/estimation/objects"
GT = "/ground_truth/objects"


def confusion(
    labels: LabelRegistry,
    frames: list[tuple[int, list[tuple[float, str]], list[tuple[float, str, float]]]],
    *,
    class_agnostic: bool = True,
) -> ConfusionMatrix:
    store = make_metric_scene(labels, frames)
    match = CenterDistanceMatchingSystem.between(
        EST,
        GT,
        threshold=1.0,
        class_agnostic=class_agnostic,
    )
    metric = ConfusionMatrixSystem.on(match.target, EST, GT)
    Pipeline([match, metric]).run(
        SystemContext(store, FRAME, labels=labels),
        TimeRange.everything(),
    )
    return store.range(
        metric.target,
        timeline=FRAME,
        time_range=TimeRange.everything(),
    ).materialize(ConfusionMatrix)


class TestConfusionMatrix:
    def test_builds_from_rows_and_reads_a_cell(self) -> None:
        result = ConfusionMatrix.from_rows([(0, 0, 2), (0, 1, 3)])

        assert result.at(0, 0) == 2
        assert result.at(0, 1) == 3

    def test_converts_long_form_to_a_dense_matrix(self) -> None:
        result = ConfusionMatrix.from_rows(
            [
                (0, 0, 2),
                (0, 1, 1),
                (1, BACKGROUND_CLASS_ID, 3),
                (BACKGROUND_CLASS_ID, 0, 4),
            ]
        )

        np.testing.assert_array_equal(
            result.as_matrix([0, 1]),
            np.asarray(
                [
                    [2, 1, 0],
                    [0, 0, 3],
                    [4, 0, 0],
                ],
                dtype=np.int64,
            ),
        )

    def test_dense_conversion_can_drop_background(self) -> None:
        result = ConfusionMatrix.from_rows(
            [(0, 0, 1), (0, BACKGROUND_CLASS_ID, 2)],
        )

        np.testing.assert_array_equal(
            result.as_matrix([0], include_background=False),
            np.asarray([[1]], dtype=np.int64),
        )

    def test_duplicate_long_form_cells_are_added(self) -> None:
        result = ConfusionMatrix.from_rows([(0, 0, 1), (0, 0, 2)])

        np.testing.assert_array_equal(result.as_matrix([0]), np.asarray([[3, 0], [0, 0]]))

    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            BatchCount([-1])

    def test_background_does_not_collide_with_unknown_class(self) -> None:
        assert BACKGROUND_CLASS_ID == -2

    def test_background_class_id_is_reserved(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            LabelRegistry((ClassInfo(BACKGROUND_CLASS_ID, "background"),))

    def test_empty_matrix_is_allowed(self) -> None:
        result = ConfusionMatrix.empty()

        assert len(result) == 0
        assert result.as_matrix().shape == (1, 1)

    def test_arrow_round_trip(self) -> None:
        original = ConfusionMatrix.from_rows([(0, 1, 2), (1, 0, 3)])
        chunk = original.to_chunk("/metrics/confusion_matrix", is_static=True)

        restored_chunk, _ = chunk_from_table(chunk_to_table(chunk))

        assert ConfusionMatrix.from_chunk(restored_chunk) == original


class TestConfusionMatrixSystem:
    def test_default_target(self) -> None:
        metric = ConfusionMatrixSystem.on("/m", EST, GT)

        assert str(metric.target) == "/metrics/confusion_matrix"

    def test_perfect_matches_are_on_the_diagonal(self, labels: LabelRegistry) -> None:
        result = confusion(
            labels,
            [(0, [(0.0, "car"), (10.0, "truck")], [(0.1, "car", 0.9), (10.1, "truck", 0.8)])],
        )

        assert result.at(labels.class_id("car"), labels.class_id("car")) == 1
        assert result.at(labels.class_id("truck"), labels.class_id("truck")) == 1
        assert result.count.values.sum() == 2

    def test_class_mismatch_is_an_off_diagonal_cell(self, labels: LabelRegistry) -> None:
        result = confusion(
            labels,
            [(0, [(0.0, "car")], [(0.1, "truck", 0.9)])],
        )

        assert result.at(labels.class_id("car"), labels.class_id("truck")) == 1
        assert result.at(labels.class_id("car"), BACKGROUND_CLASS_ID) == 0
        assert result.at(BACKGROUND_CLASS_ID, labels.class_id("truck")) == 0

    def test_false_negatives_use_the_background_column(self, labels: LabelRegistry) -> None:
        result = confusion(labels, [(0, [(0.0, "car")], [])])

        assert result.at(labels.class_id("car"), BACKGROUND_CLASS_ID) == 1

    def test_false_positives_use_the_background_row(self, labels: LabelRegistry) -> None:
        result = confusion(labels, [(0, [], [(0.0, "car", 0.9)])])

        assert result.at(BACKGROUND_CLASS_ID, labels.class_id("car")) == 1

    def test_class_aware_matching_cannot_retain_the_confused_pair(
        self,
        labels: LabelRegistry,
    ) -> None:
        result = confusion(
            labels,
            [(0, [(0.0, "car")], [(0.1, "truck", 0.9)])],
            class_agnostic=False,
        )

        assert result.at(labels.class_id("car"), labels.class_id("truck")) == 0
        assert result.at(labels.class_id("car"), BACKGROUND_CLASS_ID) == 1
        assert result.at(BACKGROUND_CLASS_ID, labels.class_id("truck")) == 1

    def test_counts_across_frames(self, labels: LabelRegistry) -> None:
        result = confusion(
            labels,
            [
                (0, [(0.0, "car")], [(0.1, "car", 0.9)]),
                (1, [(1.0, "car")], [(1.1, "car", 0.8)]),
            ],
        )

        assert result.at(labels.class_id("car"), labels.class_id("car")) == 2

    def test_emits_a_complete_matrix_for_registered_classes(
        self,
        labels: LabelRegistry,
    ) -> None:
        result = confusion(labels, [(0, [], [])])

        size = len(labels) + 1
        assert len(result) == size * size
        assert result.count.values.tolist() == [0] * (size * size)
        assert result.as_matrix([info.class_id for info in labels.classes]).shape == (size, size)

    def test_empty_store_still_uses_the_label_registry(self, labels: LabelRegistry) -> None:
        store = Store()
        metric = ConfusionMatrixSystem.on("/matching/center_distance", EST, GT)

        Pipeline([metric]).run(
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        result = store.range(
            metric.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(ConfusionMatrix)
        assert len(result) == (len(labels) + 1) ** 2
        assert result.count.values.sum() == 0
