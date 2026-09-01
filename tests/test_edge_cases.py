"""Cross-cutting edge cases every archetype and column must agree on.

These are the cases the original package handled inconsistently per task: an empty frame,
a duplicated or reversed selection, and non-finite values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pytest
from conftest import make_detection, make_prediction, make_tracking

from t4perceval import (
    FRAME,
    BatchClassification2D,
    BatchDetection2D,
    BatchDetection3D,
    BatchMatchResult,
    BatchPrediction3D,
    BatchSemanticSegmentation2D,
    BatchSemanticSegmentation3D,
    BatchTracking2D,
    BatchTracking3D,
    BatchTrajectory3D,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.component import BatchConfidence, BatchPosition3D, BatchVelocity, MatchStatus
from t4perceval.io import chunk_from_table, chunk_to_table
from t4perceval.system import CenterDistanceMatchingSystem, FilterByDistanceSystem, SystemContext

if TYPE_CHECKING:
    from t4perceval.core.archetype import Archetype


def empty_archetypes() -> dict[str, Callable[[], Archetype]]:
    """A zero-row instance of every archetype, keyed for readable test ids."""
    return {
        "BatchDetection3D": lambda: make_detection([]),
        "BatchDetection2D": lambda: BatchDetection2D(roi=[], class_id=[], confidence=[]),
        "BatchTracking3D": lambda: make_tracking([], []),
        "BatchTracking2D": lambda: BatchTracking2D(
            roi=[], class_id=[], confidence=[], instance_id=[]
        ),
        "BatchPrediction3D": lambda: make_prediction([], []),
        "BatchClassification2D": lambda: BatchClassification2D(class_id=[], confidence=[]),
        "BatchSemanticSegmentation2D": lambda: BatchSemanticSegmentation2D(pixel=[], class_id=[]),
        "BatchSemanticSegmentation3D": lambda: BatchSemanticSegmentation3D(point=[], class_id=[]),
        "BatchTrajectory3D": lambda: BatchTrajectory3D.empty(num_modes=2, num_timesteps=3),
        "BatchMatchResult": BatchMatchResult.empty,
    }


def populated_archetypes() -> dict[str, Callable[[], Archetype]]:
    """A three-row instance of every archetype that has a row-wise selection."""
    return {
        "BatchDetection3D": lambda: make_detection(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [0, 1, 2],
        ),
        "BatchDetection2D": lambda: BatchDetection2D(
            roi=[[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]],
            class_id=[0, 1, 2],
            confidence=[0.1, 0.2, 0.3],
        ),
        "BatchTracking3D": lambda: make_tracking(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [10, 11, 12],
        ),
        "BatchPrediction3D": lambda: make_prediction(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [10, 11, 12],
        ),
        "BatchClassification2D": lambda: BatchClassification2D(
            class_id=[0, 1, 2],
            confidence=[0.1, 0.2, 0.3],
        ),
        "BatchSemanticSegmentation2D": lambda: BatchSemanticSegmentation2D(
            pixel=[0, 1, 2],
            class_id=[0, 1, 2],
        ),
        "BatchMatchResult": lambda: BatchMatchResult(
            est_index=[0, 1, -1],
            gt_index=[0, -1, 2],
            matching_score=[0.5, np.nan, np.nan],
            match_status=[MatchStatus.TP, MatchStatus.FP, MatchStatus.FN],
            threshold=[1.0, 1.0, 1.0],
        ),
    }


EMPTY = empty_archetypes()
POPULATED = populated_archetypes()


class TestEmptyBatches:
    """Zero rows is allowed everywhere: an empty frame is ordinary in real data."""

    @pytest.mark.parametrize("name", sorted(EMPTY))
    def test_every_archetype_can_be_empty(self, name: str) -> None:
        archetype = EMPTY[name]()

        assert len(archetype) == 0
        for descriptor, column in archetype.as_components().items():
            assert len(column) == 0, f"{descriptor.component} is not empty"

    @pytest.mark.parametrize("name", sorted(EMPTY))
    def test_selecting_from_an_empty_batch_stays_empty(self, name: str) -> None:
        archetype = EMPTY[name]()

        assert len(archetype.select([])) == 0
        assert len(archetype.select(slice(None))) == 0

    @pytest.mark.parametrize("name", sorted(EMPTY))
    def test_an_empty_batch_survives_a_chunk_round_trip(self, name: str) -> None:
        archetype = EMPTY[name]()

        chunk = archetype.to_chunk("/x", at=TimePoint.at(frame=0))

        assert chunk.num_rows == 0
        assert type(archetype).from_chunk(chunk) == archetype

    @pytest.mark.parametrize("name", sorted(EMPTY))
    def test_an_empty_batch_survives_an_arrow_round_trip(self, name: str) -> None:
        chunk = EMPTY[name]().to_chunk("/x", at=TimePoint.at(frame=0))

        restored, _ = chunk_from_table(chunk_to_table(chunk))

        assert restored == chunk

    @pytest.mark.parametrize("name", sorted(EMPTY))
    def test_an_empty_batch_can_be_logged_and_queried(self, name: str) -> None:
        archetype = EMPTY[name]()
        store = Store()

        store.log("/x", archetype, at=TimePoint.at(frame=0))
        view = store.latest_at("/x", timeline=FRAME, at=0)

        assert len(view) == 0
        assert view.materialize(type(archetype)) == archetype

    def test_an_empty_frame_flows_through_a_system(self) -> None:
        store = Store()
        store.log("/estimation/objects", make_detection([]), at=TimePoint.at(frame=0))
        ctx = SystemContext(store, FRAME)

        (chunk,) = FilterByDistanceSystem.on("/estimation/objects")(ctx, 0)

        assert chunk.num_rows == 0

    def test_an_empty_column_selects_to_empty(self) -> None:
        assert len(BatchPosition3D.empty().select([])) == 0

    def test_an_out_of_range_selection_on_an_empty_batch_is_rejected(self) -> None:
        with pytest.raises(IndexError, match="out of range for length 0"):
            make_detection([]).select([0])


class TestSelectionOrderAndDuplication:
    @pytest.mark.parametrize("name", sorted(POPULATED))
    def test_a_reversed_selection_reverses_every_column(self, name: str) -> None:
        archetype = POPULATED[name]()

        reversed_batch = archetype.select([2, 1, 0])

        assert len(reversed_batch) == 3
        for descriptor, column in reversed_batch.as_components().items():
            original = archetype.as_components()[descriptor]
            np.testing.assert_array_equal(
                column.values,
                original.values[[2, 1, 0]],
                err_msg=f"{descriptor.component} was not reversed",
            )

    @pytest.mark.parametrize("name", sorted(POPULATED))
    def test_a_duplicated_selection_duplicates_every_column(self, name: str) -> None:
        archetype = POPULATED[name]()

        duplicated = archetype.select([1, 1, 1])

        assert len(duplicated) == 3
        for descriptor, column in duplicated.as_components().items():
            original = archetype.as_components()[descriptor]
            np.testing.assert_array_equal(column.values, original.values[[1, 1, 1]])

    @pytest.mark.parametrize("name", sorted(POPULATED))
    def test_a_negative_selection_counts_from_the_end(self, name: str) -> None:
        archetype = POPULATED[name]()

        assert archetype.select([-1]).as_components() == archetype.select([2]).as_components()

    @pytest.mark.parametrize("name", sorted(POPULATED))
    def test_an_out_of_range_selection_is_rejected(self, name: str) -> None:
        with pytest.raises(IndexError, match="out of range for length 3"):
            POPULATED[name]().select([3])

    def test_a_reversed_selection_within_one_partition_is_allowed_on_a_chunk(self) -> None:
        chunk = make_detection(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ).to_chunk("/x", at=TimePoint.at(frame=0))

        selected = chunk.select([1, 0])

        assert selected.columns[chunk.descriptors[0]].values[0].tolist() == [1.0, 0.0, 0.0]

    def test_a_lazy_view_composes_duplication(self) -> None:
        chunk = make_detection(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ).to_chunk("/x", at=TimePoint.at(frame=0))

        from t4perceval import EntityView

        view = EntityView.over(chunk).select([1, 1]).select([0, 1])

        assert view.indices.tolist() == [1, 1]


class TestNonFiniteValues:
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_bounded_column_rejects_them(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            BatchConfidence([bad])

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_an_unbounded_column_accepts_them(self, bad: float) -> None:
        # Geometry columns carry whatever the model produced; rejecting non-finite values
        # here would hide a model defect behind a construction error.
        velocity = BatchVelocity([[bad, 0.0, 0.0]])

        assert not np.isfinite(velocity.values[0, 0])

    def test_a_matching_score_uses_nan_as_its_sentinel(self) -> None:
        result = BatchMatchResult(
            est_index=[0],
            gt_index=[-1],
            matching_score=[np.nan],
            match_status=[MatchStatus.FP],
            threshold=[1.0],
        )

        assert np.isnan(result.matching_score.values[0])

    def test_nan_sentinels_compare_equal(self) -> None:
        left = BatchMatchResult(
            est_index=[0],
            gt_index=[-1],
            matching_score=[np.nan],
            match_status=[MatchStatus.FP],
            threshold=[1.0],
        )
        right = BatchMatchResult(
            est_index=[0],
            gt_index=[-1],
            matching_score=[np.nan],
            match_status=[MatchStatus.FP],
            threshold=[1.0],
        )

        assert left == right

    def test_nan_sentinels_survive_an_arrow_round_trip(self) -> None:
        result = BatchMatchResult(
            est_index=[0],
            gt_index=[-1],
            matching_score=[np.nan],
            match_status=[MatchStatus.FP],
            threshold=[1.0],
        )
        chunk = result.to_chunk("/matching/x", at=TimePoint.at(frame=0))

        restored, _ = chunk_from_table(chunk_to_table(chunk))

        assert restored == chunk

    def test_a_non_finite_position_does_not_match(self) -> None:
        store = Store()
        store.log(
            "/ground_truth/objects",
            make_detection([[0.0, 0.0, 0.0]], [0]),
            at=TimePoint.at(frame=0),
        )
        store.log(
            "/estimation/objects",
            make_detection([[np.nan, 0.0, 0.0]], [0]),
            at=TimePoint.at(frame=0),
        )
        system = CenterDistanceMatchingSystem.between(
            "/estimation/objects",
            "/ground_truth/objects",
        )

        (chunk,) = system(SystemContext(store, FRAME), 0)
        result = BatchMatchResult.from_chunk(chunk)

        assert (result.num_tp, result.num_fp, result.num_fn) == (0, 1, 1)

    def test_a_non_finite_position_is_filtered_out(self) -> None:
        store = Store()
        store.log(
            "/estimation/objects",
            make_detection([[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            at=TimePoint.at(frame=0),
        )

        (chunk,) = FilterByDistanceSystem.on("/estimation/objects", max_distance=10.0)(
            SystemContext(store, FRAME),
            0,
        )

        from t4perceval.descriptors import MASK

        assert chunk.columns[MASK].values.tolist() == [False, True]


class TestStoreEdges:
    def test_an_empty_store_answers_every_query(self) -> None:
        store = Store()

        assert store.entity_paths() == ()
        assert store.timelines() == ()
        assert len(store.latest_at("/x", timeline=FRAME, at=0)) == 0
        assert len(store.range("/x", timeline=FRAME, time_range=TimeRange.everything())) == 0

    def test_a_single_frame_scene_still_ranges(self) -> None:
        store = Store()
        store.log("/x", make_detection([[0.0, 0.0, 0.0]]), at=TimePoint.at(frame=0))

        view = store.range("/x", timeline=FRAME, time_range=TimeRange.everything())

        assert len(view) == 1
        assert view.to_chunk().num_partitions == 1

    @pytest.mark.parametrize("archetype", [BatchDetection3D, BatchTracking3D, BatchPrediction3D])
    def test_materializing_the_wrong_archetype_is_reported(
        self,
        archetype: type,
        scene_store: Store,
    ) -> None:
        view = scene_store.latest_at("/estimation/objects", timeline=FRAME, at=0)

        if archetype is BatchDetection3D:
            assert len(view.materialize(archetype)) == 2
        else:
            with pytest.raises(ValueError, match="missing required component"):
                view.materialize(archetype)
