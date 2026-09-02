from __future__ import annotations

import numpy as np
import pytest
from conftest import make_detections, make_trackings

from t4perceval import (
    FRAME,
    Detections3D,
    Trackings3D,
    Chunk,
    EntityView,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.component import BatchTimeOffset
from t4perceval.descriptors import CONFIDENCE, INSTANCE_ID, POSITION, TIME_OFFSET


def four_row_view() -> EntityView:
    chunk = make_detections(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        [0, 1, 2, 3],
    ).to_chunk("/estimation/objects", at=TimePoint.at(frame=0), frame_id="base_link")
    return EntityView.over(chunk)


class TestLaziness:
    def test_a_fresh_view_covers_every_row(self) -> None:
        view = four_row_view()

        assert len(view) == 4
        assert view.indices.tolist() == [0, 1, 2, 3]

    def test_select_composes_indices_without_copying(self) -> None:
        view = four_row_view()

        narrowed = view.select(slice(None, None, 2)).select([1])

        assert narrowed.indices.tolist() == [2]
        assert narrowed.chunk is view.chunk, "the source chunk must be shared, not copied"

    @pytest.mark.parametrize(
        ("selection", "expected"),
        [
            (slice(1, 3), [1.0, 2.0]),
            ([3, 0], [3.0, 0.0]),
            (np.array([False, True, False, True]), [1.0, 3.0]),
            ([-1], [3.0]),
        ],
    )
    def test_accepts_every_selection_form(
        self,
        selection: object,
        expected: list[float],
    ) -> None:
        view = four_row_view().select(selection)

        assert view.component(POSITION).values[:, 0].tolist() == expected

    def test_materializing_a_column_produces_independent_data(self) -> None:
        view = four_row_view()

        column = view.component(POSITION)

        assert not np.shares_memory(column.values, view.chunk.columns[POSITION].values)
        assert not column.values.flags.writeable

    def test_reports_what_it_exposes(self) -> None:
        view = four_row_view()

        assert view.has(POSITION, CONFIDENCE)
        assert not view.has(INSTANCE_ID)
        assert view.component(INSTANCE_ID) is None
        assert str(view.entity_path) == "/estimation/objects"
        assert view.frame_id == "base_link"


class TestMaterialize:
    def test_builds_the_requested_archetype(self) -> None:
        view = four_row_view().select([0, 1])

        detection = view.materialize(Detections3D)

        assert isinstance(detection, Detections3D)
        assert len(detection) == 2

    def test_reports_a_missing_required_component(self) -> None:
        view = four_row_view()

        with pytest.raises(ValueError, match="missing required component"):
            view.materialize(Trackings3D)

    def test_a_richer_archetype_materializes_as_a_narrower_one(self) -> None:
        chunk = make_trackings([[0.0, 0.0, 0.0]], [7]).to_chunk("/x", at=TimePoint.at(frame=0))

        detection = EntityView.over(chunk).materialize(Detections3D)

        assert isinstance(detection, Detections3D)
        assert len(detection) == 1


class TestToChunk:
    def test_keeps_the_partition_structure(self, scene_store: Store) -> None:
        view = scene_store.range(
            "/estimation/objects",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )

        chunk = view.to_chunk()

        assert isinstance(chunk, Chunk)
        assert (chunk.num_rows, chunk.num_partitions) == (4, 2)
        assert chunk.offsets.tolist() == [0, 2, 4]

    def test_a_narrowed_view_narrows_the_chunk(self, scene_store: Store) -> None:
        view = scene_store.range(
            "/estimation/objects",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).select([0, 2])

        chunk = view.to_chunk()

        assert chunk.offsets.tolist() == [0, 1, 2]


class TestTimeAndPartitions:
    def test_reports_the_time_of_every_row(self, scene_store: Store) -> None:
        view = scene_store.range(
            "/estimation/objects",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )

        assert view.times(FRAME).tolist() == [0, 0, 1, 1]
        assert view.select([1, 2]).times(FRAME).tolist() == [0, 1]

    def test_reports_the_partition_of_every_row(self, scene_store: Store) -> None:
        view = scene_store.range(
            "/estimation/objects",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )

        assert view.partition_ids().tolist() == [0, 0, 1, 1]


class TestStaticColumns:
    def test_static_data_reaches_the_view(self) -> None:
        chunk = make_detections([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).to_chunk(
            "/x",
            at=TimePoint.at(frame=0),
        )
        view = EntityView.over(chunk, static={TIME_OFFSET: BatchTimeOffset([[0, 100]])})

        assert view.has(TIME_OFFSET)
        assert view.component(TIME_OFFSET).values.tolist() == [[0, 100], [0, 100]]

    def test_broadcasting_follows_the_narrowed_length(self) -> None:
        chunk = make_detections([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).to_chunk(
            "/x",
            at=TimePoint.at(frame=0),
        )
        view = EntityView.over(chunk, static={TIME_OFFSET: BatchTimeOffset([[0, 100]])})

        assert len(view.select([0]).component(TIME_OFFSET)) == 1
