from __future__ import annotations

import numpy as np
import pytest

from t4perceval import FRAME, TIMESTAMP, Chunk, ComponentDescriptor, TimeColumn, concat_chunks
from t4perceval.component import BatchConfidence, BatchPosition3D

POSITION = ComponentDescriptor("position")
CONFIDENCE = ComponentDescriptor("confidence")
MISSING = ComponentDescriptor("nope")


def two_frame_chunk() -> Chunk:
    """Frame 0 holds two rows, frame 1 holds three."""
    return Chunk(
        "/estimation/objects",
        (TimeColumn.of(FRAME, [0, 1]),),
        [0, 2, 5],
        {
            POSITION: BatchPosition3D(np.arange(15, dtype=np.float64).reshape(5, 3)),
            CONFIDENCE: BatchConfidence([0.9, 0.8, 0.7, 0.6, 0.5]),
        },
        frame_id="base_link",
    )


class TestLayout:
    def test_rows_are_objects_and_offsets_are_frames(self) -> None:
        chunk = two_frame_chunk()

        assert chunk.num_rows == len(chunk) == 5
        assert chunk.num_partitions == 2
        assert chunk.partition_sizes().tolist() == [2, 3]
        assert chunk.partition_ids().tolist() == [0, 0, 1, 1, 1]
        assert chunk.partition(1) == slice(2, 5)
        assert chunk.partition(-1) == slice(2, 5)

    def test_reports_its_schema(self) -> None:
        chunk = two_frame_chunk()

        assert chunk.timelines == (FRAME,)
        assert set(chunk.descriptors) == {POSITION, CONFIDENCE}
        assert chunk.has(POSITION, CONFIDENCE)
        assert not chunk.has(MISSING)
        assert chunk.component(MISSING) is None
        assert chunk.index(TIMESTAMP) is None

    def test_expands_partition_times_over_rows(self) -> None:
        assert two_frame_chunk().times_for_rows(FRAME).tolist() == [0, 0, 1, 1, 1]

    def test_reports_a_missing_timeline(self) -> None:
        with pytest.raises(KeyError, match="no index column for timeline"):
            two_frame_chunk().times_for_rows(TIMESTAMP)

    def test_rejects_an_out_of_range_partition(self) -> None:
        with pytest.raises(IndexError, match="partition index out of range"):
            two_frame_chunk().partition(2)

    def test_from_columns_builds_a_single_partition(self) -> None:
        chunk = Chunk.from_columns("/x", {CONFIDENCE: BatchConfidence([0.1, 0.2])})

        assert (chunk.num_rows, chunk.num_partitions) == (2, 1)

    def test_from_columns_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="mismatched lengths"):
            Chunk.from_columns(
                "/x",
                {POSITION: BatchPosition3D.empty(), CONFIDENCE: BatchConfidence([0.1])},
            )


class TestInvariants:
    def test_offsets_must_start_at_zero(self) -> None:
        with pytest.raises(ValueError, match="must start with 0"):
            Chunk("/x", (), [1, 2], {})

    def test_offsets_must_not_decrease(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            Chunk("/x", (), [0, 2, 1], {})

    def test_a_column_must_cover_every_row(self) -> None:
        with pytest.raises(ValueError, match="has length 0, expected 2"):
            Chunk("/x", (), [0, 2], {POSITION: BatchPosition3D.empty()})

    def test_an_index_must_cover_every_partition(self) -> None:
        with pytest.raises(ValueError, match="has 2 times, expected 1"):
            Chunk("/x", (TimeColumn.of(FRAME, [0, 1]),), [0, 2], {})

    def test_rejects_duplicate_timelines(self) -> None:
        with pytest.raises(ValueError, match="duplicate timelines"):
            Chunk("/x", (TimeColumn.of(FRAME, [0]), TimeColumn.of(FRAME, [1])), [0, 0], {})

    def test_static_data_carries_no_time(self) -> None:
        static = Chunk.from_columns("/x", {CONFIDENCE: BatchConfidence([1.0])}, is_static=True)

        assert static.is_static
        assert static.timelines == ()

        with pytest.raises(ValueError, match="must not carry index columns"):
            Chunk("/x", (TimeColumn.of(FRAME, [0]),), [0, 1], {}, is_static=True)
        with pytest.raises(ValueError, match="exactly one partition"):
            Chunk("/x", (), [0, 1, 2], {}, is_static=True)


class TestSelect:
    def test_a_mask_keeps_the_partition_structure(self) -> None:
        chunk = two_frame_chunk()

        selected = chunk.select(np.array([True, False, True, True, False]))

        assert selected.offsets.tolist() == [0, 1, 3]
        assert selected.columns[CONFIDENCE].values.tolist() == [0.9, 0.7, 0.6]
        assert selected.index(FRAME).times.tolist() == [0, 1], "the time axis must survive"
        assert selected.frame_id == "base_link"

    def test_an_emptied_partition_keeps_its_place(self) -> None:
        selected = two_frame_chunk().select([2, 3, 4])

        assert selected.offsets.tolist() == [0, 0, 3]
        assert selected.num_partitions == 2

    def test_selecting_nothing_yields_empty_partitions(self) -> None:
        assert two_frame_chunk().select([]).offsets.tolist() == [0, 0, 0]

    def test_rejects_a_selection_that_reorders_partitions(self) -> None:
        with pytest.raises(ValueError, match="does not reorder partitions"):
            two_frame_chunk().select([4, 0])

    def test_reordering_within_one_partition_is_allowed(self) -> None:
        selected = two_frame_chunk().select([1, 0])

        assert selected.columns[CONFIDENCE].values.tolist() == [0.8, 0.9]

    def test_select_partitions_drops_whole_frames(self) -> None:
        selected = two_frame_chunk().select_partitions([1])

        assert selected.offsets.tolist() == [0, 3]
        assert selected.index(FRAME).times.tolist() == [1]
        assert selected.columns[CONFIDENCE].values.tolist() == [0.7, 0.6, 0.5]

    def test_select_partitions_rejects_a_descending_selection(self) -> None:
        with pytest.raises(ValueError, match="ascending selection"):
            two_frame_chunk().select_partitions([1, 0])

    def test_with_columns_adds_and_replaces(self) -> None:
        chunk = two_frame_chunk()
        extra = ComponentDescriptor("extra")

        updated = chunk.with_columns({extra: BatchConfidence([0.1] * 5)})

        assert updated.has(POSITION, extra)
        assert chunk.component(extra) is None, "the original must be untouched"


class TestConcat:
    def test_appends_partitions_in_order(self) -> None:
        chunk = two_frame_chunk()

        joined = concat_chunks([chunk, chunk])

        assert joined.num_rows == 10
        assert joined.num_partitions == 4
        assert joined.index(FRAME).times.tolist() == [0, 1, 0, 1]
        assert joined.offsets.tolist() == [0, 2, 5, 7, 10]

    def test_a_single_chunk_passes_through(self) -> None:
        chunk = two_frame_chunk()

        assert concat_chunks([chunk]) is chunk

    def test_requires_at_least_one_chunk(self) -> None:
        with pytest.raises(ValueError, match="at least one chunk"):
            concat_chunks([])

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            ({"entity_path": "/other"}, "different entities"),
            ({"frame_id": "map"}, "different frames"),
        ],
    )
    def test_rejects_incompatible_chunks(self, mutation: dict, message: str) -> None:
        chunk = two_frame_chunk()
        other = Chunk(
            mutation.get("entity_path", chunk.entity_path),
            chunk.indexes,
            chunk.offsets,
            chunk.columns,
            frame_id=mutation.get("frame_id", chunk.frame_id),
        )

        with pytest.raises(ValueError, match=message):
            concat_chunks([chunk, other])

    def test_rejects_mixing_static_and_temporal_data(self) -> None:
        columns = {CONFIDENCE: BatchConfidence([1.0])}
        temporal = Chunk("/x", (), [0, 1], columns)
        static = Chunk("/x", (), [0, 1], columns, is_static=True)

        with pytest.raises(ValueError, match="static and temporal"):
            concat_chunks([temporal, static])

    def test_rejects_a_different_timeline_set(self) -> None:
        chunk = two_frame_chunk()
        other = Chunk("/estimation/objects", (), [0, 5], chunk.columns, frame_id="base_link")

        with pytest.raises(ValueError, match="different timelines"):
            concat_chunks([chunk, other])

    def test_rejects_a_different_column_set(self) -> None:
        chunk = two_frame_chunk()
        other = Chunk(
            chunk.entity_path,
            chunk.indexes,
            chunk.offsets,
            {POSITION: chunk.columns[POSITION]},
            frame_id=chunk.frame_id,
        )

        with pytest.raises(ValueError, match="different columns"):
            concat_chunks([chunk, other])
