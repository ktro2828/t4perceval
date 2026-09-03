from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pytest
from conftest import make_detections, make_predictions, make_trackings

from t4perceval import (
    FRAME,
    TIMESTAMP,
    Detections3D,
    Predictions3D,
    Trackings3D,
    Chunk,
    LabelRegistry,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.component import BatchTimeOffset, FrameId
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import CLASS_ID, CONFIDENCE, INSTANCE_ID, POSITION, TIME_OFFSET
from t4perceval.io import (
    METADATA_KEY,
    chunk_from_table,
    chunk_to_table,
    component_types,
    read_parquet,
    resolve_component_type,
    write_parquet,
)

if TYPE_CHECKING:
    from pathlib import Path


def tracking_chunk() -> Chunk:
    return make_trackings(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        [100, 101],
        [1, 2],
    ).to_chunk(
        "/estimation/objects",
        at=TimePoint.at(frame=3, timestamp_ns=1_000),
        frame_id="base_link",
    )


class TestArrowSchema:
    def test_one_component_becomes_one_field(self) -> None:
        table = chunk_to_table(tracking_chunk())

        assert set(table.schema.names) == {
            "position",
            "quaternion",
            "size",
            "class_id",
            "confidence",
            "instance_id",
        }

    def test_vectors_are_fixed_size_lists(self) -> None:
        table = chunk_to_table(tracking_chunk())

        assert table.schema.field("position").type == pa.list_(pa.float64(), 3)
        assert table.schema.field("quaternion").type == pa.list_(pa.float64(), 4)

    def test_nested_shapes_survive_as_nested_lists(self) -> None:
        chunk = make_predictions([[0.0, 0.0, 0.0]], [1], num_modes=2, num_timesteps=4).to_chunk(
            "/x",
            at=TimePoint.at(frame=0),
        )

        table = chunk_to_table(chunk)

        assert table.schema.field("waypoints").type == pa.list_(
            pa.list_(pa.list_(pa.float64(), 3), 4),
            2,
        )

    def test_dtypes_are_pinned(self) -> None:
        table = chunk_to_table(tracking_chunk())

        assert table.schema.field("class_id").type == pa.int32()
        assert table.schema.field("instance_id").type == pa.int64()
        assert table.schema.field("confidence").type == pa.float64()

    def test_columns_are_declared_non_nullable(self) -> None:
        table = chunk_to_table(tracking_chunk())

        assert all(not field.nullable for field in table.schema)

    def test_non_row_shaped_data_lives_in_the_metadata(self) -> None:
        import json

        table = chunk_to_table(tracking_chunk())
        metadata = json.loads(table.schema.metadata[METADATA_KEY].decode())

        assert metadata["entity_path"] == "/estimation/objects"
        assert metadata["frame_id"] == "base_link"
        assert metadata["offsets"] == [0, 2]
        assert {index["name"] for index in metadata["indexes"]} == {"frame", "timestamp_ns"}
        assert {column["archetype"] for column in metadata["columns"]} == {"Trackings3D"}


class TestArrowRoundTrip:
    @pytest.mark.parametrize(
        ("archetype", "kind"),
        [
            (make_detections([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], [1, 2]), Detections3D),
            (make_trackings([[0.0, 0.0, 0.0]], [7], [1]), Trackings3D),
            (make_predictions([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [7, 8]), Predictions3D),
        ],
    )
    def test_preserves_every_archetype(self, archetype: Archetype, kind: type) -> None:
        chunk = archetype.to_chunk("/x", at=TimePoint.at(frame=0), frame_id="base_link")

        restored, labels = chunk_from_table(chunk_to_table(chunk))

        assert restored == chunk
        assert kind.from_chunk(restored) == archetype
        assert labels is None

    def test_preserves_rows_without_misalignment(self) -> None:
        chunk = tracking_chunk()

        restored, _ = chunk_from_table(chunk_to_table(chunk))

        assert restored.num_rows == chunk.num_rows
        np.testing.assert_array_equal(
            restored.columns[POSITION].values,
            chunk.columns[POSITION].values,
        )
        np.testing.assert_array_equal(
            restored.columns[INSTANCE_ID].values,
            chunk.columns[INSTANCE_ID].values,
        )

    def test_preserves_component_dtypes(self) -> None:
        restored, _ = chunk_from_table(chunk_to_table(tracking_chunk()))

        assert restored.columns[POSITION].values.dtype == np.float64
        assert restored.columns[CLASS_ID].values.dtype == np.int32
        assert restored.columns[CONFIDENCE].values.dtype == np.float64
        assert restored.columns[INSTANCE_ID].values.dtype == np.int64

    def test_preserves_the_time_axes(self) -> None:
        restored, _ = chunk_from_table(chunk_to_table(tracking_chunk()))

        assert restored.index(FRAME).times.tolist() == [3]
        assert restored.index(TIMESTAMP).times.tolist() == [1_000]
        assert restored.index(TIMESTAMP).timeline.kind is TIMESTAMP.kind

    def test_preserves_multiple_partitions(self, scene_store: Store) -> None:
        chunk = scene_store.range(
            "/estimation/objects",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).to_chunk()

        restored, _ = chunk_from_table(chunk_to_table(chunk))

        assert restored.num_partitions == 2
        assert restored.offsets.tolist() == [0, 2, 4]
        assert restored.index(FRAME).times.tolist() == [0, 1]

    def test_preserves_static_chunks(self) -> None:
        chunk = Chunk.from_columns(
            "/x",
            {TIME_OFFSET: BatchTimeOffset([[0, 100]])},
            is_static=True,
        )

        restored, _ = chunk_from_table(chunk_to_table(chunk))

        assert restored.is_static
        assert restored == chunk

    def test_preserves_a_text_column(self) -> None:
        # A text column is `string` in the schema, not something inferred from the value,
        # and a mono component comes back as itself rather than as a one-row column.
        frames = FrameId.descriptor()
        chunk = Chunk.from_columns(
            "/tf/lidar",
            {frames: FrameId("LIDAR_TOP")},
            is_static=True,
            frame_id="base_link",
        )

        table = chunk_to_table(chunk)
        restored, _ = chunk_from_table(table)

        assert table.schema.field("FrameId").type == pa.string()
        assert restored == chunk
        assert restored.component(frames).name == "LIDAR_TOP"
        assert restored.frame_id == "base_link"

    def test_preserves_an_empty_chunk(self) -> None:
        chunk = make_detections([]).to_chunk("/x", at=TimePoint.at(frame=0))

        restored, _ = chunk_from_table(chunk_to_table(chunk))

        assert restored.num_rows == 0
        assert restored == chunk

    def test_carries_a_label_registry(self, labels: LabelRegistry) -> None:
        restored, restored_labels = chunk_from_table(
            chunk_to_table(tracking_chunk(), labels=labels),
        )

        assert restored_labels == labels


class TestParquet:
    def test_round_trips_through_a_file(self, tmp_path: Path, labels: LabelRegistry) -> None:
        chunk = tracking_chunk()
        path = tmp_path / "chunk.parquet"

        write_parquet(chunk, path, labels=labels)
        restored, restored_labels = read_parquet(path)

        assert restored == chunk
        assert restored_labels == labels
        assert Trackings3D.from_chunk(restored) == Trackings3D.from_chunk(chunk)

    def test_round_trips_nested_shapes_through_a_file(self, tmp_path: Path) -> None:
        chunk = make_predictions([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [7, 8]).to_chunk(
            "/estimation/objects",
            at=TimePoint.at(frame=0),
        )
        path = tmp_path / "prediction.parquet"

        write_parquet(chunk, path)
        restored, _ = read_parquet(path)

        assert restored == chunk

    def test_round_trips_a_multi_frame_scene(self, tmp_path: Path, scene_store: Store) -> None:
        chunk = scene_store.range(
            "/estimation/objects",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).to_chunk()
        path = tmp_path / "scene.parquet"

        write_parquet(chunk, path)
        restored, _ = read_parquet(path)

        assert restored == chunk
        assert restored.num_partitions == 2


class TestErrors:
    def test_reports_a_table_without_metadata(self) -> None:
        with pytest.raises(ValueError, match="missing the 't4perceval' schema metadata"):
            chunk_from_table(pa.table({"class_id": pa.array([1], type=pa.int32())}))

    def test_reports_an_unsupported_schema_version(self) -> None:
        table = chunk_to_table(tracking_chunk())
        broken = table.replace_schema_metadata({METADATA_KEY: b'{"version": 999}'})

        with pytest.raises(ValueError, match="Unsupported chunk schema version"):
            chunk_from_table(broken)


class TestComponentRegistry:
    def test_resolves_components_by_name(self) -> None:
        from t4perceval.component import BatchPosition3D

        assert resolve_component_type("BatchPosition3D") is BatchPosition3D

    def test_covers_every_public_component(self) -> None:
        import t4perceval.component as components

        exported = {name for name in dir(components) if name.startswith("Batch")}

        assert exported <= set(component_types())

    def test_reports_an_unknown_name(self) -> None:
        with pytest.raises(KeyError, match="Unknown component type 'BatchNope'"):
            resolve_component_type("BatchNope")
