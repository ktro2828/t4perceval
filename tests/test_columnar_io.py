from __future__ import annotations

import numpy as np
import pyarrow as pa

from t4perceval.dataclass import BatchTracking3D, Header


def arrow_detection_table() -> pa.Table:
    return pa.table(
        {
            "position": pa.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
            "quaternion": pa.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
            "size": pa.array([[4.0, 2.0, 1.5], [1.0, 1.0, 2.0]]),
            "class_id": pa.array([1, 2], type=pa.int32()),
            "confidence": pa.array([0.9, 0.8], type=pa.float64()),
            "velocity": pa.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "instance_id": pa.array([100, 101], type=pa.int64()),
        }
    )


def numpy_column(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist())


def tracking_from_arrow(table: pa.Table) -> BatchTracking3D:
    return BatchTracking3D(
        header=Header(1_000, "map"),
        position=numpy_column(table, "position"),
        quaternion=numpy_column(table, "quaternion"),
        size=numpy_column(table, "size"),
        class_id=numpy_column(table, "class_id"),
        confidence=numpy_column(table, "confidence"),
        velocity=numpy_column(table, "velocity"),
        instance_id=numpy_column(table, "instance_id"),
    )


def tracking_to_arrow(tracking: BatchTracking3D) -> pa.Table:
    def array(values: np.ndarray) -> pa.Array:
        return pa.array(values.tolist())

    return pa.table(
        {
            "position": array(tracking.position.values),
            "quaternion": array(tracking.quaternion.values),
            "size": array(tracking.size.values),
            "class_id": array(tracking.class_id.values),
            "confidence": array(tracking.confidence.values),
            "velocity": (
                None if tracking.velocity is None else array(tracking.velocity.values)
            ),
            "instance_id": array(tracking.instance_id.values),
        }
    )


def test_arrow_columns_round_trip_without_row_misalignment() -> None:
    source = arrow_detection_table()
    tracking = tracking_from_arrow(source)
    restored = tracking_to_arrow(tracking)

    assert restored.num_rows == source.num_rows == len(tracking)
    for name in source.column_names:
        assert restored[name].to_pylist() == source[name].to_pylist()


def test_arrow_columns_preserve_component_dtypes() -> None:
    tracking = tracking_from_arrow(arrow_detection_table())

    assert tracking.position.values.dtype == np.float64
    assert tracking.class_id.values.dtype == np.int32
    assert tracking.confidence.values.dtype == np.float64
    assert tracking.instance_id.values.dtype == np.int64
