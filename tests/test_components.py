from __future__ import annotations

import numpy as np
import pytest
from attrs import define

from t4perceval import ANY, ColumnarComponent
from t4perceval.component import (
    BatchClassId,
    BatchConfidence,
    BatchMask,
    BatchPosition2D,
    BatchPosition3D,
    BatchQuaternion,
    BatchRoi,
    BatchSize3D,
    BatchTimeOffset,
    BatchVector2D,
    BatchVector3D,
    BatchVelocity,
    BatchWaypoints3D,
    VisibilityLevel,
)


class TestColumnLayout:
    def test_a_column_owns_read_only_memory_of_its_own(self) -> None:
        source = np.arange(9, dtype=np.float64).reshape(3, 3)

        position = BatchPosition3D(source)

        assert not position.values.flags.writeable
        assert not np.shares_memory(position.values, source), "must not alias the caller"
        assert source.flags.writeable, "must not freeze the caller's array"
        assert len(position) == 3

    def test_normalizes_dtype_and_layout(self) -> None:
        non_contiguous = np.arange(12, dtype=np.float32).reshape(3, 4)[:, :3]

        position = BatchPosition3D(non_contiguous)

        assert position.values.dtype == np.float64
        assert position.values.flags.c_contiguous

    def test_accessors_expose_the_backing_column(self) -> None:
        position = BatchPosition3D([[1.0, 2.0, 3.0]])

        assert position.as_array() is position.values
        assert np.shares_memory(position.x, position.values)
        assert position.y.tolist() == [2.0]
        assert position.z.tolist() == [3.0]
        assert position.row_shape == (3,)

    @pytest.mark.parametrize("component", [BatchPosition3D, BatchSize3D, BatchVelocity])
    def test_a_3d_column_rejects_the_wrong_width(
        self,
        component: type[BatchVector3D],
    ) -> None:
        with pytest.raises(ValueError, match=rf"{component.__name__} must have shape \(N, 3\)"):
            component(np.zeros((3, 2)))

    def test_a_2d_column_rejects_the_wrong_width(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
            BatchPosition2D(np.zeros((3, 3)))

    def test_a_wildcard_dimension_is_inferred_and_reported(self) -> None:
        waypoints = BatchWaypoints3D(np.zeros((2, 4, 5, 3)))

        assert (waypoints.num_modes, waypoints.num_timesteps) == (4, 5)
        with pytest.raises(ValueError, match=r"shape \(N, \*, \*, 3\)"):
            BatchWaypoints3D(np.zeros((2, 4, 5, 2)))

    def test_2d_and_3d_vectors_are_unrelated_types(self) -> None:
        assert not issubclass(BatchVector3D, BatchVector2D)
        assert not issubclass(BatchVector2D, BatchVector3D)

    def test_empty_builds_a_zero_row_column(self) -> None:
        assert BatchPosition3D.empty().values.shape == (0, 3)
        assert BatchClassId.empty().values.shape == (0,)
        assert BatchWaypoints3D.empty(2, 5).values.shape == (0, 2, 5, 3)

    def test_empty_requires_a_size_per_wildcard(self) -> None:
        with pytest.raises(ValueError, match="expects 2 size"):
            BatchWaypoints3D.empty(2)


class TestSelection:
    def test_select_preserves_the_semantic_type(self) -> None:
        position = BatchPosition3D(np.arange(12, dtype=np.float64).reshape(4, 3))

        selected = position.select(np.array([3, 1], dtype=np.int64))

        assert isinstance(selected, BatchPosition3D)
        np.testing.assert_array_equal(selected.values, position.values[[3, 1]])

    @pytest.mark.parametrize(
        "selection",
        [slice(1, 3), [1, 2], np.array([False, True, True, False])],
    )
    def test_select_always_produces_independent_data(self, selection: object) -> None:
        position = BatchPosition3D(np.arange(12, dtype=np.float64).reshape(4, 3))

        selected = position.select(selection)

        assert not np.shares_memory(selected.values, position.values)
        assert not selected.values.flags.writeable


class TestValidation:
    def test_confidence_rejects_values_outside_the_unit_interval(self) -> None:
        with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
            BatchConfidence([1.5])

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_confidence_rejects_non_finite_values(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            BatchConfidence([bad])

    def test_waypoints_reject_non_finite_padding(self) -> None:
        with pytest.raises(ValueError, match="only finite values"):
            BatchWaypoints3D(np.full((1, 1, 2, 3), np.nan))

    def test_time_offsets_must_increase(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            BatchTimeOffset([[0, 100, 100]])

    def test_time_offsets_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            BatchTimeOffset([[-1, 100]])


class TestEquality:
    def test_columns_compare_by_value(self) -> None:
        assert BatchPosition3D([[1.0, 2.0, 3.0]]) == BatchPosition3D([[1.0, 2.0, 3.0]])
        assert BatchPosition3D([[1.0, 2.0, 3.0]]) != BatchPosition3D([[1.0, 2.0, 4.0]])

    def test_columns_of_different_types_are_not_equal(self) -> None:
        assert BatchPosition3D([[1.0, 2.0, 3.0]]) != BatchVelocity([[1.0, 2.0, 3.0]])


class TestQuaternion:
    def test_uses_xyzw_order_consistently(self) -> None:
        half_sqrt = np.sqrt(0.5)
        quaternion = BatchQuaternion([[0.0, 0.0, half_sqrt, half_sqrt]])

        rotated = quaternion.as_rotation().apply([[1.0, 0.0, 0.0]])

        np.testing.assert_allclose(rotated, [[0.0, 1.0, 0.0]], atol=1e-12)
        np.testing.assert_array_equal(quaternion.qw, [half_sqrt])
        np.testing.assert_allclose(quaternion.yaw(), [np.pi / 2])

    def test_normalized_rejects_a_zero_quaternion(self) -> None:
        with pytest.raises(ValueError, match="zero quaternion"):
            BatchQuaternion(np.zeros((1, 4))).normalized()

    def test_normalized_scales_to_unit_norm(self) -> None:
        normalized = BatchQuaternion([[0.0, 0.0, 0.0, 2.0]]).normalized()

        np.testing.assert_allclose(np.linalg.norm(normalized.values, axis=1), [1.0])

    def test_yaw_of_an_empty_column_is_empty(self) -> None:
        assert BatchQuaternion.empty().yaw().shape == (0,)


class TestSemanticColumns:
    def test_class_ids_are_int32(self) -> None:
        assert BatchClassId([1, 2]).values.dtype == np.int32

    def test_velocity_exposes_speed(self) -> None:
        np.testing.assert_allclose(BatchVelocity([[3.0, 4.0, 0.0]]).speed, [5.0])

    def test_roi_uses_the_dataset_layout(self) -> None:
        roi = BatchRoi([[10, 20, 30, 40]])

        assert (roi.x_min[0], roi.y_min[0], roi.height[0], roi.width[0]) == (10, 20, 30, 40)
        assert roi.x_max[0] == 50
        assert roi.y_max[0] == 50
        assert roi.area()[0] == 1200

    def test_mask_reports_what_it_keeps(self) -> None:
        mask = BatchMask([True, False, True])

        assert mask.num_selected == 2
        assert mask.indices().tolist() == [0, 2]

    def test_visibility_is_ordered_so_a_threshold_works(self) -> None:
        assert VisibilityLevel.UNAVAILABLE < VisibilityLevel.NONE < VisibilityLevel.FULL


class TestArrowRoundTrip:
    @pytest.mark.parametrize(
        "column",
        [
            BatchPosition3D([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            BatchClassId([1, 2, 3]),
            BatchMask([True, False]),
            BatchWaypoints3D(np.arange(24, dtype=np.float64).reshape(1, 2, 4, 3)),
            BatchPosition3D.empty(),
        ],
    )
    def test_round_trips_through_arrow(self, column: ColumnarComponent) -> None:
        restored = type(column).from_arrow(column.to_arrow())

        assert restored == column
        assert restored.values.dtype == column.values.dtype
        assert restored.values.shape == column.values.shape

    def test_an_explicit_row_shape_recovers_lost_list_sizes(self) -> None:
        column = BatchWaypoints3D(np.arange(24, dtype=np.float64).reshape(1, 2, 4, 3))

        restored = BatchWaypoints3D.from_arrow(column.to_arrow(), row_shape=(2, 4, 3))

        assert restored == column

    def test_rejects_nulls(self) -> None:
        import pyarrow as pa

        with pytest.raises(ValueError, match="must not contain nulls"):
            BatchClassId.from_arrow(pa.array([1, None], type=pa.int32()))


class TestSubclassing:
    def test_a_new_column_type_needs_only_its_layout(self) -> None:
        @define(frozen=True, slots=True)
        class BatchCovariance3D(ColumnarComponent):
            SHAPE = (3, 3)

        covariance = BatchCovariance3D(np.eye(3)[None, ...])

        assert covariance.row_shape == (3, 3)
        assert len(covariance.select([0])) == 1
        assert BatchCovariance3D.from_arrow(covariance.to_arrow()) == covariance

    def test_a_wildcard_layout_is_reported_in_the_error(self) -> None:
        @define(frozen=True, slots=True)
        class BatchRagged(ColumnarComponent):
            SHAPE = (ANY,)

        with pytest.raises(ValueError, match=r"shape \(N, \*\)"):
            BatchRagged(np.zeros(3))
