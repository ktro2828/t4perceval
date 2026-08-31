from __future__ import annotations

import numpy as np
import pytest

from t4perceval.dataclass.component import (
    BatchPosition2D,
    BatchPosition3D,
    BatchQuaternion,
    BatchSize3D,
    BatchVector3D,
    BatchVelocity,
)


class TestVectorComponents:
    def test_position3d_owns_a_fixed_width_column(self) -> None:
        values = np.arange(9, dtype=np.float64).reshape(3, 3)

        position = BatchPosition3D(values)

        assert position.values is values
        assert position.as_array() is position.values
        assert np.shares_memory(position.x, position.values)
        np.testing.assert_array_equal(position.y, values[:, 1])
        assert len(position) == 3

    def test_component_normalizes_dtype_and_layout(self) -> None:
        values = np.arange(12, dtype=np.float32).reshape(3, 4)[:, :3]

        position = BatchPosition3D(values)

        assert position.values.dtype == np.float64
        assert position.values.flags.c_contiguous

    @pytest.mark.parametrize("component", [BatchPosition3D, BatchSize3D, BatchVelocity])
    def test_3d_component_rejects_invalid_shape(
        self, component: type[BatchVector3D]
    ) -> None:
        with pytest.raises(ValueError, match=rf"{component.__name__} must have shape \(N, 3\)"):
            component(np.zeros((3, 2)))

    def test_2d_component_rejects_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
            BatchPosition2D(np.zeros((3, 3)))

    def test_select_preserves_semantic_component_type(self) -> None:
        position = BatchPosition3D(np.arange(12).reshape(4, 3))

        selected = position.select(np.array([3, 1], dtype=np.int64))

        assert isinstance(selected, BatchPosition3D)
        np.testing.assert_array_equal(selected.values, position.values[[3, 1]])


class TestQuaternion:
    def test_uses_xyzw_order_consistently(self) -> None:
        half_sqrt = np.sqrt(0.5)
        quaternion = BatchQuaternion([[0.0, 0.0, half_sqrt, half_sqrt]])

        rotated = quaternion.as_rotation().apply([[1.0, 0.0, 0.0]])

        np.testing.assert_allclose(rotated, [[0.0, 1.0, 0.0]], atol=1e-12)
        np.testing.assert_array_equal(quaternion.as_array(), quaternion.values)
        np.testing.assert_array_equal(quaternion.qw, [half_sqrt])

    def test_normalized_rejects_zero_quaternion(self) -> None:
        with pytest.raises(ValueError, match="zero quaternion"):
            BatchQuaternion(np.zeros((1, 4))).normalized()
