from __future__ import annotations

import numpy as np
import pytest

from t4perceval.core.selection import normalize_selection


class TestAcceptedInputs:
    @pytest.mark.parametrize(
        ("selection", "expected"),
        [
            (slice(1, 3), [1, 2]),
            (slice(None, None, 2), [0, 2, 4]),
            (slice(None, None, -1), [4, 3, 2, 1, 0]),
            ([0, 2], [0, 2]),
            (np.array([0, 2], dtype=np.int64), [0, 2]),
            (np.array([True, False, True, False, False]), [0, 2]),
            ([True, False, True, False, False], [0, 2]),
            ([-1, -5], [4, 0]),
            ([], []),
            ([2, 2, 2], [2, 2, 2]),
            ([4, 3], [4, 3]),
        ],
    )
    def test_normalizes_every_documented_form(
        self,
        selection: object,
        expected: list[int],
    ) -> None:
        result = normalize_selection(selection, length=5)

        assert result.dtype == np.int64
        assert result.tolist() == expected

    def test_duplicate_and_reversed_indices_are_allowed(self) -> None:
        assert normalize_selection([3, 1, 1], length=4).tolist() == [3, 1, 1]


class TestRejectedInputs:
    def test_rejects_an_out_of_range_index(self) -> None:
        with pytest.raises(IndexError, match="index 5 is out of range for length 5"):
            normalize_selection([5], length=5)

    def test_rejects_a_too_negative_index(self) -> None:
        with pytest.raises(IndexError, match="index -6 is out of range"):
            normalize_selection([-6], length=5)

    def test_rejects_a_mismatched_boolean_mask(self) -> None:
        with pytest.raises(ValueError, match="Boolean selection has length 3, expected 5"):
            normalize_selection(np.array([True, False, True]), length=5)

    def test_rejects_a_multidimensional_selection(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            normalize_selection(np.zeros((2, 2), dtype=np.int64), length=5)

    def test_rejects_a_float_selection(self) -> None:
        with pytest.raises(TypeError, match="got dtype float64"):
            normalize_selection([0.5], length=5)

    def test_rejects_a_negative_length(self) -> None:
        with pytest.raises(ValueError, match="length must be non-negative"):
            normalize_selection(slice(None), length=-1)
