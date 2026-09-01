"""Matching thresholds, uniform and per class."""

from __future__ import annotations

import numpy as np
import pytest

from t4perceval import LabelRegistry
from t4perceval.system import Thresholds


class TestCoercion:
    def test_a_number_becomes_a_uniform_threshold(self) -> None:
        threshold = Thresholds.coerce(1.5)

        assert threshold.default == 1.5
        assert threshold.is_uniform

    def test_an_int_is_accepted(self) -> None:
        assert Thresholds.coerce(2).default == 2.0

    def test_an_existing_threshold_passes_through(self) -> None:
        threshold = Thresholds(1.0, by_class={"car": 2.0})

        assert Thresholds.coerce(threshold) is threshold

    def test_a_mapping_needs_a_default(self) -> None:
        with pytest.raises(ValueError, match="needs a default for the classes it omits"):
            Thresholds.coerce({"car": 2.0})

    def test_a_mapping_with_a_default_is_accepted(self) -> None:
        threshold = Thresholds.coerce({"car": 2.0}, default=1.0)

        assert threshold.default == 1.0
        assert dict(threshold.by_class) == {"car": 2.0}

    def test_an_empty_mapping_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            Thresholds.coerce({}, default=1.0)

    @pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
    def test_a_non_finite_threshold_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            Thresholds(bad)

    def test_a_non_finite_per_class_threshold_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            Thresholds(1.0, by_class={"car": np.inf})


class TestResolve:
    def test_a_uniform_threshold_applies_everywhere(self) -> None:
        resolved = Thresholds.coerce(1.5).resolve([0, 1, 2])

        assert resolved.tolist() == [1.5, 1.5, 1.5]
        assert resolved.dtype == np.float64

    def test_named_classes_override_the_default(self, labels: LabelRegistry) -> None:
        threshold = Thresholds(1.0, by_class={"car": 2.0, "pedestrian": 0.5})

        resolved = threshold.resolve([0, 1, 2, 0], labels)

        assert resolved.tolist() == [2.0, 1.0, 0.5, 2.0]

    def test_class_ids_need_no_registry(self) -> None:
        threshold = Thresholds(1.0, by_class={0: 9.0})

        assert threshold.resolve([0, 1]).tolist() == [9.0, 1.0]

    def test_names_and_ids_can_be_mixed(self, labels: LabelRegistry) -> None:
        threshold = Thresholds(1.0, by_class={"car": 2.0, 2: 0.5})

        assert threshold.resolve([0, 1, 2], labels).tolist() == [2.0, 1.0, 0.5]

    def test_names_need_a_registry(self) -> None:
        threshold = Thresholds(1.0, by_class={"car": 2.0})

        with pytest.raises(ValueError, match="require a LabelRegistry"):
            threshold.resolve([0])

    def test_an_unknown_name_raises_rather_than_leaving_the_default(
        self,
        labels: LabelRegistry,
    ) -> None:
        threshold = Thresholds(1.0, by_class={"spaceship": 2.0})

        with pytest.raises(KeyError, match="Unknown class name 'spaceship'"):
            threshold.resolve([0], labels)

    def test_an_empty_column_resolves_to_an_empty_result(self) -> None:
        assert Thresholds.coerce(1.0).resolve([]).shape == (0,)

    def test_a_class_the_mapping_omits_keeps_the_default(self, labels: LabelRegistry) -> None:
        threshold = Thresholds(1.0, by_class={"car": 2.0})

        assert threshold.resolve([1], labels).tolist() == [1.0]
