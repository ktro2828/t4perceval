"""The scaffolding both importers share, exercised without either optional extra."""

from __future__ import annotations

import numpy as np
import pytest

from t4perceval import InstanceRegistry
from t4perceval.importer._columns import (
    NAN3,
    TrajectoryColumns,
    column,
    encode_instance_ids,
    normalize_quaternions,
    padded_trajectory,
    resolve_emit,
    select_kept,
    wxyz_to_xyzw,
)
from t4perceval.importer._importer import (
    import_metadata,
    narrow,
    package_version,
    pin_trajectory_shape,
    single_frame_id,
)
from t4perceval.recording import SourceInfo


class TestColumn:
    def test_stacks_values_to_the_row_shape(self) -> None:
        stacked = column([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 2, (3,), np.float64)

        assert stacked.shape == (2, 3)
        assert stacked.dtype == np.float64

    def test_an_empty_batch_keeps_the_row_shape(self) -> None:
        # `np.asarray([])` would collapse to (0,), which a (N, 3) component rejects.
        assert column([], 0, (3,), np.float64).shape == (0, 3)
        assert column([], 0, (), np.int32).shape == (0,)

    def test_nan3_is_a_missing_vector(self) -> None:
        assert NAN3.shape == (3,)
        assert np.isnan(NAN3).all()


class TestSelectKept:
    def test_returns_the_kept_items_and_their_indices(self) -> None:
        kept, indices = select_kept(["a", "b", "c"], np.array([True, False, True]))

        assert kept == ["a", "c"]
        assert indices.tolist() == [0, 2]
        assert indices.dtype == np.int64

    def test_an_empty_mask_yields_an_empty_int64_index(self) -> None:
        kept, indices = select_kept([], np.empty(0, dtype=np.bool_))

        assert kept == []
        assert indices.shape == (0,)
        assert indices.dtype == np.int64


class TestResolveEmit:
    @pytest.mark.parametrize("setting", ["always", "never"])
    def test_an_explicit_setting_is_kept(self, setting: str) -> None:
        assert resolve_emit(setting, present=True) == setting  # type: ignore[arg-type]
        assert resolve_emit(setting, present=False) == setting  # type: ignore[arg-type]

    def test_auto_follows_the_source(self) -> None:
        assert resolve_emit("auto", present=True) == "always"
        assert resolve_emit("auto", present=False) == "never"


class TestQuaternions:
    def test_wxyz_to_xyzw_handles_one_and_many(self) -> None:
        assert wxyz_to_xyzw(np.array([1.0, 2.0, 3.0, 4.0])).tolist() == [2.0, 3.0, 4.0, 1.0]
        assert wxyz_to_xyzw(np.array([[1.0, 2.0, 3.0, 4.0]])).tolist() == [[2.0, 3.0, 4.0, 1.0]]

    def test_normalizes_to_unit_rows(self) -> None:
        unit = normalize_quaternions(np.array([[0.0, 0.0, 0.0, 2.0]]), what="Box")

        assert unit.tolist() == [[0.0, 0.0, 0.0, 1.0]]

    def test_an_empty_column_passes_through(self) -> None:
        empty = np.empty((0, 4), dtype=np.float64)

        assert normalize_quaternions(empty, what="Box") is empty

    def test_rejects_a_zero_row_naming_the_source(self) -> None:
        rows = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="Object 1 has a zero quaternion"):
            normalize_quaternions(rows, what="Object")


class TestEncodeInstanceIds:
    def test_a_namespace_keeps_two_sources_apart(self) -> None:
        instances = InstanceRegistry()

        gt = encode_instance_ids(instances, ["a"], namespace="gt")
        est = encode_instance_ids(instances, ["a"], namespace="est")

        assert gt.tolist() != est.tolist()
        assert "gt/a" in instances
        assert "est/a" in instances
        assert "a" not in instances

    def test_no_namespace_interns_the_bare_name(self) -> None:
        instances = InstanceRegistry()

        ids = encode_instance_ids(instances, ["a", "b"], namespace="")

        assert "a" in instances
        assert "b" in instances
        assert len(instances) == 2
        assert ids.tolist() == [instances.intern("a"), instances.intern("b")]


class TestPaddedTrajectory:
    def test_holds_station_and_is_fully_masked(self) -> None:
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        columns = padded_trajectory(positions, num_modes=2, num_timesteps=3)

        assert isinstance(columns, TrajectoryColumns)
        assert columns.waypoints.shape == (2, 2, 3, 3)
        assert (columns.waypoints[1] == [4.0, 5.0, 6.0]).all()
        assert not columns.mode_valid.any()
        assert not columns.timestep_valid.any()
        assert (columns.mode_confidence == 0.0).all()
        assert columns.time_offset.tolist() == [[1, 2, 3], [1, 2, 3]]

    def test_the_arrays_are_writable_for_the_caller_to_fill(self) -> None:
        columns = padded_trajectory(np.zeros((1, 3)), num_modes=1, num_timesteps=1)

        columns.mode_valid[0, 0] = True

        assert columns.mode_valid.tolist() == [[True]]

    def test_zero_rows_is_well_formed(self) -> None:
        columns = padded_trajectory(np.empty((0, 3)), num_modes=2, num_timesteps=4)

        assert columns.waypoints.shape == (0, 2, 4, 3)
        assert columns.time_offset.shape == (0, 4)


class TestNarrow:
    def test_none_keeps_everything_with_positions(self) -> None:
        assert narrow(["a", "b", "c"], None) == ((0, "a"), (1, "b"), (2, "c"))

    def test_a_slice_keeps_positions_in_the_full_sequence(self) -> None:
        assert narrow(["a", "b", "c", "d"], slice(1, 3)) == ((1, "b"), (2, "c"))

    def test_explicit_positions_are_taken_in_the_given_order(self) -> None:
        assert narrow(["a", "b", "c"], [2, 0]) == ((2, "c"), (0, "a"))


class TestSingleFrameId:
    def test_no_objects_means_no_frame(self) -> None:
        assert single_frame_id([], what="Scene") is None

    def test_one_frame_is_returned(self) -> None:
        assert single_frame_id(["map", "map"], what="Scene") == "map"

    def test_a_mixture_is_refused_naming_the_selection(self) -> None:
        with pytest.raises(
            ValueError, match=r"Topic '/x' mixes coordinate frames: \['base_link', 'map'\]"
        ):
            single_frame_id(["map", "base_link"], what="Topic '/x'")


class TestPinTrajectoryShape:
    def test_unpinned_takes_the_fitted_shape(self) -> None:
        assert pin_trajectory_shape((3, 8), num_modes=None, num_timesteps=None) == (3, 8)

    def test_a_pin_overrides_its_axis_only(self) -> None:
        assert pin_trajectory_shape((3, 8), num_modes=6, num_timesteps=None) == (6, 8)
        assert pin_trajectory_shape((3, 8), num_modes=None, num_timesteps=1) == (3, 1)


class TestImportMetadata:
    def test_stamps_provenance_and_leaves_the_fingerprint_to_the_recording(self) -> None:
        source = SourceInfo("t4", "/data")

        metadata = import_metadata(source, frame_id="map")

        assert metadata.sources == (source,)
        assert metadata.frame_id == "map"
        assert metadata.t4perceval_version == package_version()
        assert metadata.created_at_ns > 0
        assert metadata.labels_fingerprint == ""
