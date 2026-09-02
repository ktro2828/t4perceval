from __future__ import annotations

import numpy as np
import pytest
from conftest import make_predictions

from t4perceval import (
    Detections3D,
    Predictions3D,
    SemanticSegmentation2D,
    SemanticSegmentation3D,
    Trajectories3D,
    Trackings3D,
    TrajectoryMode3D,
)
from t4perceval.archetype import Detections3D as ArchetypeDetections3D
from t4perceval.archetype import MatchResults
from t4perceval.component import MatchStatus


def modes(*confidences: float, time_offsets: list[int] | None = None) -> list[TrajectoryMode3D]:
    offsets = time_offsets or [1, 2]
    return [
        TrajectoryMode3D(
            confidence=confidence,
            time_offset_ns=offsets,
            position=[[float(index), 0.0, 0.0] for index in range(len(offsets))],
        )
        for confidence in confidences
    ]


class TestPublicApi:
    def test_the_archetype_module_exposes_the_canonical_types(self) -> None:
        assert ArchetypeDetections3D is Detections3D

    def test_archetypes_use_semantic_names_instead_of_the_batch_prefix(self) -> None:
        archetypes = (
            Detections3D,
            Trackings3D,
            Predictions3D,
            Trajectories3D,
            SemanticSegmentation2D,
            SemanticSegmentation3D,
            MatchResults,
        )

        assert all(not archetype.__name__.startswith("Batch") for archetype in archetypes)


class TestTrajectoryMode:
    def test_validates_confidence(self) -> None:
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            TrajectoryMode3D(confidence=1.1, time_offset_ns=[1], position=[[0.0, 0.0, 0.0]])

    def test_validates_state_alignment(self) -> None:
        with pytest.raises(ValueError, match="position has length 1, expected 2"):
            TrajectoryMode3D(confidence=1.0, time_offset_ns=[1, 2], position=[[0.0, 0.0, 0.0]])

    def test_requires_a_strictly_increasing_time_axis(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            TrajectoryMode3D(
                confidence=1.0,
                time_offset_ns=[2, 1],
                position=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            )

    def test_requires_at_least_one_timestep(self) -> None:
        with pytest.raises(ValueError, match="at least one timestep"):
            TrajectoryMode3D(confidence=1.0, time_offset_ns=[], position=np.zeros((0, 3)))


class TestTrajectories3D:
    def test_builds_a_dense_batch_from_modes(self) -> None:
        trajectories = Trajectories3D.from_modes([modes(0.6, 0.4), modes(0.7, 0.3)])

        assert len(trajectories) == 2
        assert (trajectories.num_modes, trajectories.num_timesteps) == (2, 2)
        assert trajectories.waypoints.values.shape == (2, 2, 2, 3)
        assert trajectories.mode_confidence.values.tolist() == [[0.6, 0.4], [0.7, 0.3]]
        assert trajectories.time_offset.values.tolist() == [[1, 2], [1, 2]]

    def test_round_trips_back_to_modes(self) -> None:
        trajectories = Trajectories3D.from_modes([modes(0.6, 0.4), modes(0.7, 0.3)])

        restored = trajectories.to_modes(0)

        assert [mode.confidence for mode in restored] == [0.6, 0.4]
        assert restored[0].time_offset_ns.tolist() == [1, 2]

    def test_requires_a_shared_time_axis(self) -> None:
        with pytest.raises(ValueError, match="same time_offset_ns"):
            Trajectories3D.from_modes(
                [modes(1.0), modes(1.0, time_offsets=[1, 3])],
            )

    def test_requires_a_uniform_mode_count(self) -> None:
        with pytest.raises(ValueError, match="same number of trajectory modes"):
            Trajectories3D.from_modes([modes(0.5, 0.5), modes(1.0)])

    def test_rejects_an_empty_object_list(self) -> None:
        with pytest.raises(ValueError, match="at least one object"):
            Trajectories3D.from_modes([])

    def test_rejects_an_object_without_modes(self) -> None:
        with pytest.raises(ValueError, match="at least one trajectory mode"):
            Trajectories3D.from_modes([[]])

    def test_empty_keeps_a_fixed_mode_and_time_shape(self) -> None:
        empty = Trajectories3D.empty(num_modes=3, num_timesteps=4)

        assert len(empty) == 0
        assert (empty.num_modes, empty.num_timesteps) == (3, 4)

    def test_empty_rejects_a_degenerate_shape(self) -> None:
        with pytest.raises(ValueError, match="must both be positive"):
            Trajectories3D.empty(num_modes=0, num_timesteps=4)

    def test_reports_an_out_of_range_object(self) -> None:
        trajectories = Trajectories3D.from_modes([modes(1.0)])

        with pytest.raises(IndexError, match="object index out of range"):
            trajectories.to_modes(1)

    def test_mode_confidences_are_not_forced_to_sum_to_one(self) -> None:
        # Models frequently emit unnormalized scores; normalizing here would silently
        # change the metric, so only the [0, 1] range is enforced.
        trajectories = Trajectories3D.from_modes([modes(0.9, 0.9)])

        assert trajectories.mode_confidence.values.sum() == pytest.approx(1.8)


class TestTrajectoryShapeAgreement:
    def base(self, **overrides: object) -> dict[str, object]:
        columns: dict[str, object] = {
            "waypoints": np.zeros((1, 2, 3, 3)),
            "mode_confidence": [[0.5, 0.5]],
        }
        columns.update(overrides)
        return columns

    def test_accepts_agreeing_masks(self) -> None:
        trajectories = Trajectories3D(
            **self.base(
                mode_valid=[[True, False]],
                timestep_valid=np.ones((1, 2, 3), dtype=bool),
                time_offset=[[1, 2, 3]],
            ),
        )

        assert len(trajectories) == 1

    def test_rejects_a_mode_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="mode_confidence has 3 modes, expected 2"):
            Trajectories3D(**self.base(mode_confidence=[[0.3, 0.3, 0.4]]))

    def test_rejects_a_mode_mask_mismatch(self) -> None:
        with pytest.raises(ValueError, match=r"mode_valid has row shape \(3,\)"):
            Trajectories3D(**self.base(mode_valid=[[True, True, True]]))

    def test_rejects_a_timestep_mask_mismatch(self) -> None:
        with pytest.raises(ValueError, match=r"timestep_valid has row shape \(2, 4\)"):
            Trajectories3D(**self.base(timestep_valid=np.ones((1, 2, 4), dtype=bool)))

    def test_rejects_a_time_axis_mismatch(self) -> None:
        with pytest.raises(ValueError, match=r"time_offset has row shape \(2,\)"):
            Trajectories3D(**self.base(time_offset=[[1, 2]]))


class TestPrediction:
    def test_selects_dense_trajectories_along_with_the_boxes(self) -> None:
        prediction = make_predictions([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], [10, 11])

        selected = prediction.select(np.array([1], dtype=np.int64))

        assert isinstance(selected, Predictions3D)
        assert selected.instance_id.values.tolist() == [11]
        assert selected.waypoints.values.shape == (1, 2, 3, 3)
        assert selected.mode_confidence.values.shape == (1, 2)
        assert selected.time_offset.values.tolist() == [[100, 200, 300]]

    def test_validates_the_trajectory_row_count(self) -> None:
        with pytest.raises(ValueError, match="waypoints has length 1, expected 2"):
            Predictions3D(
                position=np.zeros((2, 3)),
                quaternion=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
                size=np.ones((2, 3)),
                class_id=[1, 1],
                confidence=[0.9, 0.9],
                instance_id=[1, 2],
                waypoints=np.zeros((1, 2, 3, 3)),
                mode_confidence=[[0.5, 0.5]],
            )

    def test_reports_its_trajectory_shape(self) -> None:
        prediction = make_predictions([[0.0, 0.0, 0.0]], [1], num_modes=4, num_timesteps=6)

        assert (prediction.num_modes, prediction.num_timesteps) == (4, 6)


class TestSemanticSegmentation:
    def test_a_class_per_pixel(self) -> None:
        segmentation = SemanticSegmentation2D(
            pixel=np.arange(6, dtype=np.int32),
            class_id=[0, 1, 1, 2, 2, 0],
        )

        assert len(segmentation) == 6
        assert segmentation.pixel.values.dtype == np.int32
        np.testing.assert_array_equal(segmentation.class_id.values.reshape(2, 3)[1], [2, 2, 0])

    def test_a_class_per_point(self) -> None:
        segmentation = SemanticSegmentation3D(
            point=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            class_id=[1, 2],
        )

        assert len(segmentation) == 2

    def test_validates_point_class_alignment(self) -> None:
        with pytest.raises(ValueError, match="class_id has length 1, expected 2"):
            SemanticSegmentation3D(
                point=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                class_id=[1],
            )

    def test_segmentation_supports_the_shared_selection_api(self) -> None:
        segmentation = SemanticSegmentation2D(pixel=[0, 1, 2], class_id=[0, 1, 2])

        assert segmentation.select([2, 0]).class_id.values.tolist() == [2, 0]


class TestMatchResult:
    def test_counts_each_verdict(self) -> None:
        result = MatchResults(
            est_index=[0, 1, -1],
            gt_index=[0, -1, 2],
            matching_score=[0.5, np.nan, np.nan],
            match_status=[MatchStatus.TP, MatchStatus.FP, MatchStatus.FN],
            threshold=[1.0, 1.0, 1.0],
        )

        assert (result.num_tp, result.num_fp, result.num_fn) == (1, 1, 1)
        assert result.count(MatchStatus.TP) == 1

    def test_empty_has_no_rows(self) -> None:
        assert len(MatchResults.empty()) == 0

    def test_rejects_an_unknown_status(self) -> None:
        with pytest.raises(ValueError, match="unknown values"):
            MatchResults(
                est_index=[0],
                gt_index=[0],
                matching_score=[0.0],
                match_status=[9],
                threshold=[1.0],
            )
