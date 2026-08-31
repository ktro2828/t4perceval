from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from t4perceval.dataclass import (
    BatchDetection3D,
    BatchPrediction3D,
    BatchTracking3D,
    BatchTrajectory3D,
    Header,
    SemanticSegmentation2D,
    SemanticSegmentation3D,
    TrajectoryMode3D,
)
from t4perceval.dataclass.archetype import BatchDetection3D as ArchetypeBatchDetection3D


def detection_columns() -> dict[str, Any]:
    return {
        "header": Header(timestamp_ns=1_000, frame_id="map"),
        "position": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        "quaternion": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "size": [[4.0, 2.0, 1.5], [1.0, 1.0, 2.0]],
        "class_id": [1, 2],
        "confidence": [0.9, 0.8],
        "velocity": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    }


def make_trajectories() -> BatchTrajectory3D:
    return BatchTrajectory3D(
        positions=np.arange(36, dtype=np.float64).reshape(2, 2, 3, 3),
        confidences=[[0.6, 0.4], [1.0, 0.0]],
        time_offsets_ns=[1, 3, 6],
    )


class TestObjectTasks:
    def test_detection_validates_component_lengths(self) -> None:
        with pytest.raises(ValueError, match="confidence has length 1, expected 2"):
            BatchDetection3D(
                header=Header(1_000, "map"),
                position=np.zeros((2, 3)),
                quaternion=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
                size=np.ones((2, 3)),
                class_id=[1, 1],
                confidence=[0.9],
            )

    def test_tracking_inherits_detection_and_keeps_columns_aligned(self) -> None:
        tracking = BatchTracking3D(**detection_columns(), instance_id=[10, 11])

        selected = tracking.select(np.array([False, True]))

        assert isinstance(tracking, BatchDetection3D)
        assert isinstance(selected, BatchTracking3D)
        assert len(selected) == 1
        assert selected.instance_id.values.tolist() == [11]
        assert selected.class_id.values.tolist() == [2]
        np.testing.assert_array_equal(selected.position.values, [[1.0, 2.0, 3.0]])

    def test_tracking_validates_instance_id_length(self) -> None:
        with pytest.raises(ValueError, match="instance_id has length 1, expected 2"):
            BatchTracking3D(**detection_columns(), instance_id=[10])

    def test_archetype_import_is_the_canonical_type(self) -> None:
        assert ArchetypeBatchDetection3D is BatchDetection3D


class TestPrediction:
    def test_builds_columnar_trajectories_from_modes(self) -> None:
        objects = [
            [
                TrajectoryMode3D(
                    confidence=0.6,
                    time_offset_ns=[1, 2],
                    position=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                ),
                TrajectoryMode3D(
                    confidence=0.4,
                    time_offset_ns=[1, 2],
                    position=[[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                ),
            ],
            [
                TrajectoryMode3D(
                    confidence=0.7,
                    time_offset_ns=[1, 2],
                    position=[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                ),
                TrajectoryMode3D(
                    confidence=0.3,
                    time_offset_ns=[1, 2],
                    position=[[2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
                ),
            ],
        ]

        trajectories = BatchTrajectory3D.from_modes(objects)

        assert len(trajectories) == 2
        assert trajectories.positions.shape == (2, 2, 2, 3)
        assert trajectories.confidences.tolist() == [[0.6, 0.4], [0.7, 0.3]]
        assert trajectories.time_offsets_ns.tolist() == [1, 2]
        restored = trajectories.to_modes(0)
        assert [mode.confidence for mode in restored] == [0.6, 0.4]
        np.testing.assert_array_equal(restored[0].position.values, objects[0][0].position.values)

    def test_mode_validates_confidence_and_state_alignment(self) -> None:
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            TrajectoryMode3D(
                confidence=1.1,
                time_offset_ns=[1],
                position=[[0.0, 0.0, 0.0]],
            )

    def test_batch_validates_dense_shapes(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(N, M, T, 3\)"):
            BatchTrajectory3D(
                positions=np.zeros((2, 3, 3)),
                confidences=np.ones((2, 3)),
                time_offsets_ns=[1, 2, 3],
            )

        with pytest.raises(ValueError, match=r"expected \(2, 3\)"):
            BatchTrajectory3D(
                positions=np.zeros((2, 3, 4, 3)),
                confidences=np.ones((2, 2)),
                time_offsets_ns=[1, 2, 3, 4],
            )

    def test_from_modes_requires_a_shared_time_axis(self) -> None:
        with pytest.raises(ValueError, match="same time_offset_ns"):
            BatchTrajectory3D.from_modes(
                [
                    [
                        TrajectoryMode3D(
                            confidence=1.0,
                            time_offset_ns=[1, 2],
                            position=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        )
                    ],
                    [
                        TrajectoryMode3D(
                            confidence=1.0,
                            time_offset_ns=[1, 3],
                            position=[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                        )
                    ],
                ]
            )

        with pytest.raises(ValueError, match="position has length 1, expected 2"):
            TrajectoryMode3D(
                confidence=1.0,
                time_offset_ns=[1, 2],
                position=[[0.0, 0.0, 0.0]],
            )

    def test_inherits_tracking_and_selects_dense_trajectories(self) -> None:
        trajectories = make_trajectories()
        prediction = BatchPrediction3D(
            **detection_columns(),
            instance_id=[10, 11],
            trajectories=trajectories,
        )

        selected = prediction.select(np.array([1], dtype=np.int64))

        assert isinstance(prediction, BatchTracking3D)
        assert isinstance(selected, BatchPrediction3D)
        assert selected.instance_id.values.tolist() == [11]
        assert selected.trajectories.positions.shape == (1, 2, 3, 3)
        assert selected.trajectories.confidences.tolist() == [[1.0, 0.0]]
        assert selected.trajectories.time_offsets_ns.tolist() == [1, 3, 6]
        np.testing.assert_array_equal(
            selected.trajectories.positions,
            trajectories.positions[1:2],
        )

    def test_validates_trajectory_object_count(self) -> None:
        trajectories = make_trajectories().select(np.array([0]))

        with pytest.raises(ValueError, match="trajectories has 1 objects, expected 2"):
            BatchPrediction3D(
                **detection_columns(),
                instance_id=[10, 11],
                trajectories=trajectories,
            )

    def test_rejects_non_increasing_mode_timestamps(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            BatchTrajectory3D(
                positions=np.zeros((1, 1, 2, 3)),
                confidences=[[1.0]],
                time_offsets_ns=[1, 1],
            )


class TestSemanticSegmentation:
    def test_semantic_2d_validates_pixel_class_alignment(self) -> None:
        segmentation = SemanticSegmentation2D(
            header=Header(1_000, "camera"),
            pixel=np.arange(6, dtype=np.int32),
            class_id=[0, 1, 1, 2, 2, 0],
        )

        assert len(segmentation.pixel) == len(segmentation.class_id)
        np.testing.assert_array_equal(segmentation.class_id.values.reshape(2, 3)[1], [2, 2, 0])

    def test_semantic_3d_validates_point_class_alignment(self) -> None:
        with pytest.raises(ValueError, match="class_id has length 1, expected 2"):
            SemanticSegmentation3D(
                header=Header(1_000, "lidar"),
                point=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                class_id=[1],
            )
