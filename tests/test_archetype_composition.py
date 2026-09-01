from __future__ import annotations

import numpy as np
import pytest
from conftest import make_detection, make_prediction, make_tracking

from t4perceval import FRAME, BatchDetection3D, BatchPrediction3D, BatchTracking3D, TimePoint
from t4perceval.archetype import BatchClassification2D, BatchDetection2D, BatchTracking2D
from t4perceval.descriptors import CLASS_ID, INSTANCE_ID, POSITION, VELOCITY, WAYPOINTS


class TestCompositionInsteadOfInheritance:
    def test_tracking_is_not_a_subclass_of_detection(self) -> None:
        assert not issubclass(BatchTracking3D, BatchDetection3D)
        assert not issubclass(BatchPrediction3D, BatchTracking3D)

    def test_but_it_carries_every_detection_component(self) -> None:
        tracking = make_tracking([[0.0, 0.0, 0.0]], [7])

        assert tracking.has(*BatchDetection3D.required_descriptors())
        assert tracking.has(INSTANCE_ID)

    def test_a_detection_is_not_mistaken_for_a_tracking(self) -> None:
        detection = make_detection([[0.0, 0.0, 0.0]])

        assert not detection.has(INSTANCE_ID)

    def test_prediction_carries_box_instance_and_trajectory_components(self) -> None:
        prediction = make_prediction([[0.0, 0.0, 0.0]], [7])

        assert prediction.has(*BatchDetection3D.required_descriptors())
        assert prediction.has(*BatchTracking3D.required_descriptors())
        assert prediction.has(WAYPOINTS)

    def test_the_same_field_resolves_to_the_same_descriptor_everywhere(self) -> None:
        for archetype in (BatchDetection3D, BatchTracking3D, BatchPrediction3D):
            assert archetype.descriptor_of("position") == POSITION
            assert archetype.descriptor_of("class_id") == CLASS_ID


class TestIntrospection:
    def test_separates_required_from_optional(self) -> None:
        assert BatchDetection3D.required_descriptors() == (
            POSITION,
            BatchDetection3D.descriptor_of("quaternion"),
            BatchDetection3D.descriptor_of("size"),
            CLASS_ID,
            BatchDetection3D.descriptor_of("confidence"),
        )
        assert VELOCITY in BatchDetection3D.optional_descriptors()
        assert VELOCITY not in BatchDetection3D.required_descriptors()

    def test_descriptors_is_the_union(self) -> None:
        descriptors = set(BatchDetection3D.descriptors())

        assert descriptors == set(BatchDetection3D.required_descriptors()) | set(
            BatchDetection3D.optional_descriptors(),
        )

    def test_archetype_name_defaults_to_the_class_name(self) -> None:
        assert BatchDetection3D.archetype_name() == "BatchDetection3D"

    def test_reports_an_unknown_field(self) -> None:
        with pytest.raises(KeyError, match="no component field named 'nope'"):
            BatchDetection3D.descriptor_of("nope")


class TestGenericSelect:
    @pytest.mark.parametrize(
        "archetype",
        [
            make_detection([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], [1, 2]),
            make_tracking([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], [10, 11], [1, 2]),
            make_prediction([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], [10, 11]),
        ],
    )
    def test_one_implementation_serves_every_archetype(self, archetype: object) -> None:
        selected = archetype.select([1])

        assert type(selected) is type(archetype)
        assert len(selected) == 1
        for descriptor, column in selected.as_components().items():
            assert len(column) == 1, f"{descriptor.component} was not narrowed"

    def test_select_narrows_every_column_consistently(self) -> None:
        tracking = make_tracking([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], [10, 11], [1, 2])

        selected = tracking.select(np.array([False, True]))

        assert selected.instance_id.values.tolist() == [11]
        assert selected.class_id.values.tolist() == [2]
        np.testing.assert_array_equal(selected.position.values, [[1.0, 2.0, 3.0]])

    def test_absent_optional_columns_stay_absent(self) -> None:
        detection = make_detection([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        assert detection.select([0]).velocity is None

    def test_present_optional_columns_are_narrowed_too(self) -> None:
        detection = make_detection(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            velocity=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

        assert detection.select([1]).velocity.values.tolist() == [[0.0, 1.0, 0.0]]

    def test_select_produces_independent_data(self) -> None:
        detection = make_detection([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        selected = detection.select(slice(0, 1))

        assert not np.shares_memory(selected.position.values, detection.position.values)


class TestComponentMapping:
    @pytest.mark.parametrize(
        ("archetype", "kind"),
        [
            (make_detection([[0.0, 0.0, 0.0]], [1]), BatchDetection3D),
            (make_tracking([[0.0, 0.0, 0.0]], [10]), BatchTracking3D),
            (make_prediction([[0.0, 0.0, 0.0]], [10]), BatchPrediction3D),
            (
                BatchDetection2D(roi=[[0, 0, 10, 10]], class_id=[1], confidence=[0.5]),
                BatchDetection2D,
            ),
            (
                BatchTracking2D(
                    roi=[[0, 0, 10, 10]], class_id=[1], confidence=[0.5], instance_id=[3]
                ),
                BatchTracking2D,
            ),
            (BatchClassification2D(class_id=[1], confidence=[0.5]), BatchClassification2D),
        ],
    )
    def test_round_trips_through_a_descriptor_mapping(self, archetype: object, kind: type) -> None:
        assert kind.from_components(archetype.as_components()) == archetype

    def test_omits_absent_optional_components(self) -> None:
        detection = make_detection([[0.0, 0.0, 0.0]])

        assert VELOCITY not in detection.as_components()

    def test_names_every_missing_required_component(self) -> None:
        with pytest.raises(ValueError, match="missing required component"):
            BatchDetection3D.from_components({POSITION: make_detection([[0.0, 0.0, 0.0]]).position})


class TestValidation:
    def test_rejects_mismatched_column_lengths(self) -> None:
        with pytest.raises(ValueError, match="confidence has length 1, expected 2"):
            BatchDetection3D(
                position=np.zeros((2, 3)),
                quaternion=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
                size=np.ones((2, 3)),
                class_id=[1, 1],
                confidence=[0.9],
            )

    def test_rejects_a_mismatched_optional_column(self) -> None:
        with pytest.raises(ValueError, match="velocity has length 1, expected 2"):
            make_detection([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], velocity=[[1.0, 0.0, 0.0]])

    def test_coerces_raw_arrays_into_components(self) -> None:
        detection = make_detection([[0.0, 0.0, 0.0]])

        assert detection.position.values.dtype == np.float64
        assert detection.class_id.values.dtype == np.int32


class TestChunkRoundTrip:
    def test_to_chunk_carries_the_time_and_frame(self) -> None:
        detection = make_detection([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        chunk = detection.to_chunk(
            "/estimation/objects",
            at=TimePoint.at(frame=4),
            frame_id="base_link",
        )

        assert str(chunk.entity_path) == "/estimation/objects"
        assert chunk.index(FRAME).times.tolist() == [4]
        assert (chunk.num_rows, chunk.num_partitions) == (2, 1)
        assert chunk.frame_id == "base_link"
        assert BatchDetection3D.from_chunk(chunk) == detection

    def test_static_data_needs_no_time(self) -> None:
        chunk = make_detection([[0.0, 0.0, 0.0]]).to_chunk("/x", is_static=True)

        assert chunk.is_static
        assert chunk.timelines == ()

    def test_requires_a_time_unless_static(self) -> None:
        with pytest.raises(ValueError, match="either a TimePoint or is_static"):
            make_detection([[0.0, 0.0, 0.0]]).to_chunk("/x")
