from __future__ import annotations

import pytest

from t4perceval import ComponentDescriptor
from t4perceval.archetype import BatchDetection3D, BatchTracking3D
from t4perceval.descriptors import POSITION


class TestIdentity:
    def test_only_the_component_name_decides_identity(self) -> None:
        tagged = ComponentDescriptor("position", archetype="BatchDetection3D")
        bare = ComponentDescriptor("position")

        assert tagged == bare
        assert hash(tagged) == hash(bare)
        assert len({tagged, bare}) == 1

    def test_tagging_annotates_provenance_without_changing_identity(self) -> None:
        tagged = POSITION.tagged("BatchDetection3D")

        assert tagged == POSITION
        assert tagged.archetype == "BatchDetection3D"
        assert tagged.component == POSITION.component

    def test_tagging_with_the_same_archetype_is_a_no_op(self) -> None:
        tagged = POSITION.tagged("X")

        assert tagged.tagged("X") is tagged

    def test_the_same_column_is_addressable_from_every_archetype(self) -> None:
        detection = BatchDetection3D.descriptor_of("position")
        tracking = BatchTracking3D.descriptor_of("position")

        assert detection == tracking == POSITION
        assert detection.archetype == "BatchDetection3D"
        assert tracking.archetype == "BatchTracking3D"

    def test_rejects_an_empty_name(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ComponentDescriptor("")


class TestNaming:
    def test_of_builds_a_qualified_name(self) -> None:
        descriptor = ComponentDescriptor.of("BatchDetection3D", "position")

        assert descriptor.component == "BatchDetection3D:position"
        assert descriptor.field_name == "position"

    def test_field_name_falls_back_to_the_whole_name(self) -> None:
        assert ComponentDescriptor("position").field_name == "position"

    def test_str_is_the_component_name(self) -> None:
        assert str(POSITION) == "position"
