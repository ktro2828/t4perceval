from __future__ import annotations

import pytest

from t4perceval import UNKNOWN_CLASS_ID, ClassInfo, InstanceRegistry, LabelRegistry


class TestLabelRegistry:
    def test_assigns_ids_in_order(self, labels: LabelRegistry) -> None:
        assert labels.names == ("car", "truck", "pedestrian")
        assert labels.class_id("car") == 0
        assert labels.class_id("pedestrian") == 2
        assert labels.name(1) == "truck"
        assert len(labels) == 3

    def test_ignores_repeated_names(self) -> None:
        registry = LabelRegistry.from_names(["car", "truck", "car"])

        assert registry.names == ("car", "truck")

    def test_reports_an_unknown_name(self, labels: LabelRegistry) -> None:
        assert labels.class_id_or("bus", UNKNOWN_CLASS_ID) == UNKNOWN_CLASS_ID
        assert "bus" not in labels
        assert "car" in labels

        with pytest.raises(KeyError, match="Unknown class name 'bus'"):
            labels.class_id("bus")

    def test_reports_an_unknown_id(self, labels: LabelRegistry) -> None:
        with pytest.raises(KeyError, match="Unknown class id 9"):
            labels.name(9)

    def test_encodes_and_decodes_columns(self, labels: LabelRegistry) -> None:
        encoded = labels.encode(["truck", "car"])

        assert encoded.tolist() == [1, 0]
        assert str(encoded.dtype) == "int32"
        assert labels.decode(encoded) == ("truck", "car")

    def test_lenient_encoding_maps_unknowns(self, labels: LabelRegistry) -> None:
        assert labels.encode(["bus"], strict=False).tolist() == [UNKNOWN_CLASS_ID]

    def test_strict_encoding_rejects_unknowns(self, labels: LabelRegistry) -> None:
        with pytest.raises(KeyError):
            labels.encode(["bus"])

    def test_carries_colours(self) -> None:
        registry = LabelRegistry.from_names(["car"], colors={"car": (255, 0, 0)})

        assert registry.info(0).color == (255, 0, 0)

    def test_rejects_a_reserved_class_id(self) -> None:
        with pytest.raises(ValueError, match="reserved for the unknown class"):
            LabelRegistry((ClassInfo(UNKNOWN_CLASS_ID, "car"),))

    def test_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="duplicate class ids"):
            LabelRegistry((ClassInfo(0, "car"), ClassInfo(0, "truck")))
        with pytest.raises(ValueError, match="duplicate class names"):
            LabelRegistry((ClassInfo(0, "car"), ClassInfo(1, "car")))

    def test_rejects_a_dangling_alias(self) -> None:
        with pytest.raises(ValueError, match="points at unknown class id"):
            LabelRegistry((ClassInfo(0, "car"),), aliases={"auto": 9})

    def test_rejects_an_empty_name(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ClassInfo(0, "")

    def test_rejects_a_malformed_colour(self) -> None:
        with pytest.raises(ValueError, match="RGB triple"):
            ClassInfo(0, "car", (255, 0))  # type: ignore[arg-type]


class TestMerging:
    def test_collapses_a_group_into_one_class(self, labels: LabelRegistry) -> None:
        merged = labels.merged({"vehicle": ["car", "truck"]})

        assert merged.names == ("vehicle", "pedestrian")
        assert merged.class_id("car") == merged.class_id("truck") == merged.class_id("vehicle")

    def test_keeps_ungrouped_classes(self, labels: LabelRegistry) -> None:
        merged = labels.merged({"vehicle": ["car", "truck"]})

        assert merged.class_id("pedestrian") == 1

    def test_records_the_merge_as_aliases(self, labels: LabelRegistry) -> None:
        merged = labels.merged({"vehicle": ["car", "truck"]})

        assert dict(merged.aliases) == {"car": 0, "truck": 0}

    def test_produces_deterministic_ids(self, labels: LabelRegistry) -> None:
        first = labels.merged({"vehicle": ["car", "truck"]})
        second = labels.merged({"vehicle": ["car", "truck"]})

        assert first == second

    def test_reports_an_unknown_member(self, labels: LabelRegistry) -> None:
        with pytest.raises(KeyError, match="unknown class name"):
            labels.merged({"vehicle": ["spaceship"]})

    def test_merging_twice_keeps_earlier_aliases_reachable(self, labels: LabelRegistry) -> None:
        once = labels.merged({"vehicle": ["car", "truck"]})

        twice = once.merged({"road_user": ["vehicle", "pedestrian"]})

        assert twice.class_id("car") == twice.class_id("road_user")


class TestMetadata:
    def test_round_trips(self, labels: LabelRegistry) -> None:
        merged = labels.merged({"vehicle": ["car", "truck"]})

        assert LabelRegistry.from_metadata(merged.to_metadata()) == merged

    def test_round_trips_colours(self) -> None:
        registry = LabelRegistry.from_names(["car"], colors={"car": (1, 2, 3)})

        assert LabelRegistry.from_metadata(registry.to_metadata()) == registry

    def test_is_json_compatible(self, labels: LabelRegistry) -> None:
        import json

        assert LabelRegistry.from_metadata(json.loads(json.dumps(labels.to_metadata()))) == labels


class TestInstanceRegistry:
    def test_interns_uuids_stably(self) -> None:
        registry = InstanceRegistry()

        assert registry.encode(["a", "b", "a"]).tolist() == [0, 1, 0]
        assert len(registry) == 2
        assert registry.intern("a") == 0

    def test_decodes_back_to_uuids(self) -> None:
        registry = InstanceRegistry()
        registry.encode(["a", "b"])

        assert registry.decode([1, 0]) == ("b", "a")
        assert registry.uuid(0) == "a"

    def test_reports_membership(self) -> None:
        registry = InstanceRegistry()
        registry.intern("a")

        assert "a" in registry
        assert "z" not in registry

    def test_reports_an_unknown_id(self) -> None:
        with pytest.raises(KeyError, match="Unknown instance id 0"):
            InstanceRegistry().uuid(0)

    def test_ids_are_int64(self) -> None:
        assert str(InstanceRegistry().encode(["a"]).dtype) == "int64"
