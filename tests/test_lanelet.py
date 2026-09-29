"""Reading a Lanelet2 map into region polygons."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from shapely.geometry import Point

from t4perceval.lanelet import LaneletMap
from tests.lanelet_builder import write_fixture_map, write_osm

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def lanelet_map(tmp_path: Path) -> LaneletMap:
    return LaneletMap.load(write_fixture_map(tmp_path / "lanelet2_map.osm"))


class TestLoad:
    def test_reads_lanelets(self, lanelet_map: LaneletMap) -> None:
        assert [lanelet.id for lanelet in lanelet_map.lanelets] == [1, 2]
        assert lanelet_map.subtypes == ("crosswalk", "road")

    def test_a_lanelet_polygon_is_its_two_bounds(self, lanelet_map: LaneletMap) -> None:
        road = lanelet_map.lanelets[0]

        assert road.polygon.area == pytest.approx(20.0)
        assert road.polygon.contains(Point(5.0, 1.0))
        assert not road.polygon.contains(Point(5.0, 3.0))

    def test_tags_are_kept(self, lanelet_map: LaneletMap) -> None:
        assert lanelet_map.lanelets[0].tags["speed_limit"] == "30"
        assert lanelet_map.lanelets[0].tags["location"] == "urban"

    def test_a_node_without_local_coordinates_is_refused(self, tmp_path: Path) -> None:
        path = write_osm(
            tmp_path / "m.osm",
            {1: (0.0, 0.0), 2: None, 3: (1.0, 1.0), 4: (0.0, 1.0)},
            {10: ([1, 2], {}), 11: ([4, 3], {})},
            {1: (11, 10, {"subtype": "road"})},
        )

        with pytest.raises(ValueError, match="lanelet 1 references node 2, which has no local_x"):
            LaneletMap.load(path)

    def test_a_degenerate_lanelet_is_skipped(self, tmp_path: Path) -> None:
        # Both bounds are the same two points: the ring has no area.
        path = write_osm(
            tmp_path / "m.osm",
            {1: (0.0, 0.0), 2: (1.0, 0.0)},
            {10: ([1, 2], {}), 11: ([1, 2], {})},
            {1: (10, 11, {"subtype": "road"})},
        )

        assert LaneletMap.load(path).lanelets == ()

    def test_a_self_touching_ring_is_repaired(self, tmp_path: Path) -> None:
        # Left bound runs right, right bound also runs right: the naive ring is a bow tie.
        path = write_osm(
            tmp_path / "m.osm",
            {1: (0.0, 0.0), 2: (2.0, 0.0), 3: (0.0, 1.0), 4: (2.0, 1.0)},
            {10: ([1, 2], {}), 11: ([4, 3], {})},
            {1: (10, 11, {"subtype": "road"})},
        )

        (lanelet,) = LaneletMap.load(path).lanelets

        assert lanelet.polygon.is_valid
        assert lanelet.polygon.area > 0.0

    def test_a_lanelet_missing_a_bound_is_skipped(self, tmp_path: Path) -> None:
        path = write_osm(
            tmp_path / "m.osm",
            {1: (0.0, 0.0), 2: (1.0, 0.0)},
            {10: ([1, 2], {})},
            {1: (10, 99, {"subtype": "road"})},
        )

        assert LaneletMap.load(path).lanelets == ()


class TestRegion:
    def test_everything_by_default(self, lanelet_map: LaneletMap) -> None:
        region = lanelet_map.region()

        assert region.area == pytest.approx(20.0 + 20.0)

    def test_selects_by_subtype(self, lanelet_map: LaneletMap) -> None:
        region = lanelet_map.region(subtypes=("road",))

        assert region.contains(Point(5.0, 1.0))
        assert not region.contains(Point(5.0, 3.0))

    def test_several_subtypes_are_a_union(self, lanelet_map: LaneletMap) -> None:
        region = lanelet_map.region(subtypes=("road", "crosswalk"))

        assert region.area == pytest.approx(40.0)
        assert region.contains(Point(5.0, 3.0))

    def test_a_token_the_map_lacks_contributes_nothing(self, lanelet_map: LaneletMap) -> None:
        # The vocabulary is the map's own, so one selection applies across scenes.
        assert lanelet_map.region(subtypes=("walkway",)).is_empty
        assert lanelet_map.region(subtypes=("raod",)).is_empty
        assert lanelet_map.region(subtypes=("road", "walkway")).area == pytest.approx(20.0)

    def test_any_subtype_a_map_carries_is_selectable(self, tmp_path: Path) -> None:
        path = write_osm(
            tmp_path / "m.osm",
            {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.0, 1.0), 4: (1.0, 1.0)},
            {10: ([1, 2], {}), 11: ([3, 4], {})},
            {1: (11, 10, {"subtype": "runway"})},
        )
        loaded = LaneletMap.load(path)

        assert loaded.subtypes == ("runway",)
        assert not loaded.region(subtypes=("runway",)).is_empty
