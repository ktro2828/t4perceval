"""Reading a Lanelet2 map into region polygons.

A T4 scene ships ``map/lanelet2_map.osm`` and an Autoware stack drives on one. The file is
OSM XML: ``<node>`` elements carry the position, ``<way>`` elements string nodes into a
line or a closed ring, and a ``<relation type="lanelet">`` names a left and a right bound.
This module reads only what a spatial filter needs -- the polygon of every lanelet, keyed
by its ``subtype`` -- and hands it over as shapely geometry in the map frame.

Coordinates come from each node's ``local_x`` / ``local_y`` tags, which Autoware's map
tools write alongside ``lat`` / ``lon``. They *are* the ``map`` frame the ego poses are
recorded in, so no projection is needed and none is done: a map without them is refused
rather than guessed at. ``lanelet2`` itself is not a dependency; the standard-library XML
parser is enough for a file that is read once.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import shapely
from attrs import define, field
from shapely.geometry import MultiPolygon, Polygon

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from pathlib import Path

    from shapely.geometry.base import BaseGeometry
    from typing_extensions import Self

__all__ = ("Lanelet", "LaneletMap")


# -- the OSM primitives, as the file spells them; private to the parser ----------------------


@define(frozen=True, slots=True)
class _Node:
    """One ``<node>``: a point in the map frame."""

    id: int
    x: float
    y: float
    tags: Mapping[str, str] = field(converter=dict)

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@define(frozen=True, slots=True)
class _Way:
    """One ``<way>``: a line or ring through nodes, such as a lane bound."""

    id: int
    node_ids: tuple[int, ...] = field(converter=tuple)
    tags: Mapping[str, str] = field(converter=dict)


@define(frozen=True, slots=True)
class _Member:
    """One ``<member>`` of a relation."""

    type: str
    """``"node"``, ``"way"`` or ``"relation"``."""

    ref: int
    role: str


@define(frozen=True, slots=True)
class _Relation:
    """One ``<relation>``: members with roles, plus tags. A lanelet is one of these."""

    id: int
    members: tuple[_Member, ...] = field(converter=tuple)
    tags: Mapping[str, str] = field(converter=dict)

    def way(self, role: str) -> int | None:
        """Return the id of the way member holding ``role``, or ``None``."""
        for member in self.members:
            if member.type == "way" and member.role == role:
                return member.ref
        return None


# -- what a filter reads -----------------------------------------------------------------


@define(frozen=True, slots=True)
class Lanelet:
    """One lanelet: a road segment between a left and a right bound."""

    id: int
    subtype: str
    """The relation's ``subtype`` tag -- ``"road"``, ``"crosswalk"`` -- or ``""``."""

    polygon: Polygon
    """Left bound followed by the reversed right bound, in the map frame."""

    tags: Mapping[str, str] = field(converter=dict)
    """Every tag on the relation, for callers who want ``speed_limit`` or ``location``."""


@define(frozen=True, slots=True)
class LaneletMap:
    """The region polygons of one Lanelet2 map, in the map frame.

    Build one with :meth:`load` and ask :meth:`region` for the union of the parts a filter
    should keep::

        lanelet_map = LaneletMap.load(data_root / "map" / "lanelet2_map.osm")
        on_road = lanelet_map.region(subtypes=("road", "crosswalk"))
    """

    lanelets: tuple[Lanelet, ...] = field(converter=tuple)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read an ``.osm`` file.

        Raises:
            ValueError: When a way references a node without ``local_x`` / ``local_y``:
                the map has no local coordinates and this package does not project
                ``lat`` / ``lon``.
        """
        nodes, ways, relations = _parse(str(path))

        def ring(way: _Way, *, what: str) -> list[tuple[float, float]]:
            points = []
            for node_id in way.node_ids:
                node = nodes.get(node_id)
                if node is None:
                    raise ValueError(
                        f"{path}: {what} references node {node_id}, which has no local_x / "
                        f"local_y; t4perceval reads a map's local coordinates, not lat / lon",
                    )
                points.append(node.xy)
            return points

        lanelets = []
        for relation in relations.values():
            left = ways.get(relation.way("left") or -1)
            right = ways.get(relation.way("right") or -1)
            if left is None or right is None:
                continue
            what = f"lanelet {relation.id}"
            polygon = _polygon(ring(left, what=what) + ring(right, what=what)[::-1])
            if polygon is not None:
                lanelets.append(
                    Lanelet(relation.id, relation.tags.get("subtype", ""), polygon, relation.tags),
                )

        return cls(lanelets)

    @property
    def subtypes(self) -> tuple[str, ...]:
        """Return the distinct lanelet subtypes this map has, sorted."""
        return tuple(sorted({lanelet.subtype for lanelet in self.lanelets}))

    def region(self, *, subtypes: Collection[str] | None = None) -> BaseGeometry:
        """Return the union of the selected lanelets.

        Args:
            subtypes: Lanelet subtypes to include, e.g. ``("road", "crosswalk")``;
                ``None`` uses every lanelet.

        The vocabulary is the map's own -- see :attr:`subtypes` -- so a subtype this map
        does not have contributes nothing rather than raising: one selection applies
        unchanged across scenes whose maps differ. An empty result is an empty region, and
        a filter built on it keeps nothing.
        """
        chosen = frozenset(subtypes) if subtypes is not None else None
        polygons = [
            lanelet.polygon
            for lanelet in self.lanelets
            if chosen is None or lanelet.subtype in chosen
        ]
        return shapely.union_all(polygons) if polygons else Polygon()


# -- parsing ------------------------------------------------------------------------------


def _parse(path: str) -> tuple[dict[int, _Node], dict[int, _Way], dict[int, _Relation]]:
    """One streaming pass: nodes with local coordinates, every way, lanelet relations."""
    nodes: dict[int, _Node] = {}
    ways: dict[int, _Way] = {}
    relations: dict[int, _Relation] = {}

    # `iterparse` keeps memory flat on a 50 MB map, provided each element is cleared
    # once it has been read.
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "node":
            tags = _tags(element)
            if "local_x" in tags and "local_y" in tags:
                node = _Node(_id(element), float(tags["local_x"]), float(tags["local_y"]), tags)
                nodes[node.id] = node
        elif element.tag == "way":
            way = _Way(
                _id(element),
                (int(nd.get("ref", 0)) for nd in element.findall("nd")),
                _tags(element),
            )
            ways[way.id] = way
        elif element.tag == "relation":
            tags = _tags(element)
            if tags.get("type") == "lanelet":
                members = (
                    _Member(m.get("type", ""), int(m.get("ref", 0)), m.get("role", ""))
                    for m in element.findall("member")
                )
                relation = _Relation(_id(element), members, tags)
                relations[relation.id] = relation
        else:
            continue
        element.clear()

    return nodes, ways, relations


def _id(element: ET.Element) -> int:
    return int(element.get("id", 0))


def _tags(element: ET.Element) -> dict[str, str]:
    return {tag.get("k", ""): tag.get("v", "") for tag in element.findall("tag")}


def _polygon(ring: list[tuple[float, float]]) -> Polygon | None:
    """Build a valid polygon from a ring, repairing a self-touching one; ``None`` if degenerate."""
    if len(ring) < 3:
        return None
    polygon = Polygon(ring)
    if not polygon.is_valid:
        repaired = shapely.make_valid(polygon)
        polygon = _largest_polygon(repaired)
        if polygon is None:
            return None
    return polygon if polygon.area > 0.0 else None


def _largest_polygon(geometry: BaseGeometry) -> Polygon | None:
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda part: part.area)
    if hasattr(geometry, "geoms"):
        candidates = [
            part for part in (_largest_polygon(piece) for piece in geometry.geoms) if part
        ]
        return max(candidates, key=lambda part: part.area) if candidates else None
    return None
