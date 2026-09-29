"""Writing a minimal Autoware-style Lanelet2 ``.osm`` for the tests.

Real maps are megabytes and not vendored; the loader only reads node coordinates, way node
lists and lanelet relations, so a few dozen lines of XML exercise every path it has.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.sax.saxutils import quoteattr

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

Point = tuple[float, float]


def osm_text(
    nodes: Mapping[int, Point | None],
    ways: Mapping[int, tuple[Sequence[int], Mapping[str, str]]],
    lanelets: Mapping[int, tuple[int, int, Mapping[str, str]]],
) -> str:
    """Render nodes, ways and lanelet relations as OSM XML.

    Args:
        nodes: ``id -> (local_x, local_y)``; ``None`` writes a node with ``lat``/``lon``
            only, the case the loader must refuse.
        ways: ``id -> (node ids, tags)``.
        lanelets: ``id -> (left way id, right way id, tags)``; ``type=lanelet`` is added.
    """
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm generator="test">']
    for node_id, point in nodes.items():
        lines.append(f'  <node id="{node_id}" lat="35.0" lon="139.0">')
        if point is not None:
            lines.append(f'    <tag k="local_x" v="{point[0]}"/>')
            lines.append(f'    <tag k="local_y" v="{point[1]}"/>')
            lines.append('    <tag k="ele" v="0.0"/>')
        lines.append("  </node>")
    for way_id, (refs, tags) in ways.items():
        lines.append(f'  <way id="{way_id}">')
        lines.extend(f'    <nd ref="{ref}"/>' for ref in refs)
        lines.extend(_tags(tags))
        lines.append("  </way>")
    for relation_id, (left, right, tags) in lanelets.items():
        lines.append(f'  <relation id="{relation_id}">')
        lines.append(f'    <member type="way" role="left" ref="{left}"/>')
        lines.append(f'    <member type="way" role="right" ref="{right}"/>')
        lines.extend(_tags({"type": "lanelet", **tags}))
        lines.append("  </relation>")
    lines.append("</osm>")
    return "\n".join(lines) + "\n"


def _tags(tags: Mapping[str, str]) -> list[str]:
    return [f"    <tag k={quoteattr(key)} v={quoteattr(value)}/>" for key, value in tags.items()]


def write_osm(
    path: Path,
    nodes: Mapping[int, Point | None],
    ways: Mapping[int, tuple[Sequence[int], Mapping[str, str]]],
    lanelets: Mapping[int, tuple[int, int, Mapping[str, str]]],
) -> Path:
    path.write_text(osm_text(nodes, ways, lanelets))
    return path


def write_fixture_map(path: Path) -> Path:
    """Two lanelets side by side, plus a crosswalk polygon way the loader must ignore.

    ::

        y
        4 +---------+          way 300 (crosswalk_polygon, area=yes): x 12..14, y 0..4
        2 | road 1  |   [xx]   road 1 (id 1):      x 0..10, y 0..2
        0 +---------+   [xx]   crosswalk (id 2):   x 0..10, y 2..4
          0        10  12 14
    """
    nodes = {
        1: (0.0, 0.0),
        2: (10.0, 0.0),
        3: (0.0, 2.0),
        4: (10.0, 2.0),
        5: (0.0, 4.0),
        6: (10.0, 4.0),
        7: (12.0, 0.0),
        8: (14.0, 0.0),
        9: (14.0, 4.0),
        10: (12.0, 4.0),
    }
    ways = {
        100: ([1, 2], {"type": "line_thin", "subtype": "solid"}),
        101: ([3, 4], {"type": "line_thin", "subtype": "dashed"}),
        102: ([5, 6], {"type": "line_thin", "subtype": "solid"}),
        300: ([7, 8, 9, 10, 7], {"type": "crosswalk_polygon", "area": "yes"}),
    }
    lanelets = {
        1: (101, 100, {"subtype": "road", "speed_limit": "30", "location": "urban"}),
        2: (102, 101, {"subtype": "crosswalk"}),
    }
    return write_osm(path, nodes, ways, lanelets)
