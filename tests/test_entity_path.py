from __future__ import annotations

import pytest

from t4perceval import EntityPath
from t4perceval.core.entity import as_entity_path


class TestParsing:
    @pytest.mark.parametrize(
        ("text", "parts"),
        [
            ("/ground_truth/objects", ("ground_truth", "objects")),
            ("ground_truth/objects", ("ground_truth", "objects")),
            ("/estimation/objects/", ("estimation", "objects")),
            ("/", ()),
            ("", ()),
            ("  /a/b  ", ("a", "b")),
        ],
    )
    def test_accepts_equivalent_spellings(self, text: str, parts: tuple[str, ...]) -> None:
        assert EntityPath.parse(text).parts == parts

    def test_round_trips_through_str(self) -> None:
        assert str(EntityPath.parse("/estimation/objects")) == "/estimation/objects"
        assert str(EntityPath.root()) == "/"

    def test_rejects_empty_segments(self) -> None:
        with pytest.raises(ValueError, match="empty segments"):
            EntityPath.parse("/a//b")

    def test_rejects_non_string(self) -> None:
        with pytest.raises(TypeError, match="expects a str"):
            EntityPath.parse(3)  # type: ignore[arg-type]

    def test_rejects_separator_inside_a_part(self) -> None:
        with pytest.raises(ValueError, match="must not contain"):
            EntityPath(("a/b",))


class TestNavigation:
    def test_joins_strings_and_paths(self) -> None:
        base = EntityPath.parse("/estimation")

        assert str(base / "objects") == "/estimation/objects"
        assert str(base / EntityPath.parse("objects/camera")) == "/estimation/objects/camera"
        assert str(base / "filter/distance") == "/estimation/filter/distance"

    def test_exposes_name_and_parent(self) -> None:
        path = EntityPath.parse("/estimation/objects")

        assert path.name == "objects"
        assert str(path.parent) == "/estimation"
        assert EntityPath.root().name is None
        assert EntityPath.root().parent is None

    def test_distinguishes_descendant_from_equal(self) -> None:
        parent = EntityPath.parse("/estimation")
        child = EntityPath.parse("/estimation/objects")

        assert child.is_descendant_of(parent)
        assert not parent.is_descendant_of(parent)
        assert parent.starts_with(parent)
        assert child.starts_with(parent)
        assert not parent.starts_with(child)

    def test_is_hashable_and_compares_by_value(self) -> None:
        assert EntityPath.parse("/a/b") == EntityPath.parse("/a/b/")
        assert len({EntityPath.parse("/a"), EntityPath.parse("a")}) == 1

    def test_coercion_helper_is_idempotent(self) -> None:
        path = EntityPath.parse("/a")

        assert as_entity_path(path) is path
        assert as_entity_path("/a") == path
