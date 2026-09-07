"""Turning T4 category names into class ids.

The dataset's ``category`` table decides the vocabulary and its order; the name -> id
step itself is :func:`~t4perceval.importer._labels.encode_class_ids`, shared with every
other importer so ``unknown_labels`` means one thing throughout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from t4perceval.importer._labels import UnknownLabels, encode_class_ids
from t4perceval.label import LabelRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ("UnknownLabels", "encode_class_ids", "label_registry_from_categories")


def label_registry_from_categories(
    categories: Sequence[object],
    *,
    prefix: str = "autoware",
    colors: Mapping[str, tuple[int, int, int]] | None = None,
) -> LabelRegistry:
    """Build a registry from a T4 dataset's own ``category`` table.

    Ordered by ``Category.index`` when every category has one -- that is the dataset's
    own numbering, the same one lidarseg uses -- and by table order otherwise. Either way
    the ids are a property of the dataset rather than of the order frames happened to be
    read in, so two imports of the same dataset produce comparable ``class_id`` columns.

    This is discovery, deliberately kept separate from importing. Passing the result to an
    importer is the caller's explicit step, because an id assignment that is derived
    silently on each side is exactly how two sources come to disagree about what class
    ``3`` means.

    Args:
        categories: ``Category`` records, or anything with a ``name`` attribute.
        prefix: Label prefix recorded on the registry.
        colors: Optional display colors, keyed by category name.

    Returns:
        A registry covering every category in the table.
    """
    indexed = [(getattr(category, "index", None), category) for category in categories]
    if indexed and all(index is not None for index, _ in indexed):
        indexed.sort(key=lambda entry: entry[0])  # type: ignore[arg-type,return-value]

    names = [str(category.name) for _, category in indexed]  # type: ignore[attr-defined]
    return LabelRegistry.from_names(names, prefix=prefix, colors=colors)
