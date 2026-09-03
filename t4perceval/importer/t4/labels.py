"""Turning T4 category names into class ids.

Two things here that the obvious implementation gets wrong.

:class:`~t4perceval.label.LabelRegistry` is frozen, so it cannot be grown while walking a
dataset -- the vocabulary has to be settled before the first box is converted. And its
``encode`` resolves each name by scanning the class list, which is fine for one call and
quadratic over a scene, so a dict is built once instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from t4perceval.label import UNKNOWN_CLASS_ID, LabelRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from t4perceval.typing import NDArrayBool, NDArrayI32

__all__ = ("UnknownLabels", "encode_class_ids", "label_registry_from_categories")

UnknownLabels: TypeAlias = Literal["error", "unknown", "drop"]
"""What to do with a category the registry does not know.

``"error"`` raises, ``"unknown"`` encodes :data:`~t4perceval.label.UNKNOWN_CLASS_ID` and
keeps the row, ``"drop"`` removes the row.
"""


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


def encode_class_ids(
    labels: LabelRegistry,
    names: Sequence[str],
    *,
    unknown: UnknownLabels = "error",
) -> tuple[NDArrayI32, NDArrayBool]:
    """Encode category names, returning the id column and a keep mask.

    Args:
        labels: The registry that decides the ids.
        names: One category name per object.
        unknown: How to treat a name the registry does not know.

    Returns:
        ``(class_id, keep)``. ``class_id`` is already filtered by ``keep``, so the two
        have the same length only when nothing was dropped; ``keep`` is indexed against
        ``names`` so a caller can filter its other columns the same way.

    Raises:
        KeyError: With ``unknown="error"``, when a name is not in the registry.
    """
    lookup = _lookup(labels)
    count = len(names)

    if count == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.bool_)

    raw = np.fromiter(
        (lookup.get(name, UNKNOWN_CLASS_ID) for name in names),
        dtype=np.int32,
        count=count,
    )
    missing = raw == UNKNOWN_CLASS_ID

    if unknown == "error" and missing.any():
        unknown_names = sorted({names[index] for index in np.flatnonzero(missing)})
        raise KeyError(
            f"Categories not in the label registry: {unknown_names}. "
            f"The registry knows {list(labels.names)}. "
            f'Pass unknown_labels="drop" to skip these objects, or "unknown" to keep '
            f"them with an unknown class.",
        )

    if unknown == "drop":
        keep = ~missing
        return raw[keep], keep

    return raw, np.ones(count, dtype=np.bool_)


def _lookup(labels: LabelRegistry) -> dict[str, int]:
    """Return a name -> class id dict, aliases included.

    ``LabelRegistry.class_id_or`` scans the class list per call, which turns a scene-wide
    encode into O(objects x classes); this pays for the scan once.
    """
    lookup = {info.name: info.class_id for info in labels.classes}
    lookup.update({name: class_id for name, class_id in labels.aliases})
    return lookup
