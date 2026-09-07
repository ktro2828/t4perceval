"""Encoding category names against a registry, shared by every importer.

:class:`~t4perceval.label.LabelRegistry` is frozen, so it cannot be grown while walking a
source -- the vocabulary has to be settled before the first object is converted. And its
``encode`` resolves each name by scanning the class list, which is fine for one call and
quadratic over a scene, so a dict is built once instead.

Each importer owns the step *before* this one -- turning its source's notion of a class
(a T4 category record, an Autoware ``uint8`` enum) into a canonical name. Only the final
name -> id step is common, and it lives here so the ``unknown_labels`` policy means the
same thing whichever source produced the names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from t4perceval.label import UNKNOWN_CLASS_ID

if TYPE_CHECKING:
    from collections.abc import Sequence

    from t4perceval.label import LabelRegistry
    from t4perceval.typing import NDArrayBool, NDArrayI32

__all__ = ("UnknownLabels", "encode_class_ids")

UnknownLabels: TypeAlias = Literal["error", "unknown", "drop"]
"""What to do with a category the registry does not know.

``"error"`` raises, ``"unknown"`` encodes :data:`~t4perceval.label.UNKNOWN_CLASS_ID` and
keeps the row, ``"drop"`` removes the row.
"""


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
