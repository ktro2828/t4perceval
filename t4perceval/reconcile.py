"""Expressing one registry's class ids in another's.

Needed whenever two sets of columns were encoded against different registries: two
importers that were each handed their own registry, a recording read back against a newer
registry, or a registry narrowed by
:meth:`~t4perceval.label.LabelRegistry.merged`, which deliberately produces a different id
space over the same names.

Preventing the mismatch is better than repairing it -- pass one registry to every importer
-- but the repair has to exist, because ``merged()`` needs it regardless.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from attrs import evolve

from t4perceval.descriptors import CLASS_ID, ESTIMATION_CLASS_ID, GROUND_TRUTH_CLASS_ID
from t4perceval.label import BACKGROUND_CLASS_ID, UNKNOWN_CLASS_ID

if TYPE_CHECKING:
    from collections.abc import Sequence

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.label import LabelRegistry
    from t4perceval.typing import NDArrayI32

__all__ = ("CLASS_ID_DESCRIPTORS", "class_id_lut", "remap_class_ids")

#: Every descriptor whose values are class ids.
CLASS_ID_DESCRIPTORS: tuple[ComponentDescriptor, ...] = (
    CLASS_ID,
    GROUND_TRUTH_CLASS_ID,
    ESTIMATION_CLASS_ID,
)

#: How far ids are shifted to index the lookup table. The two sentinels are negative, so
#: offsetting by their magnitude lets one fancy-index cover sentinels and real classes
#: alike, with no branch.
_OFFSET = -BACKGROUND_CLASS_ID


def class_id_lut(
    source: LabelRegistry,
    target: LabelRegistry,
    *,
    on_missing: Literal["raise", "unknown"] = "raise",
) -> NDArrayI32:
    """Return a lookup table mapping ``source`` class ids onto ``target`` ids.

    Index it as ``lut[source_id + 2]``, or use :func:`remap_class_ids`, which does.

    Args:
        source: The registry the data was encoded against.
        target: The registry it should be expressed in.
        on_missing: What to do with a name ``target`` does not know -- raise, or map it to
            :data:`~t4perceval.label.UNKNOWN_CLASS_ID`.

    Returns:
        An ``int32`` table covering both sentinels and every source class id.

    Raises:
        KeyError: With ``on_missing="raise"``, when ``target`` lacks one of the names.
    """
    highest = max((info.class_id for info in source.classes), default=-1)
    lut = np.full(highest + 1 + _OFFSET, UNKNOWN_CLASS_ID, dtype=np.int32)

    # The sentinels mean the same thing in every registry, so they map to themselves.
    lut[BACKGROUND_CLASS_ID + _OFFSET] = BACKGROUND_CLASS_ID
    lut[UNKNOWN_CLASS_ID + _OFFSET] = UNKNOWN_CLASS_ID

    missing: list[str] = []
    for info in source.classes:
        mapped = target.class_id_or(info.name, UNKNOWN_CLASS_ID)
        if mapped == UNKNOWN_CLASS_ID and on_missing == "raise":
            missing.append(info.name)
        lut[info.class_id + _OFFSET] = mapped

    if missing:
        raise KeyError(
            f"Target registry does not know {sorted(missing)}; it has {list(target.names)}. "
            f'Pass on_missing="unknown" to map them to the unknown class instead.',
        )
    return lut


def remap_class_ids(
    chunk: Chunk,
    lut: NDArrayI32,
    *,
    descriptors: Sequence[ComponentDescriptor] = CLASS_ID_DESCRIPTORS,
) -> Chunk:
    """Return a chunk whose class-id columns are expressed in the target registry.

    Only the class-id columns are rebuilt; every other column is shared with the original,
    so the cost is one ``int32`` array per remapped column.
    """
    replacements = {}
    for descriptor in descriptors:
        column = chunk.columns.get(descriptor)
        if column is None:
            continue
        values = np.asarray(column.values, dtype=np.int64)
        replacements[descriptor] = type(column)(lut[values + _OFFSET])

    if not replacements:
        return chunk
    return evolve(chunk, columns={**dict(chunk.columns), **replacements})
