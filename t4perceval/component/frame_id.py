"""Coordinate-frame names.

The one text component in the model. Everything else is numeric because it is *per
object*: interning a name into an integer plus a registry -- what
:class:`~t4perceval.label.LabelRegistry` does for classes -- pays for itself across
:math:`10^5` object rows. A frame name names one end of one relationship, so the rule that
admits this component is:

    text is admissible where there is one value per *edge*, not one per *object*.

Interning frames instead would leave :attr:`~t4perceval.core.chunk.Chunk.frame_id` a
string while its counterpart became an integer -- two encodings of one concept, with
nothing checking that they agree -- and would thread a registry through ``Store``,
``SystemContext``, ``Recording`` and the Arrow schema to save a few hundred bytes.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pyarrow as pa
from attrs import define

from t4perceval.core.component import ColumnarComponent, MonoComponent

__all__ = ("BatchFrameId", "FrameId")


@define(frozen=True, slots=True)
class BatchFrameId(ColumnarComponent):
    """Coordinate-frame names with shape ``(N,)``.

    The storage form. :class:`FrameId` is what an archetype declares; this is what a chunk
    holds, so a query spanning several samples of one edge -- or a hand-built chunk naming
    several children of one parent -- reads back as a column like any other.

    Note:
        Validated row by row, unlike every other component. ``object`` is the only dtype
        that holds a variable-length string, and numpy coerces *anything* into it, so
        ``BatchFrameId([1])`` would otherwise encode as an Arrow ``int64`` column that never
        compares equal to a frame name. The check is O(N) where N counts edge samples, not
        objects -- do not copy this into a hot column.
    """

    #: Variable-length ``str``. Never a fixed-width ``<U*``: numpy truncates silently, so
    #: ``traffic_light_left_camera_optical_link`` would become a different, shorter frame,
    #: and two sensors could collapse into one.
    DTYPE: ClassVar[object] = object

    #: Bound on one name, so a column built from the wrong data fails here rather than
    #: becoming megabytes of pointers.
    MAX_LENGTH: ClassVar[int] = 256

    def __attrs_post_init__(self) -> None:
        for index, value in enumerate(self.values.tolist()):
            if not isinstance(value, str):
                raise ValueError(
                    f"{type(self).__name__} row {index} is {type(value).__name__}, not "
                    f"str; a non-string name encodes as a numeric Arrow column that never "
                    f"compares equal to a frame",
                )
            if not value:
                raise ValueError(
                    f"{type(self).__name__} row {index} is empty; a frame needs a name",
                )
            if len(value) > self.MAX_LENGTH:
                raise ValueError(
                    f"{type(self).__name__} row {index} is {len(value)} characters, over "
                    f"the {self.MAX_LENGTH} allowed",
                )

    def names(self) -> tuple[str, ...]:
        """Return the column as strings, in row order."""
        return tuple(self.values.tolist())

    def matching(self, name: str) -> np.ndarray:
        """Return the indices of the rows naming ``name``."""
        return np.flatnonzero(self.values == name).astype(np.int64, copy=False)

    def to_arrow(self) -> pa.Array:
        """Encode the column as an Arrow ``string`` array.

        The type is pinned rather than inferred, because inference on an ``object`` array
        depends on the *values*: a zero-row column infers ``null``, which
        :func:`~t4perceval.io.arrow.chunk_to_table` then rejects for declaring its fields
        non-nullable. A zero-row frame column is ordinary -- an entity can exist before it
        has any edge -- so the schema must not depend on the row count.
        """
        return pa.array(self.values.reshape(-1), type=pa.string())


@define(frozen=True, slots=True)
class FrameId(BatchFrameId, MonoComponent):
    """One coordinate-frame name.

    What an archetype declares -- a transform relates exactly two frames, so its child is
    a name rather than a column of names. Stored as :class:`BatchFrameId`.

    A name is opaque: this package never parses one, so ``/`` is allowed and a
    ROS-namespaced ``/robot1/base_link`` means exactly what it says.

    Validation, and the reason for it, are inherited from :class:`BatchFrameId`.
    """

    DTYPE: ClassVar[object] = object
    BATCH: ClassVar[type[ColumnarComponent]] = BatchFrameId

    @property
    def name(self) -> str:
        """Return the frame name."""
        return str(self.value)
