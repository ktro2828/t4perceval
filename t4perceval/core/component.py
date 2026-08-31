from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
from attrs import Converter, cmp_using, define, field

from t4perceval.core.selection import normalize_selection

if TYPE_CHECKING:
    from collections.abc import Sized

    from numpy.typing import NDArray
    from typing_extensions import Self

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.typing import ArrayLike, SelectionLike

__all__ = (
    "ANY",
    "ColumnarComponent",
    "Component",
    "values_equal",
    "validate_lengths",
)

#: Wildcard for a per-row dimension whose size is inferred from the data.
ANY = -1


@runtime_checkable
class Component(Protocol):
    """The unit of data stored in a column.

    A component owns exactly one column: ``N`` rows of a fixed per-row shape and dtype.
    """

    DESCRIPTOR: ClassVar[ComponentDescriptor]
    values: NDArray[Any]

    def __len__(self) -> int: ...

    def select(self, selection: SelectionLike) -> Self: ...

    def to_arrow(self) -> pa.Array: ...

    @classmethod
    def from_arrow(cls, array: pa.Array) -> Self: ...


def values_equal(left: NDArray[Any], right: NDArray[Any]) -> bool:
    """Compare two columns by value, treating ``NaN`` as equal to itself.

    ``NaN`` is a meaningful sentinel here -- an unmatched row's score, for instance -- so
    two columns holding the same sentinel in the same place must compare equal, which
    plain ``==`` would deny.
    """
    if np.issubdtype(left.dtype, np.inexact) and np.issubdtype(right.dtype, np.inexact):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _describe_shape(shape: tuple[int, ...]) -> str:
    parts = ("N", *("*" if dim == ANY else str(dim) for dim in shape))
    return f"({', '.join(parts)})"


def _coerce_column(value: ArrayLike, self_: ColumnarComponent) -> NDArray[Any]:
    """Normalize dtype and layout, validate the per-row shape, and freeze the array.

    The returned array is always read-only. Memory is never shared with a *writable*
    array owned by the caller, so freezing can never surprise them; an input that is
    already read-only and correctly laid out is adopted without copying.
    """
    cls = type(self_)
    name = cls.__name__
    source = value if isinstance(value, np.ndarray) else None

    try:
        array = np.asarray(value, dtype=cls.DTYPE)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} cannot be interpreted as {np.dtype(cls.DTYPE)}: {error}",
        ) from error

    expected_ndim = 1 + len(cls.SHAPE)

    if array.size == 0 and array.ndim != expected_ndim and ANY not in cls.SHAPE:
        # An empty input can only mean zero rows, and with a fully concrete per-row shape
        # there is nothing to guess -- so `BatchPosition3D([])` is accepted. A wildcard
        # shape genuinely is ambiguous and still needs `empty()`.
        array = array.reshape(0, *cls.SHAPE)

    row_shape = array.shape[1:]
    shape_ok = array.ndim == expected_ndim and all(
        expected in (ANY, actual) for expected, actual in zip(cls.SHAPE, row_shape)
    )
    if not shape_ok:
        raise ValueError(
            f"{name} must have shape {_describe_shape(cls.SHAPE)}, got {array.shape}",
        )

    if cls.REQUIRE_FINITE and array.size and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")

    if cls.VALUE_RANGE is not None and array.size:
        low, high = cls.VALUE_RANGE
        if not np.isfinite(array).all() or np.any((array < low) | (array > high)):
            raise ValueError(f"{name} must contain finite values within [{low}, {high}]")

    frozen = np.ascontiguousarray(array)
    if source is not None and source.flags.writeable and np.shares_memory(frozen, source):
        frozen = frozen.copy()
    if frozen.flags.writeable:
        frozen.flags.writeable = False
    return frozen


@define(frozen=True, slots=True)
class ColumnarComponent:
    """Base class implementing the :class:`Component` protocol generically.

    Subclasses declare the column layout as class variables instead of re-implementing
    converters, ``__len__``, ``select()`` and the Arrow round-trip:

    * :attr:`SHAPE` -- per-row shape. ``()`` is a scalar column; use :data:`ANY` for a
      dimension whose size is inferred from the data.
    * :attr:`DTYPE` -- numpy dtype every value is coerced to.
    * :attr:`VALUE_RANGE` -- optional inclusive ``(low, high)`` bound.
    * :attr:`REQUIRE_FINITE` -- reject ``NaN`` / ``Inf`` when set.

    Examples:
        >>> import numpy as np
        >>> @define(frozen=True, slots=True)
        ... class BatchPosition3D(ColumnarComponent):
        ...     SHAPE = (3,)
        >>> position = BatchPosition3D(np.arange(6, dtype=np.float32).reshape(2, 3))
        >>> position.values.dtype, len(position)
        (dtype('float64'), 2)
    """

    SHAPE: ClassVar[tuple[int, ...]] = ()
    DTYPE: ClassVar[Any] = np.float64
    VALUE_RANGE: ClassVar[tuple[float, float] | None] = None
    REQUIRE_FINITE: ClassVar[bool] = False

    # NOTE: `eq=cmp_using(np.array_equal)` is required -- attrs' default equality would
    # compare arrays elementwise and then fail on the ambiguous truth value.
    values: NDArray[Any] = field(
        converter=Converter(_coerce_column, takes_self=True),
        eq=cmp_using(eq=values_equal),
    )

    @classmethod
    def descriptor(cls) -> ComponentDescriptor:
        """Return the descriptor identifying a standalone column of this type."""
        from t4perceval.core.descriptor import ComponentDescriptor

        return ComponentDescriptor(cls.__name__, component_type=cls.__name__)

    @classmethod
    def empty(cls, *row_shape: int) -> Self:
        """Return a zero-row column.

        Args:
            row_shape: Sizes for the :data:`ANY` dimensions of :attr:`SHAPE`, in order.
        """
        wildcards = [index for index, dim in enumerate(cls.SHAPE) if dim == ANY]
        if len(row_shape) != len(wildcards):
            raise ValueError(
                f"{cls.__name__}.empty() expects {len(wildcards)} size(s) for its "
                f"wildcard dimension(s), got {len(row_shape)}",
            )
        resolved = list(cls.SHAPE)
        for index, size in zip(wildcards, row_shape):
            resolved[index] = size
        return cls(np.empty((0, *resolved), dtype=cls.DTYPE))

    @classmethod
    def from_array(cls, array: ArrayLike) -> Self:
        """Alias of the constructor, kept for symmetry with :meth:`as_array`."""
        return cls(array)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def row_shape(self) -> tuple[int, ...]:
        """Return the concrete per-row shape of this column."""
        return tuple(int(dim) for dim in self.values.shape[1:])

    def as_array(self) -> NDArray[Any]:
        """Return the backing array. It is read-only."""
        return self.values

    def select(self, selection: SelectionLike) -> Self:
        """Return a new component holding an independent copy of the selected rows."""
        indices = normalize_selection(selection, length=len(self))
        picked = self.values[indices]
        # Fancy indexing always allocates; freeze it here so the converter adopts it.
        picked.flags.writeable = False
        return type(self)(picked)

    def to_arrow(self) -> pa.Array:
        """Encode the column as a (possibly nested) Arrow fixed-size list array."""
        flat = pa.array(self.values.reshape(-1))
        for size in reversed(self.row_shape):
            flat = pa.FixedSizeListArray.from_arrays(flat, size)
        return flat

    @classmethod
    def from_arrow(
        cls,
        array: pa.Array | pa.ChunkedArray,
        *,
        row_shape: tuple[int, ...] | None = None,
    ) -> Self:
        """Decode a column produced by :meth:`to_arrow`.

        Args:
            array: The encoded column. A ``ChunkedArray`` (what reading a Parquet file
                yields) is combined first.
            row_shape: Per-row shape to reshape to. Only needed when the encoding lost the
                fixed-size-list sizes, e.g. after a round-trip through a format that
                stores variable-size lists.
        """
        if isinstance(array, pa.ChunkedArray):
            array = array.combine_chunks()

        if array.null_count:
            raise ValueError(
                f"{cls.__name__} columns must not contain nulls, got {array.null_count}",
            )

        dims: list[int] = []
        current = array
        while pa.types.is_fixed_size_list(current.type) or pa.types.is_list(current.type):
            if pa.types.is_fixed_size_list(current.type):
                dims.append(current.type.list_size)
            current = current.flatten()

        flat = current.to_numpy(zero_copy_only=False)
        shape = row_shape if row_shape is not None else tuple(dims)
        return cls(flat.reshape(-1, *shape) if shape else flat)


def validate_lengths(expected: int, **columns: Sized | None) -> None:
    """Raise when any non-``None`` column does not have ``expected`` rows."""
    for name, column in columns.items():
        if column is not None and len(column) != expected:
            raise ValueError(f"{name} has length {len(column)}, expected {expected}")
