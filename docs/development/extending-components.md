# Extending components

Adding a new column type.

## The declaration

Subclass `ColumnarComponent` and declare the layout as class variables. The base supplies the
converter, `__len__`, `select()`, the Arrow round-trip and the read-only guarantee.

```python
from __future__ import annotations

from typing import ClassVar

import numpy as np
from attrs import define

from t4perceval.core.component import ColumnarComponent

__all__ = ("BatchAcceleration",)


@define(frozen=True, slots=True)
class BatchAcceleration(ColumnarComponent):
    """Columnar 3D accelerations with shape ``(N, 3)``, in m/s²."""

    SHAPE: ClassVar[tuple[int, ...]] = (3,)
    DTYPE: ClassVar[object] = np.float64
    REQUIRE_FINITE: ClassVar[bool] = True
```

| Class variable   | Meaning                                                                                     |
| :--------------- | :------------------------------------------------------------------------------------------ |
| `SHAPE`          | the **per-row** shape. `()` is a scalar column; `ANY` is a dimension inferred from the data |
| `DTYPE`          | the dtype every value is coerced to                                                         |
| `VALUE_RANGE`    | an inclusive `(low, high)` bound. Numeric columns only                                      |
| `REQUIRE_FINITE` | reject `NaN` / `Inf`. Numeric columns only                                                  |
| `MONO`           | one value rather than a column of them -- use `MonoComponent` instead                       |

`VALUE_RANGE` and `REQUIRE_FINITE` call `np.isfinite`, which raises on a text column, so leave them
unset for one.

## Give it a descriptor

A column type is not a meaning. Add the descriptor to `t4perceval/descriptors.py`, named for what
the column **means**, not for the archetype that will hold it:

```python
ACCELERATION = ComponentDescriptor("acceleration", component_type="BatchAcceleration")
```

The same column type may appear under several descriptors -- `BatchClassId` is `class_id`,
`ground_truth_class_id` and `estimation_class_id` -- and two archetypes sharing a descriptor is
exactly what lets one system serve both.

If your column is a relationship rather than a property of an object, give it a _distinct_ name the
way `TRANSLATION` differs from `POSITION`, so a system asking for the other one cannot be pointed at
it and appear to work.

## Register it for IO

`t4perceval.io.registry` resolves a component class by name when decoding an Arrow table. A class
reachable from `t4perceval.component` is picked up automatically:

```python
from t4perceval.io.registry import component_types, resolve_component_type

"BatchAcceleration" in component_types()
resolve_component_type("BatchAcceleration")
```

Export it from your module's `__all__` and make sure the module is star-imported in
`t4perceval/component/__init__.py`, as the existing ones are.

## Nested shapes

Use `ANY` for a dimension whose size comes from the data:

```python
from t4perceval.core.component import ANY


@define(frozen=True, slots=True)
class BatchModeAcceleration(ColumnarComponent):
    """Per-mode acceleration with shape ``(N, M)``."""

    SHAPE: ClassVar[tuple[int, ...]] = (ANY,)
```

`empty()` then needs those sizes: `BatchModeAcceleration.empty(6)`.

Arrow encodes nested vectors as fixed-size lists, so a wildcard dimension must be **consistent
within one column** -- it is inferred once, not per row.

## Mono components

A component describing one value, not `N` of them, subclasses its `Batch*` counterpart **and**
`MonoComponent`:

```python
from t4perceval.core.component import MonoComponent


@define(frozen=True, slots=True)
class Acceleration(BatchAcceleration, MonoComponent):
    """One 3D acceleration, written and read as a ``(3,)`` value."""
```

`.value` returns the bare value, and `.as_batch()` widens it into the columnar form -- which is what
happens on the way into a chunk, so storage stays columnar either way.

Only do this when "what if it has three rows?" is a question the type should not be able to be
asked. Today `Transform3D` is the only archetype where that is true.

## Helper properties

Add derived accessors on the class rather than making callers index columns:

```python
@property
def magnitude(self) -> np.ndarray:
    """Return the L2 norm of each acceleration."""
    return np.linalg.norm(self.values, axis=1)
```

Follow the existing convention: `BatchVelocity.speed` and `BatchRoi.x_max` are properties;
`BatchRoi.area()`, `BatchQuaternion.yaw()` and `BatchMask.indices()` are methods. Prefer a property
for a pure reinterpretation and a method for a computation.

## Make it frame-dependent

A component is carried through a coordinate transform unchanged unless
`t4perceval.transform.TRANSFORM_KINDS` says otherwise. If yours is geometry _in_ the frame, register
how it moves:

```python
from t4perceval.transform import TRANSFORM_KINDS, TransformKind

TRANSFORM_KINDS[BatchAcceleration] = TransformKind.DIRECTION  # rotate, never translate
```

`POINT` rotates and translates (a position), `DIRECTION` rotates only (a velocity), `ROTATION`
composes with the frame's rotation (an orientation), `DROP` omits the column (a mask -- a claim about
the source frame). Lookup walks the MRO, so a subclass of a registered component inherits its kind
and a mono component inherits its batch class's. Do not register a shared base such as
`BatchVector3D`: `BatchSize3D` shares it and must not rotate.

## Testing

- Round-trip through Arrow and Parquet.
- A zero-row column, including `empty()` with any wildcard sizes.
- That `values` is read-only and does not alias the input array.
- Every validation you declared, including the failure message.

`tests/test_components.py` and `tests/test_columnar_io.py` are the models.

## Where to go next

- [Extending archetypes](extending-archetypes.md) -- bundling it.
- [Serialization format](serialization-format.md) -- how it reaches Arrow.
