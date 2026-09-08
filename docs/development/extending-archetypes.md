# Extending archetypes

Adding a new validated bundle of components.

## The declaration

```python
from __future__ import annotations

from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import BatchAcceleration, BatchClassId, BatchPosition3D
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import ACCELERATION, CLASS_ID, POSITION

__all__ = ("Motions3D",)


@define(frozen=True, slots=True)
class Motions3D(Archetype):
    """3D positions carrying an acceleration."""

    position = component_field(POSITION, BatchPosition3D)
    class_id = component_field(CLASS_ID, BatchClassId)
    acceleration = component_field(ACCELERATION, BatchAcceleration, optional=True, kw_only=True)
```

`component_field(descriptor, component_type, *, optional=False, kw_only=False)` stores the
descriptor in the field metadata, which is how the base implements `select()`, `as_components()` and
the chunk round-trip generically. You do not write any of those.

The base checks, on construction, that every column has the same length and that the required ones
are present.

## Rules

### Compose; do not inherit

Re-declare the components of a related archetype rather than subclassing it. They resolve to the
**same descriptors**, so:

```python
motion.has(*Detections3D.required_descriptors())
```

works without anyone knowing a hierarchy exists. `Trackings3D` re-declares `Detections3D`'s box
components for exactly this reason, and `Predictions3D` composes box, instance **and** trajectory
columns so that a system needing only one group applies directly.

### Optional fields are keyword-only

Positional arguments are the required columns, in a stable order. An optional field defaults to
`None`, so a consumer checks `archetype.velocity is not None` or `view.has(VELOCITY)`.

### Reuse descriptors

Use the canonical descriptor from `t4perceval.descriptors`. Inventing a second name for a column
that already has one is what stops an existing system from running against your archetype -- which
is the whole point of the model.

Add a new descriptor only for a genuinely new meaning, and then add it to
[the components catalogue](../components/index.md).

### Cross-column validation goes in `__attrs_post_init__`

Call `super()` first, then check anything the field converters cannot:

```python
def __attrs_post_init__(self) -> None:
    super().__attrs_post_init__()
    validate_trajectory_shapes(
        self.waypoints,
        self.mode_confidence,
        self.mode_valid,
        self.timestep_valid,
        self.time_offset,
    )
```

`Trajectories3D` uses this to check that every trajectory column agrees with `waypoints` on `M` and
`T`; `MatchResults` uses it to reject an unknown `match_status`.

### Zero rows must work

Provide `empty()` when the shape needs parameters:

```python
@classmethod
def empty(cls, *, num_modes: int, num_timesteps: int) -> Self: ...
```

## Convenience constructors and accessors

Add readable constructors and readers when the columnar form is awkward to build or read by hand:

```python
MetricValues.from_rows([(0, 1.0, 0.93, 120)])
ConfusionMatrix.from_rows([(0, 1, 3)])
Trajectories3D.from_modes([[mode_a, mode_b]])
matrix.as_matrix()
values.of_class(class_id)
```

`TrajectoryMode3D` is the pattern for a row-at-a-time helper that is **not** a component -- it exists
to make tests and adapters readable, and the columnar form remains the real thing.

## Export it

Add it to your module's `__all__`, and make sure the module is star-imported in
`t4perceval/archetype/__init__.py`. Re-export from `t4perceval/__init__.py` if it is part of the
common vocabulary.

## Document it

An archetype is schema, so it gets a page in [the catalogue](../archetypes/index.md) with a
requirement table, not just a docstring. State what each optional component _unlocks_ -- that table
is how a reader works out which systems will run against their data.

## Testing

- Construction with only the required columns, and with every optional one.
- Length mismatch raises.
- Zero rows.
- `select()`, `as_components()`, and the chunk round-trip.
- `has(*Other.required_descriptors())` for whichever archetypes yours is meant to be usable as.

`tests/test_archetype_composition.py` and `tests/test_task_models.py` are the models.

## Where to go next

- [Extending systems](extending-systems.md) -- doing something with it.
- [Archetypes](../archetypes/index.md) -- the catalogue.
