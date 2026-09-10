# Extending systems

Most extensions are a [custom filter](../recipes/custom-filter.md),
[matcher](../recipes/custom-matcher.md) or [metric](../recipes/custom-metric.md), and those recipes
cover the common case. This page is for the rest: implementing the `System` protocol directly.

## The protocol

```python
@runtime_checkable
class System(Protocol):
    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]]
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...] | Passthrough]

    def requires_for(self, index: int) -> tuple[ComponentDescriptor, ...]: ...

    @property
    def sources(self) -> tuple[EntityPath, ...]: ...

    @property
    def target(self) -> EntityPath: ...

    @property
    def targets(self) -> tuple[EntityPath, ...]: ...

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]: ...
```

`EntitySystem` is the base that supplies the source/target wiring; subclass it and implement
`__call__`.

## When to go direct

| Situation                                         | Base to use                             |
| :------------------------------------------------ | :-------------------------------------- |
| one source, a boolean verdict per row             | `MaskSystem`                            |
| an estimation and a ground truth, pairwise scores | `MatchingSystem`                        |
| a matching plus the two entities it matched       | `MetricSystem`                          |
| two row-aligned label entities, no matching       | `SegmentationMetricSystem`              |
| **anything else**                                 | `EntitySystem`, implementing `__call__` |

The last row covers: several sources (`CombineMasksSystem`), materializing rows
(`ApplyMaskSystem`), reading metric entities rather than objects
(`MeanAveragePrecisionSystem`), and expressing an entity in another frame
(`TransformEntitySystem`).

## The shape of `__call__`

```python
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define

from t4perceval.core.chunk import Chunk
from t4perceval.core.timeline import TimeRange
from t4perceval.system.base import EntitySystem, require


@define(slots=True)
class MySystem(EntitySystem):
    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (SOME_DESCRIPTOR,)
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = (OTHER_DESCRIPTOR,)

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 2:
            raise ValueError(
                f"{type(self).__name__} needs exactly two sources, got {len(self.sources)}"
            )

    def __call__(self, ctx, at):
        first, second = self.sources
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)

        left = ctx.store.range(first, timeline=ctx.timeline, time_range=time_range)
        right = ctx.store.range(second, timeline=ctx.timeline, time_range=time_range)

        if len(left):
            require(left, *self.REQUIRES)

        chunk = left.to_chunk()
        return (
            Chunk(
                self.target,
                chunk.indexes,
                chunk.offsets,
                {OTHER_DESCRIPTOR: ...},
                frame_id=chunk.frame_id,
            ),
        )
```

Six things every system does:

1. **Normalise `at`.** It is an `int` or a `TimeRange`; `TimeRange.single(at)` covers the first.
2. **Query, do not reach past the store.** `ctx.store.range` / `latest_at` on `ctx.timeline`.
3. **Check the components you declared,** with `require()`, and only when the view is non-empty --
   an empty frame is ordinary, not a wiring error.
4. **Preserve the time index.** Reuse the source chunk's `indexes` and `offsets` so your output
   lines up frame-for-frame with its input.
5. **Carry `frame_id` through.** Use `resolve_frame(...)` for one frame, `require_same_frame(...)`
   when two inputs must agree. Dropping it makes a downstream guard unable to fire.
6. **Return chunks; do not write.** `Pipeline.run` sends them. A system that calls `send_chunk`
   itself breaks the caller's ability to inspect what a stage produced.

## Declaring REQUIRES and PROVIDES honestly

`Pipeline` validates wiring with them, once, up front:

- If a system reads an entity a **later** system writes, that raises at construction.
- If a system reads an entity an **earlier** system writes whose columns are **known**, the known
  set must cover what this one requires of that source (`requires_for(index)`, which defaults to
  `REQUIRES`).
- If the earlier writer's columns are **opaque** -- it passed through something the pipeline cannot
  see -- the consumer is checked at run time instead, like a store-sourced entity.

Components expected to come from the store rather than from another system are checked at run time
by `require()`.

### Passthrough systems

A system that carries its source's columns through -- a mask applier, a frame transform -- cannot
enumerate them where the class is defined. Say so instead of declaring `()`:

```python
PROVIDES: ClassVar[tuple[ComponentDescriptor, ...] | Passthrough] = Passthrough(
    0,  # index into `sources` of the entity carried through
    drops=(MASK,),  # columns deliberately not carried
    adds=(),  # columns written in addition
)
```

`Pipeline` then treats the target as carrying the source's columns: known when the source's are
known, opaque otherwise. Declaring `()` would tell it the target holds _nothing_, and every consumer
would be rejected. `ApplyMaskSystem` and `TransformEntitySystem` are the two in the library.

### Requirements that differ per source

`REQUIRES` applies to every source. When the sources play different roles, override `requires_for`:

```python
def requires_for(self, index: int) -> tuple[ComponentDescriptor, ...]:
    return () if index == 0 else (MASK,)  # ApplyMaskSystem: data, then mask
```

`MetricSystem` does the same for `(matching, estimation, ground_truth)`.

## Several targets from one computation

Override `targets` and return one chunk per target:

```python
@property
def targets(self) -> tuple[EntityPath, ...]:
    root = as_entity_path(self.target)
    return (root / "mota", root / "motp", root / "id_switch")
```

`Pipeline` uses `targets`, not `target`, to work out who produces what.

## Constructors

Follow the family conventions, so a system reads the same way whoever wrote it:

| Base                       | Constructor                                                         | Default target                  |
| :------------------------- | :------------------------------------------------------------------ | :------------------------------ |
| `MaskSystem`               | `.on(source, *, name=None, **params)`                               | `<source>/filter/<FILTER_NAME>` |
| `MatchingSystem`           | `.between(estimation, ground_truth, *, target=None, **params)`      | `/matching/<MATCHING_NAME>`     |
| `MetricSystem`             | `.on(matching, estimation, ground_truth, *, target=None, **params)` | `/metrics/<METRIC_NAME>`        |
| `SegmentationMetricSystem` | `.between(estimation, ground_truth, *, target=None, **params)`      | `/metrics/<METRIC_NAME>`        |
| `TransformEntitySystem`    | `.of(source, *, target_frame, target=None, resolver=None)`          | `<source>/in/<target_frame>`    |
| others                     | `.of(sources, target, ...)`                                         | explicit                        |

Parameters are attrs fields, keyword-only, validated in `__attrs_post_init__`.

## Using the registries

`ctx.labels` and `ctx.instances` are how a system takes class names and UUIDs instead of raw ids.
Resolve them at call time, not at construction, and raise when the context lacks the registry your
parameters need.

## A preset, not a config

If your systems are usually run together, ship a **function returning a list**, like
`average_precision_sweep`. The result is ordinary data a caller can print, edit or extend -- which is
the difference between sugar and the `EvaluationTask` enum this package set out to remove.

## Testing

- The wiring: `sources`, `target`, `targets`.
- An empty range, and an entity with no rows in range.
- A missing required component raises, naming it.
- The output chunk's `indexes`, `offsets` and `frame_id` match the input's.
- `Pipeline` rejects a bad order involving your system.

## Where to go next

- [Evaluation pipeline](../concepts/evaluation-pipeline.md) -- the model.
- [Custom filter](../recipes/custom-filter.md) · [matcher](../recipes/custom-matcher.md) · [metric](../recipes/custom-metric.md)
