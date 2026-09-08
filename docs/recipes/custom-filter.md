# Implement a custom filter

**Goal:** keep or drop rows on a criterion the built-in filters do not cover.

## The contract

Subclass `MaskSystem` and supply four things:

| Thing         | What it is                                             |
| :------------ | :----------------------------------------------------- |
| `REQUIRES`    | the descriptors the source entity must carry           |
| `FILTER_NAME` | the last path segment, `<source>/filter/<FILTER_NAME>` |
| attrs fields  | your parameters, keyword-only                          |
| `keep()`      | a boolean array of length `len(view)`                  |

The base class handles the range query, the component check, the shape check and writing the chunk.

## Example: bound a 2D box's area

```python
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define, field

from t4perceval.descriptors import ROI
from t4perceval.system import MaskSystem

if TYPE_CHECKING:
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.view import EntityView
    from t4perceval.system import SystemContext
    from t4perceval.typing import NDArrayBool


@define(slots=True)
class FilterByRoiAreaSystem(MaskSystem):
    """Keep 2D detections whose pixel area is within ``[min, max]``."""

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (ROI,)
    FILTER_NAME: ClassVar[str] = "roi_area"

    min_area: int = field(default=0, kw_only=True)
    max_area: int | None = field(default=None, kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.min_area < 0:
            raise ValueError(f"min_area must be non-negative, got {self.min_area}")
        if self.max_area is not None and self.max_area < self.min_area:
            raise ValueError("max_area must not be below min_area")

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        del ctx
        area = view.component(ROI).area()
        upper = np.inf if self.max_area is None else self.max_area
        return (area >= self.min_area) & (area <= upper)
```

Use it exactly like a built-in one:

```python
from t4perceval.system import Pipeline

big = FilterByRoiAreaSystem.on("/estimation/objects", min_area=400)
big.target  # /estimation/objects/filter/roi_area

Pipeline([big]).run(ctx, TimeRange.everything())
```

## Rules to follow

- **Return a mask, do not drop rows.** The point of a mask is that the verdict stays inspectable.
  Materializing is `ApplyMaskSystem`'s job.
- **Return exactly `len(view)` booleans.** The base class checks this and raises with the shapes if
  you do not.
- **Validate parameters in `__attrs_post_init__`**, and call `super()` first -- the base checks
  that there is exactly one source.
- **Do not touch the store.** `keep()` gets a view; everything it needs should be a column or a
  parameter.
- **An empty view never reaches `keep()`.** A frame with no rows is an ordinary empty frame, not a
  wiring error, so the base skips the component check and emits an empty mask.

## Using the registries

Parameters that name classes or instances are resolved through the context, not stored as ids:

```python
def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
    if ctx.labels is None:
        raise ValueError("FilterByMySystem needs SystemContext(labels=...)")
    wanted = {ctx.labels.class_id(name) for name in self.labels}
    return np.isin(view.component(CLASS_ID).values, list(wanted))
```

`LabelRegistry.class_id` raises on an unknown name, which is what you want -- a typo should fail
rather than match nothing.

## Combining with the built-ins

Your filter is an ordinary system, so it composes:

```python
from t4perceval.system import CombineMasksSystem, FilterByConfidenceSystem

big = FilterByRoiAreaSystem.on(SOURCE, min_area=400)
confident = FilterByConfidenceSystem.on(SOURCE, min_confidence=0.3)
keep = CombineMasksSystem.of([big.target, confident.target], f"{SOURCE}/filter/keep")
```

## A filter over several entities

`MaskSystem` requires exactly one source. If you need more -- "keep estimations near _any_ ground
truth", say -- implement the `System` protocol directly, as `CombineMasksSystem` does. See
[Extending systems](../development/extending-systems.md).

## Where to go next

- [Filtering](../user-guide/filtering.md) -- the built-in family.
- [Write a custom matcher](custom-matcher.md) · [Write a custom metric](custom-metric.md)
