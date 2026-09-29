# Systems

A system is one stage of an evaluation: it reads entities from the store, declares the components
it needs, and writes its result back as a new entity. Filters write a mask, matchers write verdicts,
metrics write numbers, and a few systems rewrite geometry or rows.

This section is a **catalogue**: what each system computes, what it requires, and where it writes.
How to use them is in the [user guide](../user-guide/filtering.md); the Python signatures are in the
[API reference](../reference/api/system.md).

## The catalogue

| Family               | Page                           | Base class                 | Writes                                |
| :------------------- | :----------------------------- | :------------------------- | :------------------------------------ |
| Filters              | [Filters](filters.md)          | `MaskSystem`               | a `mask` at `<source>/filter/<name>`  |
| Mask composition     | [Filters](filters.md)          | --                         | a `mask`, or the rows a mask kept     |
| Matchers             | [Matchers](matchers.md)        | `MatchingSystem`           | `MatchResults` at `/matching/<mode>`  |
| Metrics              | [Metrics](metrics.md)          | `MetricSystem`             | `MetricValues` at `/metrics/<name>`   |
| Segmentation metrics | [Metrics](metrics.md)          | `SegmentationMetricSystem` | `MetricValues` / `ConfusionMatrix`    |
| Frames and points    | [Frames and points](frames.md) | --                         | the source's rows, moved or reordered |

## Shared surface

Every system has the same handful of things on it:

```python
from t4perceval.system import FilterByDistanceSystem

near = FilterByDistanceSystem.on("/estimation/objects", max_distance=50.0)

near.sources  # (EntityPath('/estimation/objects'),)   what it reads
near.target  # EntityPath('/estimation/objects/filter/distance')   where it writes
near.REQUIRES  # (POSITION,)   components the source must carry
near.PROVIDES  # (MASK,)       components the target will carry
near(ctx, at)  # the chunks it produced, for one time or a TimeRange
```

Three constructor names recur. `on(source, ...)` takes one source; `between(estimation,
ground_truth, ...)` takes a pair; `of(sources, ...)` takes a list. Each defaults the target to a
conventional path and accepts `target=` or `name=` to change it.

Parameters are keyword-only fields, validated when the system is built, so a bad threshold fails
before any data is read. `Pipeline` checks the wiring of every `REQUIRES` against every `PROVIDES`
the same way, in order, before it runs.

## Where to go next

- [Evaluation pipeline](../concepts/evaluation-pipeline.md) -- how systems compose into a run.
- [Extending systems](../development/extending-systems.md) -- adding your own.
