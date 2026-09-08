# Evaluation pipeline

A **system** is the "S" of ECS: a transformation from components to components. It declares what it
needs and what it produces, and runs against any entity that carries the right columns.

```python
class MySystem:
    REQUIRES: tuple[ComponentDescriptor, ...]  # must be present on each source
    PROVIDES: tuple[ComponentDescriptor, ...]  # written to the target

    sources: tuple[EntityPath, ...]  # what it reads
    target: EntityPath  # what it writes
    targets: tuple[EntityPath, ...]  # every path it writes

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]: ...
```

That is what replaces the `EvaluationTask` enum and the single `evaluation_config_dict` of the
original package. **An evaluation task is not a value to branch on -- it is the set of components
present plus the pipeline you compose.**

## The shape of a run

```text
                  ┌──────────────┐
   Dataset ──────▶│   Importer   │──────┐
                  └──────────────┘      │
                                        ▼
                                   ┌─────────┐
   Model output ───────────────────▶│  Store  │◀─────────────┐
                                   └────┬────┘               │
                                        │ range query        │ send_chunk
                        ┌───────────────┼───────────────┐    │
                        ▼               ▼               ▼    │
                     Filter          Matcher         Metric ─┘
                   (→ mask)        (→ verdicts)    (→ values)
                        │               │               │
                        └───────────────┴───────────────┘
                                        │
                                        ▼
                                  every stage's output
                                  is a queryable entity
```

Each stage writes its result **back into the store**, so which rows a filter dropped and what a
matcher scored stay available instead of being discarded.

## The four stages

| Stage        | Reads                                   | Writes                               | Page                                                  |
| :----------- | :-------------------------------------- | :----------------------------------- | :---------------------------------------------------- |
| **Filter**   | one entity                              | `mask` at `<source>/filter/<name>`   | [Filtering](../user-guide/filtering.md)               |
| **Match**    | an estimation and a ground truth        | `MatchResults` at `/matching/<mode>` | [Matching](../user-guide/matching.md)                 |
| **Metric**   | a matching plus the entities it matched | `MetricValues` at `/metrics/<name>`  | [Metrics](../user-guide/metrics.md)                   |
| **Analysis** | anything already in the store           | -- (queries, not systems)            | [Offline analysis](../user-guide/offline-analysis.md) |

A filter **masks rather than drops**, which keeps the verdict inspectable. When a metric needs the
filtered set to be an entity in its own right -- recall divides by the ground-truth count --
`ApplyMaskSystem` materializes it.

## SystemContext

Everything a system reads that is not one of its own parameters:

```python
from t4perceval import FRAME, Store
from t4perceval.system import SystemContext

ctx = SystemContext(store, FRAME, labels=labels, instances=instances)
```

The two registries give meaning to the integer id columns, so a system can be configured with class
names and UUIDs instead of raw ids:

```python
FilterByLabelSystem.on("/estimation/objects", labels=["car", "pedestrian"])
```

## Pipeline

`Pipeline` is an ordered list of systems whose **wiring is checked once, up front**.

```python
from t4perceval.system import Pipeline

pipeline = Pipeline([matcher, metric])
pipeline.run(ctx, TimeRange.everything())
```

The check is about _order_: if a system reads an entity that a **later** system writes, that is a
bug in the pipeline and it is reported at construction time rather than as an empty result at run
time.

```text
ValueError: AveragePrecisionSystem reads /matching/center_distance before a later system
writes it; reorder the pipeline
```

Components expected to come from the store rather than from another system are checked when the
pipeline runs, by `require`:

```text
ValueError: /estimation/objects is missing required component(s): instance_id
```

`Pipeline.run` sends every produced chunk into `ctx.store` and returns them in production order.

!!! note "Two runs for a materialized filter"

    `ApplyMaskSystem` copies whatever columns its source holds, so it cannot declare `PROVIDES`, and
    the up-front check would reject a matcher reading its target. Run the narrowing and the
    evaluation as two pipelines. See
    [Filtering](../user-guide/filtering.md#materializing-a-filtered-set).

## Presets are plain functions

```python
from t4perceval.system import average_precision_sweep

systems = average_precision_sweep(
    "/estimation/objects",
    "/ground_truth/objects",
    thresholds=[0.5, 1.0, 2.0, 4.0],
    heading=True,
)
[type(s).__name__ for s in systems]
```

The result is ordinary data you can print, edit or extend before handing it to a `Pipeline`. That is
the difference between sugar and the enum this package set out to remove.

## Where a metric gets its inputs

A metric reads three entities -- the matching result and the two entities it matched -- through a
`MatchJoin`, which lines up every match row with the estimation and ground-truth rows it points at:

```text
MatchResults row:  est_index=3  gt_index=7  score=0.42  status=TP  threshold=1.0
                        │            │
        ┌───────────────┘            └───────────────┐
        ▼                                            ▼
/estimation/objects row 3                 /ground_truth/objects row 7
   class_id, confidence, quaternion, ...     class_id, quaternion, ...
```

`-1` means "no counterpart", so a false positive is a row with `gt_index == -1` and a false negative
one with `est_index == -1`. The `threshold` column is the one thing a later stage could not recover
by following the indices back, because only the matcher knew it.

## Where to go next

- [Detection 3D](../evaluation/detection-3d.md) -- a whole task, end to end.
- [Extending systems](../development/extending-systems.md) -- writing your own filter, matcher or metric.
- [System design](../development/design/en/system.md) -- the long-form rationale.
