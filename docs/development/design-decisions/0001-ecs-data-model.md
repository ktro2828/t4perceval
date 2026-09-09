# 0001: An ECS-inspired data model

## Status

Accepted.

## Context

[`autoware_perception_evaluation`](https://github.com/tier4/autoware_perception_evaluation) modelled
one detected object as one Python object, `DynamicObject`. That shape caused five distinct problems:

- `DynamicObject` held position, orientation, shape, velocity, `tracked_*` and `predicted_*` as more
  than twenty fields on a single class. Whichever task ran, the fields it did not use sat there
  filled with `None`.
- Work happened by iterating a `List[DynamicObject]` in Python, so nothing vectorized.
- The `EvaluationTask` enum leaked as `if` branches into config, matching, metrics and
  visualization. Adding a task meant touching all four.
- `Catalog → Scenario → Scene → PerceptionFrameResult` was a nest of Python lists, which made
  cross-frame queries -- CLEAR, HOTA, ADE -- awkward to write.
- Intermediate products, which rows a filter dropped and what a matcher scored, were discarded and
  could not be re-analysed.

## Decision

Implement [Rerun](https://github.com/rerun-io/rerun)'s data model: data lives at an **entity path**,
is made of **component** columns identified by a **descriptor**, is bundled into **archetypes**,
stored as **chunks**, indexed along **timelines** inside a **store**, and transformed by
**systems** -- the "S" of ECS.

An evaluation task is therefore **the set of components present plus the pipeline you compose**,
not a value to branch on. There is no `EvaluationTask` enum and no single config dict.

## Rationale

Every one of the five problems dissolves in the same move:

| Problem                 | What the model does about it                                       |
| :---------------------- | :----------------------------------------------------------------- |
| unused fields           | a column absent means the property is absent; nothing is `None`    |
| Python loops            | a frame is NumPy columns; a metric is a vectorized call            |
| the task enum           | a system declares `REQUIRES` and runs against anything carrying it |
| the nested containers   | one frame is `latest_at`, a scene is `range`, over the same store  |
| discarded intermediates | every stage writes a chunk back, so its output stays queryable     |

Descriptors are the load-bearing piece. Because `POSITION` means "position" in `Detections3D` and in
`Trackings3D` alike, one system serves both without knowing either exists. Naming a column by
meaning rather than by its declaring archetype is what makes composition work.

The [benchmark](../benchmarks.md) puts numbers on the vectorization: 67x to 224x faster on the
metric workloads, 16x smaller in retained memory.

## Alternatives considered

**Keep `DynamicObject`, vectorize internally.** Each metric would convert a list of objects into
arrays on entry. That fixes speed and nothing else: the enum, the `None` fields and the discarded
intermediates all remain, and every metric pays the conversion.

**A dataframe (pandas / Polars) per frame.** Good ergonomics, but a dataframe has no notion of a
required column, a per-row shape, or a nested `(M, T, 3)` value. Validation would move into every
consumer, and the trajectory columns would not fit at all.

**Depend on Rerun itself.** The vocabulary is Rerun's and the semantics are deliberately close, so
that anyone who knows Rerun already knows how to file and query data here. But evaluation needs
three things a visualization log does not: archetypes that _validate_, registries that give integer
columns meaning, and systems that write results back into the same store. Depending on Rerun would
mean adapting around a model built for a different purpose, and taking a large dependency for the
part that is easy.

**An actual ECS library.** The "E" and "C" carry their weight here; the runtime scheduling,
archetype storage and change detection that an ECS library exists to provide do not. What is wanted
is the data layout and the systems-as-functions idea, both of which are a few hundred lines.

## Consequences

**Good.**

- A new evaluation task is a new pipeline, often with no new code in the core.
- A filter's verdict, a matcher's score and a metric's value are all queryable after the fact, so
  "why is this number low?" is a query rather than a re-run.
- Per-frame and per-scene answers come from the same computation, differing only by the query range.
- Serialization is nearly free: columns are already Arrow-shaped.

**Costs.**

- The vocabulary is unfamiliar. Someone who wants "the third object" has to learn that there is no
  such thing, only row 3 of every column. That is why [Concepts](../../concepts/overview.md) exists
  and why it comes before the user guide.
- Nested shapes -- `(N, M, T, 3)` -- are harder to read than a list of trajectory objects, which is
  why `TrajectoryMode3D` exists as a non-component constructor.
- Composition instead of inheritance means the box components are literally re-declared in three
  archetypes. That duplication is deliberate, and a comment says so at each site.
- `Pipeline` can only validate wiring up front for entities whose columns it can know. A
  passthrough system (`PROVIDES = Passthrough(...)`) of a store-sourced entity yields an _opaque_
  target, whose consumers are checked at run time instead -- deferred, not rejected.

## Where it is written down

[Data model design](../design/en/data_model.md) ([日本語](../design/ja/data_model.md)) and
[System design](../design/en/system.md) ([日本語](../design/ja/system.md)).
