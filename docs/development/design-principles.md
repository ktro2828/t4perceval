# Design principles

The rules the code is written to. Each one exists because breaking it caused a specific problem in
the package this one replaces.

## 1. A task is a composition, not a value

There is no `EvaluationTask` enum and no single config dict. An evaluation task **is** the set of
components present plus the pipeline you compose.

In `autoware_perception_evaluation` the enum leaked as `if` branches into config, matching, metrics
and visualization; adding a task meant touching all four. Here a system declares
`REQUIRES = (POSITION,)` and runs against anything carrying a position.

The corollary: **presets are plain functions returning lists**. `average_precision_sweep` returns
ordinary data you can print, edit or extend. That is the difference between sugar and the enum.

## 2. Columns, not objects

One frame of detections is a set of NumPy columns. `DynamicObject` held twenty-plus fields, most of
them `None` for any given task, and every operation was a Python loop.

- A component that holds several rows is prefixed `Batch`; its one-value counterpart is not.
- The batch dimension is explicit: `(N, M, T, 3)` says objects, modes, timesteps, dimensions.
- A batch of zero rows is legal everywhere.

## 3. Meaning lives in the descriptor, not the type

`BatchClassId` is a column of `i32`. Whether it means `class_id`, `ground_truth_class_id` or
`estimation_class_id` is the descriptor's job, and the descriptor is named for meaning rather than
for the archetype that declared it.

That is what lets one system serve several archetypes -- and it is why `TRANSLATION` is deliberately
**not** `POSITION`: a transform is a relationship between frames, not a thing in a frame, and a
distance filter should not appear to work on one.

## 4. Composition over inheritance

`Trackings3D` re-declares `Detections3D`'s components rather than inheriting them. They resolve to
the same descriptors, so `tracking.has(*Detections3D.required_descriptors())` is `True` without
anyone knowing a hierarchy exists.

Inheritance would make `Predictions3D` an `is-a` chain that every consumer has to reason about.
Composition lets a consumer state a requirement and stop caring.

## 5. Nothing is discarded

Which rows a filter dropped and what a matcher scored are written back into the store as entities.
The original package threw both away, so "why is this number low?" meant re-running with print
statements.

Consequences:

- A filter **masks**; it does not drop.
- A match result stores **row indices**, not object references, so it is storable and re-analysable.
- A metric records the **threshold** it was reached at, because that is the one thing a later stage
  could not recover.
- A per-frame answer and a whole-scene answer differ only by the query range.

## 6. Immutable data, cheap movement

Columns are read-only and never alias a writable array a caller passed in; chunks are frozen. So
moving an entity between stores references the same objects rather than copying, and a view handed
to a metric cannot be mutated under it.

## 7. Fail loudly on a silent-wrong

The failures this domain produces are _plausible numbers_, not exceptions. Wherever that is
possible, the code raises instead:

| Silent wrong                                       | What happens instead                              |
| :------------------------------------------------- | :------------------------------------------------ |
| comparing geometry across coordinate frames        | `ValueError` naming both entities and both frames |
| a per-class threshold mapping with no default      | `ValueError` asking for an explicit `Thresholds`  |
| a pipeline reading an entity a later system writes | `ValueError` at construction, not an empty result |
| an unreachable frame in a transform lookup         | raises, rather than resolving to identity         |
| two recordings disagreeing about class ids         | raises, unless `reconcile=True`                   |
| a category the registry does not know              | raises, unless `unknown_labels="ignore"`          |

The counterpart rule: **an absence is not a disagreement.** An unstated `frame_id` does not raise,
because it cannot be shown to conflict with anything.

## 8. Optional dependencies are enforced by packaging

The evaluation core never imports `t4_devkit` or `mcap`. Keeping them out of the base install is
what makes that structural rather than aspirational -- an accidental import fails for anyone
without the extra, and a test asserts it.

## 9. NaN is an answer

A class with no ground truth in range gets a `NaN` value with `support == 0` rather than being
dropped from the report. The shape of a result should not depend on the scene.

## 10. Pinned tooling

The Ruff rule set is pinned rather than inherited, because an unpinned project starts failing on
code that was clean when it was written. Widening it is a deliberate change, made in
`pyproject.toml`, not a side effect of an upgrade.

## Where to go next

- [Architecture](architecture.md) -- the layers these principles produced.
- [Design decisions](design-decisions/index.md) -- the individual calls, with alternatives.
