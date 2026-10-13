# Concepts overview

`t4perceval` is built on a data model borrowed from [Rerun](https://github.com/rerun-io/rerun) and
adapted to perception evaluation. Rerun itself is not a dependency; the model is reimplemented here.

Reading this section is optional for running your first evaluation, and close to mandatory for
extending the library. It explains _why_ the APIs look the way they do.

## The one idea

The original [`autoware_perception_evaluation`](https://github.com/tier4/autoware_perception_evaluation)
modelled one detected object as one Python object (`DynamicObject`) with more than twenty fields,
most of them `None` for any given task. `t4perceval` inverts that: **one frame of objects is a set
of columns**, and whether an entity can be used for an operation depends on the components it
carries, not its archetype type. For example, centre-distance matching requires `position` and
`class_id`, while the `Detections3D` archetype requires `position`, `quaternion`, `size`, `class_id`
and `confidence`.

```text
      One object per row                     One column per property
      (autoware_perception_evaluation)       (t4perceval)

      DynamicObject                          position    (N, 3) f64
        .position                            quaternion  (N, 4) f64
        .orientation                         size        (N, 3) f64
        .shape                               class_id    (N,)   i32
        .semantic_label                      confidence  (N,)   f64
        .semantic_score
        .uuid            × N objects
        .tracked_paths
        .predicted_paths
        ... 20+ fields
```

Everything else follows from that choice: vectorized work instead of Python loops, an evaluation
task that is a _composition_ instead of an enum, and intermediate products that stay queryable
instead of being discarded.

## The six abstractions

```text
Entity path        /ground_truth/objects        where data is filed
   │
   ▼
Component          one column, fixed shape      position, class_id, mask, ...
   │
   ▼
Archetype          a validated bundle           Detections3D, MatchResults, ...
   │
   ▼
Chunk              one write                     temporal or static columns
   │
   ▼
Store              collection and query layer    chunks filed by entity and time
   │
   ▼
System             components → components      filter, match, metric
```

| Page                                                          | Answers                                              |
| :------------------------------------------------------------ | :--------------------------------------------------- |
| [Data model](data-model.md)                                   | Why columns, and what a descriptor is for            |
| [Entity, component, archetype](entity-component-archetype.md) | How data is named, shaped and bundled                |
| [Store](store.md)                                             | Where chunks live, and how static data differs       |
| [Timeline](timeline.md)                                       | `FRAME` vs `TIMESTAMP`, and querying a range         |
| [Coordinate system](coordinate-system.md)                     | `frame_id`, `Transform3D`, and the cross-frame guard |
| [Evaluation pipeline](evaluation-pipeline.md)                 | Systems, `Pipeline`, and how a task is composed      |

## How the layers map to modules

Each abstraction above is one module, and each module has a short list of modules it must not
know about. The table is in [Architecture](../development/for-developers/architecture.md#module-responsibilities).

`importer` converts an external representation into this one; `io` moves an already-native
recording to and from storage. **Reading a saved recording is `io`; reading a dataset is
`importer`.**

## Relationship to Rerun

The vocabulary -- entity path, component, archetype, chunk, timeline, store -- is Rerun's, and the
semantics are deliberately close, so anyone who knows Rerun already knows how to file and query data
here. The implementation is separate because evaluation needs things a visualization log does not:
validated archetypes, per-class registries, and systems that write results back into the same store.

For the rationale, see [Architecture](../development/for-developers/architecture.md) and
[Design principles](../development/for-developers/design-principles.md).
