# API reference

Generated from the docstrings, so a signature change cannot leave this page behind. Use it to answer
"what exactly does this do"; use [Concepts](../../concepts/overview.md) to answer "why is it shaped
like this".

## By module

| Module                                   | Contents                                                       |
| :--------------------------------------- | :------------------------------------------------------------- |
| [`t4perceval.core`](core.md)             | entity paths, descriptors, chunks, timelines, the store, views |
| [`t4perceval.component`](component.md)   | the concrete column types                                      |
| [`t4perceval.archetype`](archetype.md)   | the concrete bundles                                           |
| [`t4perceval.system`](system.md)         | filters, matchers, metrics, `Pipeline`                         |
| [`t4perceval.geometry`](geometry.md)     | vectorized, pairwise box geometry                              |
| [`t4perceval.transform`](transform.md)   | frame graph discovery, lookup and composition                  |
| [`t4perceval.label`](label.md)           | `LabelRegistry`, `InstanceRegistry`, reconciliation            |
| [`t4perceval.recording`](recording.md)   | `Recording` and its metadata                                   |
| [`t4perceval.evaluation`](evaluation.md) | assembling recordings into an evaluation store                 |
| [`t4perceval.align`](align.md)           | pairing two recordings' frames by timestamp                    |
| [`t4perceval.importer`](importer.md)     | the T4 and MCAP ROS bag importers                              |
| [`t4perceval.io`](io.md)                 | Arrow and Parquet persistence                                  |

## The top-level namespace

The names most code needs are re-exported from `t4perceval` itself:

```python
from t4perceval import (
    ANY,
    FRAME,
    TIMESTAMP,
    Archetype,
    Chunk,
    ColumnarComponent,
    Component,
    ComponentDescriptor,
    MonoComponent,
    EntityPath,
    EntityView,
    Store,
    TimeColumn,
    TimeKind,
    TimePoint,
    TimeRange,
    Timeline,
    concat_chunks,
    Classifications2D,
    ConfusionMatrix,
    Detections2D,
    Detections3D,
    MatchResults,
    MetricValues,
    Predictions3D,
    SemanticSegmentation2D,
    SemanticSegmentation3D,
    Trackings2D,
    Trackings3D,
    Trajectories3D,
    TrajectoryMode3D,
    Transform3D,
    BACKGROUND_CLASS_ID,
    UNKNOWN_CLASS_ID,
    ClassInfo,
    InstanceRegistry,
    LabelRegistry,
    Recording,
    RecordingMetadata,
    SourceInfo,
)
```

The component classes live in `t4perceval.component`, the canonical descriptors in
`t4perceval.descriptors`, and everything in the system layer in `t4perceval.system`.

## Conventions the reference assumes

- Every column is read-only and never aliases an array you passed in.
- A `Batch*` component holds `N` rows; its unprefixed counterpart holds one value.
- A descriptor names a column by meaning, not by the archetype that declared it.
- Anywhere an entity path is taken, a plain string works.
- A time argument is either an `int` on the chosen timeline or a `TimeRange`.
