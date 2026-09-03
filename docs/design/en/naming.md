# Archetype and Component Naming

Use plural, domain-oriented names for collection archetypes, but do not mechanically pluralize every
type whose current name starts with `Batch`.

Rerun's public archetypes use names such as `Points3D`, `Boxes3D`, and `Scalars`; batching is implicit
in the archetype. A single `Boxes3D` value can contain one or many boxes. This matches the behavior
of `Detections3D`.

## Archetype Names

| Previous                      | Canonical                |
| ----------------------------- | ------------------------ |
| `BatchDetection3D`            | `Detections3D`           |
| `BatchDetection2D`            | `Detections2D`           |
| `BatchTracking3D`             | `Trackings3D`            |
| `BatchTracking2D`             | `Trackings2D`            |
| `BatchPrediction3D`           | `Predictions3D`          |
| `BatchTrajectory3D`           | `Trajectories3D`         |
| `BatchClassification2D`       | `Classifications2D`      |
| `BatchMatchResult`            | `MatchResults`           |
| `BatchMetric`                 | `MetricValues`           |
| `BatchSemanticSegmentation2D` | `SemanticSegmentation2D` |
| `BatchSemanticSegmentation3D` | `SemanticSegmentation3D` |

Some archetype names should remain singular. A semantic segmentation describes one complete labeling
result even though it contains many pixels or points, similar to Rerun's singular `Image` and
`SegmentationImage` archetypes. The naming rule should be semantic rather than purely grammatical.

`Trackings3D` and `Classifications2D` intentionally favor concise process-result plurals, keeping the
API visually parallel with `Detections3D` and `Predictions3D`. A tracking row still represents a
tracked object at one observation time rather than a complete time-series track.

## Components Need a Different Rule

Rerun distinguishes between:

- A semantic component type, such as `Position3D`, `ClassId`, or `Color`
- A batch representation, such as `Position3DBatch`
- A collection archetype, such as `Points3D`

Therefore, component types should not be renamed by mechanically converting them to plurals:

```python
# Not recommended: easily confused with collection archetypes.
BatchPosition3D -> Positions3D
BatchClassId -> ClassIds
```

Two conventions were settled by the transform work. `BatchFrameId` follows the rule above --
singular, `Batch`-prefixed -- and is worth noting because it is the only component that is not a
numeric array; the test for a text column is what it counts, a frame name being per _edge_ of the
transform graph whereas a class or instance name is per _object_ and stays interned in a registry.
And a component that holds **one** value rather than a column of them drops the prefix: `FrameId`,
`Position3D` and `Quaternion` are the mono counterparts of `BatchFrameId`, `BatchPosition3D` and
`BatchQuaternion` -- the same distinction Rerun draws between `Position3D` and `Position3DBatch`.

If component types are renamed, suffixing them with `Batch` more closely follows Rerun's distinction:

```python
BatchPosition3D -> Position3DBatch
BatchClassId -> ClassIdBatch
BatchConfidence -> ConfidenceBatch
```

Alternatively, retain the existing component names initially and rename only the archetypes. This is
the safer and more focused first step.

## Why Rename the Archetypes

- `Batch` exposes storage mechanics rather than domain meaning.
- `Detections3D(...)` reads naturally at call sites.
- One-row and zero-row values no longer sound exceptional.
- The public API becomes more consistent with the Rerun-inspired data model.
- The project is still at version `0.1.0`, making this the appropriate time for a breaking naming
  cleanup.

For example:

```python
store.log(
    "/estimation/objects",
    Detections3D(
        position=...,
        quaternion=...,
        size=...,
        class_id=...,
        confidence=...,
    ),
    at=...,
)
```

## Migration

The archetype rename is a breaking API change; the previous names are not retained as aliases.
Component naming remains a separate decision. If components are renamed later, serialized
component-name aliases must be retained because Parquet metadata records component class names.

The resulting rule is: **rename `BatchDetection3D` to `Detections3D`, but do not blindly convert every
`Batch*` class into a plural.** Archetypes should describe collections semantically, while components
should follow a distinct component/batch convention.
