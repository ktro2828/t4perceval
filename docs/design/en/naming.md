# Archetype and Component Naming

Use plural, domain-oriented names for collection archetypes, but do not mechanically pluralize every
type whose current name starts with `Batch`.

Rerun's public archetypes use names such as `Points3D`, `Boxes3D`, and `Scalars`; batching is implicit
in the archetype. A single `Boxes3D` value can contain one or many boxes. This matches what
`BatchDetection3D` does today.

## Recommended Archetype Names

| Current                       | Proposed                 |
| ----------------------------- | ------------------------ |
| `BatchDetection3D`            | `Detections3D`           |
| `BatchDetection2D`            | `Detections2D`           |
| `BatchTracking3D`             | `TrackedObjects3D`       |
| `BatchTracking2D`             | `TrackedObjects2D`       |
| `BatchPrediction3D`           | `Predictions3D`          |
| `BatchTrajectory3D`           | `Trajectories3D`         |
| `BatchClassification2D`       | `ClassificationResults2D` |
| `BatchMatchResult`            | `MatchResults`           |
| `BatchMetric`                 | `MetricValues`           |
| `BatchSemanticSegmentation2D` | `SemanticSegmentation2D` |
| `BatchSemanticSegmentation3D` | `SemanticSegmentation3D` |

Some archetype names should remain singular. A semantic segmentation describes one complete labeling
result even though it contains many pixels or points, similar to Rerun's singular `Image` and
`SegmentationImage` archetypes. The naming rule should be semantic rather than purely grammatical.

For tracking, `TrackedObjects3D` is preferable to `Trackings3D`. `Tracks3D` is shorter, but it could
imply that each row contains a complete time-series track, whereas the current archetype contains
tracked objects at one observation time.

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

## Migration Approach

1. Rename the archetypes first.
2. Keep deprecated aliases for one release:

   ```python
   BatchDetection3D = Detections3D
   ```

3. Update the documentation and the naming rule in `TODO.md`.
4. Decide component naming separately.
5. If components are renamed, retain serialized component-name aliases because Parquet metadata
   records component class names.

The resulting rule is: **rename `BatchDetection3D` to `Detections3D`, but do not blindly convert every
`Batch*` class into a plural.** Archetypes should describe collections semantically, while components
should follow a distinct component/batch convention.
