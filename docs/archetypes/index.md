# Archetypes

An archetype is a **validated bundle of components**: which are required, which are optional, and
what each field is called. Constructing one checks that every column has the same length and the
shape and dtype the schema says.

This section is a **schema catalogue** -- the semantic meaning of each bundle and how it relates to
the others, not a reproduction of the Python signatures. For those, see the
[API reference](../reference/api/archetype.md).

## The catalogue

| Archetype                | Page                                | What it is                                        |
| :----------------------- | :---------------------------------- | :------------------------------------------------ |
| `Detections2D`           | [Detection 2D](detection-2d.md)     | image-plane regions with a class and a confidence |
| `Detections3D`           | [Detection 3D](detection-3d.md)     | 3D boxes with a class and a confidence            |
| `Trackings2D`            | [Tracking](tracking.md)             | `Detections2D` plus a persistent instance id      |
| `Trackings3D`            | [Tracking](tracking.md)             | `Detections3D` plus a persistent instance id      |
| `Predictions3D`          | [Prediction](prediction.md)         | `Trackings3D` plus multi-modal futures            |
| `Trajectories3D`         | [Trajectory](trajectory.md)         | futures on their own, no box                      |
| `Classifications2D`      | [Classification](classification.md) | a class and a confidence, no geometry             |
| `SemanticSegmentation2D` | [Segmentation](segmentation.md)     | a class per labelled pixel                        |
| `SemanticSegmentation3D` | [Segmentation](segmentation.md)     | a class per labelled point                        |
| `Transform3D`            | [Transform 3D](transform-3d.md)     | the pose of a child frame in its parent           |
| `MatchResults`           | [Matching](matching.md)             | the outcome of matching two entities              |
| `MetricValues`           | [Metrics](metrics.md)               | one metric's values, by class and threshold       |
| `ConfusionMatrix`        | [Metrics](metrics.md)               | long-form detection confusion matrix              |

## The object family

The five object archetypes form a chain by **composition**, not inheritance:

```text
Detections2D  ──(+ instance_id)──▶  Trackings2D

Detections3D  ──(+ instance_id)──▶  Trackings3D  ──(+ waypoints, mode_confidence)──▶  Predictions3D
```

Each re-declares the components of the one before it, and they resolve to the **same descriptors**.
So:

```python
from t4perceval import Detections3D, Predictions3D, Trackings3D

prediction.has(*Detections3D.required_descriptors())  # True
prediction.has(*Trackings3D.required_descriptors())  # True
```

Any system that requires a detection's components runs unchanged against a tracking or a prediction.
Inheritance would have forced every consumer to know the hierarchy; composition lets a system state
its requirement and stop caring.

## Reading an archetype's schema

```python
from t4perceval import Detections3D

Detections3D.archetype_name()  # 'Detections3D'
[d.component for d in Detections3D.required_descriptors()]
[d.component for d in Detections3D.optional_descriptors()]
Detections3D.descriptor_of("position")  # the POSITION descriptor
```

## Shared surface

```python
detections = Detections3D(...)

len(detections)                       # rows
detections.has(POSITION, CLASS_ID)    # are these columns present?
detections.as_components()            # descriptor -> component
detections.select(mask)               # independent data
detections.to_chunk("/path", at=TimePoint.at(frame=0), frame_id="base_link")
Detections3D.from_chunk(chunk)
Detections3D.from_components({POSITION: ..., ...})
```

An optional field is `None` when absent, so `detections.velocity` is either a column or `None`.

## Where to go next

- [Components](../components/index.md) -- the columns these bundle.
- [Extending archetypes](../development/extending-archetypes.md) -- adding your own.
