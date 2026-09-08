# Implement a custom importer

**Goal:** read a format `t4perceval` does not ship an importer for -- your own annotation JSON, a
KITTI-style directory, a database.

## What an importer is

A function from an external representation to a [`Recording`](../user-guide/persistence.md): an
immutable log plus the registries that give its integer columns meaning.

```text
your format ──▶ [ source ] ──▶ [ convert ] ──▶ Store ──▶ Recording
                    │               │
      the only module that     archetypes and
      knows the external       entity paths
      library
```

Everything downstream -- `build_evaluation_store`, the pipeline, the metrics -- already works, so
this is the only piece you write.

## The module layout the built-ins use

| Module          | Responsibility                                         |
| :-------------- | :----------------------------------------------------- |
| `source.py`     | **the only** module that talks to the external library |
| `labels.py`     | the external class vocabulary → canonical names        |
| `convert.py`    | external records → archetypes                          |
| `paths.py`      | where the data lands in the entity hierarchy           |
| `transforms.py` | the frame graph, if the format has one                 |
| `importer.py`   | the public class: selection, options, and the loop     |

Copy it. Isolating the external library in `source.py` is what keeps the rest of the importer -- and
all of the evaluation core -- free of a dependency it should not have.

## The minimum

```python
from __future__ import annotations

from attrs import define

from t4perceval import (
    Detections3D,
    InstanceRegistry,
    LabelRegistry,
    Recording,
    RecordingMetadata,
    SourceInfo,
    Store,
    TimePoint,
)


@define(frozen=True, slots=True)
class MyImporter:
    path: str

    @classmethod
    def open(cls, path: str) -> MyImporter:
        return cls(path)

    def label_registry(self) -> LabelRegistry:
        """Every class this source can produce, in a stable order."""
        return LabelRegistry.from_names(sorted(read_categories(self.path)))

    def import_scene(
        self,
        *,
        labels: LabelRegistry,
        instances: InstanceRegistry | None = None,
    ) -> Recording:
        instances = instances if instances is not None else InstanceRegistry()
        store = Store()

        for frame, record in enumerate(read_frames(self.path)):
            store.log(
                "/ground_truth/objects",
                Detections3D(
                    position=record["xyz"],
                    quaternion=record["quat_xyzw"],
                    size=record["wlh"],
                    class_id=labels.encode(record["names"]),
                    confidence=[1.0] * len(record["xyz"]),
                ),
                at=TimePoint.at(frame=frame, timestamp_ns=record["stamp_ns"]),
                frame_id="base_link",
            )

        return Recording.of(
            store,
            labels=labels,
            instances=instances,
            metadata=RecordingMetadata(
                sources=(
                    SourceInfo(
                        kind="myformat",
                        uri=self.path,
                        entity_path="/ground_truth/objects",
                        extra=(("frames", str(len(store.times("/ground_truth/objects", FRAME)))),),
                    ),
                ),
                frame_id="base_link",
            ),
        )
```

## Rules to follow

- **Take the registry as an argument, do not invent one.** Class ids are assigned in first-seen
  order, so an importer that derives its own is silently incompatible with every other source.
  Offer `label_registry()` so a caller _can_ derive one, and then require it to be passed back in.
- **Log both timelines.** `TimePoint.at(frame=..., timestamp_ns=...)`. Only the timestamp axis is
  comparable with another recording, and [alignment](align-frames.md) needs it.
- **Log empty frames.** A frame with no objects is a real frame; skipping it removes its false
  negatives from the denominator.
- **State `frame_id`.** Even for an empty frame -- an unstated frame makes a system's output frame
  flicker across a scene, which `concat_chunks` then refuses to join.
- **Fill in the metadata.** `SourceInfo` is what makes a result interpretable later: which file,
  which version, which options.
- **Return a `Recording`, not a `Store`.** Read-only is the point: an evaluation materializes what
  it needs into a fresh, writable store, so a saved result is about the evaluation.

## Behind an optional dependency

If your format needs a library not everyone has, add an extra in `pyproject.toml` and use the helper
that turns a missing dependency into an actionable message:

```python
from t4perceval.importer._optional import require

pandas = require("pandas", extra="myformat")
```

```text
ImportError: 'pandas' is required by this importer but is not installed.
Install it with:  pip install 't4perceval[myformat]'
```

Keep that call inside `source.py`, so importing your package does not pull the dependency in.

## Selection and options

The built-in importers take two frozen attrs classes -- a _selection_ (which slice of the source)
and _options_ (how the conversion is done). Follow that split: a selection changes what you get,
an option changes how it is represented.

```python
@define(frozen=True, slots=True)
class MySelection:
    scene: str | int | None = None
    frames: slice | Sequence[int] | None = None


@define(frozen=True, slots=True)
class MyOptions:
    kind_3d: str = "detections"
    coords: str = "base_link"
    unknown_labels: str = "error"
```

`unknown_labels="error"` versus `"ignore"` is worth copying: a category the registry does not know
should fail loudly by default, because the alternative is a class silently becoming
`UNKNOWN_CLASS_ID` and scoring as a miss.

## Transforms

If your format records poses, log them as [`Transform3D`](../archetypes/transform-3d.md) edges and
they will compose in the frame graph for free:

```python
store.log_static("/tf/lidar", extrinsic, frame_id="base_link")  # a calibration
store.log("/tf/base_link", ego_pose, at=at, frame_id="map")  # an ego pose
```

## Testing it

The built-in importers are tested against **synthetic fixtures** built in-process
(`tests/t4_builder.py`, `tests/rosbag_builder.py`) rather than against a checked-in dataset. Do the
same: a builder that writes a minimal valid source is faster, and makes the edge cases -- an empty
frame, an unknown category, a missing sensor -- easy to construct.

## Where to go next

- [Dataset importers](../user-guide/dataset-importers.md) -- what the built-in ones do.
- [Persistence](../user-guide/persistence.md) -- what a `Recording` is.
