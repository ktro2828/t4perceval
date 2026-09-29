# Filter by map region

**Goal:** evaluate only the objects on the road, using the Lanelet2 map a T4 scene ships with.

## The problem

A dataset annotates everything the sensors see: parked cars in a lot, pedestrians on a far
pavement, a bus behind a fence. A detector is not expected to find those, and counting them as
false negatives says nothing about driving. `FilterByDistanceSystem` cuts by range but not by
where the road is; the map knows where the road is.

## The map

A T4 scene keeps its map at `<data_root>/map/lanelet2_map.osm`. `LaneletMap.load` reads the
lanelets (a `subtype` each: `road`, `crosswalk`, `walkway`, `road_shoulder`, ...) into shapely
polygons in the `map` frame. It reads the nodes' `local_x` / `local_y` tags, which _are_ the map
frame, so no projection library is needed and none is used.

```python
from t4perceval.lanelet import LaneletMap

lanelet_map = LaneletMap.load("/data/t4/db_v1/map/lanelet2_map.osm")
lanelet_map.subtypes  # ('crosswalk', 'road', 'walkway')
```

`region(...)` returns the union of the subtypes you select. The vocabulary is the map's own:
`subtypes` lists what this map carries, and a subtype it lacks contributes nothing, so one
selection applies across scenes whose maps differ. Check the mask counts after a run; a typo shows
up as a region that keeps nothing.

```python
road = lanelet_map.region(subtypes=("road", "crosswalk"))
```

## The filter

Ground truth from a T4 import is in `base_link`, the map is in `map`. `FilterByMapSystem`
reconciles that itself: it looks the ego pose up per frame at `/tf/base_link`,
which the importer records, and moves each object's position into `map` before the test. The mask
lands under the source like any other filter's, so the source stays in `base_link` and every
other filter and matcher reads it unchanged.

```python
from t4perceval import TimeRange
from t4perceval.evaluation import SourceSpec, build_evaluation_store_from
from t4perceval.importer.t4 import T4Importer
from t4perceval.system import (
    ApplyMaskSystem,
    FilterByMapSystem,
    Pipeline,
    average_precision_sweep,
)
from t4perceval.transform import TransformResolver

importer = T4Importer.open("/data/t4/db_v1")
labels = importer.label_registry()
ground_truth = importer.import_scene(labels=labels)
estimation = ...  # a recording of your model's output over the same scene, in base_link

setup = build_evaluation_store_from(
    [
        SourceSpec.of(ground_truth, "/ground_truth/objects"),
        SourceSpec.of(estimation, "/estimation/objects"),
    ],
)

on_road = FilterByMapSystem.on_lanelet(
    "/ground_truth/objects",
    lanelet_map,
    subtypes=("road", "crosswalk"),
    resolver=TransformResolver.of(ground_truth),  # the frame tree lives in the recording
)
kept = ApplyMaskSystem.of("/ground_truth/objects", on_road.target)

Pipeline(
    [
        on_road,
        kept,
        *average_precision_sweep("/estimation/objects", kept.target, thresholds=(1.0, 2.0)),
    ]
).run(setup.context(), TimeRange.everything())
```

The resolver is built from the recording because that is where the importer logged the ego
poses, at `/tf/base_link`. The alternative is to carry that entity into the evaluation store with
its own `SourceSpec` and let the filter build the resolver itself; the store then holds a `map`
entity next to `base_link` ones, so the frame check has to be told that is intended:

```python
setup = build_evaluation_store_from(
    [
        SourceSpec.of(ground_truth, "/ground_truth/objects"),
        SourceSpec.of(ground_truth, "/tf/base_link"),
        SourceSpec.of(estimation, "/estimation/objects"),
    ],
    require_same_frame_id=False,
)
on_road = FilterByMapSystem.on_lanelet("/ground_truth/objects", lanelet_map, subtypes=("road",))
```

## Filtering the estimation too

A detector that reports a parked car off the road is not wrong about the world, only about what
was asked. Apply the same region to the estimation so both sides are cut alike, and combine it
with the usual range cut. Both masks land under the estimation entity, so they combine directly:

```python
from t4perceval.system import CombineMasksSystem, FilterByDistanceSystem

EST = "/estimation/objects"
est_on_road = FilterByMapSystem.on_lanelet(
    EST, lanelet_map, subtypes=("road", "crosswalk"), resolver=TransformResolver.of(ground_truth)
)
est_near = FilterByDistanceSystem.on(EST, max_distance=80.0)  # in base_link: from the ego
est_keep = CombineMasksSystem.of([est_on_road.target, est_near.target], f"{EST}/filter/keep")
est_kept = ApplyMaskSystem.of(EST, est_keep.target)
```

The estimation borrows the ground truth's ego poses here because both recordings describe the
same drive; a bag import records its own `/tf`, in which case build the resolver from that
recording instead.

## Reading the verdict back

The mask is data. After the run, the rows the map rejected are still there, with the reason next
to them:

```python
mask = setup.store.range(on_road.target, timeline=FRAME, time_range=TimeRange.everything())
mask.component(MASK).values  # one bool per ground-truth object, in row order
```

## What the test is

The object's _centre_ is tested, nothing else. A truck whose centre is on the pavement while its
tail hangs over the road is out. If that matters, select the `road_shoulder` and `walkway`
subtypes too.
