# Filtering

A filter **masks rather than drops**. It writes a boolean column to a new entity, so its verdict
stays queryable instead of becoming a discarded intermediate.

```text
/estimation/objects                       the rows
/estimation/objects/filter/distance       → mask: which rows passed
/estimation/objects/filter/confidence     → mask: which rows passed
/estimation/objects/filter/keep           → mask: the AND of the two
/estimation/objects/kept                  → the surviving rows, materialized
```

## The filter family

Every filter is built with `.on(source, **params)` and writes its mask to
`<source>/filter/<name>`.

| System                          | Requires      | Parameters                                                              |
| :------------------------------ | :------------ | :---------------------------------------------------------------------- |
| `FilterByDistanceSystem`        | `position`    | `min_distance=0.0`, `max_distance=inf`, `bev=False`                     |
| `FilterByRegionSystem`          | `position`    | `min_xy=(-inf, -inf)`, `max_xy=(inf, inf)`                              |
| `FilterByPolarGridSystem`       | `position`    | `min_distance=0.0`, `max_distance=inf`, `min_angle=-pi`, `max_angle=pi` |
| `FilterByMapSystem`             | `position`    | `polygon=None`, `resolver=None`                                         |
| `FilterPointsByDistanceSystem`  | `point`       | as `FilterByDistanceSystem`                                             |
| `FilterPointsByRegionSystem`    | `point`       | as `FilterByRegionSystem`                                               |
| `FilterPointsByPolarGridSystem` | `point`       | as `FilterByPolarGridSystem`                                            |
| `FilterPointsByMapSystem`       | `point`       | as `FilterByMapSystem`                                                  |
| `FilterByLabelSystem`           | `class_id`    | `labels=None`, `exclude=None`                                           |
| `FilterByConfidenceSystem`      | `confidence`  | `min_confidence=0.0`, `max_confidence=1.0`                              |
| `FilterByInstanceSystem`        | `instance_id` | `instances=None`, `exclude=None`                                        |
| `FilterBySpeedSystem`           | `velocity`    | `min_speed=0.0`, `max_speed=inf`                                        |
| `FilterByNumPointsSystem`       | `num_points`  | `min_num_points=0`, `max_num_points=None`                               |
| `FilterByVisibilitySystem`      | `visibility`  | `min_visibility=VisibilityLevel.NONE`                                   |

```python
from t4perceval.system import FilterByDistanceSystem

near = FilterByDistanceSystem.on("/estimation/objects", max_distance=50.0)
near.target  # /estimation/objects/filter/distance
```

Give `name=` to place the mask somewhere else:

```python
FilterByDistanceSystem.on("/estimation/objects", max_distance=50.0, name="near")
# → /estimation/objects/filter/near
```

Two conventions hold across the family. Bounds are inclusive on both ends, so a filter built with
its defaults is a guaranteed no-op; the original package compared strictly, which made
`min_distance=0` reject an object at the origin. And a missing column is an error, not a pass:
`t4_devkit` lets a box with no velocity through its speed filter, here that is a wiring mistake.

## Distance, region and speed

`FilterByDistanceSystem` measures from the origin **of the chunk's coordinate frame**, so positions
must already be relative to the point you are measuring from -- normally `base_link`, which puts the
ego at the origin.

```python
FilterByDistanceSystem.on(path, max_distance=50.0)  # full 3D norm (t4-devkit's rule)
FilterByDistanceSystem.on(path, max_distance=50.0, bev=True)  # xy plane only
```

`FilterByRegionSystem` bounds an axis-aligned box in xy. Note the difference between the two
constructors:

```python
FilterByRegionSystem.on(path, max_xy=(100.0, 50.0))  # upper bound only
FilterByRegionSystem.symmetric(path, max_xy=(100.0, 50.0))  # ±100 by ±50, mirrored about the origin
```

`symmetric` is the region the original package's `max_x_position` / `max_y_position` described.

These, the polar grid and the map filter below judge an object's `position`. A point cloud such as
`SemanticSegmentation3D` carries `point` instead, and the same predicates over it are
`FilterPointsByDistanceSystem`, `FilterPointsByRegionSystem`, `FilterPointsByPolarGridSystem` and
`FilterPointsByMapSystem`, with the same parameters:

```python
FilterPointsByDistanceSystem.on("/ground_truth/points", max_distance=50.0)
FilterPointsByRegionSystem.symmetric("/ground_truth/points", max_xy=(100.0, 50.0))
```

## Polar grid

`FilterByPolarGridSystem` bounds one cell of the polar grid around the origin: a distance range
crossed with an azimuth range. It narrows a score to "within 50 m, ahead" or "the far left" in
one filter.

```python
import numpy as np
from t4perceval.system import FilterByPolarGridSystem

# Within 50 m, in the 90 degree cone ahead.
front_near = FilterByPolarGridSystem.on(
    "/ground_truth/objects",
    max_distance=50.0,
    min_angle=np.deg2rad(-45.0),
    max_angle=np.deg2rad(45.0),
)
front_near.target  # /ground_truth/objects/filter/polar_grid

# 20 to 80 m, in a 20 degree cone across the rear.
rear = FilterByPolarGridSystem.on(
    "/ground_truth/objects",
    min_distance=20.0,
    max_distance=80.0,
    min_angle=np.deg2rad(170.0),
    max_angle=np.deg2rad(190.0),
)
```

Distance is measured in the xy plane. The angle is in radians, counter-clockwise from `+x`; in
`base_link` 0 is straight ahead and `+pi/2` is the left. `min_angle` may lie anywhere and
`max_angle` runs counter-clockwise from it, at most a full turn, so a cell may cross the `±pi`
seam as `rear` does above. Like the rest of the family the bounds are inclusive, and the defaults
pass every row. The grid is laid out in the frame the source declares, and `z` is ignored.

`FilterBySpeedSystem` uses the L2 norm of the `velocity` column.

## Map regions

`FilterByMapSystem` keeps objects whose xy position lies inside a polygon, and the polygon may be
stated in a frame other than the one the objects are recorded in. That is what makes it usable
with a map: objects are usually in `base_link`, a Lanelet2 map is in `map`.

```python
from t4perceval.lanelet import LaneletMap
from t4perceval.system import FilterByMapSystem

lanelet_map = LaneletMap.load("/data/t4/db_v1/map/lanelet2_map.osm")
on_road = FilterByMapSystem.on_lanelet(
    "/ground_truth/objects",
    lanelet_map,
    subtypes=("road", "crosswalk"),  # default: ("road", "road_shoulder", "crosswalk")
)
on_road.target  # /ground_truth/objects/filter/lanelet
```

`on_lanelet(...)` is `on(name="lanelet", polygon=lanelet_map.region(...))`. The polygon is always in `map`. When
the source declares another frame, the filter looks the ego pose up per frame -- the same lookup
`TransformEntitySystem` makes -- and moves the _positions_ into `map` before the test. The mask is
still written under the source, so it combines with every other mask on that entity, and the
source itself is untouched. A source already in `map` needs no lookup.

This is the one filter that reconciles frames, and the reason is the predicate: a distance or a
box is a claim about the frame it is measured in, so those filters must test where the data is.
Whether an object is on the road is a fact about the world, the same in every frame, so the
verdict belongs under the source and the transform is only how it is evaluated.

The lookup reads `/tf` from the store the pipeline runs on. When the evaluation store carries no
transforms, or they live on another timeline, pass a `resolver` built from the original recording,
as for `TransformEntitySystem`:

```python
from t4perceval import TIMESTAMP
from t4perceval.transform import TransformResolver

FilterByMapSystem.on_lanelet(
    "/estimation/objects",
    lanelet_map,
    subtypes=("road",),
    resolver=TransformResolver.of(estimation, timeline=TIMESTAMP),
)
```

Any polygon works, not only a map's. `polygon=` takes a shapely `Polygon` / `MultiPolygon` or one
or more rings of `(x, y)` points, in `map`:

```python
FilterByMapSystem.on(path, polygon=[[0, -5], [60, -5], [60, 5], [0, 5]])
```

The test is the object's centre only; an object straddling the edge is in or out by where its
centre is. See [Filter by map region](../recipes/filter-by-lanelet.md) for the full recipe.

## Labels and instances

Both accept names _or_ raw ids, and resolve names through the registry on the context:

```python
from t4perceval.system import FilterByLabelSystem, SystemContext

vehicles = FilterByLabelSystem.on("/ground_truth/objects", labels=["car", "truck", "bus"])
not_unknown = FilterByLabelSystem.on("/ground_truth/objects", exclude=["unknown"])

ctx = SystemContext(store, FRAME, labels=labels, instances=instances)
```

A name the registry does not know raises rather than matching nothing. Pass `instances=` on the
context for `FilterByInstanceSystem`.

## Ground-truth quality

`num_points` and `visibility` come from the dataset, not from a model, so these two filters apply to
ground truth:

```python
from t4perceval.component import VisibilityLevel
from t4perceval.system import FilterByNumPointsSystem, FilterByVisibilitySystem

FilterByNumPointsSystem.on("/ground_truth/objects", min_num_points=5)
FilterByVisibilitySystem.on("/ground_truth/objects", min_visibility=VisibilityLevel.PARTIAL)
```

`VisibilityLevel` is ordered -- `NONE < PARTIAL < MOST < FULL` -- and the filter keeps everything
_at least as visible as_ the level you give. Objects annotated `UNAVAILABLE` always pass, so a
dataset that does not record visibility is not emptied out.

## Combining masks

`CombineMasksSystem` reduces several masks to one, with `mode="all"` (AND, the default) or
`mode="any"` (OR):

```python
from t4perceval.system import CombineMasksSystem

near = FilterByDistanceSystem.on(SOURCE, max_distance=50.0)
confident = FilterByConfidenceSystem.on(SOURCE, min_confidence=0.5)
keep = CombineMasksSystem.of([near.target, confident.target], f"{SOURCE}/filter/keep")
```

The sources must describe the same rows, which is checked. This is also how a _per-class_ filter
is expressed -- AND within a class, then `mode="any"` across classes replaces the old
`min_point_numbers: [5, 0, 0]` list. Matching takes per-class thresholds directly, but not every
filter requires `class_id`, so composition does the job:

```python
CombineMasksSystem.of(
    [
        FilterByLabelSystem.on(SOURCE, labels=["car"], name="is_car").target,
        FilterByDistanceSystem.on(SOURCE, max_distance=80.0, name="car_range").target,
    ],
    f"{SOURCE}/filter/car_in_range",
    mode="all",
)
```

## Reading a mask

```python
from t4perceval.descriptors import MASK

mask = store.range(near.target, timeline=FRAME, time_range=TimeRange.everything())
mask.component(MASK).values  # array([ True,  True, False, ...])
```

For a lazy narrowed view of the source, without materializing anything:

```python
from t4perceval.system import masked_view

view = masked_view(
    store, "/estimation/objects", near.target, timeline=FRAME, time_range=TimeRange.everything()
)
```

`masked_view` raises if the mask and the source disagree about row count -- normally a sign the two
were computed over different ranges.

## Materializing a filtered set

A mask is enough for inspection, but **a metric needs the filtered set to be an entity**: recall
divides by the number of ground-truth objects, so the denominator has to be the filtered
ground truth. `ApplyMaskSystem` writes the surviving rows to a new path:

```python
from t4perceval.system import ApplyMaskSystem, Pipeline

near = FilterByDistanceSystem.on("/ground_truth/objects", max_distance=50.0)
kept = ApplyMaskSystem.of("/ground_truth/objects", near.target)
kept.target  # /ground_truth/objects/kept

Pipeline([near, kept]).run(ctx, TimeRange.everything())
```

Point the matcher **and** the metric at the same materialized entity, so the row indices a match
result stores refer to the rows both of them see.

!!! note "One pipeline is enough"

    `ApplyMaskSystem` carries over whatever columns its source holds, and says so:
    `PROVIDES = Passthrough(0)`. `Pipeline` therefore treats `/estimation/objects/kept` as carrying
    the source's columns -- checked up front when an earlier system declared them, at run time by
    `require()` when the source came from the store -- so the narrowing and the evaluation go in one
    pipeline:

    ```python
    Pipeline(
        [
            near_gt,
            kept_gt,
            near_est,
            kept_est,
            *average_precision_sweep("/estimation/objects/kept", "/ground_truth/objects/kept"),
        ]
    ).run(ctx, scene)
    ```

    Two pipelines still work; they are a choice, not a requirement. See
    [Evaluation pipeline](../concepts/evaluation-pipeline.md#pipeline) for what `Pipeline` knows
    about a passthrough target.

## A complete narrowing stage

```python
from t4perceval.system import (
    ApplyMaskSystem,
    CombineMasksSystem,
    FilterByConfidenceSystem,
    FilterByDistanceSystem,
    FilterByLabelSystem,
    Pipeline,
)

SOURCE = "/estimation/objects"

near = FilterByDistanceSystem.on(SOURCE, max_distance=50.0)
confident = FilterByConfidenceSystem.on(SOURCE, min_confidence=0.3)
wanted = FilterByLabelSystem.on(SOURCE, labels=["car", "pedestrian"])
keep = CombineMasksSystem.of(
    [near.target, confident.target, wanted.target], f"{SOURCE}/filter/keep"
)
kept = ApplyMaskSystem.of(SOURCE, keep.target)

Pipeline([near, confident, wanted, keep, kept]).run(ctx, TimeRange.everything())
```

Every one of those four masks is still in the store afterwards, so "how many did the confidence
threshold cost me?" is a query, not a re-run.

## Where to go next

- [Matching](matching.md) -- pairing what survived.
- [Filter by map region](../recipes/filter-by-lanelet.md) -- the Lanelet2 recipe end to end.
- [Filters catalogue](../systems/filters.md) -- every system, its requirements and defaults.
- [Write a custom filter](../recipes/custom-filter.md).
