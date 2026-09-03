# System Layer Design

## Purpose

In the original package the evaluation flow was concentrated in `PerceptionEvaluationManager`, and the
`EvaluationTask` enum (`detection`, `tracking`, `prediction`, `detection2d`, `tracking2d`,
`classification2d`) leaked as `if` branches into config, matching, metrics and visualization. On top of
that, thresholds, label settings and TP/FP criteria all lived in one `evaluation_config_dict`, so
whether a combination made sense could only be discovered as a runtime error.

The system layer resolves this structurally: **an evaluation task is not an enum value, it is the set
of components present plus the pipeline you compose.**

## The System protocol

```python
@runtime_checkable
class System(Protocol):
    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]]  # components it needs
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]]  # components it writes

    @property
    def sources(self) -> tuple[EntityPath, ...]: ...  # entities it reads
    @property
    def target(self) -> EntityPath: ...  # entity it writes

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]: ...
```

`SystemContext` carries everything a system reads that is not one of its own parameters.

```python
@define(frozen=True, slots=True)
class SystemContext:
    store: Store
    timeline: Timeline = FRAME
    labels: LabelRegistry | None = None
    instances: InstanceRegistry | None = None
```

The two registries let a system be configured with class names and UUIDs instead of the raw ids
stored in the columns.

`EntitySystem` is the base class implementing the `sources` / `target` wiring.

### Nothing intermediate is thrown away

Every system returns its result as a `Chunk`, and `Pipeline` sends it into the store. The reason a
filter dropped a row and the score a matcher produced therefore stay in the store, available for
re-analysis and visualization. The original package discarded them.

A filter **does not delete rows; it emits a boolean mask**. Because the rows remain, you can still ask
later why a given object was excluded from evaluation.

## Pipeline

```python
pipeline = Pipeline(
    [
        FilterByDistanceSystem.on("/estimation/objects", max_distance=102.4),
        CenterDistanceMatchingSystem.between(
            "/estimation/objects", "/ground_truth/objects", threshold=1.0
        ),
    ]
)
pipeline.run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())
```

### What is checked at construction time

The check is about **order**.

- A system reading the `target` of a _later_ system is an error, reported immediately rather than as
  an empty result at run time.
- When a system's source is an earlier system's `target`, that earlier system's `PROVIDES` must cover
  this system's `REQUIRES`.

Components expected to come from the store are checked at run time by `require()`, because the store's
contents are not knowable at construction time.

```python
# The filter writes only a mask, so its entity cannot be a matching source
Pipeline(
    [
        FilterByDistanceSystem.on("/estimation/objects"),
        CenterDistanceMatchingSystem.between(
            "/estimation/objects/filter/distance", "/ground_truth/objects"
        ),
    ]
)
# ValueError: ... reads /estimation/objects/filter/distance for component(s)
#             class_id, position, which no earlier system provides there
```

## Filter systems

### `MaskSystem` — the shared base

A filter writes only its predicate. The query, the validation, the empty-frame case and the
chunk construction are implemented once, in `MaskSystem`.

```python
@define(slots=True)
class FilterByConfidenceSystem(MaskSystem):
    REQUIRES = (CONFIDENCE,)
    FILTER_NAME = "confidence"  # target defaults to <source>/filter/confidence

    min_confidence: float = field(default=0.0, kw_only=True)
    max_confidence: float = field(default=1.0, kw_only=True)

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        confidence = view.component(CONFIDENCE).values
        return (confidence >= self.min_confidence) & (confidence <= self.max_confidence)
```

`MaskSystem.on(source, *, name=None, **params)` does the wiring, and `target` becomes
`<source>/filter/<name>`. Putting the mask under the source path means one prefix query finds an
entity together with every verdict recorded about it.

### Conventions across the family

- **Bounds are inclusive on both ends**, so a filter constructed with its default parameters passes
  every row. The original package compared strictly (`score > threshold`,
  `abs(x) < max_x_position`), which makes `min_distance=0.0` reject an object at the origin. The
  boundary itself is measure-zero in float arithmetic, and a default that is a guaranteed no-op is
  worth more than matching it exactly.
- **A missing column is an error, not a pass.** `t4_devkit` lets a box with no velocity through its
  speed filter; here, filtering on a component the entity does not carry means the pipeline is wired
  wrong, and `require()` says so. A frame with zero rows is an ordinary empty frame, so the check is
  skipped there.

### Implemented filters

| System                     | `REQUIRES`    | Parameters                          | Old config                            |
| :------------------------- | :------------ | :---------------------------------- | :------------------------------------ |
| `FilterByDistanceSystem`   | `POSITION`    | `min_distance` `max_distance` `bev` | `max_distance` / `min_distance`       |
| `FilterByRegionSystem`     | `POSITION`    | `min_xy` `max_xy`                   | `max_x_position` / `max_y_position`   |
| `FilterByLabelSystem`      | `CLASS_ID`    | `labels` `exclude`                  | `target_labels` / `ignore_attributes` |
| `FilterByConfidenceSystem` | `CONFIDENCE`  | `min_confidence` `max_confidence`   | `confidence_threshold`                |
| `FilterByInstanceSystem`   | `INSTANCE_ID` | `instances` `exclude`               | `target_uuids`                        |
| `FilterBySpeedSystem`      | `VELOCITY`    | `min_speed` `max_speed`             | —                                     |
| `FilterByNumPointsSystem`  | `NUM_POINTS`  | `min_num_points` `max_num_points`   | `min_point_numbers`                   |
| `FilterByVisibilitySystem` | `VISIBILITY`  | `min_visibility`                    | —                                     |

Per-filter notes:

- `FilterByDistanceSystem` measures in the coordinate frame the source chunk declares, so positions
  must already be expressed relative to the point being measured from — normally `base_link`. The
  default is the full 3D norm, matching `t4_devkit.filtering.FilterByDistance`; `bev=True` measures
  in the xy plane only, which is what the original package's `max_distance` did.
- `FilterByRegionSystem.symmetric(source, max_xy=(102.4, 102.4))` expresses the symmetric
  region the original config described with `max_x_position` / `max_y_position`. Unlike
  `on(max_xy=...)`, which bounds from above only, it also pins `min_xy` to `-max_xy`.
- `FilterByLabelSystem` and `FilterByInstanceSystem` accept class names and UUIDs as well as raw
  ids, resolving names through `ctx.labels` / `ctx.instances`. **An unknown name raises**: a typo
  that silently turned a filter into a no-op would skew every downstream metric.
- `FilterByVisibilitySystem` always passes objects annotated `UNAVAILABLE`, matching
  `t4_devkit.filtering.FilterByVisibility`. Rejecting them would empty out any dataset that does not
  annotate visibility at all.

### `CombineMasksSystem` — composing masks

```python
CombineMasksSystem.of([mask_a, mask_b], target, mode="all")  # intersection (AND)
CombineMasksSystem.of([mask_a, mask_b], target, mode="any")  # union (OR)
```

Having both modes is what makes **per-class thresholds expressible without a per-class threshold
parameter**. The original `min_point_numbers: [5, 0, 0]` — a list aligned with `target_labels` —
becomes an AND of a label filter with a threshold filter per class, then an OR across the classes.

```python
# Cars must have 50+ points; every other class is admitted regardless.
is_car = FilterByLabelSystem.on(src, labels=["car"], name="is_car")
not_car = FilterByLabelSystem.on(src, exclude=["car"], name="not_car")
many_points = FilterByNumPointsSystem.on(src, min_num_points=50, name="pts50")
car_ok = CombineMasksSystem.of(
    [is_car.target, many_points.target], f"{src}/filter/car_ok", mode="all"
)
keep = CombineMasksSystem.of([car_ok.target, not_car.target], f"{src}/filter/keep", mode="any")

Pipeline([is_car, not_car, many_points, car_ok, keep]).run(ctx, TimeRange.everything())
```

Every source mask must describe the same rows — they normally come from filters on one shared source
entity — and this is checked rather than assumed.

### `masked_view()` — only the rows that passed

```python
view = masked_view(
    store, "/estimation/objects", keep.target, timeline=FRAME, time_range=TimeRange.everything()
)
passed = view.materialize(Detections3D)
```

The result is a lazy view still referring to the original chunk, so nothing is copied until a column
is asked for.

## Matching systems

### `MatchingSystem` — the shared base

A matcher writes only its score matrix. The frame loop, the feasibility rules and the assignment are
implemented once, in `MatchingSystem`.

```python
@define(slots=True)
class CenterDistanceMatchingSystem(MatchingSystem):
    REQUIRES = (POSITION, CLASS_ID)
    MATCHING_NAME = "center_distance"  # target defaults to /matching/center_distance
    DEFAULT_THRESHOLD = 1.0
    # HIGHER_IS_BETTER = False            # a distance, so smaller is better (the default)

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        est = est_view.component(POSITION).values
        gt = gt_view.component(POSITION).values
        return np.linalg.norm(est[:, None, :] - gt[None, :, :], axis=-1)
```

`MatchingSystem.between(estimation, ground_truth, *, target=None, **params)` does the wiring.

### The assignment

A **globally optimal one-to-one assignment** is solved per frame with
`scipy.optimize.linear_sum_assignment`. A greedy matcher would lose viable pairs to an earlier choice.

A pair is never assigned when:

- its score fails the threshold — feasible means `score <= threshold` when
  `HIGHER_IS_BETTER=False`, and `score >= threshold` when it is `True`;
- its score is not finite, as for an estimation with a `NaN` position;
- the classes disagree and `class_agnostic=False`.

Since `linear_sum_assignment` cannot represent forbidden pairs, the implementation fills them with a
cost above any feasible one and rejects them after solving. Modes where `HIGHER_IS_BETTER` negate the
score, turning the minimization into a maximization.

`est_index` and `gt_index` are **row indices within their frame**, and `-1` means "no counterpart".
`matching_score` holds **the mode's natural value** — metres for a distance, a ratio for an IoU —
not the internal cost, and is `NaN` on an unmatched row.

### Implemented modes

| System                            | `REQUIRES`                                | Better | Default | Target                          | Old config                       |
| :-------------------------------- | :---------------------------------------- | :----- | :------ | :------------------------------ | :------------------------------- |
| `CenterDistanceMatchingSystem`    | `POSITION` `CLASS_ID`                     | lower  | 1.0     | `/matching/center_distance`     | `center_distance_thresholds`     |
| `CenterDistanceBEVMatchingSystem` | `POSITION` `CLASS_ID`                     | lower  | 1.0     | `/matching/center_distance_bev` | `center_distance_bev_thresholds` |
| `PlaneDistanceMatchingSystem`     | `POSITION` `QUATERNION` `SIZE` `CLASS_ID` | lower  | 2.0     | `/matching/plane_distance`      | `plane_distance_thresholds`      |
| `IoUBEVMatchingSystem`            | `POSITION` `QUATERNION` `SIZE` `CLASS_ID` | higher | 0.5     | `/matching/iou_bev`             | `iou_2d_thresholds` (3D tasks)   |
| `IoU3DMatchingSystem`             | `POSITION` `QUATERNION` `SIZE` `CLASS_ID` | higher | 0.5     | `/matching/iou_3d`              | `iou_3d_thresholds`              |
| `IoURoiMatchingSystem`            | `ROI` `CLASS_ID`                          | higher | 0.5     | `/matching/iou_roi`             | `iou_2d_thresholds` (2D tasks)   |

Each mode has its own target, so several can run over the same frame and be compared afterwards.

Per-mode notes:

- **Why BEV centre distance is its own class.** A matching mode names the entity a metric later reads,
  so the 3D and BEV distances must not share a target. This differs from the filter layer's `bev`
  flag because a filter's mask is about the entity being filtered, not about the identity of a mode.
- **`PlaneDistanceMatchingSystem`** compares the _nearest faces_ of the two boxes: the root mean
  square of the gaps between their left and right corners on the face closest to the ego. An error
  only on the far side scores 0, so the mode measures what the sensor could actually see, which is a
  different property from centre distance. Positions must be in the frame the distance from the
  origin is measured in — normally `base_link`.
- **Why IoU is two classes.** BEV IoU requires `POSITION` / `QUATERNION` / `SIZE`; image IoU requires
  `ROI`. Different required components mean a different system. A single class inspecting which
  components happen to be present is exactly the hidden branching this design removes.
- **Rotated footprints** cannot be handled by an axis-aligned overlap, so
  `t4perceval.geometry` clips polygons through shapely's vectorized API — otherwise two boxes at
  right angles would be scored as a perfect match.

### Per-class thresholds

`threshold` accepts a number or a class-keyed `Thresholds`. Matching systems already require
`CLASS_ID`, so this needs no extra component.

```python
CenterDistanceMatchingSystem.between(
    "/estimation/objects",
    "/ground_truth/objects",
    threshold=Thresholds(1.0, by_class={"car": 2.0, "pedestrian": 0.5}),
)
```

The original `center_distance_thresholds: [[1.0, 1.0, 1.0, 1.0]]` required knowing the order of
`target_labels` to know what a number applied to. `Thresholds` is keyed by class explicitly, so it
says what it applies to.

The threshold is looked up by the **ground truth's** class. Under `class_agnostic=True` the two sides
can disagree, so one of them has to decide, and the ground truth is the authority. Names are resolved
through `ctx.labels`, and an unknown name raises — silently falling back to the default would look
like the per-class threshold had been applied.

### Geometry

`t4perceval.geometry` holds the vectorized box geometry. Everything works on whole columns and on
_pairs_ of columns: a matcher needs an `(N, M)` score, so the pairwise helpers build that matrix
directly rather than being called per pair from Python.

| Function                                             | Returns                                                |
| :--------------------------------------------------- | :----------------------------------------------------- |
| `bev_corners(position, quaternion, size)`            | `(N, 4, 2)` footprint corners                          |
| `bev_area(size)` / `volume(size)`                    | `(N,)`                                                 |
| `pairwise_bev_iou(...)` / `pairwise_volume_iou(...)` | `(N, M)`                                               |
| `pairwise_roi_iou(est_roi, gt_roi)`                  | `(N, M)`, axis-aligned so no polygon clipping          |
| `pairwise_plane_distance(...)`                       | `(N, M)`                                               |
| `canonical_bev_corners(corners)`                     | corners reordered counter-clockwise about the centroid |

## Coordinate frames

Geometry across two frames is meaningless, and nothing downstream notices: subtracting a `map`
position from a `base_link` one produces a number, not an error, and the metric built on it looks
entirely plausible. The system layer therefore refuses the comparison.

```python
from t4perceval.system.base import require_same_frame, resolve_frame
```

| Helper                       | Behaviour                                                                       |
| :--------------------------- | :------------------------------------------------------------------------------ |
| `require_same_frame(*views)` | returns the single stated frame; raises when two views state different ones     |
| `resolve_frame(*views)`      | returns the first stated frame, without checking -- for a system with one input |

Two seams cover the whole surface: `MatchingSystem` calls it, so all six matchers are checked, and
`MatchJoin.of` calls it, so every geometric metric is checked through the join it already goes
through.

- An **unstated** frame is not a disagreement. A view carries `None` when the entity had nothing in
  range, when the data was logged without a frame, or when it holds a metric rather than geometry.
  Only two _different, stated_ frames raise.
- Row count is deliberately not consulted. A frame with no objects is still recorded in a frame, and
  skipping it would make the output frame flicker across a scene -- which `concat_chunks` then
  refuses to join.
- Only the **temporal** chunk's frame is consulted. A static column's frame need not describe the
  rows it broadcasts over -- a transform states its edge's _parent_ there -- so a static
  `time_offset` logged in `base_link` must not make a comparison in `map` raise.
- `check_frames=False` opts out per system, mirroring the `require_same_frame_id` escape hatch that
  store assembly already offers. Without it, a store assembled with that flag would be un-matchable.

This is a guard, not a fix. Resolving a transform _is_ implemented --
`TransformResolver.lookup(target_frame=..., source_frame=..., at=...)`, see "Transforms" in
[data_model.md](data_model.md) -- but rewriting an entity's rows into another frame is not, because a
passthrough system cannot yet declare the columns it carries. Until it can, bringing the inputs into
one frame is the caller's job and this guard is what stops the alternative from looking plausible.

## Where the remaining systems fit

All of the following sit on the same protocol; only the predicate or cost computation differs. The
metric systems are implemented apart from `HotaSystem`; `PassFailSystem` is not.

### Metrics (`PROVIDES = /metrics/*` descriptors)

| System                                                      | `REQUIRES`                                                 | Source                                        |
| :---------------------------------------------------------- | :--------------------------------------------------------- | :-------------------------------------------- |
| `MeanAveragePrecisionSystem`                                | `MATCH_STATUS` `MATCHING_SCORE` `CONFIDENCE` `CLASS_ID`    | `/matching/<mode>` plus the estimation entity |
| `ClearSystem` (MOTA / MOTP / IDSwitch)                      | `MATCH_STATUS` `EST_INDEX` `GT_INDEX` `INSTANCE_ID`        | a whole `TimeRange`                           |
| `HotaSystem`                                                | as above                                                   | a whole `TimeRange`                           |
| `PathDisplacementSystem` (ADE / FDE)                        | `MATCH_STATUS` `WAYPOINTS` `MODE_CONFIDENCE` `TIME_OFFSET` | a whole `TimeRange`                           |
| `ClassificationSystem` (accuracy / precision / recall / F1) | `MATCH_STATUS` `CLASS_ID`                                  | frame or scene                                |

Cross-frame metrics (CLEAR, HOTA, ADE) take a `TimeRange` as `at` and read the whole scene with a
single `store.range()` call. There is no `List[PerceptionFrameResult]` to walk in Python.

### Pass/fail

| System           | `REQUIRES`                                                    |
| :--------------- | :------------------------------------------------------------ |
| `PassFailSystem` | `MATCH_STATUS` plus the `MASK` of the critical-object filters |

`CriticalObjectFilterConfig` becomes a set of filter systems used to mark critical objects, and
`PerceptionPassFailConfig` becomes `PassFailSystem`'s parameters.

## Dismantling the enum and the config dict

### The enum

| Old `evaluation_task` | New expression                                         |
| :-------------------- | :----------------------------------------------------- |
| `detection`           | `Detections3D`'s components + matching + mAP           |
| `tracking`            | as above + `INSTANCE_ID` + CLEAR / HOTA                |
| `prediction`          | as above + `WAYPOINTS` / `MODE_CONFIDENCE` + ADE / FDE |
| `detection2d`         | `Detections2D`'s components + IoU2D matching + mAP     |
| `tracking2d`          | as above + `INSTANCE_ID` + CLEAR                       |
| `classification2d`    | `Classifications2D`'s components + classification      |

The branches disappear because a system only asks whether the components it needs are present. To
evaluate tracking data as detection, run the tracking entity through a detection pipeline unchanged --
`tracking.has(*Detections3D.required_descriptors())` is True.

### The config dict

One large dict becomes a parameter dataclass per system.

```python
# Before
evaluation_config_dict = {
    "evaluation_task": "detection",
    "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
    "max_x_position": 102.4,
    "max_y_position": 102.4,
    "min_point_numbers": [0, 0, 0, 0],
    "label_prefix": "autoware",
    "merge_similar_labels": False,
    "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
    "iou_3d_thresholds": [0.5],
    ...
}

# After
labels = LabelRegistry.from_names(["car", "bicycle", "pedestrian", "motorbike"])
pipeline = Pipeline([
    FilterByRegionSystem.symmetric("/ground_truth/objects", max_xy=(102.4, 102.4)),
    FilterByNumPointsSystem.on("/ground_truth/objects", min_num_points=0),
    CenterDistanceMatchingSystem.between("/estimation/objects", "/ground_truth/objects",
                                         threshold=1.0),
    IoU3DMatchingSystem.between("/estimation/objects", "/ground_truth/objects",
                                threshold=0.5),
    MeanAveragePrecisionSystem.on("/matching/center_distance"),
])
```

What this buys:

- Parameter validity is checked at construction time by each system's `__attrs_post_init__`, instead
  of arriving at run time as `RuntimeError: Either max x/y position or max/min distance should be
specified`.
- Which filters are in force is readable from the pipeline itself.
- The same mode at a different threshold, or a different mode at the same threshold, can run in
  parallel by changing the target.
