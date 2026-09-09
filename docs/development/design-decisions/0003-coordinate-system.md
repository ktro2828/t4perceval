# 0003: Coordinate frames as recorded data

## Status

Accepted.

## Context

Perception data is expressed in a frame -- `base_link`, `map`, a camera channel -- and evaluation
compares two sets of objects that may not be in the same one. Comparing them anyway produces a
number, not an error: subtracting a `map` position from a `base_link` one is arithmetically fine, and
the metric built on it looks plausible.

The original package held frames as state outside the data: an ego pose was a field on a frame
object, a sensor extrinsic came from configuration, and whether two things were comparable was
something the caller was expected to know.

## Decision

Three parts:

1. **A chunk states the frame its rows are expressed in**, as `frame_id`.
2. **A transform is recorded data like everything else** -- one `Transform3D` row per edge of the
   frame graph, split the way ROS splits a `TransformStamped`: the chunk's `frame_id` is the
   **parent**, `child_frame_id` is a column. A static extrinsic and a temporal ego pose are the
   same archetype; `static` says only that the value does not depend on a timeline.
3. **The system layer refuses to compare geometry across two different, stated frames**, rather than
   converting silently or producing a plausible number.

Frame names are **not** entity paths, and edges are discovered by reading chunks rather than by
parsing paths.

## Rationale

Making a transform a row rather than state means it persists, it is queryable, it has a time axis
when it needs one, and it travels with the data it describes. `TransformResolver` then composes
static and temporal edges in one graph:

```text
T_map_lidar(t) = T_map_base_link(t) @ T_base_link_lidar
```

Splitting parent onto the chunk is not arbitrary: `frame_id` already means "the frame these rows are
expressed in", and that is exactly true of a transform's own row. Matching ROS's split also means a
bag's `/tf` maps across with no reinterpretation.

Keeping the frame **out of the entity path** matters because the two are different kinds of name. A
path says where data is filed; a frame names a node of a graph. Conflating them would force frame
names to be path-safe -- ruling out ROS's `/robot1/base_link` -- and would stop a graph being
re-filed without being renamed.

The guard is the part that earns its keep. This domain's failure mode is a _plausible number_, so
the check has to be somewhere no caller can forget it: every matcher goes through `MatchingSystem`
and every geometric metric through `MatchJoin`, so one check covers all of them.

Two rules keep it from firing spuriously:

- **Only two _different, stated_ frames raise.** An unstated frame is not a disagreement -- the
  entity may have had nothing in range, or may hold a metric rather than geometry.
- **An empty frame is still consulted.** An importer logs an empty frame with its `frame_id` intact;
  ignoring that would make a system's output frame flicker across a scene, which `concat_chunks`
  then refuses to join.

## Alternatives considered

**Convert automatically when frames differ.** Tempting, and wrong at this stage: it needs a
resolver, a timeline, a lookup policy and a decision about which side to move, none of which the
matcher has any basis to choose. Guessing produces exactly the silent-wrong the guard exists to
prevent. `TransformEntitySystem` is that decision made explicit -- a stage the user puts in the
pipeline, naming the target frame and, when the frame graph is not in the store, the resolver.

**Require one global frame.** Simple, and it makes 2D evaluation impossible: a camera ROI is only
meaningful in its own image.

**Keep transforms outside the store**, in a resolver built from configuration. Then a saved
evaluation cannot be reinterpreted without the configuration that produced it, and there is no
answer to "what was the ego pose at frame 40?" from the data alone.

**Put the frame in the entity path** -- `/base_link/objects`. Reads nicely, and makes a frame name
path-shaped, breaks on ROS-namespaced names, and conflates filing with geometry.

**Both frames as columns.** Symmetric, and duplicates what `frame_id` already says, so the two can
disagree.

## Consequences

**Good.**

- A frame graph is discovered from the data, so an importer records it once and everything
  downstream can use it.
- Static and temporal edges compose without the caller knowing which is which.
- A cross-frame comparison fails with a message naming both entities and both frames, instead of
  returning a number.
- A frame name may contain a `/`, and where a transform is filed is a filing decision.

**Costs.**

- A cross-frame evaluation needs an explicit `TransformEntitySystem` stage, and the user has to know
  which side to move and which frame graph answers for it -- the pipeline cannot guess.
- `Transform3D` is the only archetype with **mono** components, which is a special case in the model
  even though storage stays columnar underneath.
- Bag transforms live on `TIMESTAMP` only, because a `/tf` sample between two object messages has no
  frame index, so a caller has to know which axis to resolve on.
- `check_frames=False` exists, and an escape hatch is a thing that can be reached for too readily.

## Where it is written down

[Coordinate system](../../concepts/coordinate-system.md),
[Working with transforms](../../user-guide/transforms.md),
[Transform3D](../../archetypes/transform-3d.md).
