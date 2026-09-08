#!/usr/bin/env python
"""End-to-end t4perceval usage against a real T4 dataset.

Run it with::

    uv run python example.py
    uv run python example.py /path/to/another/t4dataset

Needs the ``t4`` extra::

    uv sync --group dev            # or: pip install 't4perceval[t4]'

The four sections mirror the documentation:

1. Inspect      -- what the dataset holds                docs/user-guide/dataset-importers.md
2. Import       -- what lands in a Recording             docs/concepts/store.md
3. Frames       -- the coordinate-frame graph            docs/concepts/coordinate-system.md
4. Evaluate     -- filter, match, metrics                docs/evaluation/detection-3d.md
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from t4perceval import (
    FRAME,
    TIMESTAMP,
    LabelRegistry,
    MatchResults,
    MetricValues,
    Recording,
    RecordingMetadata,
    SourceInfo,
    Store,
    TimePoint,
    TimeRange,
    Trackings3D,
)
from t4perceval.descriptors import MASK
from t4perceval.evaluation import build_evaluation_store
from t4perceval.importer.t4 import SceneSelection, T4Importer
from t4perceval.system import (
    ApplyMaskSystem,
    FilterByDistanceSystem,
    Pipeline,
    average_precision_sweep,
)
from t4perceval.transform import FrameGraph, TransformResolver

if TYPE_CHECKING:
    from t4perceval.system.base import SystemContext


#: Entity paths the evaluation reads. The T4 importer writes the first one.
GROUND_TRUTH = "/ground_truth/objects"
ESTIMATION = "/estimation/objects"

SCENE = TimeRange.everything()


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------------------
# 1. Inspect the dataset
# ---------------------------------------------------------------------------------------


def inspect(importer: T4Importer) -> LabelRegistry:
    """Report what the dataset holds, and return the registry every source must share."""
    banner("1. Inspect")

    # The registry is an *input* to the import, never derived per source. Class ids are
    # assigned in first-seen order, so two sources that each build their own registry are
    # both valid and silently incompatible -- the disagreement surfaces as plausible
    # numbers, not as an error. Build it once, hand this object to everything.
    labels = importer.label_registry()

    print(f"scenes    : {len(importer.scene_tokens())}")
    print(f"channels  : {len(importer.source.channels())}")
    print(f"classes   : {len(labels.names)}")
    for name in labels.names[:6]:
        print(f"            {labels.class_id(name):>3}  {name}")
    if len(labels.names) > 6:
        print(f"            ...  ({len(labels.names) - 6} more)")

    # Datasets differ in how fine-grained their categories are. A nuScenes-style release
    # names them `vehicle.car`, `pedestrian.adult`, ...; `merged()` collapses groups onto
    # one class and keeps the originals working as aliases, so the raw column still
    # encodes. Shown rather than applied -- which grouping you want is an evaluation
    # decision, not a property of the dataset.
    groups: dict[str, list[str]] = {}
    for name in labels.names:
        head, _, rest = name.partition(".")
        if rest:
            groups.setdefault(head, []).append(name)
    if groups:
        print(f"\nhierarchical category names -- {', '.join(sorted(groups))}")
        print("to evaluate them coarsely:")
        print(
            f"    labels = labels.merged({{{', '.join(f'{k!r}: [...]' for k in sorted(groups))}}})"
        )
        print(f"    # -> {len(labels.merged(groups).names)} classes instead of {len(labels.names)}")
    return labels


# ---------------------------------------------------------------------------------------
# 2. Import a scene
# ---------------------------------------------------------------------------------------


def import_scene(importer: T4Importer, labels: LabelRegistry) -> Recording:
    """Import the first scene into a read-only Recording."""
    banner("2. Import")

    # `channel_3d` defaults to LIDAR_CONCAT -- the concatenated cloud a T4 release
    # annotates against. Name a different one here if your dataset publishes its lidar
    # under another name; an unknown channel raises rather than falling back.
    recording = importer.import_scene(
        labels=labels,
        selection=SceneSelection(scene=0),
    )

    frames = recording.times(GROUND_TRUTH, FRAME)
    stamps = recording.times(GROUND_TRUTH, TIMESTAMP)
    objects = recording.range(GROUND_TRUTH, timeline=FRAME, time_range=SCENE)

    print(f"frames    : {len(frames)}  ({frames[0]}..{frames[-1]})")
    print(f"duration  : {(stamps[-1] - stamps[0]) / 1e9:.1f} s")
    print(f"objects   : {len(objects)} rows over the whole scene")
    print(f"columns   : {', '.join(sorted(d.component for d in objects.descriptors))}")
    print(f"frame_id  : {recording.metadata.frame_id}")
    print(f"provenance: {dict(recording.metadata.sources[0].extra)}")

    entities = [str(path) for path in recording.entity_paths()]
    print(f"entities  : {len(entities)}")
    print(f"            {GROUND_TRUTH}")
    print(f"            {len(entities) - 1}x /tf/<frame>  (the coordinate-frame graph)")
    return recording


# ---------------------------------------------------------------------------------------
# 3. Coordinate frames
# ---------------------------------------------------------------------------------------


def coordinate_frames(recording: Recording) -> None:
    """Walk the frame graph the importer recorded alongside the objects."""
    banner("3. Coordinate frames")

    # Edges are discovered by reading chunks, not by parsing entity paths: an ego pose is
    # a temporal Transform3D row, a sensor extrinsic a static one, and both compose in one
    # graph as  T_map_sensor(t) = T_map_base_link(t) @ T_base_link_sensor.
    graph = FrameGraph.of(recording)
    static = [edge for edge in graph.edges if edge.is_static]
    temporal = [edge for edge in graph.edges if not edge.is_static]

    print(f"frames    : {len(graph.frames())}")
    print(f"edges     : {len(static)} static (extrinsics), {len(temporal)} temporal (ego pose)")

    resolver = TransformResolver.of(recording, timeline=FRAME)
    frames = recording.times(GROUND_TRUTH, FRAME)

    start = resolver.lookup(target_frame="map", source_frame="base_link", at=int(frames[0]))
    end = resolver.lookup(target_frame="map", source_frame="base_link", at=int(frames[-1]))
    travelled = float(np.linalg.norm(end.translation.value - start.translation.value))
    print(f"ego moved : {travelled:.1f} m between frame {frames[0]} and frame {frames[-1]}")

    for channel in ("LIDAR_CONCAT", "CAM_FRONT"):
        if channel in graph.frames():
            pose = resolver.lookup(target_frame="base_link", source_frame=channel, at=0)
            print(f"{channel:<12}: {pose.translation.value.round(3)} in base_link")


# ---------------------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------------------


def fake_estimation(ground_truth: Recording, *, offset: float = 0.3) -> Recording:
    """Nudge the ground truth into a stand-in 'estimation', to exercise the pipeline.

    Replace this with your model's output. It exists so the section below runs end to end
    on any dataset: a perfect detector shifted by ``offset`` metres should score close to
    1.0, which is the cheapest way to prove an evaluation setup is wired up correctly.
    """
    store = Store()
    rng = np.random.default_rng(0)

    for frame in ground_truth.times(GROUND_TRUTH, FRAME).tolist():
        view = ground_truth.latest_at(GROUND_TRUTH, timeline=FRAME, at=frame)
        if not len(view):
            continue
        objects = view.materialize(Trackings3D)
        store.log(
            ESTIMATION,
            Trackings3D(
                position=objects.position.values
                + rng.normal(0.0, offset, objects.position.values.shape),
                quaternion=objects.quaternion.values,
                size=objects.size.values,
                class_id=objects.class_id.values,
                confidence=rng.uniform(0.5, 1.0, len(objects)),
                instance_id=objects.instance_id.values,
            ),
            at=TimePoint.at(frame=frame),
            frame_id=view.frame_id,
        )

    return Recording.of(
        store,
        labels=ground_truth.labels,
        instances=ground_truth.instances,
        metadata=RecordingMetadata(
            sources=(SourceInfo(kind="synthetic", uri="fake_estimation", entity_path=ESTIMATION),),
            frame_id=ground_truth.metadata.frame_id,
        ),
    )


def evaluate(ground_truth: Recording, estimation: Recording) -> None:
    """Filter, match and score -- the shape every detection evaluation takes."""
    banner("4. Evaluate")

    if not len(ground_truth.range(GROUND_TRUTH, timeline=FRAME, time_range=SCENE)):
        print("This dataset carries no 3D box annotations, so there is nothing to score.")
        print("`sample_annotation.json` is empty -- it is a lidarseg dataset, labelling")
        print("points rather than objects. Every metric below would be NaN over 0 rows.")
        print()
        print("The pipeline code is left here as a template; point it at a dataset whose")
        print("`sample_annotation.json` is populated and it produces real numbers.")
        print("For point-wise labels see docs/evaluation/segmentation-3d.md -- the")
        print("archetypes exist, the metric systems do not yet.")
        return

    # A Recording is read-only and `Pipeline.run` writes results back into the store it
    # reads from, so the entities an evaluation needs are materialized into a fresh,
    # writable store first. Only what is named moves.
    setup = build_evaluation_store(ground_truth, estimation)
    ctx: SystemContext = setup.context()

    # Narrow both sides identically, and *materialize*: recall divides by the ground-truth
    # count, so the denominator has to be the filtered set rather than a mask over it.
    narrowing = []
    for path in (GROUND_TRUTH, ESTIMATION):
        near = FilterByDistanceSystem.on(path, max_distance=50.0)
        narrowing += [near, ApplyMaskSystem.of(path, near.target)]
    Pipeline(narrowing).run(ctx, SCENE)

    kept = setup.store.range(f"{GROUND_TRUTH}/filter/distance", timeline=FRAME, time_range=SCENE)
    mask = kept.component(MASK)
    print(f"within 50 m: {mask.num_selected} of {len(mask)} ground-truth objects")

    # `average_precision_sweep` is a plain function returning a list of systems -- ordinary
    # data you can print, edit or extend, not a config value to branch on.
    systems = average_precision_sweep(
        f"{ESTIMATION}/kept",
        f"{GROUND_TRUTH}/kept",
        thresholds=[0.5, 1.0, 2.0, 4.0],
        heading=True,
    )
    Pipeline(systems).run(ctx, SCENE)

    matches = setup.store.range(
        "/matching/center_distance/1", timeline=FRAME, time_range=SCENE
    ).materialize(MatchResults)
    print(f"at 1.0 m   : TP={matches.num_tp}  FP={matches.num_fp}  FN={matches.num_fn}")

    for path in ("/metrics/map", "/metrics/maph"):
        values = setup.store.range(path, timeline=FRAME, time_range=SCENE).materialize(MetricValues)
        print(f"{path:<12}: {values.aggregate:.4f}")

    per_class = setup.store.range("/metrics/map", timeline=FRAME, time_range=SCENE).materialize(
        MetricValues
    )
    scored = [
        (ground_truth.labels.name(int(class_id)), value, support)
        for class_id, value, support in zip(
            per_class.class_id.values,
            per_class.value.values,
            per_class.support.values,
            strict=True,
        )
        if class_id >= 0 and support
    ]
    width = max((len(name) for name, _, _ in scored), default=0)
    print("\nper class (support > 0):")
    for name, value, support in sorted(scored, key=lambda row: -row[2]):
        print(f"  {name:<{width}}  {value:.4f}  ({support} objects)")

    # Every intermediate is still queryable -- masks, verdicts, per-threshold AP.
    print(f"\nentities produced: {len(setup.store.entity_paths())}")

    # Freeze the whole thing, provenance included.
    result = setup.into_recording(pipeline=systems)
    print(f"pipeline recorded: {len(result.metadata.pipeline)} systems")


def main(data_root: str) -> None:
    print(f"dataset: {data_root}")
    importer = T4Importer.open(data_root)

    labels = inspect(importer)
    ground_truth = import_scene(importer, labels)
    coordinate_frames(ground_truth)
    evaluate(ground_truth, fake_estimation(ground_truth))


if __name__ == "__main__":
    if not sys.argv[1:]:
        raise SystemExit("no data root specified")
    main(sys.argv[1])
