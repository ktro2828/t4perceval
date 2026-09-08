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
from t4_devkit.schema import SensorModality
from t4_devkit.viewer import RerunViewer, ViewerBuilder

from t4perceval import (
    FRAME,
    TIMESTAMP,
    ConfusionMatrix,
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
    Transform3D,
)
from t4perceval.descriptors import CLASS_ID, INSTANCE_ID, MASK, POSITION, QUATERNION, SIZE, VELOCITY
from t4perceval.evaluation import build_evaluation_store
from t4perceval.importer.t4 import SceneSelection, T4Importer, T4Source
from t4perceval.system import (
    ApplyMaskSystem,
    ConfusionMatrixSystem,
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
    stamps = dict(
        zip(
            ground_truth.times(GROUND_TRUTH, FRAME).tolist(),
            ground_truth.times(GROUND_TRUTH, TIMESTAMP).tolist(),
            strict=True,
        )
    )

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
            # Both axes, as an importer would: TIMESTAMP is what the viewer's time
            # slider uses, and what `t4perceval.align` needs if the two sides were ever
            # produced independently.
            at=TimePoint.at(frame=frame, timestamp_ns=stamps[frame]),
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
    max_distance = 50.0
    for path in (GROUND_TRUTH, ESTIMATION):
        near = FilterByDistanceSystem.on(path, max_distance=max_distance)
        narrowing += [near, ApplyMaskSystem.of(path, near.target)]
    Pipeline(narrowing).run(ctx, SCENE)

    gt_mask = setup.store.range(
        f"{GROUND_TRUTH}/filter/distance", timeline=FRAME, time_range=SCENE
    ).component(MASK)
    est_mask = setup.store.range(
        f"{ESTIMATION}/filter/distance", timeline=FRAME, time_range=SCENE
    ).component(MASK)

    print(
        f"within {max_distance} m:\n"
        f"  GT=  {gt_mask.num_selected}/{len(gt_mask)} objects\n"
        f"  EST=  {est_mask.num_selected}/{len(est_mask)} objects"
    )

    # `average_precision_sweep` is a plain function returning a list of systems -- ordinary
    # data you can print, edit or extend, not a config value to branch on.
    thresholds = [0.5, 1.0, 2.0, 4.0]
    systems = average_precision_sweep(
        f"{ESTIMATION}/kept",
        f"{GROUND_TRUTH}/kept",
        thresholds=thresholds,
        heading=True,
    )

    # One confusion matrix per matching run, indexed like the sweep indexes its own targets.
    # The default target is shared, and the store *appends*: four systems writing to one
    # entity would concatenate their cells and quadruple every count.
    for index in range(len(thresholds)):
        systems.append(
            ConfusionMatrixSystem.on(
                f"/matching/center_distance/{index}",
                f"{ESTIMATION}/kept",
                f"{GROUND_TRUTH}/kept",
                target=f"/metrics/confusion_matrix/{index}",
            )
        )

    Pipeline(systems).run(ctx, SCENE)

    print("\nTP/FP/FN:")
    for index, threshold in enumerate(thresholds):
        matches = setup.store.range(
            f"/matching/center_distance/{index}", timeline=FRAME, time_range=SCENE
        ).materialize(MatchResults)
        print(f"  @{threshold} m : TP={matches.num_tp}  FP={matches.num_fp}  FN={matches.num_fn}")

    print("\nmAP/mAPH:")
    for name, path in (("mAP", "/metrics/map"), ("mAPH", "/metrics/maph")):
        values = setup.store.range(path, timeline=FRAME, time_range=SCENE).materialize(MetricValues)
        print(f"    {name:<4}: {values.aggregate:.4f}")

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
        print(f"    {name:<{width}}  {value:.4f}  ({support} objects)")

    # Ground truth down the rows, estimation across the columns. The trailing `background`
    # column holds false negatives and the `background` row false positives. The sweep's
    # matcher is class-aware, so a misclassification is already an FP plus an FN and every
    # off-diagonal class cell stays zero -- pass `class_agnostic=True` to the sweep to see
    # which classes get confused with which.
    for index, threshold in enumerate(thresholds):
        confusion = setup.store.range(
            f"/metrics/confusion_matrix/{index}", timeline=FRAME, time_range=SCENE
        ).materialize(ConfusionMatrix)
        print(f"\nconfusion matrix @{threshold} m (rows: GT, columns: EST):")
        _print_confusion_matrix(ground_truth.labels, confusion)

    # Every intermediate is still queryable -- masks, verdicts, per-threshold AP.
    print(f"\nentities produced: {len(setup.store.entity_paths())}")

    # Freeze the whole thing, provenance included.
    result = setup.into_recording(pipeline=systems)
    print(f"pipeline recorded: {len(result.metadata.pipeline)} systems")


def _print_confusion_matrix(labels: LabelRegistry, confusion: ConfusionMatrix) -> None:
    """Print a dense confusion matrix with class names on both axes."""
    class_ids = [info.class_id for info in labels.classes]
    matrix = confusion.as_matrix(class_ids)
    names = [labels.name(class_id) for class_id in class_ids] + ["background"]
    width = max(len(name) for name in names)
    cell = max(width, len(str(int(matrix.max()))) if matrix.size else 1)

    print(f"  {'':<{width}}  " + "  ".join(f"{name:>{cell}}" for name in names))
    for name, row in zip(names, matrix, strict=True):
        print(f"  {name:<{width}}  " + "  ".join(f"{int(count):>{cell}}" for count in row))


def _viewer_seconds(recording: Recording, path: str) -> dict[int, float]:
    """Map each FRAME of one entity to the seconds the viewer's time slider uses.

    TIMESTAMP when the stream carries it -- an importer always logs both axes -- and the
    frame index otherwise, so a hand-built estimation logged on FRAME alone still lines up
    with the ground truth instead of failing.
    """
    frames = recording.times(path, FRAME)
    stamps = recording.times(path, TIMESTAMP)
    if len(stamps) == len(frames):
        return {int(f): int(t) / 1e9 for f, t in zip(frames, stamps, strict=True)}
    return {int(f): float(f) for f in frames}


def _viewer_frame(frame_id: str | None, role: str) -> str:
    """Where one stream's boxes hang in the viewer's entity tree.

    `render_box3ds` always prefixes `map` and `render_ego` logs the map->base_link
    transform, so rows already in `map` sit directly under it while rows in `base_link`
    have to hang below that transform. The trailing role is what keeps the two streams
    apart -- logged to the same entity path, one would overwrite the other.
    """
    if frame_id in (None, "map"):
        return role
    return f"{frame_id}/{role}"


def _render_calibration(
    viewer: RerunViewer,
    recording: Recording,
    source: T4Source | None = None,
) -> int:
    """Draw each sensor's frame, from the extrinsics the recording already carries.

    The importer records every ``base_link -> <channel>`` calibration as a *static*
    ``Transform3D`` edge, so the poses come out of the recording itself -- no dataset
    handle needed, and it works just as well for a recording restored from disk.

    Intrinsics are the exception: they are not part of t4perceval's data model, because
    nothing in the evaluation pipeline consumes them. Pass the importer's ``source`` to
    add camera pinholes on top; without it every sensor is drawn as a bare set of axes.
    """
    calibrations = {} if source is None else source.extrinsics()
    rendered = 0

    for edge in FrameGraph.of(recording).edges:
        # Only the fixed ego->sensor edges. The temporal map->base_link one is the ego
        # pose, already rendered, and re-logging it here as static would pin the vehicle
        # in place for the whole scene.
        if not edge.is_static or edge.parent != "base_link":
            continue

        pose = Transform3D.from_chunk(recording.static_chunks(edge.entity_path)[0])
        is_camera = source is not None and source.is_camera(edge.child)

        viewer.render_calibration(
            channel=edge.child,
            modality=SensorModality.CAMERA if is_camera else SensorModality.LIDAR,
            translation=pose.translation.value,
            rotation=np.roll(pose.rotation.value, 1),
            camera_intrinsic=(calibrations[edge.child].camera_intrinsic if is_camera else None),
        )
        rendered += 1

    return rendered


def _render_boxes(viewer: RerunViewer, recording: Recording, path: str, role: str) -> int:
    """Log one stream's 3D boxes, one frame at a time."""
    seconds = _viewer_seconds(recording, path)
    rendered = 0

    for frame in recording.times(path, FRAME).tolist():
        view = recording.latest_at(path, timeline=FRAME, at=int(frame))
        if not len(view):
            continue

        instances = view.component(INSTANCE_ID)
        velocity = view.component(VELOCITY)

        viewer.render_box3ds(
            seconds[int(frame)],
            _viewer_frame(view.frame_id, role),
            centers=view.component(POSITION).values,
            # t4perceval stores quaternions xyzw (SciPy's convention); the viewer hands
            # rotations to pyquaternion, which reads wxyz. Rolling by one is the whole
            # conversion -- get it wrong and every box is silently mis-oriented.
            rotations=np.roll(view.component(QUATERNION).values, 1, axis=1),
            # Both sides agree on (width, length, height), so `size` passes straight
            # through.
            sizes=view.component(SIZE).values,
            class_ids=[int(value) for value in view.component(CLASS_ID).values],
            velocities=None if velocity is None else velocity.values,
            uuids=(
                None if instances is None else list(recording.instances.decode(instances.values))
            ),
        )
        rendered += len(view)

    return rendered


def visualize(
    ground_truth: Recording,
    estimation: Recording,
    *,
    source: T4Source | None = None,
    save_dir: str | None = None,
) -> None:
    """Render both streams into Rerun through t4-devkit's viewer.

    Spawns the Rerun viewer, or writes ``<save_dir>/t4perceval.rrd`` when ``save_dir`` is
    given -- which is what you want on a headless machine.

    The two streams land on separate entity paths, so they can be toggled independently
    in the viewer:

        map/base_link                      the ego pose, per frame
        map/base_link/<channel>            each sensor's calibration, static
        map/base_link/ground_truth/box
        map/base_link/estimation/box

    ``source`` is optional and only adds camera pinholes -- see :func:`_render_calibration`.
    """
    # `class_id` columns are integers; this is the mapping that gives them names in the
    # viewer's legend. It is the same registry both recordings were encoded with.
    label2id = {info.name: info.class_id for info in ground_truth.labels.classes}
    viewer = (
        ViewerBuilder()
        .with_labels(label2id)
        .with_spatial3d()
        .build("t4perceval", save_dir=save_dir)
    )

    # The ego pose is what makes base_link-relative boxes land in the right place: it is
    # logged as the map->base_link transform every box entity below it inherits.
    if "base_link" in FrameGraph.of(ground_truth).frames():
        resolver = TransformResolver.of(ground_truth, timeline=FRAME)
        seconds = _viewer_seconds(ground_truth, GROUND_TRUTH)
        for frame, at in sorted(seconds.items()):
            ego = resolver.lookup(target_frame="map", source_frame="base_link", at=frame)
            viewer.render_ego(
                seconds=at,
                translation=ego.translation.value,
                rotation=np.roll(ego.rotation.value, 1),
            )
        print(f"ego poses : {len(seconds)}")

    print(f"sensors   : {_render_calibration(viewer, ground_truth, source)}")
    print(f"gt boxes  : {_render_boxes(viewer, ground_truth, GROUND_TRUTH, 'ground_truth')}")
    print(f"est boxes : {_render_boxes(viewer, estimation, ESTIMATION, 'estimation')}")
    if save_dir is not None:
        print(f"written   : {save_dir}/t4perceval.rrd  (open with `rerun <file>`)")


def main(data_root: str, *, viewer: str | None = None) -> None:
    print(f"dataset: {data_root}")
    importer = T4Importer.open(data_root)

    labels = inspect(importer)
    ground_truth = import_scene(importer, labels)
    coordinate_frames(ground_truth)

    estimation = fake_estimation(ground_truth)
    evaluate(ground_truth, estimation)

    if viewer is not None:
        banner("5. Visualize")
        visualize(
            ground_truth,
            estimation,
            source=importer.source,
            save_dir=None if viewer == "spawn" else viewer,
        )


if __name__ == "__main__":
    if not sys.argv[1:]:
        raise SystemExit(
            "usage: evaluate_t4.py <data_root> [--visualize [SAVE_DIR]]\n"
            "  --visualize            spawn the Rerun viewer\n"
            "  --visualize SAVE_DIR   write SAVE_DIR/t4perceval.rrd instead (headless)"
        )

    argv = sys.argv[1:]
    show = "--visualize" in argv
    where = None
    if show:
        index = argv.index("--visualize")
        rest = argv[index + 1 :]
        where = rest[0] if rest else "spawn"
        argv = argv[:index]

    main(argv[0], viewer=where)
