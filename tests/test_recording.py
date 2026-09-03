"""The recording wrapper, class-id reconciliation, and materialization."""

from __future__ import annotations

import json

import numpy as np
import pytest
from attrs import evolve

from t4perceval import (
    FRAME,
    Detections3D,
    InstanceRegistry,
    LabelRegistry,
    Store,
    TimePoint,
    TimeRange,
    Trackings3D,
)
from t4perceval.descriptors import CLASS_ID, POSITION
from t4perceval.evaluation import SourceSpec, build_evaluation_store, build_evaluation_store_from
from t4perceval.reconcile import class_id_lut, remap_class_ids
from t4perceval.recording import Recording, RecordingMetadata, SourceInfo

EVERYTHING = TimeRange.everything()


def detections(labels: LabelRegistry, names: list[str], *, x: float = 0.0) -> Detections3D:
    count = len(names)
    return Detections3D(
        np.tile([x, 0.0, 0.0], (count, 1)),
        np.tile([0.0, 0.0, 0.0, 1.0], (count, 1)),
        np.ones((count, 3)),
        labels.encode(names),
        np.ones(count),
    )


def recording(
    labels: LabelRegistry,
    *,
    path: str = "/ground_truth/objects",
    names: list[str] | None = None,
    frame_id: str = "base_link",
    instances: InstanceRegistry | None = None,
) -> Recording:
    store = Store()
    store.log(
        path,
        detections(labels, names if names is not None else ["car"]),
        at=TimePoint.at(frame=0, timestamp_ns=1_000),
        frame_id=frame_id,
    )
    return Recording.of(store, labels=labels, instances=instances)


def tracked_recording(
    labels: LabelRegistry,
    *,
    path: str = "/ground_truth/objects",
    instances: InstanceRegistry | None = None,
) -> Recording:
    """A recording whose rows carry instance ids."""
    registry = instances if instances is not None else InstanceRegistry()
    store = Store()
    store.log(
        path,
        Trackings3D(
            np.zeros((1, 3)),
            np.tile([0.0, 0.0, 0.0, 1.0], (1, 1)),
            np.ones((1, 3)),
            labels.encode(["car"]),
            np.ones(1),
            instance_id=registry.encode(["a"]),
        ),
        at=TimePoint.at(frame=0, timestamp_ns=1_000),
        frame_id="base_link",
    )
    return Recording.of(store, labels=labels, instances=registry)


class TestRecording:
    def test_exposes_no_writers(self) -> None:
        held = recording(LabelRegistry.from_names(["car"]))

        assert not any(hasattr(held, name) for name in ("send_chunk", "log", "log_static"))

    def test_delegates_queries(self) -> None:
        held = recording(LabelRegistry.from_names(["car"]))

        assert [str(path) for path in held.entity_paths()] == ["/ground_truth/objects"]
        assert len(held.range("/ground_truth/objects", timeline=FRAME, time_range=EVERYTHING)) == 1

    def test_stamps_the_registry_fingerprint(self) -> None:
        labels = LabelRegistry.from_names(["car"])
        held = recording(labels)

        assert held.metadata.labels_fingerprint == labels.fingerprint()

    def test_agreement_follows_the_fingerprint(self) -> None:
        first = recording(LabelRegistry.from_names(["car", "pedestrian"]))
        same = recording(LabelRegistry.from_names(["car", "pedestrian"]))
        reordered = recording(LabelRegistry.from_names(["pedestrian", "car"]))

        assert first.agrees_with(same)
        assert not first.agrees_with(reordered)


class TestMetadata:
    def test_round_trips_through_json(self) -> None:
        metadata = RecordingMetadata(
            t4perceval_version="0.1.0",
            created_at_ns=123,
            sources=(SourceInfo("t4", "/data", scene="s1", extra={"channel_3d": "LIDAR_TOP"}),),
            labels_fingerprint="abc",
            pipeline=("CenterDistanceMatchingSystem",),
            frame_id="base_link",
            tags={"run": "nightly"},
        )

        assert RecordingMetadata.from_json(json.loads(json.dumps(metadata.to_json()))) == metadata

    def test_with_metadata_returns_a_copy(self) -> None:
        held = recording(LabelRegistry.from_names(["car"]))
        tagged = held.with_metadata(notes="checked")

        assert tagged.metadata.notes == "checked"
        assert held.metadata.notes == ""


class TestReconcile:
    def test_maps_names_onto_the_target_ids(self) -> None:
        source = LabelRegistry.from_names(["car", "pedestrian", "bicycle"])
        target = LabelRegistry.from_names(["pedestrian", "bicycle", "car"])
        lut = class_id_lut(source, target)

        assert [int(lut[source.class_id(name) + 2]) for name in source.names] == [
            target.class_id(name) for name in source.names
        ]

    def test_sentinels_map_to_themselves(self) -> None:
        lut = class_id_lut(LabelRegistry.from_names(["car"]), LabelRegistry.from_names(["car"]))

        assert (int(lut[-1 + 2]), int(lut[-2 + 2])) == (-1, -2)

    def test_a_name_the_target_lacks_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="does not know"):
            class_id_lut(
                LabelRegistry.from_names(["car", "bus"]),
                LabelRegistry.from_names(["car"]),
            )

    def test_a_missing_name_can_become_unknown(self) -> None:
        lut = class_id_lut(
            LabelRegistry.from_names(["car", "bus"]),
            LabelRegistry.from_names(["car"]),
            on_missing="unknown",
        )

        assert int(lut[1 + 2]) == -1

    def test_remapping_touches_only_the_class_column(self) -> None:
        source = LabelRegistry.from_names(["car", "pedestrian"])
        target = LabelRegistry.from_names(["pedestrian", "car"])
        chunk = detections(source, ["car", "pedestrian"]).to_chunk("/o", at=TimePoint.at(frame=0))

        remapped = remap_class_ids(chunk, class_id_lut(source, target))

        assert remapped.columns[CLASS_ID].values.tolist() == [1, 0]
        assert remapped.columns[POSITION] is chunk.columns[POSITION]


class TestMaterialization:
    def test_moves_only_the_named_entities(self) -> None:
        labels = LabelRegistry.from_names(["car"])
        store = Store()
        for path in ("/ground_truth/objects", "/sensors/lidar"):
            store.log(
                path,
                detections(labels, ["car"]),
                at=TimePoint.at(frame=0),
                frame_id="base_link",
            )
        held = Recording.of(store, labels=labels)

        setup = build_evaluation_store_from(
            (SourceSpec.of(held, "/ground_truth/objects"),),
        )

        assert [str(path) for path in setup.store.entity_paths()] == ["/ground_truth/objects"]

    def test_repathing_shares_the_columns(self) -> None:
        labels = LabelRegistry.from_names(["car"])
        held = recording(labels)

        setup = build_evaluation_store_from(
            (SourceSpec.of(held, "/ground_truth/objects", "/estimation/objects"),),
        )
        moved = setup.store.chunks("/estimation/objects")[0]
        original = held.chunks("/ground_truth/objects")[0]

        assert moved.columns[POSITION] is original.columns[POSITION]

    def test_static_columns_come_across(self) -> None:
        # Static data lives in the store, not in a chunk, and takes precedence over a
        # temporal column with the same descriptor -- so losing it changes results
        # silently rather than raising.
        labels = LabelRegistry.from_names(["car"])
        store = Store()
        store.log(
            "/ground_truth/objects",
            detections(labels, ["car"]),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        store.log_static("/ground_truth/objects", detections(labels, ["car"], x=9.0))
        held = Recording.of(store, labels=labels)

        setup = build_evaluation_store_from((SourceSpec.of(held, "/ground_truth/objects"),))

        assert setup.store.static("/ground_truth/objects")

    def test_log_order_is_preserved(self) -> None:
        labels = LabelRegistry.from_names(["car"])
        store = Store()
        for x in (1.0, 2.0):
            store.log(
                "/ground_truth/objects",
                detections(labels, ["car"], x=x),
                at=TimePoint.at(frame=0),
                frame_id="base_link",
            )
        held = Recording.of(store, labels=labels)

        setup = build_evaluation_store_from((SourceSpec.of(held, "/ground_truth/objects"),))
        view = setup.store.latest_at("/ground_truth/objects", timeline=FRAME, at=0)

        # `latest_at` prefers the most recently logged chunk on a tie, so a reordered
        # transfer would quietly return the wrong frame.
        assert view.component(POSITION).values[0, 0] == pytest.approx(2.0)

    def test_disagreeing_registries_are_rejected(self) -> None:
        first = recording(LabelRegistry.from_names(["car", "pedestrian"]))
        second = recording(
            LabelRegistry.from_names(["pedestrian", "car"]),
            path="/estimation/objects",
        )

        with pytest.raises(ValueError, match="disagree about class ids"):
            build_evaluation_store(first, second)

    def test_disagreeing_registries_can_be_reconciled(self) -> None:
        first = recording(LabelRegistry.from_names(["car", "pedestrian"]), names=["pedestrian"])
        second = recording(
            LabelRegistry.from_names(["pedestrian", "car"]),
            path="/estimation/objects",
            names=["pedestrian"],
        )

        setup = build_evaluation_store(first, second, reconcile=True)
        view = setup.store.latest_at("/estimation/objects", timeline=FRAME, at=0)

        # Both rows are pedestrians, so both must carry the reference registry's id for it.
        assert view.component(CLASS_ID).values.tolist() == [first.labels.class_id("pedestrian")]

    def test_mismatched_coordinate_frames_are_rejected(self) -> None:
        labels = LabelRegistry.from_names(["car"])
        instances = InstanceRegistry()
        first = recording(labels, frame_id="base_link", instances=instances)
        second = recording(
            labels,
            path="/estimation/objects",
            frame_id="map",
            instances=instances,
        )

        # Nothing downstream would complain: a matcher takes whichever frame it sees
        # first, so this yields meaningless distances and a plausible near-zero score.
        with pytest.raises(ValueError, match="different coordinate frames"):
            build_evaluation_store(first, second)

    def test_mismatched_frames_can_be_opted_into(self) -> None:
        labels = LabelRegistry.from_names(["car"])
        instances = InstanceRegistry()
        first = recording(labels, frame_id="base_link", instances=instances)
        second = recording(
            labels,
            path="/estimation/objects",
            frame_id="map",
            instances=instances,
        )

        setup = build_evaluation_store(first, second, require_same_frame_id=False)

        assert len(setup.store.entity_paths()) == 2

    def test_separate_instance_registries_are_rejected_when_ids_are_moved(self) -> None:
        labels = LabelRegistry.from_names(["car"])
        first = tracked_recording(labels, instances=InstanceRegistry())
        second = tracked_recording(
            labels,
            path="/estimation/objects",
            instances=InstanceRegistry(),
        )

        with pytest.raises(ValueError, match="different InstanceRegistry"):
            build_evaluation_store(first, second)

    def test_separate_instance_registries_are_fine_without_ids(self) -> None:
        # Detections carry no instance ids, so there is nothing to renumber and nothing
        # to conflict -- demanding a shared registry here would reject a well-defined
        # evaluation.
        labels = LabelRegistry.from_names(["car"])
        first = recording(labels, instances=InstanceRegistry())
        second = recording(labels, path="/estimation/objects", instances=InstanceRegistry())

        setup = build_evaluation_store(first, second)

        assert len(setup.store.entity_paths()) == 2

    def test_no_sources_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one source"):
            build_evaluation_store_from(())


class TestSetup:
    def test_context_carries_the_registries(self) -> None:
        labels = LabelRegistry.from_names(["car"])
        held = recording(labels)
        setup = build_evaluation_store_from((SourceSpec.of(held, "/ground_truth/objects"),))

        context = setup.context()

        assert context.labels is labels
        assert context.store is setup.store

    def test_into_recording_records_the_pipeline(self) -> None:
        labels = LabelRegistry.from_names(["car"])
        held = recording(labels)
        setup = build_evaluation_store_from((SourceSpec.of(held, "/ground_truth/objects"),))

        frozen = setup.into_recording(pipeline=[evolve(SourceSpec.of(held, "/o"))])

        assert frozen.metadata.pipeline == ("SourceSpec",)
        assert isinstance(frozen, Recording)
