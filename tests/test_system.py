from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pytest

from t4perceval import FRAME, LabelRegistry, Store, TimeRange
from t4perceval.archetype import BatchMatchResult
from t4perceval.descriptors import CLASS_ID, MASK, POSITION
from t4perceval.io import read_parquet, write_parquet
from t4perceval.system import (
    CenterDistanceMatchingSystem,
    EntitySystem,
    FilterByDistanceSystem,
    Pipeline,
    System,
    SystemContext,
    require,
    resolve_times,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.descriptor import ComponentDescriptor


def context(store: Store, labels: LabelRegistry | None = None) -> SystemContext:
    return SystemContext(store, FRAME, labels=labels)


class TestProtocol:
    def test_the_reference_systems_satisfy_the_protocol(self) -> None:
        assert isinstance(FilterByDistanceSystem.on("/x"), System)
        assert isinstance(CenterDistanceMatchingSystem.between("/a", "/b"), System)

    def test_a_system_declares_what_it_reads_and_writes(self) -> None:
        system = FilterByDistanceSystem.on("/estimation/objects", max_distance=40.0)

        assert system.REQUIRES == (POSITION,)
        assert system.PROVIDES == (MASK,)
        assert [str(path) for path in system.sources] == ["/estimation/objects"]
        assert str(system.target) == "/estimation/objects/filter/distance"

    def test_require_names_every_missing_component(self, scene_store: Store) -> None:
        view = scene_store.latest_at("/estimation/objects", timeline=FRAME, at=0)

        from t4perceval.descriptors import INSTANCE_ID, WAYPOINTS

        with pytest.raises(ValueError, match="missing required component.*instance_id"):
            require(view, POSITION, INSTANCE_ID, WAYPOINTS)

    def test_resolve_times_honours_the_query(self, scene_store: Store) -> None:
        ctx = context(scene_store)

        assert resolve_times(ctx, "/estimation/objects", TimeRange.everything()) == [0, 1]
        assert resolve_times(ctx, "/estimation/objects", 1) == [1]
        assert resolve_times(ctx, "/estimation/objects", 9) == []


class TestPipeline:
    def test_runs_every_system_and_stores_the_results(self, scene_store: Store) -> None:
        pipeline = Pipeline(
            [
                FilterByDistanceSystem.on("/estimation/objects", max_distance=40.0),
                CenterDistanceMatchingSystem.between(
                    "/estimation/objects",
                    "/ground_truth/objects",
                    threshold=1.0,
                ),
            ],
        )

        produced = pipeline.run(context(scene_store), TimeRange.everything())

        assert len(produced) == 2
        assert len(pipeline) == 2
        assert {str(path) for path in scene_store.entity_paths()} == {
            "/ground_truth/objects",
            "/estimation/objects",
            "/estimation/objects/filter/distance",
            "/matching/center_distance",
        }

    def test_reports_a_system_reading_a_later_target(self) -> None:
        matcher = CenterDistanceMatchingSystem.between("/a", "/b")
        reader = CenterDistanceMatchingSystem.between(
            "/matching/center_distance",
            "/b",
            target="/matching/second",
        )

        with pytest.raises(ValueError, match="before a later system writes it"):
            Pipeline([reader, matcher])

    def test_reports_an_unsatisfied_requirement_from_an_earlier_system(self) -> None:
        # The filter writes only a mask, so a matcher reading that entity can never find
        # the position and class it needs.
        filter_system = FilterByDistanceSystem.on("/estimation/objects")
        matcher = CenterDistanceMatchingSystem.between(
            filter_system.target,
            "/ground_truth/objects",
        )

        with pytest.raises(ValueError, match="which no earlier system provides there"):
            Pipeline([filter_system, matcher])

    def test_accepts_a_satisfied_chain(self) -> None:
        class MaskConsumer(EntitySystem):
            REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (MASK,)
            PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)

            def __call__(self, ctx: SystemContext, at: object) -> Iterable[Chunk]:
                return ()

        filter_system = FilterByDistanceSystem.on("/estimation/objects")
        consumer = MaskConsumer((filter_system.target,), "/derived")

        assert len(Pipeline([filter_system, consumer])) == 2

    def test_a_system_may_announce_several_targets(self) -> None:
        """One shared computation can feed several entities, and the pipeline sees them."""
        from t4perceval.system import ClearSystem

        clear = ClearSystem.on("/matching/center_distance", SOURCE, "/ground_truth/objects")

        assert len(clear.targets) == 3
        assert clear.target not in clear.targets, "the root is a prefix, not an output"

    def test_a_single_target_system_announces_just_itself(self) -> None:
        system = FilterByDistanceSystem.on(SOURCE)

        assert system.targets == (system.target,)

    def test_ordering_is_checked_against_every_target(self) -> None:
        from t4perceval.system import ClearSystem

        clear = ClearSystem.on("/matching/center_distance", SOURCE, "/ground_truth/objects")
        reader = ClearSystem.on(
            str(clear.targets[1]),
            SOURCE,
            "/ground_truth/objects",
            target="/metrics/second",
        )

        with pytest.raises(ValueError, match="before a later system writes it"):
            Pipeline([reader, clear])

    def test_an_empty_pipeline_is_valid(self, scene_store: Store) -> None:
        assert Pipeline([]).run(context(scene_store), 0) == ()


class TestEndToEnd:
    def test_a_scene_flows_from_logging_to_a_stored_verdict(
        self,
        scene_store: Store,
        labels: LabelRegistry,
        tmp_path: Path,
    ) -> None:
        """The whole stack in one pass: log, filter, match, query back, persist."""
        pipeline = Pipeline(
            [
                FilterByDistanceSystem.on("/estimation/objects", max_distance=40.0),
                CenterDistanceMatchingSystem.between(
                    "/estimation/objects",
                    "/ground_truth/objects",
                    threshold=1.0,
                ),
            ],
        )

        pipeline.run(context(scene_store, labels), TimeRange.everything())

        # The filter verdict is queryable data, not a discarded intermediate.
        mask_view = scene_store.range(
            "/estimation/objects/filter/distance",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )
        assert mask_view.component(MASK).values.tolist() == [True, False, True, True]

        # A whole-scene aggregate.
        scene = scene_store.range(
            "/matching/center_distance",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(BatchMatchResult)
        assert (scene.num_tp, scene.num_fp, scene.num_fn) == (2, 2, 1)

        # The same store answers per-frame questions with no re-computation.
        frame_one = scene_store.range(
            "/matching/center_distance",
            timeline=FRAME,
            time_range=TimeRange.single(1),
        ).materialize(BatchMatchResult)
        assert (frame_one.num_tp, frame_one.num_fp, frame_one.num_fn) == (1, 1, 0)

        # And the verdict survives a round-trip to disk.
        chunk = scene_store.range(
            "/matching/center_distance",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).to_chunk()
        path = tmp_path / "matching.parquet"
        write_parquet(chunk, path, labels=labels)
        restored, restored_labels = read_parquet(path)

        assert restored == chunk
        assert restored_labels == labels
        assert BatchMatchResult.from_chunk(restored).num_tp == 2

    def test_the_task_is_the_pipeline_not_an_enum(self, scene_store: Store) -> None:
        """The same components serve a different task by composing different systems."""
        tight = Pipeline(
            [
                CenterDistanceMatchingSystem.between(
                    "/estimation/objects",
                    "/ground_truth/objects",
                    threshold=0.2,
                    target="/matching/tight",
                ),
            ],
        )
        loose = Pipeline(
            [
                CenterDistanceMatchingSystem.between(
                    "/estimation/objects",
                    "/ground_truth/objects",
                    threshold=2.0,
                    class_agnostic=True,
                    target="/matching/loose",
                ),
            ],
        )

        ctx = context(scene_store)
        tight.run(ctx, TimeRange.everything())
        loose.run(ctx, TimeRange.everything())

        def hits(path: str) -> int:
            return (
                scene_store.range(path, timeline=FRAME, time_range=TimeRange.everything())
                .materialize(BatchMatchResult)
                .num_tp
            )

        assert hits("/matching/tight") == 1
        assert hits("/matching/loose") == 2
