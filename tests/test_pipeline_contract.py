"""The passthrough contract: what `Pipeline` knows about a target, and when it checks.

`tests/test_system.py` keeps the original wiring rules as the regression guard. This file
covers what `Passthrough` and `requires_for` add: known, opaque and store-sourced entities,
and requirements that differ per source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pytest
from conftest import make_detections

from t4perceval import (
    FRAME,
    Chunk,
    MatchResults,
    SemanticSegmentation3D,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.component import BatchMask
from t4perceval.core.timeline import TimeColumn
from t4perceval.descriptors import (
    CLASS_ID,
    CONFIDENCE,
    EST_INDEX,
    GT_INDEX,
    MASK,
    MATCH_STATUS,
    POSITION,
)
from t4perceval.system import (
    ApplyMaskSystem,
    AveragePrecisionSystem,
    CenterDistanceMatchingSystem,
    EntitySystem,
    FilterByDistanceSystem,
    Passthrough,
    Pipeline,
    System,
    SystemContext,
    TransformEntitySystem,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from t4perceval.core.descriptor import ComponentDescriptor

EST = "/estimation/objects"
GT = "/ground_truth/objects"
EVERYTHING = TimeRange.everything()


def passthrough(source: str, target: str, *, adds=(), drops=(), index: int = 0) -> EntitySystem:  # noqa: ANN001
    """A system that declares it carries ``source`` through, and does nothing when run."""

    class Carrier(EntitySystem):
        REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = ()
        PROVIDES: ClassVar[tuple[ComponentDescriptor, ...] | Passthrough] = Passthrough(
            index, adds=adds, drops=drops
        )

        def __call__(self, ctx: SystemContext, at: object) -> Iterable[Chunk]:
            return ()

    return Carrier((source,), target)


class MaskConsumer(EntitySystem):
    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (MASK,)
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...] | Passthrough] = (CLASS_ID,)

    def __call__(self, ctx: SystemContext, at: object) -> Iterable[Chunk]:
        return ()


class MaskAndClassConsumer(EntitySystem):
    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (MASK, CLASS_ID)
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...] | Passthrough] = ()

    def __call__(self, ctx: SystemContext, at: object) -> Iterable[Chunk]:
        return ()


class TestPassthroughValue:
    def test_defaults(self) -> None:
        assert Passthrough(0) == Passthrough(source=0, adds=(), drops=())
        assert hash(Passthrough(0)) == hash(Passthrough(0))

    def test_apply_narrows_and_widens_a_known_set(self) -> None:
        contract = Passthrough(0, adds=(CLASS_ID,), drops=(MASK,))
        assert contract.apply({MASK, POSITION}) == {POSITION, CLASS_ID}

    def test_apply_mask_carries_its_data_source_through(self) -> None:
        assert ApplyMaskSystem.PROVIDES == Passthrough(0)

    def test_the_transform_system_drops_masks(self) -> None:
        assert TransformEntitySystem.PROVIDES == Passthrough(0, drops=(MASK,))

    def test_tuple_contracts_are_untouched(self) -> None:
        assert FilterByDistanceSystem.PROVIDES == (MASK,)
        assert CenterDistanceMatchingSystem.PROVIDES == MatchResults.required_descriptors()

    def test_passthrough_systems_satisfy_the_protocol(self) -> None:
        assert isinstance(ApplyMaskSystem.of(EST, "/m"), System)
        assert isinstance(TransformEntitySystem.of(EST, target_frame="map"), System)


class TestPerSourceRequirements:
    def test_requires_for_defaults_to_requires(self) -> None:
        assert FilterByDistanceSystem.on(EST).requires_for(0) == (POSITION,)
        matcher = CenterDistanceMatchingSystem.between(EST, GT)
        assert matcher.requires_for(0) == matcher.requires_for(1) == matcher.REQUIRES

    def test_apply_mask_needs_the_mask_only_on_the_mask_source(self) -> None:
        kept = ApplyMaskSystem.of(EST, "/mask")
        assert kept.requires_for(0) == ()
        assert kept.requires_for(1) == (MASK,)

    def test_metric_sources_have_their_own_requirements(self) -> None:
        metric = AveragePrecisionSystem.on("/matching/x", EST, GT)
        assert metric.requires_for(0) == metric.REQUIRES
        assert {EST_INDEX, GT_INDEX, MATCH_STATUS} <= set(metric.requires_for(0))
        assert metric.requires_for(1) == AveragePrecisionSystem.REQUIRES_ESTIMATION
        assert metric.requires_for(2) == AveragePrecisionSystem.REQUIRES_GROUND_TRUTH
        assert CLASS_ID in metric.requires_for(1)
        assert CONFIDENCE in metric.requires_for(1)

    def test_apply_mask_data_may_come_from_a_known_set_without_a_mask(self) -> None:
        # The matcher's target is known and carries no MASK; that must not matter for the
        # data source -- only the mask source has to have one.
        near = FilterByDistanceSystem.on(EST)
        matcher = CenterDistanceMatchingSystem.between(EST, GT)
        assert len(Pipeline([near, matcher, ApplyMaskSystem.of(matcher.target, near.target)])) == 3

    def test_apply_mask_mask_source_is_still_checked(self) -> None:
        matcher = CenterDistanceMatchingSystem.between(EST, GT)
        with pytest.raises(
            ValueError,
            match=r"ApplyMaskSystem reads /matching/center_distance for component\(s\) mask",
        ):
            Pipeline([matcher, ApplyMaskSystem.of(EST, matcher.target)])

    def test_a_metric_estimation_source_is_checked_against_its_own_needs(self) -> None:
        near = FilterByDistanceSystem.on(EST)
        matcher = CenterDistanceMatchingSystem.between(EST, GT)
        metric = AveragePrecisionSystem.on(matcher.target, near.target, GT)
        with pytest.raises(ValueError) as info:
            Pipeline([near, matcher, metric])
        message = str(info.value)
        assert "class_id" in message and "confidence" in message
        assert "est_index" not in message, "the matching requirements belong to source 0"

    def test_a_metric_matching_source_is_checked_as_before(self) -> None:
        near = FilterByDistanceSystem.on(EST)
        with pytest.raises(ValueError, match="match_status"):
            Pipeline([near, AveragePrecisionSystem.on(near.target, EST, GT)])


class TestOpaqueTargets:
    def test_apply_mask_then_matcher_in_one_pipeline(self, scene_store: Store, labels) -> None:  # noqa: ANN001
        near = FilterByDistanceSystem.on(EST, max_distance=40.0)
        kept = ApplyMaskSystem.of(EST, near.target)
        matcher = CenterDistanceMatchingSystem.between(kept.target, GT, threshold=1.0)

        produced = Pipeline([near, kept, matcher]).run(
            SystemContext(scene_store, FRAME, labels=labels), EVERYTHING
        )

        assert len(produced) == 3
        matches = scene_store.range(
            matcher.target, timeline=FRAME, time_range=EVERYTHING
        ).materialize(MatchResults)
        # The far estimation (x=50) is gone, so one false positive fewer than the
        # unfiltered scene in tests/test_system.py.
        assert (matches.num_tp, matches.num_fp, matches.num_fn) == (2, 1, 1)

    def test_transform_then_matcher_in_one_pipeline(self) -> None:
        moved = TransformEntitySystem.of(EST, target_frame="map")
        assert len(Pipeline([moved, CenterDistanceMatchingSystem.between(moved.target, GT)])) == 2

    def test_a_passthrough_of_an_opaque_source_stays_opaque(self, scene_store: Store) -> None:
        near = FilterByDistanceSystem.on(EST, max_distance=40.0)
        kept = ApplyMaskSystem.of(EST, near.target)
        nearer = FilterByDistanceSystem.on(kept.target, max_distance=5.0)
        kept_again = ApplyMaskSystem.of(kept.target, nearer.target)
        matcher = CenterDistanceMatchingSystem.between(kept_again.target, GT)

        pipeline = Pipeline([near, kept, nearer, kept_again, matcher])
        assert len(pipeline.run(SystemContext(scene_store, FRAME), EVERYTHING)) == 5

    def test_opaque_consumers_are_checked_when_the_pipeline_runs(self) -> None:
        store = Store()
        store.log(
            EST,
            SemanticSegmentation3D(point=[[0.0, 0.0, 0.0]], class_id=[0]),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        store.log(
            GT, make_detections([[0.0, 0.0, 0.0]]), at=TimePoint.at(frame=0), frame_id="base_link"
        )
        store.send_chunk(
            Chunk.from_columns(
                f"{EST}/filter/all",
                {MASK: BatchMask([True])},
                indexes=(TimeColumn.of(FRAME, [0]),),
                frame_id="base_link",
            ),
        )
        kept = ApplyMaskSystem.of(EST, f"{EST}/filter/all")
        matcher = CenterDistanceMatchingSystem.between(kept.target, GT)

        pipeline = Pipeline([kept, matcher])  # constructs: the source columns are unknown
        with pytest.raises(
            ValueError,
            match=r"/estimation/objects/kept is missing required component\(s\): position",
        ):
            pipeline.run(SystemContext(store, FRAME), EVERYTHING)

    def test_a_consumer_with_no_requirements_is_never_rejected(self) -> None:
        kept = ApplyMaskSystem.of(EST, "/mask")
        assert len(Pipeline([kept, passthrough(str(kept.target), "/again")])) == 2


class TestKnownSetInheritance:
    def test_a_passthrough_of_a_known_set_inherits_it(self) -> None:
        matcher = CenterDistanceMatchingSystem.between(EST, GT)
        carrier = passthrough(str(matcher.target), "/derived")
        assert (
            len(Pipeline([matcher, carrier, AveragePrecisionSystem.on("/derived", EST, GT)])) == 3
        )

    def test_dropping_a_required_component_is_rejected_up_front(self) -> None:
        matcher = CenterDistanceMatchingSystem.between(EST, GT)
        carrier = passthrough(str(matcher.target), "/derived", drops=(MATCH_STATUS,))
        with pytest.raises(
            ValueError, match=r"reads /derived for component\(s\) match_status, which no earlier"
        ):
            Pipeline([matcher, carrier, AveragePrecisionSystem.on("/derived", EST, GT)])

    def test_adds_satisfy_a_consumer(self) -> None:
        near = FilterByDistanceSystem.on(EST)
        with pytest.raises(ValueError, match="class_id"):
            Pipeline(
                [near, passthrough(str(near.target), "/d"), MaskAndClassConsumer(("/d",), "/x")]
            )
        widened = passthrough(str(near.target), "/d", adds=(CLASS_ID,))
        assert len(Pipeline([near, widened, MaskAndClassConsumer(("/d",), "/x")])) == 3

    def test_drops_do_not_make_an_opaque_target_known(self) -> None:
        # EST comes from the store, so nothing is known about it; dropping POSITION from
        # "unknown" is still unknown, and the consumer is deferred rather than rejected.
        narrowed = passthrough(EST, "/d", drops=(POSITION,))
        assert len(Pipeline([narrowed, CenterDistanceMatchingSystem.between("/d", GT)])) == 2

    def test_transforming_a_mask_entity_loses_the_mask_up_front(self) -> None:
        near = FilterByDistanceSystem.on(EST)
        moved = TransformEntitySystem.of(near.target, target_frame="map")
        with pytest.raises(ValueError, match=r"MaskConsumer reads .* for component\(s\) mask"):
            Pipeline([near, moved, MaskConsumer((moved.target,), "/x")])

    def test_two_writers_merge_their_known_sets(self) -> None:
        first = passthrough(EST, "/d", adds=(MASK,))
        second = passthrough(GT, "/d", adds=(CLASS_ID,))
        # Both upstreams are opaque, so the target is too -- and the consumer is deferred.
        assert len(Pipeline([first, second, MaskAndClassConsumer(("/d",), "/x")])) == 3


class TestOrderingUnchanged:
    def test_a_passthrough_reading_a_later_target_is_still_reordered(self) -> None:
        moved = TransformEntitySystem.of(EST, target_frame="map")
        matcher = CenterDistanceMatchingSystem.between(moved.target, GT)
        with pytest.raises(ValueError, match="before a later system writes it"):
            Pipeline([matcher, moved])

    def test_a_filter_feeding_a_matcher_is_still_rejected(self) -> None:
        # {MASK} is a *known* set, and it lacks what a matcher needs.
        near = FilterByDistanceSystem.on(EST)
        with pytest.raises(ValueError, match="which no earlier system provides there"):
            Pipeline([near, CenterDistanceMatchingSystem.between(near.target, GT)])

    def test_a_passthrough_index_out_of_range_is_reported(self) -> None:
        with pytest.raises(ValueError, match="passes through source 3, but has 1 source"):
            Pipeline([passthrough(EST, "/d", index=3)])
