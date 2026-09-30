"""Filter systems: the predicate of each, and how masks compose."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
from conftest import make_detections, make_ego_scene, make_segmentation

from t4perceval import (
    FRAME,
    Detections3D,
    SemanticSegmentation3D,
    Trackings3D,
    Chunk,
    InstanceRegistry,
    LabelRegistry,
    Store,
    TimeColumn,
    TimePoint,
    TimeRange,
)
from t4perceval.component import BatchMask, VisibilityLevel
from t4perceval.descriptors import MASK, POSITION
from t4perceval.system import (
    ApplyMaskSystem,
    CombineMasksSystem,
    FilterByConfidenceSystem,
    FilterByDistanceSystem,
    FilterByInstanceSystem,
    FilterByLabelSystem,
    FilterByNumPointsSystem,
    FilterByMapSystem,
    FilterByRegionSystem,
    FilterBySpeedSystem,
    FilterByVisibilitySystem,
    FilterPointsByDistanceSystem,
    FilterPointsByMapSystem,
    FilterPointsByRegionSystem,
    MaskSystem,
    Pipeline,
    SystemContext,
    masked_view,
)

if TYPE_CHECKING:
    from t4perceval.core.chunk import Chunk
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.view import EntityView


SOURCE = "/estimation/objects"

#: Every filter class, for the properties that must hold across the whole family.
ALL_FILTERS: tuple[type[MaskSystem], ...] = (
    FilterByDistanceSystem,
    FilterByRegionSystem,
    FilterByMapSystem,
    FilterByLabelSystem,
    FilterByConfidenceSystem,
    FilterByInstanceSystem,
    FilterBySpeedSystem,
    FilterByNumPointsSystem,
    FilterByVisibilitySystem,
)

#: The positional filters over a cloud's ``point`` column rather than an object's ``position``.
POINT_FILTERS: tuple[type[MaskSystem], ...] = (
    FilterPointsByDistanceSystem,
    FilterPointsByRegionSystem,
    FilterPointsByMapSystem,
)


@pytest.fixture
def uuids() -> list[str]:
    return ["a", "b", "c", "d"]


@pytest.fixture
def instances(uuids: list[str]) -> InstanceRegistry:
    registry = InstanceRegistry()
    registry.encode(uuids)
    return registry


@pytest.fixture
def rich_store(labels: LabelRegistry, instances: InstanceRegistry, uuids: list[str]) -> Store:
    """One frame of four objects, carrying every column the filters need.

    | row | position     | class      | conf | uuid | speed | points | visibility  |
    | :-- | :----------- | :--------- | :--- | :--- | :---- | :----- | :---------- |
    | 0   | (1, 0, 0)    | car        | 0.90 | a    | 0     | 100    | FULL        |
    | 1   | (50, 0, 0)   | truck      | 0.20 | b    | 10    | 3      | NONE        |
    | 2   | (0, 120, 0)  | pedestrian | 0.50 | c    | 1     | 0      | UNAVAILABLE |
    | 3   | (3, 4, 0)    | car        | 0.75 | d    | 30    | 20     | MOST        |
    """
    store = Store()
    store.log(
        SOURCE,
        Trackings3D(
            position=[[1.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 120.0, 0.0], [3.0, 4.0, 0.0]],
            quaternion=[[0.0, 0.0, 0.0, 1.0]] * 4,
            size=[[1.0, 1.0, 1.0]] * 4,
            class_id=labels.encode(["car", "truck", "pedestrian", "car"]),
            confidence=[0.9, 0.2, 0.5, 0.75],
            instance_id=instances.encode(uuids),
            velocity=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
            num_points=[100, 3, 0, 20],
            visibility=[
                VisibilityLevel.FULL,
                VisibilityLevel.NONE,
                VisibilityLevel.UNAVAILABLE,
                VisibilityLevel.MOST,
            ],
        ),
        at=TimePoint.at(frame=0),
        frame_id="base_link",
    )
    return store


@pytest.fixture
def rich_scene_store(labels: LabelRegistry) -> Store:
    """Two frames of two objects each, carrying every column the filters need."""
    store = Store()
    for frame in (0, 1):
        store.log(
            SOURCE,
            Trackings3D(
                position=[[float(frame), 0.0, 0.0], [1.0 + frame, 0.0, 0.0]],
                quaternion=[[0.0, 0.0, 0.0, 1.0]] * 2,
                size=[[1.0, 1.0, 1.0]] * 2,
                class_id=labels.encode(["car", "truck"]),
                confidence=[0.9, 0.8],
                instance_id=[10, 11],
                velocity=[[1.0, 0.0, 0.0]] * 2,
                num_points=[50, 60],
                visibility=[VisibilityLevel.FULL] * 2,
            ),
            at=TimePoint.at(frame=frame),
            frame_id="base_link",
        )
    return store


@pytest.fixture
def rich_context(
    rich_store: Store,
    labels: LabelRegistry,
    instances: InstanceRegistry,
) -> SystemContext:
    return SystemContext(rich_store, FRAME, labels=labels, instances=instances)


def mask_of(system: MaskSystem, ctx: SystemContext, at: object = 0) -> list[bool]:
    (chunk,) = system(ctx, at)
    return chunk.columns[MASK].values.tolist()


class TestFamilyProperties:
    """Properties every filter must share, so a new one cannot drift."""

    @pytest.mark.parametrize("cls", ALL_FILTERS + POINT_FILTERS, ids=lambda c: c.__name__)
    def test_declares_a_single_source_and_provides_a_mask(self, cls: type[MaskSystem]) -> None:
        system = cls.on(SOURCE)

        assert len(system.sources) == 1
        assert system.PROVIDES == (MASK,)
        assert len(system.REQUIRES) == 1

    @pytest.mark.parametrize("cls", ALL_FILTERS, ids=lambda c: c.__name__)
    def test_default_parameters_are_a_no_op(
        self,
        cls: type[MaskSystem],
        rich_context: SystemContext,
    ) -> None:
        assert mask_of(cls.on(SOURCE), rich_context) == [True] * 4

    @pytest.mark.parametrize("cls", ALL_FILTERS + POINT_FILTERS, ids=lambda c: c.__name__)
    def test_writes_under_the_source_path(self, cls: type[MaskSystem]) -> None:
        system = cls.on(SOURCE)

        assert system.target.is_descendant_of(system.sources[0])
        assert str(system.target) == f"{SOURCE}/filter/{cls.FILTER_NAME}"

    @pytest.mark.parametrize("cls", ALL_FILTERS + POINT_FILTERS, ids=lambda c: c.__name__)
    def test_the_target_name_can_be_overridden(self, cls: type[MaskSystem]) -> None:
        assert str(cls.on(SOURCE, name="critical").target) == f"{SOURCE}/filter/critical"

    @pytest.mark.parametrize("cls", ALL_FILTERS, ids=lambda c: c.__name__)
    def test_masks_rather_than_drops(
        self,
        cls: type[MaskSystem],
        rich_context: SystemContext,
    ) -> None:
        (chunk,) = cls.on(SOURCE)(rich_context, 0)

        assert chunk.num_rows == 4, "excluded rows must stay addressable"

    @pytest.mark.parametrize("cls", ALL_FILTERS, ids=lambda c: c.__name__)
    def test_keeps_the_partition_structure_of_its_source(
        self,
        cls: type[MaskSystem],
        rich_context: SystemContext,
    ) -> None:
        (chunk,) = cls.on(SOURCE)(rich_context, TimeRange.everything())

        assert chunk.num_partitions == 1
        assert chunk.index(FRAME).times.tolist() == [0]
        assert chunk.frame_id == "base_link"

    @pytest.mark.parametrize("cls", ALL_FILTERS, ids=lambda c: c.__name__)
    def test_a_bare_frame_index_selects_one_frame(
        self,
        cls: type[MaskSystem],
        rich_scene_store: Store,
        labels: LabelRegistry,
    ) -> None:
        ctx = SystemContext(rich_scene_store, FRAME, labels=labels)

        (chunk,) = cls.on(SOURCE)(ctx, 0)

        assert chunk.num_rows == 2, "only frame 0"

    @pytest.mark.parametrize("cls", ALL_FILTERS, ids=lambda c: c.__name__)
    def test_a_multi_frame_range_keeps_every_partition(
        self,
        cls: type[MaskSystem],
        rich_scene_store: Store,
        labels: LabelRegistry,
    ) -> None:
        ctx = SystemContext(rich_scene_store, FRAME, labels=labels)

        (chunk,) = cls.on(SOURCE)(ctx, TimeRange.everything())

        assert chunk.num_partitions == 2
        assert chunk.offsets.tolist() == [0, 2, 4]
        assert chunk.index(FRAME).times.tolist() == [0, 1]

    @pytest.mark.parametrize("cls", ALL_FILTERS, ids=lambda c: c.__name__)
    def test_an_empty_frame_is_not_a_wiring_error(self, cls: type[MaskSystem]) -> None:
        store = Store()
        store.log(SOURCE, make_detections([]), at=TimePoint.at(frame=0))

        (chunk,) = cls.on(SOURCE)(SystemContext(store, FRAME), 0)

        assert chunk.num_rows == 0

    @pytest.mark.parametrize("cls", ALL_FILTERS + POINT_FILTERS, ids=lambda c: c.__name__)
    def test_a_missing_component_is_reported(self, cls: type[MaskSystem]) -> None:
        # An entity carrying one unrelated column is missing whatever any filter needs.
        store = Store()
        store.send_chunk(
            Chunk.from_columns(
                SOURCE,
                {MASK: BatchMask([True])},
                indexes=(TimeColumn.of(FRAME, [0]),),
            ),
        )
        system = cls.on(SOURCE)

        with pytest.raises(ValueError, match="missing required component"):
            list(system(SystemContext(store, FRAME), 0))

    @pytest.mark.parametrize("cls", ALL_FILTERS + POINT_FILTERS, ids=lambda c: c.__name__)
    def test_rejects_more_than_one_source(self, cls: type[MaskSystem]) -> None:
        with pytest.raises(ValueError, match="needs exactly one source"):
            cls(("/a", "/b"), "/out")

    def test_a_new_filter_only_writes_its_predicate(
        self,
        rich_context: SystemContext,
    ) -> None:
        from attrs import define

        @define(slots=True)
        class FilterFirstRowSystem(MaskSystem):
            REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POSITION,)
            FILTER_NAME: ClassVar[str] = "first_row"

            def keep(self, view: EntityView, ctx: SystemContext) -> np.ndarray:
                del ctx
                keep = np.zeros(len(view), dtype=np.bool_)
                keep[0] = True
                return keep

        assert mask_of(FilterFirstRowSystem.on(SOURCE), rich_context) == [
            True,
            False,
            False,
            False,
        ]

    def test_a_wrongly_shaped_predicate_is_reported(
        self,
        rich_context: SystemContext,
    ) -> None:
        from attrs import define

        @define(slots=True)
        class BrokenSystem(MaskSystem):
            REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POSITION,)

            def keep(self, view: EntityView, ctx: SystemContext) -> np.ndarray:
                del view, ctx
                return np.ones(2, dtype=np.bool_)

        with pytest.raises(ValueError, match=r"returned shape \(2,\), expected \(4,\)"):
            list(BrokenSystem.on(SOURCE)(rich_context, 0))


class TestFilterByDistance:
    def test_bounds_the_radial_distance(self, rich_context: SystemContext) -> None:
        assert mask_of(FilterByDistanceSystem.on(SOURCE, max_distance=10.0), rich_context) == [
            True,
            False,
            False,
            True,
        ]

    def test_applies_a_minimum_too(self, rich_context: SystemContext) -> None:
        assert mask_of(FilterByDistanceSystem.on(SOURCE, min_distance=2.0), rich_context) == [
            False,
            True,
            True,
            True,
        ]

    def test_bev_ignores_the_z_axis(self) -> None:
        store = Store()
        store.log(
            SOURCE,
            make_detections([[3.0, 4.0, 0.0], [3.0, 4.0, 100.0]]),
            at=TimePoint.at(frame=0),
        )
        ctx = SystemContext(store, FRAME)

        full = FilterByDistanceSystem.on(SOURCE, max_distance=10.0)
        bev = FilterByDistanceSystem.on(SOURCE, max_distance=10.0, bev=True)

        assert mask_of(full, ctx) == [True, False]
        assert mask_of(bev, ctx) == [True, True]

    def test_bounds_are_inclusive(self) -> None:
        store = Store()
        store.log(
            SOURCE, make_detections([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), at=TimePoint.at(frame=0)
        )
        ctx = SystemContext(store, FRAME)

        # An object at the origin passes the default min, and one exactly at max passes.
        assert mask_of(FilterByDistanceSystem.on(SOURCE, max_distance=10.0), ctx) == [True, True]

    def test_rejects_a_contradictory_range(self) -> None:
        with pytest.raises(ValueError, match="must not be below"):
            FilterByDistanceSystem.on(SOURCE, min_distance=10.0, max_distance=1.0)

    def test_rejects_a_negative_minimum(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            FilterByDistanceSystem.on(SOURCE, min_distance=-1.0)


class TestFilterByRegion:
    def test_bounds_the_xy_box(self, rich_context: SystemContext) -> None:
        system = FilterByRegionSystem.on(SOURCE, min_xy=(-100.0, -100.0), max_xy=(100.0, 100.0))

        assert mask_of(system, rich_context) == [True, True, False, True]

    def test_symmetric_mirrors_the_bound_about_the_origin(
        self,
        rich_context: SystemContext,
    ) -> None:
        system = FilterByRegionSystem.symmetric(SOURCE, max_xy=(10.0, 10.0))

        assert system.min_xy == (-10.0, -10.0)
        assert system.max_xy == (10.0, 10.0)
        assert mask_of(system, rich_context) == [True, False, False, True]

    def test_symmetric_differs_from_an_upper_bound_alone(self) -> None:
        """``on(max_xy=...)`` leaves ``min_xy`` unbounded; ``symmetric`` does not."""
        store = Store()
        store.log(
            SOURCE,
            make_detections([[-50.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            at=TimePoint.at(frame=0),
        )
        ctx = SystemContext(store, FRAME)

        upper_only = FilterByRegionSystem.on(SOURCE, max_xy=(10.0, 10.0))
        mirrored = FilterByRegionSystem.symmetric(SOURCE, max_xy=(10.0, 10.0))

        assert mask_of(upper_only, ctx) == [True, True]
        assert mask_of(mirrored, ctx) == [False, True]

    def test_bounds_each_axis_independently(self) -> None:
        store = Store()
        store.log(
            SOURCE,
            make_detections([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
            at=TimePoint.at(frame=0),
        )
        system = FilterByRegionSystem.on(SOURCE, max_xy=(1.0, 100.0))

        assert mask_of(system, SystemContext(store, FRAME)) == [False, True]

    def test_ignores_the_z_axis(self) -> None:
        store = Store()
        store.log(SOURCE, make_detections([[0.0, 0.0, 999.0]]), at=TimePoint.at(frame=0))
        system = FilterByRegionSystem.on(SOURCE, max_xy=(1.0, 1.0))

        assert mask_of(system, SystemContext(store, FRAME)) == [True]

    def test_rejects_a_contradictory_range(self) -> None:
        with pytest.raises(ValueError, match=r"max_xy\[y\] .* must not be below min_xy\[y\]"):
            FilterByRegionSystem.on(SOURCE, min_xy=(0.0, 10.0), max_xy=(1.0, 1.0))


class TestPointPositionFilters:
    """The positional predicates over a cloud's ``point`` column."""

    CLOUD = "/ground_truth/points"

    #: rich_store's positions, so the expected masks read the same as the object filters'.
    POINTS = [[1.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 120.0, 0.0], [3.0, 4.0, 0.0]]

    @pytest.fixture
    def cloud_store(self, labels: LabelRegistry) -> Store:
        store = Store()
        make_segmentation(
            store,
            self.CLOUD,
            0,
            ["car", "truck", "pedestrian", "car"],
            labels=labels,
            points=self.POINTS,
            frame_id="base_link",
        )
        return store

    @pytest.fixture
    def cloud_context(self, cloud_store: Store, labels: LabelRegistry) -> SystemContext:
        return SystemContext(cloud_store, FRAME, labels=labels)

    @pytest.mark.parametrize("cls", POINT_FILTERS, ids=lambda c: c.__name__)
    def test_requires_the_point_column(self, cls: type[MaskSystem]) -> None:
        from t4perceval.descriptors import POINT

        assert cls.REQUIRES == (POINT,)
        assert cls.COLUMN == POINT

    @pytest.mark.parametrize("cls", POINT_FILTERS, ids=lambda c: c.__name__)
    def test_default_parameters_are_a_no_op(
        self,
        cls: type[MaskSystem],
        cloud_context: SystemContext,
    ) -> None:
        assert mask_of(cls.on(self.CLOUD), cloud_context) == [True] * 4

    @pytest.mark.parametrize("cls", POINT_FILTERS, ids=lambda c: c.__name__)
    def test_keeps_the_partition_structure_of_the_cloud(
        self,
        cls: type[MaskSystem],
        cloud_context: SystemContext,
    ) -> None:
        (chunk,) = cls.on(self.CLOUD)(cloud_context, TimeRange.everything())

        assert chunk.num_rows == 4
        assert chunk.num_partitions == 1
        assert chunk.frame_id == "base_link"

    def test_distance_bounds_the_radial_distance(self, cloud_context: SystemContext) -> None:
        system = FilterPointsByDistanceSystem.on(self.CLOUD, max_distance=10.0)

        assert mask_of(system, cloud_context) == [True, False, False, True]

    def test_distance_applies_a_minimum_too(self, cloud_context: SystemContext) -> None:
        system = FilterPointsByDistanceSystem.on(self.CLOUD, min_distance=2.0, max_distance=60.0)

        assert mask_of(system, cloud_context) == [False, True, False, True]

    def test_distance_bev_ignores_the_z_axis(self, labels: LabelRegistry) -> None:
        store = Store()
        make_segmentation(
            store,
            self.CLOUD,
            0,
            ["car", "car"],
            labels=labels,
            points=[[3.0, 4.0, 100.0], [3.0, 4.0, 0.0]],
            frame_id="base_link",
        )
        ctx = SystemContext(store, FRAME)

        assert mask_of(FilterPointsByDistanceSystem.on(self.CLOUD, max_distance=5.0), ctx) == [
            False,
            True,
        ]
        assert mask_of(
            FilterPointsByDistanceSystem.on(self.CLOUD, max_distance=5.0, bev=True), ctx
        ) == [True, True]

    def test_region_bounds_the_xy_box(self, cloud_context: SystemContext) -> None:
        system = FilterPointsByRegionSystem.symmetric(self.CLOUD, max_xy=(5.0, 5.0))

        assert mask_of(system, cloud_context) == [True, False, False, True]

    def test_region_bounds_each_axis_independently(self, cloud_context: SystemContext) -> None:
        system = FilterPointsByRegionSystem.on(self.CLOUD, max_xy=(60.0, 1.0))

        assert mask_of(system, cloud_context) == [True, True, False, False]

    SQUARE = [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]

    def test_map_tests_xy_against_the_polygon(self, labels: LabelRegistry) -> None:
        # Already in map, so no /tf is needed.
        store = Store()
        make_segmentation(
            store,
            self.CLOUD,
            0,
            ["car"] * 4,
            labels=labels,
            points=self.POINTS,
            frame_id="map",
        )
        system = FilterPointsByMapSystem.on(self.CLOUD, polygon=self.SQUARE)

        assert mask_of(system, SystemContext(store, FRAME)) == [True, False, False, True]

    #: In map the point is at x = 1, 11, 21 on y = 1; this keeps x in [8, 15]: frame 1 only.
    MIDDLE = [[8.0, -2.0], [15.0, -2.0], [15.0, 2.0], [8.0, 2.0]]

    def test_map_moves_a_base_link_cloud_with_the_ego(self, labels: LabelRegistry) -> None:
        store = make_ego_scene(Store())
        for frame in range(3):
            make_segmentation(
                store,
                self.CLOUD,
                frame,
                ["car"],
                labels=labels,
                points=[[1.0, 1.0, 0.0]],
                frame_id="base_link",
            )
        system = FilterPointsByMapSystem.on(self.CLOUD, polygon=self.MIDDLE)
        ctx = SystemContext(store, FRAME)

        assert [mask_of(system, ctx, at=frame) for frame in range(3)] == [[False], [True], [False]]

    def test_map_lands_under_the_cloud(self) -> None:
        system = FilterPointsByMapSystem.on(self.CLOUD, polygon=self.SQUARE)

        assert str(system.target) == f"{self.CLOUD}/filter/map"

    def test_a_point_filter_refuses_an_object_entity(self, rich_context: SystemContext) -> None:
        # POINT is not POSITION: pointing the cloud filter at objects is a wiring error.
        system = FilterPointsByDistanceSystem.on(SOURCE, max_distance=10.0)

        with pytest.raises(ValueError, match="missing required component"):
            list(system(rich_context, 0))

    def test_an_object_filter_still_refuses_a_cloud(self, cloud_context: SystemContext) -> None:
        system = FilterByDistanceSystem.on(self.CLOUD, max_distance=10.0)

        with pytest.raises(ValueError, match="missing required component"):
            list(system(cloud_context, 0))

    def test_the_mask_applies_to_the_cloud(self, cloud_context: SystemContext) -> None:
        near = FilterPointsByDistanceSystem.on(self.CLOUD, max_distance=10.0)
        kept = ApplyMaskSystem.of(self.CLOUD, near.target)

        Pipeline([near, kept]).run(cloud_context, 0)
        view = cloud_context.store.range(
            kept.target, timeline=FRAME, time_range=TimeRange.everything()
        )

        assert len(view) == 2
        cloud = view.materialize(SemanticSegmentation3D)
        assert cloud.point.values[:, 0].tolist() == [1.0, 3.0]

    def test_the_mask_reads_back_as_a_view(self, cloud_context: SystemContext) -> None:
        near = FilterPointsByDistanceSystem.on(self.CLOUD, max_distance=10.0)
        (chunk,) = near(cloud_context, 0)
        cloud_context.store.send_chunk(chunk)

        view = masked_view(
            cloud_context.store,
            self.CLOUD,
            near.target,
            timeline=FRAME,
            time_range=TimeRange.single(0),
        )

        assert view.materialize(SemanticSegmentation3D).point.values[:, 0].tolist() == [1.0, 3.0]


class TestFilterByLabel:
    def test_keeps_an_allowlist(self, rich_context: SystemContext) -> None:
        system = FilterByLabelSystem.on(SOURCE, labels=["car", "pedestrian"])

        assert mask_of(system, rich_context) == [True, False, True, True]

    def test_drops_a_denylist(self, rich_context: SystemContext) -> None:
        system = FilterByLabelSystem.on(SOURCE, exclude=["truck"])

        assert mask_of(system, rich_context) == [True, False, True, True]

    def test_combines_both(self, rich_context: SystemContext) -> None:
        system = FilterByLabelSystem.on(SOURCE, labels=["car", "truck"], exclude=["truck"])

        assert mask_of(system, rich_context) == [True, False, False, True]

    def test_accepts_raw_class_ids(self, rich_store: Store) -> None:
        # No registry needed when the caller already speaks in class ids.
        system = FilterByLabelSystem.on(SOURCE, labels=[0])

        assert mask_of(system, SystemContext(rich_store, FRAME)) == [True, False, False, True]

    def test_an_unknown_name_raises_rather_than_matching_nothing(
        self,
        rich_context: SystemContext,
    ) -> None:
        system = FilterByLabelSystem.on(SOURCE, labels=["spaceship"])

        with pytest.raises(KeyError, match="Unknown class name 'spaceship'"):
            list(system(rich_context, 0))

    def test_names_need_a_registry(self, rich_store: Store) -> None:
        system = FilterByLabelSystem.on(SOURCE, labels=["car"])

        with pytest.raises(ValueError, match="require a LabelRegistry"):
            list(system(SystemContext(rich_store, FRAME), 0))

    def test_an_empty_allowlist_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="pass None to admit every class"):
            FilterByLabelSystem.on(SOURCE, labels=[])

    def test_merging_a_registry_does_not_re_encode_existing_columns(
        self,
        rich_store: Store,
        labels: LabelRegistry,
    ) -> None:
        """A merged registry resolves names against the *new* ids, not the logged ones.

        The class-id column was written with the pre-merge registry, so the truck row --
        id 1, which the merged registry reads as "pedestrian" -- does not join the vehicle
        group. Merging is a property of the encoding, so it belongs before logging.
        """
        merged = labels.merged({"vehicle": ["car", "truck"]})
        system = FilterByLabelSystem.on(SOURCE, labels=["vehicle"])

        assert merged.class_id("vehicle") == 0
        assert mask_of(system, SystemContext(rich_store, FRAME, labels=merged)) == [
            True,
            False,
            False,
            True,
        ]


class TestFilterByConfidence:
    def test_bounds_the_score(self, rich_context: SystemContext) -> None:
        system = FilterByConfidenceSystem.on(SOURCE, min_confidence=0.5)

        assert mask_of(system, rich_context) == [True, False, True, True]

    def test_applies_a_maximum_too(self, rich_context: SystemContext) -> None:
        system = FilterByConfidenceSystem.on(SOURCE, max_confidence=0.5)

        assert mask_of(system, rich_context) == [False, True, True, False]

    def test_the_minimum_is_inclusive(self, rich_context: SystemContext) -> None:
        # The original compared strictly, so 0.5 would have been dropped here.
        system = FilterByConfidenceSystem.on(SOURCE, min_confidence=0.5)

        assert mask_of(system, rich_context)[2] is True

    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_rejects_a_threshold_outside_the_unit_interval(self, value: float) -> None:
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            FilterByConfidenceSystem.on(SOURCE, min_confidence=value)

    def test_rejects_a_contradictory_range(self) -> None:
        with pytest.raises(ValueError, match="must not be below"):
            FilterByConfidenceSystem.on(SOURCE, min_confidence=0.9, max_confidence=0.1)


class TestFilterByInstance:
    def test_keeps_target_uuids(self, rich_context: SystemContext) -> None:
        system = FilterByInstanceSystem.on(SOURCE, instances=["a", "d"])

        assert mask_of(system, rich_context) == [True, False, False, True]

    def test_drops_excluded_uuids(self, rich_context: SystemContext) -> None:
        system = FilterByInstanceSystem.on(SOURCE, exclude=["b"])

        assert mask_of(system, rich_context) == [True, False, True, True]

    def test_accepts_raw_instance_ids(self, rich_store: Store) -> None:
        system = FilterByInstanceSystem.on(SOURCE, instances=[0, 3])

        assert mask_of(system, SystemContext(rich_store, FRAME)) == [True, False, False, True]

    def test_an_unknown_uuid_raises(self, rich_context: SystemContext) -> None:
        system = FilterByInstanceSystem.on(SOURCE, instances=["zzz"])

        with pytest.raises(KeyError, match="Unknown instance uuid 'zzz'"):
            list(system(rich_context, 0))

    def test_uuids_need_a_registry(self, rich_store: Store) -> None:
        system = FilterByInstanceSystem.on(SOURCE, instances=["a"])

        with pytest.raises(ValueError, match="require an InstanceRegistry"):
            list(system(SystemContext(rich_store, FRAME), 0))

    def test_an_empty_allowlist_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="pass None to admit every instance"):
            FilterByInstanceSystem.on(SOURCE, instances=[])


class TestFilterBySpeed:
    def test_bounds_the_velocity_norm(self, rich_context: SystemContext) -> None:
        system = FilterBySpeedSystem.on(SOURCE, max_speed=20.0)

        assert mask_of(system, rich_context) == [True, True, True, False]

    def test_applies_a_minimum_too(self, rich_context: SystemContext) -> None:
        system = FilterBySpeedSystem.on(SOURCE, min_speed=5.0)

        assert mask_of(system, rich_context) == [False, True, False, True]

    def test_rejects_a_negative_minimum(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            FilterBySpeedSystem.on(SOURCE, min_speed=-1.0)


class TestFilterByNumPoints:
    def test_bounds_the_point_count(self, rich_context: SystemContext) -> None:
        system = FilterByNumPointsSystem.on(SOURCE, min_num_points=5)

        assert mask_of(system, rich_context) == [True, False, False, True]

    def test_the_minimum_is_inclusive(self, rich_context: SystemContext) -> None:
        # The original documented min_point_numbers=[5] as dropping boxes with 4 or fewer.
        system = FilterByNumPointsSystem.on(SOURCE, min_num_points=20)

        assert mask_of(system, rich_context) == [True, False, False, True]

    def test_applies_a_maximum_too(self, rich_context: SystemContext) -> None:
        system = FilterByNumPointsSystem.on(SOURCE, max_num_points=20)

        assert mask_of(system, rich_context) == [False, True, True, True]

    def test_rejects_a_negative_minimum(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            FilterByNumPointsSystem.on(SOURCE, min_num_points=-1)

    def test_rejects_a_contradictory_range(self) -> None:
        with pytest.raises(ValueError, match="must not be below"):
            FilterByNumPointsSystem.on(SOURCE, min_num_points=10, max_num_points=1)


class TestFilterByVisibility:
    def test_bounds_the_level(self, rich_context: SystemContext) -> None:
        system = FilterByVisibilitySystem.on(SOURCE, min_visibility=VisibilityLevel.MOST)

        assert mask_of(system, rich_context) == [True, False, True, True]

    def test_unavailable_always_passes(self, rich_context: SystemContext) -> None:
        # Row 2 is UNAVAILABLE. Rejecting it would empty out any dataset that does not
        # annotate visibility at all.
        system = FilterByVisibilitySystem.on(SOURCE, min_visibility=VisibilityLevel.FULL)

        assert mask_of(system, rich_context) == [True, False, True, False]

    def test_accepts_a_level_by_name_or_value(self) -> None:
        assert (
            FilterByVisibilitySystem.on(
                SOURCE,
                min_visibility=2,
            ).min_visibility
            is VisibilityLevel.MOST
        )

    def test_rejects_an_incomparable_threshold(self) -> None:
        with pytest.raises(ValueError, match="not UNAVAILABLE"):
            FilterByVisibilitySystem.on(SOURCE, min_visibility=VisibilityLevel.UNAVAILABLE)


class TestCombineMasks:
    def masks(self, ctx: SystemContext) -> tuple[MaskSystem, MaskSystem]:
        return (
            FilterByDistanceSystem.on(SOURCE, max_distance=10.0),
            FilterByConfidenceSystem.on(SOURCE, min_confidence=0.8),
        )

    def test_all_is_the_intersection(self, rich_context: SystemContext) -> None:
        near, confident = self.masks(rich_context)
        combined = CombineMasksSystem.of([near.target, confident.target], "/keep", mode="all")

        Pipeline([near, confident, combined]).run(rich_context, 0)

        assert mask_of(combined, rich_context) == [True, False, False, False]

    def test_any_is_the_union(self, rich_context: SystemContext) -> None:
        near, confident = self.masks(rich_context)
        combined = CombineMasksSystem.of([near.target, confident.target], "/keep", mode="any")

        Pipeline([near, confident, combined]).run(rich_context, 0)

        assert mask_of(combined, rich_context) == [True, False, False, True]

    def test_a_single_source_passes_through(self, rich_context: SystemContext) -> None:
        near, _ = self.masks(rich_context)
        combined = CombineMasksSystem.of([near.target], "/keep")

        Pipeline([near, combined]).run(rich_context, 0)

        assert mask_of(combined, rich_context) == mask_of(near, rich_context)

    def test_per_class_thresholds_by_composition(self, rich_context: SystemContext) -> None:
        """Cars need 50+ points; every other class is admitted regardless.

        This is what replaces the original per-class ``min_point_numbers`` list: an AND of
        a label filter with a threshold filter, then an OR across the classes.
        """
        is_car = FilterByLabelSystem.on(SOURCE, labels=["car"], name="is_car")
        not_car = FilterByLabelSystem.on(SOURCE, exclude=["car"], name="not_car")
        many_points = FilterByNumPointsSystem.on(SOURCE, min_num_points=50, name="pts50")
        car_ok = CombineMasksSystem.of(
            [is_car.target, many_points.target],
            f"{SOURCE}/filter/car_ok",
            mode="all",
        )
        keep = CombineMasksSystem.of(
            [car_ok.target, not_car.target],
            f"{SOURCE}/filter/keep",
            mode="any",
        )

        Pipeline([is_car, not_car, many_points, car_ok, keep]).run(rich_context, 0)

        # row 0: car with 100 points → kept. row 3: car with 20 points → dropped.
        # rows 1, 2: not cars → kept.
        assert mask_of(keep, rich_context) == [True, True, True, False]

    def test_reports_masks_describing_different_rows(self, rich_context: SystemContext) -> None:
        near, _ = self.masks(rich_context)
        other = "/other/objects"
        rich_context.store.log(other, make_detections([[0.0, 0.0, 0.0]]), at=TimePoint.at(frame=0))
        other_mask = FilterByDistanceSystem.on(other)

        Pipeline([near, other_mask]).run(rich_context, 0)
        combined = CombineMasksSystem.of([near.target, other_mask.target], "/keep")

        with pytest.raises(ValueError, match="different rows"):
            list(combined(rich_context, 0))

    def test_keeps_the_partition_structure(self, rich_context: SystemContext) -> None:
        near, confident = self.masks(rich_context)
        combined = CombineMasksSystem.of([near.target, confident.target], "/keep")

        Pipeline([near, confident, combined]).run(rich_context, TimeRange.everything())
        (chunk,) = combined(rich_context, TimeRange.everything())

        assert chunk.num_partitions == 1
        assert chunk.frame_id == "base_link"

    def test_rejects_an_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="must be 'all' or 'any'"):
            CombineMasksSystem.of(["/a"], "/b", mode="some")

    def test_needs_at_least_one_source(self) -> None:
        with pytest.raises(ValueError, match="at least one source"):
            CombineMasksSystem((), "/b")


class TestMaskedView:
    def test_narrows_the_source_to_the_rows_that_passed(self, rich_context: SystemContext) -> None:
        near = FilterByDistanceSystem.on(SOURCE, max_distance=10.0)
        Pipeline([near]).run(rich_context, TimeRange.everything())

        view = masked_view(
            rich_context.store,
            SOURCE,
            near.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )

        assert len(view) == 2
        assert view.component(POSITION).values[:, 0].tolist() == [1.0, 3.0]

    def test_is_lazy(self, rich_context: SystemContext) -> None:
        near = FilterByDistanceSystem.on(SOURCE, max_distance=10.0)
        Pipeline([near]).run(rich_context, TimeRange.everything())

        view = masked_view(
            rich_context.store,
            SOURCE,
            near.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )

        assert view.indices.tolist() == [0, 3], "row indices are composed, not materialized"
        assert view.chunk.num_rows == 4

    def test_materializes_as_any_matching_archetype(self, rich_context: SystemContext) -> None:
        near = FilterByDistanceSystem.on(SOURCE, max_distance=10.0)
        Pipeline([near]).run(rich_context, TimeRange.everything())

        view = masked_view(
            rich_context.store,
            SOURCE,
            near.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )

        assert len(view.materialize(Detections3D)) == 2
        assert len(view.materialize(Trackings3D)) == 2

    def test_spans_several_frames(self, scene_store: Store) -> None:
        near = FilterByDistanceSystem.on(SOURCE, max_distance=40.0)
        ctx = SystemContext(scene_store, FRAME)
        Pipeline([near]).run(ctx, TimeRange.everything())

        view = masked_view(
            scene_store,
            SOURCE,
            near.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )

        assert len(view) == 3, "the far estimation in frame 0 is dropped"
        assert view.times(FRAME).tolist() == [0, 1, 1]

    def test_an_empty_source_yields_an_empty_view(self) -> None:
        store = Store()
        store.log(SOURCE, make_detections([]), at=TimePoint.at(frame=0))
        near = FilterByDistanceSystem.on(SOURCE)
        ctx = SystemContext(store, FRAME)
        Pipeline([near]).run(ctx, 0)

        view = masked_view(
            store,
            SOURCE,
            near.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )

        assert len(view) == 0

    def test_reports_a_mask_of_the_wrong_length(self, rich_context: SystemContext) -> None:
        other = "/other/objects"
        rich_context.store.log(other, make_detections([[0.0, 0.0, 0.0]]), at=TimePoint.at(frame=0))
        other_mask = FilterByDistanceSystem.on(other)
        Pipeline([other_mask]).run(rich_context, 0)

        with pytest.raises(ValueError, match=r"describes 1 row\(s\)"):
            masked_view(
                rich_context.store,
                SOURCE,
                other_mask.target,
                timeline=FRAME,
                time_range=TimeRange.everything(),
            )


class TestApplyMask:
    """Materializing a mask, so a metric can divide by the filtered ground-truth count."""

    def pipeline(self, source: str, **params: object) -> tuple[object, object]:
        near = FilterByDistanceSystem.on(source, max_distance=100.0)
        kept = ApplyMaskSystem.of(source, near.target, **params)
        return near, kept

    def test_writes_the_surviving_rows_to_a_new_entity(self, rich_context: SystemContext) -> None:
        near, kept = self.pipeline(SOURCE)

        Pipeline([near, kept]).run(rich_context, TimeRange.everything())

        view = rich_context.store.range(
            kept.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )
        assert len(view) == 3, "the object 120 m away is gone"
        assert view.component(POSITION).values[:, 0].tolist() == [1.0, 50.0, 3.0]

    def test_the_source_is_left_alone(self, rich_context: SystemContext) -> None:
        near, kept = self.pipeline(SOURCE)

        Pipeline([near, kept]).run(rich_context, TimeRange.everything())

        assert (
            len(rich_context.store.range(SOURCE, timeline=FRAME, time_range=TimeRange.everything()))
            == 4
        )

    def test_carries_every_column_across(self, rich_context: SystemContext) -> None:
        near, kept = self.pipeline(SOURCE)

        Pipeline([near, kept]).run(rich_context, TimeRange.everything())

        source = rich_context.store.range(
            SOURCE,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )
        result = rich_context.store.range(
            kept.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )
        assert set(result.descriptors) == set(source.descriptors)

    def test_defaults_to_a_child_of_the_source(self) -> None:
        _, kept = self.pipeline(SOURCE)

        assert str(kept.target) == f"{SOURCE}/kept"

    def test_the_target_can_be_named(self) -> None:
        _, kept = self.pipeline(SOURCE, target="/estimation/filtered")

        assert str(kept.target) == "/estimation/filtered"

    def test_keeps_the_partition_structure(self, scene_store: Store) -> None:
        near, kept = self.pipeline(SOURCE)
        ctx = SystemContext(scene_store, FRAME)

        Pipeline([near, kept]).run(ctx, TimeRange.everything())

        chunk = scene_store.range(
            kept.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).to_chunk()
        assert chunk.num_partitions == 2
        assert chunk.frame_id == "base_link"

    def test_an_empty_source_stays_empty(self) -> None:
        store = Store()
        store.log(SOURCE, make_detections([]), at=TimePoint.at(frame=0))
        near, kept = self.pipeline(SOURCE)

        Pipeline([near, kept]).run(SystemContext(store, FRAME), 0)

        assert len(store.range(kept.target, timeline=FRAME, time_range=TimeRange.everything())) == 0

    def test_needs_exactly_two_sources(self) -> None:
        with pytest.raises(ValueError, match="needs exactly two sources"):
            ApplyMaskSystem((SOURCE,), "/out")

    def test_applying_before_the_filter_runs_is_reported(self) -> None:
        near = FilterByDistanceSystem.on(SOURCE)
        kept = ApplyMaskSystem.of(SOURCE, near.target)

        with pytest.raises(ValueError, match="before a later system writes it"):
            Pipeline([kept, near])


class TestPipelineIntegration:
    def test_filters_chain_into_a_matcher(self, scene_store: Store, labels: LabelRegistry) -> None:
        from t4perceval.archetype import MatchResults
        from t4perceval.system import CenterDistanceMatchingSystem

        near = FilterByDistanceSystem.on(SOURCE, max_distance=40.0)
        confident = FilterByConfidenceSystem.on(SOURCE, min_confidence=0.5)
        keep = CombineMasksSystem.of(
            [near.target, confident.target],
            f"{SOURCE}/filter/keep",
        )
        matcher = CenterDistanceMatchingSystem.between(
            SOURCE,
            "/ground_truth/objects",
            threshold=1.0,
        )

        pipeline = Pipeline([near, confident, keep, matcher])
        ctx = SystemContext(scene_store, FRAME, labels=labels)
        produced = pipeline.run(ctx, TimeRange.everything())

        assert len(produced) == 4
        assert {str(chunk.entity_path) for chunk in produced} == {
            f"{SOURCE}/filter/distance",
            f"{SOURCE}/filter/confidence",
            f"{SOURCE}/filter/keep",
            "/matching/center_distance",
        }

        combined = scene_store.range(
            keep.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).component(MASK)
        assert combined.values.tolist() == [True, False, True, True]

        result = scene_store.range(
            "/matching/center_distance",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MatchResults)
        assert (result.num_tp, result.num_fp, result.num_fn) == (2, 2, 1)

    def test_combining_before_the_filters_run_is_reported(self) -> None:
        near = FilterByDistanceSystem.on(SOURCE)
        combined = CombineMasksSystem.of([near.target], "/keep")

        with pytest.raises(ValueError, match="before a later system writes it"):
            Pipeline([combined, near])


class TestApplyMaskCarriesStatic:
    """A passthrough target's static data is its source's, resolved on every read."""

    @staticmethod
    def _apply(store: Store) -> tuple[ApplyMaskSystem, SystemContext]:
        near = FilterByDistanceSystem.on(SOURCE, max_distance=10.0)
        kept = ApplyMaskSystem.of(SOURCE, near.target)
        ctx = SystemContext(store, FRAME)
        Pipeline([near, kept]).run(ctx, TimeRange.everything())
        return kept, ctx

    def test_static_columns_show_through_with_their_frame(self, rich_store: Store) -> None:
        from t4perceval.component import BatchConfidence, BatchTimeOffset
        from t4perceval.descriptors import CONFIDENCE, TIME_OFFSET

        rich_store.log_static_components(SOURCE, {TIME_OFFSET: BatchTimeOffset([[0, 100]])})
        rich_store.log_static_components(
            SOURCE, {CONFIDENCE: BatchConfidence([0.5])}, frame_id="map"
        )
        kept, _ = self._apply(rich_store)

        out = rich_store.range(kept.target, timeline=FRAME, time_range=TimeRange.everything())
        assert len(out) == 2
        assert out.component(TIME_OFFSET).values.tolist() == [[0, 100], [0, 100]]
        assert out.component(CONFIDENCE).values.tolist() == [0.5, 0.5], (
            "static wins, as at the source"
        )
        assert rich_store.static_frame_id(kept.target) == "map"

    def test_every_later_write_shows_through_in_order(self, rich_store: Store) -> None:
        from t4perceval.component import BatchConfidence
        from t4perceval.descriptors import CONFIDENCE

        kept, _ = self._apply(rich_store)
        shared = BatchConfidence([0.5])
        # A, B, then the very same A object again: three writes, the last wins -- and a
        # reused component object is still a new write.
        for column in (shared, BatchConfidence([0.9]), shared):
            rich_store.log_static_components(SOURCE, {CONFIDENCE: column})
            assert rich_store.static(kept.target)[CONFIDENCE] is column, "no rerun needed"

        assert [c.columns[CONFIDENCE].values[0] for c in rich_store.static_chunks(kept.target)] == [
            0.5,
            0.9,
            0.5,
        ]

    def test_the_targets_own_static_writes_win(self, rich_store: Store) -> None:
        from t4perceval.component import BatchConfidence
        from t4perceval.descriptors import CONFIDENCE

        near = FilterByDistanceSystem.on(SOURCE, max_distance=10.0)
        kept = ApplyMaskSystem.of(SOURCE, near.target)
        rich_store.log_static_components(kept.target, {CONFIDENCE: BatchConfidence([0.1])})
        rich_store.log_static_components(SOURCE, {CONFIDENCE: BatchConfidence([0.9])})
        Pipeline([near, kept]).run(SystemContext(rich_store, FRAME), TimeRange.everything())

        assert rich_store.static(kept.target)[CONFIDENCE].values.tolist() == [0.1]
        assert rich_store.static(SOURCE)[CONFIDENCE].values.tolist() == [0.9]

    def test_masking_in_place_neither_inherits_nor_raises(self, rich_store: Store) -> None:
        from t4perceval.component import BatchTimeOffset
        from t4perceval.descriptors import TIME_OFFSET

        rich_store.log_static_components(SOURCE, {TIME_OFFSET: BatchTimeOffset([[0, 100]])})
        near = FilterByDistanceSystem.on(SOURCE, max_distance=10.0)
        kept = ApplyMaskSystem.of(SOURCE, near.target, target=SOURCE)
        ctx = SystemContext(rich_store, FRAME)
        # A pipeline refuses this wiring (the filter would read what a later system
        # writes), so in-place masking only ever happens by calling the system directly.
        Pipeline([near]).run(ctx, TimeRange.single(0))

        (chunk,) = kept(ctx, 0)

        assert chunk.num_rows == 2 and str(chunk.entity_path) == SOURCE
        assert rich_store.static(SOURCE)[TIME_OFFSET].values.tolist() == [[0, 100]]
        assert len(rich_store.static_chunks(SOURCE)) == 1

    def test_a_second_run_changes_nothing(self, rich_store: Store) -> None:
        from t4perceval.component import BatchTimeOffset
        from t4perceval.descriptors import TIME_OFFSET

        rich_store.log_static_components(SOURCE, {TIME_OFFSET: BatchTimeOffset([[0, 100]])})
        kept, ctx = self._apply(rich_store)
        Pipeline([kept]).run(ctx, TimeRange.single(0))

        assert len(rich_store.static_chunks(kept.target)) == 1

    def test_a_multi_row_static_column_does_not_disturb_a_mask(self, rich_store: Store) -> None:
        # Three static rows against four temporal ones: nothing here asks for that column,
        # so the mask must not trip over it.
        from t4perceval.component import BatchTimeOffset
        from t4perceval.descriptors import TIME_OFFSET

        rich_store.log_static_components(
            SOURCE, {TIME_OFFSET: BatchTimeOffset([[0, 1], [0, 2], [0, 3]])}
        )
        near = FilterByDistanceSystem.on(SOURCE, max_distance=10.0)

        (chunk,) = near(SystemContext(rich_store, FRAME), 0)

        assert chunk.columns[MASK].values.tolist() == [True, False, False, True]


class TestFilterByMap:
    """Membership of a polygon in ``map``, whichever frame the source is in."""

    SQUARE = [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]

    @staticmethod
    def _map_store(positions: list[list[float]]) -> SystemContext:
        """One frame of objects already in ``map``; no ``/tf`` anywhere."""
        store = Store()
        store.log(SOURCE, make_detections(positions), at=TimePoint.at(frame=0), frame_id="map")
        return SystemContext(store, FRAME)

    #: rich_store's positions, so the expected masks read the same as the other filters'.
    POSITIONS = [[1.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 120.0, 0.0], [3.0, 4.0, 0.0]]

    def test_tests_xy_against_the_polygon(self) -> None:
        system = FilterByMapSystem.on(SOURCE, polygon=self.SQUARE)

        assert mask_of(system, self._map_store(self.POSITIONS)) == [True, False, False, True]

    def test_accepts_a_shapely_geometry(self) -> None:
        from shapely.geometry import Polygon

        system = FilterByMapSystem.on(SOURCE, polygon=Polygon(self.SQUARE))

        assert mask_of(system, self._map_store(self.POSITIONS)) == [True, False, False, True]

    def test_several_rings_are_a_union(self) -> None:
        far = [[45.0, -1.0], [55.0, -1.0], [55.0, 1.0], [45.0, 1.0]]
        system = FilterByMapSystem.on(SOURCE, polygon=[self.SQUARE, far])

        assert mask_of(system, self._map_store(self.POSITIONS)) == [True, True, False, True]

    def test_rejects_a_non_polygon(self) -> None:
        from shapely.geometry import LineString

        with pytest.raises(TypeError, match="Polygon or MultiPolygon"):
            FilterByMapSystem.on(SOURCE, polygon=LineString([(0, 0), (1, 1)]))

    def test_a_source_in_map_needs_no_lookup(self) -> None:
        # No /tf in the store at all, and none is asked for.
        system = FilterByMapSystem.on(SOURCE, polygon=self.SQUARE)

        assert mask_of(system, self._map_store([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])) == [True, False]

    def test_no_polygon_needs_no_lookup(self, rich_context: SystemContext) -> None:
        # rich_store is in base_link with no /tf; nothing is looked up when nothing is tested.
        system = FilterByMapSystem.on(SOURCE)

        assert mask_of(system, rich_context) == [True] * 4

    @staticmethod
    def _ego_store() -> Store:
        """Ego at x = 0, 10, 20; one object 1 m ahead and 1 m left of it in every frame."""
        store = make_ego_scene(Store())
        for frame in range(3):
            store.log(
                SOURCE,
                make_detections([[1.0, 1.0, 0.0]]),
                at=TimePoint.at(frame=frame, timestamp_ns=(frame + 1) * 1_000),
                frame_id="base_link",
            )
        return store

    #: In map the object is at x = 1, 11, 21 on y = 1; this keeps x in [8, 15]: frame 1 only.
    MIDDLE = [[8.0, -2.0], [15.0, -2.0], [15.0, 2.0], [8.0, 2.0]]

    def test_a_base_link_source_follows_the_ego(self) -> None:
        store = self._ego_store()
        system = FilterByMapSystem.on(SOURCE, polygon=self.MIDDLE)
        ctx = SystemContext(store, FRAME)

        assert [mask_of(system, ctx, at=frame) for frame in range(3)] == [[False], [True], [False]]

    def test_the_mask_lands_under_the_source(self) -> None:
        system = FilterByMapSystem.on(SOURCE, polygon=self.MIDDLE)

        assert str(system.target) == f"{SOURCE}/filter/map"

    def test_a_range_covers_every_partition_with_its_own_pose(self) -> None:
        store = self._ego_store()
        system = FilterByMapSystem.on(SOURCE, polygon=self.MIDDLE)

        (chunk,) = system(SystemContext(store, FRAME), TimeRange.everything())

        assert chunk.columns[MASK].values.tolist() == [False, True, False]
        assert chunk.num_partitions == 3

    def test_an_explicit_resolver_may_come_from_another_store(self) -> None:
        from t4perceval.transform import TransformResolver

        store = self._ego_store()
        objects_only = Store()
        for frame in range(3):
            objects_only.log(
                SOURCE,
                make_detections([[1.0, 1.0, 0.0]]),
                at=TimePoint.at(frame=frame),
                frame_id="base_link",
            )
        system = FilterByMapSystem.on(
            SOURCE, polygon=self.MIDDLE, resolver=TransformResolver.of(store)
        )

        assert mask_of(system, SystemContext(objects_only, FRAME), at=1) == [True]

    def test_a_source_without_a_frame_is_refused(self) -> None:
        store = make_ego_scene(Store())
        store.log(SOURCE, make_detections([[1.0, 0.0, 0.0]]), at=TimePoint.at(frame=0))
        system = FilterByMapSystem.on(SOURCE, polygon=self.SQUARE)

        with pytest.raises(ValueError, match="states no coordinate frame"):
            system(SystemContext(store, FRAME), 0)

    def test_a_missing_ego_pose_names_the_entity_and_time(
        self, rich_context: SystemContext
    ) -> None:
        # rich_store is in base_link and carries no /tf: there is no way into map.
        system = FilterByMapSystem.on(SOURCE, polygon=self.SQUARE)

        with pytest.raises(
            ValueError,
            match="cannot bring /estimation/objects from 'base_link' into 'map' at frame=0",
        ):
            mask_of(system, rich_context)

    def test_a_frame_with_no_objects_needs_no_pose(self) -> None:
        store = make_ego_scene(Store(), xs=(0.0,))
        store.log(SOURCE, make_detections([]), at=TimePoint.at(frame=0), frame_id="base_link")
        store.log(
            SOURCE,
            make_detections([[1.0, 0.0, 0.0]]),
            at=TimePoint.at(frame=5),
            frame_id="base_link",
        )
        system = FilterByMapSystem.on(SOURCE, polygon=self.SQUARE)

        # Frame 5 has no ego pose, but the lookup policy is LATEST, so frame 0's serves.
        assert mask_of(system, SystemContext(store, FRAME), at=0) == []
        assert mask_of(system, SystemContext(store, FRAME), at=5) == [True]

    def test_on_lanelet_defaults_to_where_traffic_is(self, tmp_path) -> None:  # noqa: ANN001
        from t4perceval.lanelet import LaneletMap
        from tests.lanelet_builder import write_fixture_map

        lanelet_map = LaneletMap.load(write_fixture_map(tmp_path / "lanelet2_map.osm"))
        store = self._ego_store()  # object at map x = 1, 11, 21 on y = 1

        on_road = FilterByMapSystem.on_lanelet(SOURCE, lanelet_map)

        # Road (y 0..2) and crosswalk (y 2..4) are both in; only x <= 10 lies on the map.
        ctx = SystemContext(store, FRAME)
        assert [mask_of(on_road, ctx, at=frame) for frame in range(3)] == [[True], [False], [False]]
        assert on_road.polygon.area == pytest.approx(40.0)

    def test_on_lanelet_builds_the_region_and_names_the_mask(self, tmp_path) -> None:  # noqa: ANN001
        from t4perceval.lanelet import LaneletMap
        from tests.lanelet_builder import write_fixture_map

        lanelet_map = LaneletMap.load(write_fixture_map(tmp_path / "lanelet2_map.osm"))
        store = self._ego_store()  # object at map x = 1, 11, 21 on y = 1
        on_road = FilterByMapSystem.on_lanelet(SOURCE, lanelet_map, subtypes=("road",))

        assert str(on_road.target) == f"{SOURCE}/filter/lanelet"
        ctx = SystemContext(store, FRAME)
        # Road 1 spans x 0..10, y 0..2.
        assert [mask_of(on_road, ctx, at=frame) for frame in range(3)] == [[True], [False], [False]]

    def test_map_and_range_masks_combine_on_the_source(self) -> None:
        store = self._ego_store()
        on_map = FilterByMapSystem.on(SOURCE, polygon=self.MIDDLE)
        near = FilterByDistanceSystem.on(SOURCE, max_distance=2.0)
        both = CombineMasksSystem.of([on_map.target, near.target], f"{SOURCE}/filter/keep")

        Pipeline([on_map, near, both]).run(SystemContext(store, FRAME), TimeRange.everything())

        mask = store.range(both.target, timeline=FRAME, time_range=TimeRange.everything())
        assert mask.component(MASK).values.tolist() == [False, True, False]

    def test_lanelet_then_apply_then_match_in_one_pipeline(
        self, tmp_path, labels: LabelRegistry
    ) -> None:  # noqa: ANN001
        from t4perceval.archetype import MatchResults
        from t4perceval.lanelet import LaneletMap
        from t4perceval.system import (
            ApplyMaskSystem,
            CenterDistanceMatchingSystem,
            TransformEntitySystem,
        )
        from tests.lanelet_builder import write_fixture_map

        lanelet_map = LaneletMap.load(write_fixture_map(tmp_path / "lanelet2_map.osm"))
        store = self._ego_store()
        gt = "/ground_truth/objects"
        for frame in range(3):
            store.log(
                gt,
                make_detections([[1.0 + 10.0 * frame, 1.0, 0.0]]),
                at=TimePoint.at(frame=frame, timestamp_ns=(frame + 1) * 1_000),
                frame_id="map",
            )

        gt_on_road = FilterByMapSystem.on_lanelet(gt, lanelet_map, subtypes=("road",))
        gt_kept = ApplyMaskSystem.of(gt, gt_on_road.target)
        est_on_road = FilterByMapSystem.on_lanelet(SOURCE, lanelet_map, subtypes=("road",))
        est_kept = ApplyMaskSystem.of(SOURCE, est_on_road.target)
        est_in_map = TransformEntitySystem.of(est_kept.target, target_frame="map")
        matcher = CenterDistanceMatchingSystem.between(
            est_in_map.target, gt_kept.target, threshold=0.5
        )

        Pipeline([gt_on_road, gt_kept, est_on_road, est_kept, est_in_map, matcher]).run(
            SystemContext(store, FRAME, labels=labels), TimeRange.everything()
        )

        matches = store.range(matcher.target, timeline=FRAME, time_range=TimeRange.everything())
        result = matches.materialize(MatchResults)
        assert (result.num_tp, result.num_fp, result.num_fn) == (1, 0, 0)
