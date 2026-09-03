"""Resolving a recorded frame graph into one transform.

This is the interpretation step the store deliberately does not take. A store holds
transform *facts*; a resolver walks them, inverts and composes, and answers "where is this
frame relative to that one, at this time".

It is not a :class:`~t4perceval.system.base.System`: a system returns chunks for a
pipeline to file, whereas a lookup answers a question and writes nothing. Materializing a
*transformed entity* is the system-shaped job, and it is still blocked on a separate
problem -- a passthrough system cannot declare the columns it carries.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from attrs import define, field

from t4perceval.archetype.transform import Transform3D
from t4perceval.core.timeline import FRAME
from t4perceval.descriptors import CHILD_FRAME_ID, ROTATION, TRANSLATION
from t4perceval.transform.compose import chain, identity, interpolate, invert
from t4perceval.transform.graph import DEFAULT_ROOT, FrameGraph

if TYPE_CHECKING:
    from typing_extensions import Self

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.store import Store
    from t4perceval.core.timeline import Timeline
    from t4perceval.core.view import EntityView
    from t4perceval.recording import Recording
    from t4perceval.transform.compose import Pose
    from t4perceval.transform.graph import TransformEdge

__all__ = ("LookupPolicy", "TransformResolver")


class LookupPolicy(Enum):
    """How a temporal edge picks the sample to answer with."""

    LATEST = auto()
    """The most recent sample at or before the requested time."""

    EXACT = auto()
    """The sample at exactly the requested time; anything else is an error."""

    NEAREST = auto()
    """The closest sample in either direction."""

    INTERPOLATE = auto()
    """Between the two samples bracketing the time: linear translation, ``Slerp`` rotation."""


def _pose_of(view_or_chunk: EntityView | Chunk, child: str, *, where: str) -> Pose:
    """Return the pose ``view_or_chunk`` records for ``child``.

    Reads the storage columns rather than materializing a
    :class:`~t4perceval.archetype.transform.Transform3D`, because a query may legitimately
    span several samples -- and because the row wanted is the one naming ``child``, not
    simply the first.
    """
    translation = view_or_chunk.component(TRANSLATION)
    rotation = view_or_chunk.component(ROTATION)
    frames = view_or_chunk.component(CHILD_FRAME_ID)
    if translation is None or rotation is None or frames is None:
        raise ValueError(f"{where} holds no transform for child frame {child!r}")

    rows = frames.matching(child).tolist()
    if not rows:
        raise ValueError(f"{where} holds no transform for child frame {child!r}")
    if len(rows) > 1:
        raise ValueError(
            f"{where} holds {len(rows)} transforms for child frame {child!r}; a frame "
            f"cannot be in two places at one time",
        )
    row = int(rows[0])
    return translation.values[row], rotation.values[row]


@define(frozen=True, slots=True)
class TransformResolver:
    """Answers frame-to-frame transform queries against recorded data.

    Static and temporal edges take part in one graph: a calibration recorded once and an
    ego pose recorded per frame compose exactly as ROS composes a latched and a live
    transform.

    Examples:
        >>> resolver = TransformResolver.of(recording)  # doctest: +SKIP
        >>> resolver.lookup(target_frame="map", source_frame="lidar", at=3)  # doctest: +SKIP
    """

    graph: FrameGraph
    source: Store | Recording
    timeline: Timeline = field(default=FRAME, kw_only=True)
    policy: LookupPolicy = field(default=LookupPolicy.LATEST, kw_only=True)

    @classmethod
    def of(
        cls,
        source: Store | Recording,
        *,
        timeline: Timeline = FRAME,
        policy: LookupPolicy = LookupPolicy.LATEST,
        root: EntityPathLike | None = DEFAULT_ROOT,
    ) -> Self:
        """Build a resolver over every transform ``source`` holds."""
        return cls(
            FrameGraph.of(source, root=root),
            source,
            timeline=timeline,
            policy=policy,
        )

    def lookup(
        self,
        *,
        target_frame: str,
        source_frame: str,
        at: int | None = None,
    ) -> Transform3D:
        """Return the pose of ``source_frame`` expressed in ``target_frame``.

        Args:
            target_frame: Frame to express the result in -- the parent of the answer.
            source_frame: Frame whose pose is wanted.
            at: Time on this resolver's timeline. Omit it only when every edge on the
                chain is static.

        Returns:
            A :class:`~t4perceval.archetype.transform.Transform3D` whose ``child_frame_id``
            is ``source_frame``; its parent is ``target_frame``, which the caller states
            when logging it.

        Raises:
            ValueError: When a frame is unknown, when no chain connects the two, when a
                temporal edge is on the chain and ``at`` was not given, or when the policy
                cannot be satisfied.

        Note:
            Composing ``map -> base_link`` (temporal) with ``base_link -> lidar`` (static)
            gives ``T_map_lidar(t) = T_map_base_link(t) @ T_base_link_lidar``.
        """
        hops = self.graph.path(target_frame=target_frame, source_frame=source_frame)
        poses = [self._hop_pose(edge, inverted, at=at) for edge, inverted in hops]
        translation, rotation = chain(poses) if poses else identity()
        return Transform3D(
            translation=translation,
            rotation=rotation,
            child_frame_id=source_frame,
        )

    # -- reading one edge ---------------------------------------------------------------

    def _hop_pose(self, edge: TransformEdge, inverted: bool, *, at: int | None) -> Pose:
        pose = self._edge_pose(edge, at=at)
        return invert(pose) if inverted else pose

    def _edge_pose(self, edge: TransformEdge, *, at: int | None) -> Pose:
        if edge.is_static:
            # A static edge has no time axis, so no policy applies to it. Interpolating
            # something that never changes is not an error, it is a question with one
            # answer.
            return self._static_pose(edge)
        if at is None:
            raise ValueError(
                f"{edge.parent!r} -> {edge.child!r} is recorded over time, so looking it "
                f"up needs a time on the {self.timeline.name!r} timeline",
            )
        return self._temporal_pose(edge, at=at)

    def _static_pose(self, edge: TransformEdge) -> Pose:
        where = f"Static data of {edge.entity_path}"
        for chunk in self.source.static_chunks(edge.entity_path):
            column = chunk.columns.get(CHILD_FRAME_ID)
            if column is not None and column.matching(edge.child).size:
                return _pose_of(chunk, edge.child, where=where)
        raise ValueError(f"{where} holds no transform for child frame {edge.child!r}")

    def _temporal_pose(self, edge: TransformEdge, *, at: int) -> Pose:
        if self.policy is LookupPolicy.INTERPOLATE:
            return self._interpolated_pose(edge, at=at)
        return self._sampled_pose(edge, at=self._time_for(edge, at=at))

    def _sampled_pose(self, edge: TransformEdge, *, at: int) -> Pose:
        view = self.source.latest_at(edge.entity_path, timeline=self.timeline, at=at)
        return _pose_of(view, edge.child, where=f"{edge.entity_path} at {at}")

    def _time_for(self, edge: TransformEdge, *, at: int) -> int:
        """Return the recorded time this policy answers ``at`` with."""
        if self.policy is LookupPolicy.LATEST:
            return at

        times = self.source.times(edge.entity_path, self.timeline)
        if times.size == 0:
            raise ValueError(
                f"{edge.entity_path} has no samples on the {self.timeline.name!r} timeline",
            )
        if self.policy is LookupPolicy.EXACT:
            if at not in times:
                nearest = int(times[int(np.argmin(np.abs(times - at)))])
                raise ValueError(
                    f"{edge.entity_path} has no sample at {self.timeline.name}={at}; the "
                    f"nearest is {nearest}. Use LATEST, NEAREST or INTERPOLATE to accept "
                    f"a sample from another time.",
                )
            return at
        return int(times[int(np.argmin(np.abs(times - at)))])

    def _interpolated_pose(self, edge: TransformEdge, *, at: int) -> Pose:
        times = self.source.times(edge.entity_path, self.timeline)
        if times.size == 0:
            raise ValueError(
                f"{edge.entity_path} has no samples on the {self.timeline.name!r} timeline",
            )
        # Outside the recorded span there is nothing to interpolate between, and guessing
        # beyond the last sample would invent motion: hold the nearest end instead.
        if at <= int(times[0]) or at >= int(times[-1]) or at in times:
            return self._sampled_pose(edge, at=int(times[np.argmin(np.abs(times - at))]))

        index = int(np.searchsorted(times, at))
        before, after = int(times[index - 1]), int(times[index])
        fraction = (at - before) / (after - before)
        return interpolate(
            self._sampled_pose(edge, at=before),
            self._sampled_pose(edge, at=after),
            fraction=fraction,
        )
