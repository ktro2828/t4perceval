"""Assembling recordings into a store an evaluation can run against.

:meth:`~t4perceval.system.base.Pipeline.run` writes its results back into
``ctx.store``, and :class:`~t4perceval.system.base.SystemContext` holds exactly one store.
So a read-only :class:`~t4perceval.recording.Recording` cannot be evaluated in place: the
entities an evaluation needs are materialized into a fresh, writable store first.

Only the named entities move. An importer that also recorded ego poses or sensor data does
not drag them in, so a persisted evaluation says what the evaluation actually used rather
than what happened to be loaded.

Moving an entity is close to free -- chunks are frozen and their arrays read-only, so the
new store references the same objects. Three things do *not* ride along with a chunk and
are handled explicitly here: static columns, per-path log order, and the registries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, evolve, field

from t4perceval.core.chunk import Chunk
from t4perceval.core.entity import as_entity_path
from t4perceval.core.store import Store
from t4perceval.core.timeline import FRAME
from t4perceval.descriptors import INSTANCE_ID
from t4perceval.reconcile import class_id_lut, remap_class_ids
from t4perceval.recording import RecordingMetadata
from t4perceval.system.base import SystemContext

if TYPE_CHECKING:
    from collections.abc import Sequence

    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.core.timeline import Timeline
    from t4perceval.label import InstanceRegistry, LabelRegistry
    from t4perceval.recording import Recording
    from t4perceval.system.base import System

__all__ = ("EvaluationSetup", "SourceSpec", "build_evaluation_store", "build_evaluation_store_from")


@define(frozen=True, slots=True)
class SourceSpec:
    """One recording's contribution to an evaluation store."""

    recording: Recording
    entity_path: EntityPath = field(converter=as_entity_path)
    """Where the data lives in the recording."""

    target_path: EntityPath = field(converter=as_entity_path)
    """Where it should live in the evaluation store."""

    role: str = field(default="", kw_only=True)
    """``"reference"`` or ``"query"``, informational."""

    @classmethod
    def of(
        cls,
        recording: Recording,
        entity_path: EntityPathLike,
        target_path: EntityPathLike | None = None,
        *,
        role: str = "",
    ) -> SourceSpec:
        """Build a spec, defaulting the target to the source path."""
        return cls(
            recording,
            entity_path,
            target_path if target_path is not None else entity_path,
            role=role,
        )


@define(frozen=True, slots=True)
class EvaluationSetup:
    """A writable store plus everything needed to interpret and run it."""

    store: Store
    labels: LabelRegistry
    instances: InstanceRegistry
    metadata: RecordingMetadata

    def context(self, timeline: Timeline = FRAME) -> SystemContext:
        """Return the context a pipeline runs against."""
        return SystemContext(
            self.store,
            timeline,
            labels=self.labels,
            instances=self.instances,
        )

    def into_recording(self, *, pipeline: Sequence[System] = ()) -> Recording:
        """Freeze the store once the pipeline has run.

        The result holds the inputs, the filter masks, the matching results and the
        metrics -- everything the evaluation touched, and nothing it did not.
        """
        from t4perceval.recording import Recording

        return Recording.of(
            self.store,
            labels=self.labels,
            instances=self.instances,
            metadata=evolve(
                self.metadata,
                pipeline=tuple(type(system).__name__ for system in pipeline),
            ),
        )


def build_evaluation_store_from(
    sources: Sequence[SourceSpec],
    *,
    reconcile: bool = False,
    require_same_frame_id: bool = True,
) -> EvaluationSetup:
    """Materialize several recordings into one evaluation store.

    Args:
        sources: What to move, and where it should land.
        reconcile: When the sources disagree about class ids, remap the later ones onto
            the first source's registry instead of raising.
        require_same_frame_id: Whether a coordinate-frame mismatch between sources is an
            error.

    Returns:
        A setup whose store holds exactly the named entities.

    Raises:
        ValueError: When the sources disagree about class ids, coordinate frames or
            instance registries, and the disagreement was not opted into.
    """
    if not sources:
        raise ValueError("build_evaluation_store_from() needs at least one source")

    labels = sources[0].recording.labels
    luts = {
        index: _lut_for(spec, labels, reconcile=reconcile) for index, spec in enumerate(sources)
    }
    instances = _shared_instances(sources)
    _check_frame_ids(sources, require_same_frame_id=require_same_frame_id)

    store = Store()
    for index, spec in enumerate(sources):
        lut = luts[index]

        # Static first: it is merged into the store rather than carried by a chunk, and a
        # static column takes precedence over a temporal one with the same descriptor --
        # so a forgotten one changes results instead of raising.
        static = spec.recording.static(spec.entity_path)
        if static:
            store.send_chunk(Chunk.from_columns(spec.target_path, static, is_static=True))

        # Log order matters, but only within one entity path: `latest_at` prefers the most
        # recently logged chunk on a tie and `range` sorts by (time, log order). Sources
        # land on different paths, so preserving each source's own order is enough.
        for chunk in spec.recording.chunks(spec.entity_path):
            moved = evolve(chunk, entity_path=spec.target_path)
            store.send_chunk(moved if lut is None else remap_class_ids(moved, lut))

    return EvaluationSetup(
        store,
        labels,
        instances,
        RecordingMetadata(
            t4perceval_version=sources[0].recording.metadata.t4perceval_version,
            sources=tuple(source for spec in sources for source in spec.recording.metadata.sources),
            labels_fingerprint=labels.fingerprint(),
            frame_id=sources[0].recording.metadata.frame_id,
        ),
    )


def build_evaluation_store(
    reference: Recording,
    query: Recording,
    *,
    reference_path: EntityPathLike = "/ground_truth/objects",
    query_path: EntityPathLike = "/estimation/objects",
    reference_target: EntityPathLike | None = None,
    query_target: EntityPathLike | None = None,
    reconcile: bool = False,
    require_same_frame_id: bool = True,
) -> EvaluationSetup:
    """Materialize a ground-truth and an estimation recording into one store.

    Sugar over :func:`build_evaluation_store_from` for the two-source case.
    """
    return build_evaluation_store_from(
        (
            SourceSpec.of(reference, reference_path, reference_target, role="reference"),
            SourceSpec.of(query, query_path, query_target, role="query"),
        ),
        reconcile=reconcile,
        require_same_frame_id=require_same_frame_id,
    )


def _lut_for(
    spec: SourceSpec,
    labels: LabelRegistry,
    *,
    reconcile: bool,
) -> object | None:
    """Return the class-id remapping this source needs, or ``None`` when it agrees."""
    source_labels = spec.recording.labels
    if source_labels.fingerprint() == labels.fingerprint():
        return None

    if not reconcile:
        raise ValueError(
            f"Sources disagree about class ids.\n"
            f"  target registry: names={list(labels.names)} "
            f"fingerprint={labels.fingerprint()}\n"
            f"  {spec.entity_path}: names={list(source_labels.names)} "
            f"fingerprint={source_labels.fingerprint()}\n"
            f"Pass the same LabelRegistry to both importers, or set reconcile=True to "
            f"remap this source onto the target registry.",
        )
    return class_id_lut(source_labels, labels)


def _shared_instances(sources: Sequence[SourceSpec]) -> InstanceRegistry:
    """Return the instance registry the evaluation should use.

    Two registries only conflict if instance ids are actually being moved: detections
    carry none, so requiring a shared registry for them would reject a perfectly
    well-defined evaluation. Where ids *are* present, merging is not offered -- they are
    already encoded into the columns, so renumbering would silently corrupt them.
    """
    carriers = {
        id(spec.recording.instances): spec.recording.instances
        for spec in sources
        if any(chunk.has(INSTANCE_ID) for chunk in spec.recording.chunks(spec.entity_path))
    }
    if len(carriers) > 1:
        raise ValueError(
            "Sources carrying instance ids were imported with different InstanceRegistry "
            "objects. Those ids are already encoded into the columns, so they cannot be "
            "merged after the fact -- pass one shared InstanceRegistry to every importer.",
        )
    if carriers:
        return next(iter(carriers.values()))
    return sources[0].recording.instances


def _check_frame_ids(sources: Sequence[SourceSpec], *, require_same_frame_id: bool) -> None:
    """Reject sources whose data is in different coordinate frames.

    Nothing downstream would complain: a matching system takes whichever frame id it finds
    first, so ground truth in ``base_link`` against estimations in ``map`` yields
    geometrically meaningless distances and a plausible near-zero score.
    """
    if not require_same_frame_id:
        return

    frames = {
        str(chunk.frame_id): str(spec.entity_path)
        for spec in sources
        for chunk in spec.recording.chunks(spec.entity_path)
        if chunk.frame_id is not None
    }
    if len(frames) > 1:
        detail = ", ".join(f"{path} in {frame!r}" for frame, path in sorted(frames.items()))
        raise ValueError(
            f"Sources are in different coordinate frames: {detail}. Distances between "
            f"them would be meaningless. Re-import into one frame, or pass "
            f"require_same_frame_id=False if the frames are known to coincide.",
        )
