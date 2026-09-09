"""The ``.t4eval`` directory format: a whole recording on disk.

:mod:`t4perceval.io.arrow` saves one chunk losslessly, but a store is more than its chunks.
Three things belong to the store rather than to any chunk, and this format is what carries
them: the *per-entity* log order (which ``Store.latest_at`` uses to break equal-time ties),
which files belong to which entity, and the registries -- ``InstanceRegistry`` has no other
serialization at all.

.. code-block:: text

    result.t4eval/
    ├── manifest.json     format version, provenance, both registries, chunks in log order
    └── chunks/
        ├── 000000.parquet
        └── ...

The manifest is the single authority: chunk files carry no label registry, and entity paths
are never used as filenames. "Log order" here is per entity. A live store keeps the order
chunks were logged to each entity and the order entities were first seen, but no global
sequence across entities -- so that is neither recorded nor needed, and replaying the
manifest reproduces every observable ordering. See ``docs/development/serialization-format.md``
for the layout and ``docs/development/design-decisions/0004-persistent-recording.md`` for the
decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from attrs import evolve

from t4perceval.core.entity import as_entity_path
from t4perceval.core.store import Store
from t4perceval.io.arrow import read_parquet, write_parquet
from t4perceval.label import InstanceRegistry, LabelRegistry
from t4perceval.recording import Recording, RecordingMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from t4perceval.core.chunk import Chunk

__all__ = (
    "CHUNKS_DIRNAME",
    "MANIFEST_NAME",
    "RECORDING_FORMAT_VERSION",
    "read_recording",
    "write_recording",
)

#: Bumped whenever the manifest or the directory layout changes incompatibly. Independent
#: of the chunk :data:`~t4perceval.io.arrow.SCHEMA_VERSION`, which each file checks itself.
RECORDING_FORMAT_VERSION = 1

#: Name of the manifest inside a recording directory.
MANIFEST_NAME = "manifest.json"

#: Subdirectory holding one Parquet file per chunk.
CHUNKS_DIRNAME = "chunks"

_FORMAT = "t4eval"
_CHUNK_SUFFIX = ".parquet"


def write_recording(recording: Recording, path: str | Path, *, exist_ok: bool = False) -> Path:
    """Write a recording as a ``.t4eval`` directory and return that directory.

    Chunks are written first and the manifest last, so an interrupted write leaves a
    directory the reader refuses rather than a manifest pointing at missing files. Only the
    recording's read surface is used, and the recording itself is not modified: the format
    version is stamped into the *written* metadata, the way
    :meth:`~t4perceval.recording.Recording.of` stamps the label fingerprint.

    Args:
        recording: The recording to write.
        path: The directory to create. ``.t4eval`` is a naming convention, not a
            requirement; the suffix is neither added nor checked.
        exist_ok: Write into an existing directory, overwriting ``manifest.json`` and the
            chunk files this write produces. Files it did not write are left alone -- and
            left unreferenced, since the manifest is the index of what belongs to the
            recording.

    Returns:
        The recording directory, so that ``read_recording(write_recording(...))`` composes.

    Raises:
        FileExistsError: When ``path`` exists and ``exist_ok`` is false.
        ValueError: When ``path`` exists and is not a directory.
    """
    root = Path(path)
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"{root} exists and is not a directory")
        if not exist_ok:
            raise FileExistsError(f"{root} already exists; pass exist_ok=True to write into it")
    (root / CHUNKS_DIRNAME).mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for index, chunk in enumerate(_chunks_in_log_order(recording)):
        relative = f"{CHUNKS_DIRNAME}/{index:06d}{_CHUNK_SUFFIX}"
        write_parquet(chunk, root / relative)
        entries.append(
            {
                "id": index,
                "entity_path": str(chunk.entity_path),
                "file": relative,
                "is_static": chunk.is_static,
            },
        )

    metadata = evolve(recording.metadata, format_version=RECORDING_FORMAT_VERSION)
    manifest = {
        "format": _FORMAT,
        "format_version": RECORDING_FORMAT_VERSION,
        "metadata": metadata.to_json(),
        "labels": recording.labels.to_metadata(),
        "instances": recording.instances.to_metadata(),
        "chunks": entries,
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


def read_recording(path: str | Path) -> Recording:
    """Read a recording written by :func:`write_recording`.

    The store is rebuilt by replaying the manifest's chunks in order, so every query --
    ``latest_at`` tie-breaking included -- answers exactly as it did before the write.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        ValueError: When the directory is not a recording, was written by a format version
            this build does not read, or disagrees with its own manifest.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"No such recording: {root}")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"{root} is not a t4perceval recording: no {MANIFEST_NAME}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Recording manifest {manifest_path} is not valid JSON: {error}",
        ) from error

    found = manifest.get("format") if isinstance(manifest, dict) else None
    if found != _FORMAT:
        raise ValueError(
            f"{manifest_path} is not a t4perceval recording manifest: "
            f"format is {found!r}, expected {_FORMAT!r}",
        )
    version = manifest.get("format_version")
    if isinstance(version, int) and version > RECORDING_FORMAT_VERSION:
        raise ValueError(
            f"Recording format version {version} is newer than this build supports "
            f"({RECORDING_FORMAT_VERSION}); upgrade t4perceval to read {root}",
        )
    if version != RECORDING_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported recording format version {version!r}, "
            f"expected {RECORDING_FORMAT_VERSION}",
        )

    labels = LabelRegistry.from_metadata(_require(manifest, "labels", manifest_path))
    instances = InstanceRegistry.from_metadata(_require(manifest, "instances", manifest_path))
    metadata = RecordingMetadata.from_json(_require(manifest, "metadata", manifest_path))
    entries = _require(manifest, "chunks", manifest_path)
    if not isinstance(entries, list):
        raise ValueError(f"Recording manifest {manifest_path} key 'chunks' must be a list")

    store = Store()
    for index, entry in enumerate(entries):
        store.send_chunk(_read_chunk(root, manifest_path, index, entry))

    # The constructor, not `Recording.of`: `of` stamps the label fingerprint into empty
    # metadata, and what was written must come back verbatim.
    return Recording(store=store, labels=labels, instances=instances, metadata=metadata)


def _chunks_in_log_order(recording: Recording) -> Iterator[Chunk]:
    """Yield every chunk in the order the reader must replay them.

    Every entity's temporal chunks first, then every entity's static chunks, each in its
    own log order. ``entity_paths()`` lists the entities that have temporal data first, in
    the order they were first logged, then the static-only ones -- so replaying in this
    order recreates both of the store's entity orders, and ``entity_paths()`` comes back
    identical.
    """
    paths = recording.entity_paths()
    for path in paths:
        yield from recording.chunks(path)
    for path in paths:
        yield from recording.static_chunks(path)


def _require(data: Any, key: str, manifest_path: Path) -> Any:
    if not isinstance(data, dict) or key not in data:
        raise ValueError(f"Recording manifest {manifest_path} is missing required key {key!r}")
    return data[key]


def _read_chunk(root: Path, manifest_path: Path, index: int, entry: Mapping[str, Any]) -> Chunk:
    """Read one manifest entry, checking that the file and the manifest agree about it."""
    found_id = _require(entry, "id", manifest_path)
    if found_id != index:
        raise ValueError(
            f"Recording manifest {manifest_path} lists chunk id {found_id!r} at position "
            f"{index}; chunk order is load-bearing",
        )
    relative = str(_require(entry, "file", manifest_path))
    expected_path = as_entity_path(str(_require(entry, "entity_path", manifest_path)))
    expect_static = bool(_require(entry, "is_static", manifest_path))

    file = root / relative
    if not file.resolve().is_relative_to(root.resolve()):
        raise ValueError(
            f"Recording manifest {manifest_path} lists chunk file {relative!r} outside the "
            "recording directory",
        )
    if not file.is_file():
        raise ValueError(
            f"Recording {root} is missing chunk file {relative!r} listed for entity "
            f"{str(expected_path)!r} in {MANIFEST_NAME}",
        )

    try:
        chunk, embedded = read_parquet(file)
    except (ValueError, KeyError) as error:
        # pyarrow's ArrowInvalid is a ValueError; the registry's unknown-type is a KeyError.
        raise ValueError(f"Chunk file {relative!r}: {error}") from error

    if embedded is not None:
        raise ValueError(
            f"Chunk file {relative!r} carries an embedded label registry; a recording's "
            f"registries live in {MANIFEST_NAME} only",
        )
    if chunk.entity_path != expected_path:
        raise ValueError(
            f"Chunk file {relative!r} holds entity path {str(chunk.entity_path)!r} but "
            f"{MANIFEST_NAME} lists it under {str(expected_path)!r}",
        )
    if chunk.is_static != expect_static:
        listed = "static" if expect_static else "temporal"
        raise ValueError(
            f"Chunk file {relative!r} is listed as {listed} in {MANIFEST_NAME} but the chunk "
            f"declares is_static={chunk.is_static}",
        )
    return chunk
