"""Arrow and Parquet encoding for :class:`~t4perceval.core.chunk.Chunk`.

One component becomes one Arrow field, named by its descriptor. Everything that is not
row-shaped -- the entity path, the coordinate frame, the timeline index values, the
partition offsets, an optional label registry -- goes into the schema metadata, because it
has a different length from the columns and so cannot be a column itself.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from t4perceval.core.chunk import Chunk
from t4perceval.core.descriptor import ComponentDescriptor
from t4perceval.core.timeline import TimeColumn, TimeKind, Timeline
from t4perceval.io.registry import resolve_component_type
from t4perceval.label import LabelRegistry

if TYPE_CHECKING:
    from pathlib import Path

    from t4perceval.core.component import Component

__all__ = (
    "METADATA_KEY",
    "SCHEMA_VERSION",
    "chunk_from_table",
    "chunk_to_table",
    "read_parquet",
    "write_parquet",
)

#: Schema-metadata key holding the chunk description.
METADATA_KEY = b"t4perceval"

#: Bumped whenever the metadata layout changes incompatibly.
SCHEMA_VERSION = 1


def chunk_to_table(chunk: Chunk, *, labels: LabelRegistry | None = None) -> pa.Table:
    """Encode a chunk as an Arrow table.

    Args:
        chunk: The chunk to encode.
        labels: Optional label registry to carry alongside the data.
    """
    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    column_meta: list[dict[str, Any]] = []

    for descriptor, column in chunk.columns.items():
        array = column.to_arrow()
        arrays.append(array)
        # Components never contain nulls; pinning that in the schema keeps readers honest.
        fields.append(pa.field(descriptor.component, array.type, nullable=False))
        column_meta.append(
            {
                "component": descriptor.component,
                "archetype": descriptor.archetype,
                "component_type": type(column).__name__,
                "row_shape": list(column.row_shape),
            },
        )

    metadata = {
        "version": SCHEMA_VERSION,
        "entity_path": str(chunk.entity_path),
        "frame_id": chunk.frame_id,
        "is_static": chunk.is_static,
        "offsets": chunk.offsets.tolist(),
        "indexes": [
            {
                "name": index.timeline.name,
                "kind": index.timeline.kind.name,
                "times": index.times.tolist(),
            }
            for index in chunk.indexes
        ],
        "columns": column_meta,
    }
    if labels is not None:
        metadata["labels"] = labels.to_metadata()

    schema = pa.schema(fields, metadata={METADATA_KEY: json.dumps(metadata).encode()})
    return pa.Table.from_arrays(arrays, schema=schema)


def chunk_from_table(table: pa.Table) -> tuple[Chunk, LabelRegistry | None]:
    """Decode a table produced by :meth:`chunk_to_table`.

    Returns:
        The chunk, and the label registry when the table carried one.
    """
    raw = (table.schema.metadata or {}).get(METADATA_KEY)
    if raw is None:
        raise ValueError(
            f"Table is missing the {METADATA_KEY.decode()!r} schema metadata written by "
            "chunk_to_table()",
        )

    metadata = json.loads(raw.decode())
    version = metadata.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported chunk schema version {version!r}, expected {SCHEMA_VERSION}",
        )

    columns: dict[ComponentDescriptor, Component] = {}
    for entry in metadata["columns"]:
        descriptor = ComponentDescriptor(
            entry["component"],
            archetype=entry.get("archetype"),
            component_type=entry.get("component_type"),
        )
        component_type = resolve_component_type(entry["component_type"])
        row_shape = tuple(entry.get("row_shape") or ())
        columns[descriptor] = component_type.from_arrow(
            table.column(entry["component"]),
            row_shape=row_shape,
        )

    indexes = tuple(
        TimeColumn(
            Timeline(entry["name"], TimeKind[entry["kind"]]),
            np.asarray(entry["times"], dtype=np.int64),
        )
        for entry in metadata["indexes"]
    )

    chunk = Chunk(
        metadata["entity_path"],
        indexes,
        np.asarray(metadata["offsets"], dtype=np.int64),
        columns,
        frame_id=metadata.get("frame_id"),
        is_static=bool(metadata.get("is_static", False)),
    )

    raw_labels = metadata.get("labels")
    labels = LabelRegistry.from_metadata(raw_labels) if raw_labels else None
    return chunk, labels


def write_parquet(chunk: Chunk, path: str | Path, *, labels: LabelRegistry | None = None) -> None:
    """Write a chunk to a Parquet file, metadata included."""
    pq.write_table(chunk_to_table(chunk, labels=labels), str(path))


def read_parquet(path: str | Path) -> tuple[Chunk, LabelRegistry | None]:
    """Read a chunk written by :func:`write_parquet`."""
    return chunk_from_table(pq.read_table(str(path)))
