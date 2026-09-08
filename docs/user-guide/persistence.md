# Persistence

What can be saved today, what a `Recording` is, and what is still missing.

## Recording: a log plus what its integers mean

A `Store` says what rows exist. It deliberately does not say what the integers in `class_id` and
`instance_id` mean. A `Recording` is the immutable pair:

```python
from t4perceval import Recording

recording = Recording.of(store, labels=labels, instances=instances, metadata=metadata)
```

It exposes the same read surface as a store -- `entity_paths`, `timelines`, `chunks`, `static`,
`times`, `latest_at`, `range` -- and no write surface at all.

```python
recording.range("/ground_truth/objects", timeline=FRAME, time_range=TimeRange.everything())
recording.labels.decode([0, 2])
recording.agrees_with(other)  # do the two encode class ids the same way?
```

### Metadata

```python
recording.metadata.format_version
recording.metadata.t4perceval_version
recording.metadata.created_at_ns
recording.metadata.sources  # SourceInfo per stream: kind, uri, version, scene, topic
recording.metadata.labels_fingerprint
recording.metadata.pipeline  # the system class names that produced it
recording.metadata.frame_id
recording.metadata.tags  # free-form provenance
recording.metadata.notes

recording.with_metadata(notes="baseline run, model v3")
```

`RecordingMetadata` and `SourceInfo` both round-trip through `to_json` / `from_json`.

## Arrow and Parquet

Persistence is **per chunk** today.

```python
from t4perceval.io import chunk_from_table, chunk_to_table, read_parquet, write_parquet

table = chunk_to_table(chunk, labels=labels)  # -> pyarrow.Table
chunk, labels = chunk_from_table(table)  # the registry rides along

write_parquet(chunk, "matching.parquet", labels=labels)
chunk, labels = read_parquet("matching.parquet")
```

Both readers return a `(chunk, labels)` pair; `labels` is `None` when the table carried no
registry.

Everything that pins a column's meaning travels with it: dtypes and shapes come from the schema,
nested vectors are fixed-size lists, and non-row-wise information -- the entity path, `frame_id`,
`is_static`, partition offsets and the label registry -- lives in the schema metadata. A round trip
is lossless for one chunk.

To save a query result rather than a raw chunk, materialize it first:

```python
scene = store.range(
    "/matching/center_distance", timeline=FRAME, time_range=TimeRange.everything()
).materialize(MatchResults)

write_parquet(scene.to_chunk("/matching/center_distance", at=TimePoint.at(frame=0)), "m.parquet")
```

## What is not there yet

**A whole recording cannot be saved and reopened.** There is no `write_recording` /
`read_recording`, so an evaluation's inputs, masks, verdicts and metrics survive only as long as the
process. Saving each chunk by hand loses log order, which `latest_at` depends on when two chunks
share a time, and loses the registries unless you persist them separately.

The design for it -- a directory format with a manifest, preserving chunk order, static columns,
timelines, `frame_id` and both registries -- is written up in
[Persistent recordings](../development/persistent-recordings.md) and recorded as
[ADR 0004](../development/design-decisions/0004-persistent-recording.md).

Until then, the practical options are:

- Keep the process alive and query the store (see [Offline analysis](offline-analysis.md)).
- Write the entities you care about as individual Parquet files, and persist
  `labels.to_metadata()` alongside them.

```python
import json
from pathlib import Path

Path("run/labels.json").write_text(json.dumps(labels.to_metadata()))
for path in ("/metrics/map", "/matching/center_distance"):
    chunk = store.range(path, timeline=FRAME, time_range=TimeRange.everything()).to_chunk()
    write_parquet(chunk, f"run/{path.strip('/').replace('/', '_')}.parquet", labels=labels)
```

`LabelRegistry.from_metadata` reads it back.

## Where to go next

- [Offline analysis](offline-analysis.md) -- querying a finished run.
- [Serialization format](../development/serialization-format.md) -- how a chunk maps onto Arrow.
