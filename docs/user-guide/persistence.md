# Persistence

What a `Recording` is, how one chunk maps onto Arrow, and how a whole recording is saved and
reopened.

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

One chunk at a time:

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

## Saving a whole recording

`write_recording` / `read_recording` save a `Recording` as a `.t4eval` directory and reopen it:

```python
from t4perceval.io import read_recording, write_recording

write_recording(recording, "result.t4eval")
recording = read_recording("result.t4eval")
```

Everything the store held comes back -- every temporal chunk in the order it was logged, static
chunks with their own `frame_id`, every timeline and partition offset, both registries and the
metadata -- so any query written against the live store answers identically against the reopened
recording. That includes `latest_at` on two chunks that share a time: the most recently logged one
still wins, because the manifest records the log order that saving chunks one by one would lose.

```text
result.t4eval/
  manifest.json     format version, provenance, both registries, chunks in log order
  chunks/
    000000.parquet  one chunk per file, readable on its own by any Arrow tool
    000001.parquet
    ...
```

`write_recording` refuses an existing directory unless `exist_ok=True`; then it overwrites
`manifest.json` and the chunk files it produces and leaves anything else alone. It returns the
directory, so `read_recording(write_recording(recording, path))` composes. The one thing it changes
in what it writes is `metadata.format_version`, stamped with the format version it produced.

The registries live in the manifest only. A chunk file lifted out of the directory carries no label
registry, and `read_recording` refuses one that does -- there is a single authority for what a class
id means. The layout is described in
[Serialization format](../development/serialization-format.md#a-whole-recording) and the decision
in [ADR 0004](../development/design-decisions/0004-persistent-recording.md).

## Where to go next

- [Offline analysis](offline-analysis.md) -- querying a finished run.
- [Serialization format](../development/serialization-format.md) -- how a chunk maps onto Arrow,
  and how a recording is laid out on disk.
