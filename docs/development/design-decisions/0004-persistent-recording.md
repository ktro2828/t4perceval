# 0004: Persistent evaluation recordings

## Status

**Proposed.** Not implemented. `Recording` and `RecordingMetadata` exist; `write_recording` and
`read_recording` do not.

## Context

The [data model](0001-ecs-data-model.md) keeps every intermediate -- inputs, filter masks, matching
verdicts, metric values -- in the store, precisely so a result can be re-analysed. Today that only
holds **while the process is alive**. Close Python and the analysis is gone.

[Arrow persistence](0002-arrow-storage.md) saves **one chunk** at a time. Writing every chunk by
hand does not add up to saving a store, because three things belong to the store rather than to any
chunk:

- **Log order.** `Store.latest_at` prefers the most recently logged chunk when two share a time.
- **Which files belong to which entity**, in what sequence.
- **The instance registry.** Only the label registry travels in chunk metadata, and
  `InstanceRegistry` has no serialization at all.

Without provenance the file is also hard to interpret: which dataset, which scene, which filters,
which thresholds, which coordinate frame.

## Decision

Add a **directory-based recording format** and the two functions that read and write it.

```python
write_recording(recording, "result.t4eval")
recording = read_recording("result.t4eval")
```

```text
result.t4eval/
  manifest.json          format version, entity → files, log order, registries, provenance
  chunks/
    000000.parquet
    000001.parquet
    ...
```

The manifest maps files to entity paths; **raw entity paths are not used as filenames**. A recording
must preserve every temporal chunk _including its insertion order_, static components, entity paths,
all timelines and partition offsets, `frame_id` on temporal and static chunks alike, both registries,
and the metadata already on `RecordingMetadata`.

`InstanceRegistry` gains `to_metadata()` / `from_metadata()`, matching `LabelRegistry`.

The format carries its **own** version, separate from the chunk `SCHEMA_VERSION`.

## Rationale

A directory of Parquet files rather than one container: each chunk is already a valid Parquet file,
any Arrow tool can open one, and a reader can load a single entity without parsing the whole
recording. The manifest carries exactly what a chunk cannot.

`Recording` already exposes the same read surface as `Store` -- `entity_paths`, `timelines`,
`chunks`, `static`, `times`, `latest_at`, `range` -- so **every query written against a live store
works unchanged against a reopened recording**. That is the property the format has to preserve, and
it is why the unit of persistence is a recording rather than a bag of chunks.

Preserving chunk order is not incidental: it is what makes `latest_at` deterministic when two chunks
share a time.

Stating what "analyse later" promises, in increasing cost:

1. **Inspect results with no computation** -- metrics, TP/FP/FN, masks, scores, objects.
2. **Compute new summaries without rematching** -- new groupings and slices over persisted verdicts.
3. **Recompute metrics without rerunning matching** -- possible because both the verdicts and the
   source entities are kept.
4. **Change filters or thresholds** -- reruns those stages, but still without reopening the original
   dataset.

Level 4 is the one that justifies keeping the raw inputs in the recording rather than only the
results.

## Alternatives considered

**One Parquet file with an entity column.** Forces every entity into one schema, which they do not
share -- a mask, a match result and a set of boxes have nothing in common. Would mean a union schema
mostly full of nulls, contradicting the non-nullable invariant.

**Zip or tar of the directory.** Can be added later over the same layout; it only trades
random access for one file.

**Pickle the `Store`.** Fast to write, and unreadable outside Python, unsafe from an untrusted
source, and invalidated by any refactor of the classes.

**Save only the metrics.** Small, and throws away exactly what the data model spent its effort
keeping. "Which objects were the false positives?" would be unanswerable.

**Entity paths as filenames.** A path may contain characters a filesystem does not like, and case
sensitivity differs across platforms. The manifest indirection costs one lookup and removes the
class of problem.

## Consequences

**Good.**

- An evaluation becomes an artifact: shareable, diffable between runs, and re-analysable months
  later.
- Every existing query keeps working, because `Recording` is already the read surface.
- Provenance travels with the numbers, so a result says which dataset, scene and options produced
  it.

**Costs.**

- A second version number to maintain, with a migration policy, alongside the chunk schema's.
- A directory is more awkward to move around than a single file.
- Round-trip fidelity needs real test coverage: static _and_ temporal data, empty frames and
  zero-row chunks, several timelines, equal-timestamp ordering, both registries, and identical
  metrics before and after. Unknown format versions and corrupted chunk files must produce clear
  errors.
- Multi-scene results need an index above the recording (`scenes/scene-001/...`), or identical frame
  numbers from different scenes will mix.

## Open questions

- Whether the pipeline description in the manifest stays informational, or becomes enough to
  _reconstruct_ the systems. Reconstructing arbitrary custom systems from serialized configuration
  is a later feature at best.
- Whether a higher-level `EvaluationAnalysis` API ships with the format or after it.
- Whether a CLI (`t4perceval inspect`, `metrics`, `errors`) belongs in this package.

## Where it is written down

The full plan, including the proposed MVP, is in
[Persistent recordings](../persistent-recordings.md). The gap it fills is described in
[Persistence](../../user-guide/persistence.md#what-is-not-there-yet).
