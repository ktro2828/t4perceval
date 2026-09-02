# Offline Evaluation Analysis

The architecture already supports analysis while the evaluation process is alive: inputs, filter
masks, matching results, and metrics remain in the `Store`. What is missing is a way to save and
reopen the entire evaluation recording.

This feature is defined as **persistent evaluation recordings with offline analysis**.

## What We Need

### 1. Whole-recording persistence

Current Parquet support saves only one `Chunk` at a time. Add APIs such as:

```python
write_recording(recording, "result.t4eval")
recording = read_recording("result.t4eval")
```

A recording should preserve:

- Every temporal chunk, including its insertion order
- Static components
- Entity paths
- All timelines and partition offsets
- `frame_id`
- Label and instance registries
- Raw estimations and ground truth
- Filter masks
- Matching results
- Metric chunks

Preserving chunk order matters because `Store.latest_at()` uses the most recently logged chunk when
timestamps are equal.

A directory-based format is a sensible first version:

```text
result.t4eval/
  manifest.json
  chunks/
    000000.parquet
    000001.parquet
    ...
```

The manifest maps files to entity paths; raw entity paths should not be used directly as filenames.

### 2. Evaluation metadata and provenance

Results are difficult to interpret without knowing how they were produced. Persist:

- Recording format version
- `t4perceval` version and optionally the Git commit
- Dataset name, revision, scenes, and sample selection
- Creation timestamp
- Pipeline order
- System class names
- Sources and targets
- Filter parameters
- Matching modes and thresholds
- Metric parameters
- Coordinate-frame assumptions
- Label and instance mappings
- Optional user-defined tags and notes

Introduce a recording object that owns the store and its context:

```python
EvaluationRecording(
    store=store,
    labels=labels,
    instances=instances,
    metadata=RunMetadata(...),
)
```

The pipeline description can initially be informational. Reconstructing and rerunning arbitrary
custom systems from serialized configuration should be a later feature.

### 3. Instance-registry serialization

`LabelRegistry` already has metadata serialization, but `InstanceRegistry` does not. Add equivalent
`to_metadata()` and `from_metadata()` methods.

This is essential for showing object UUIDs in offline error analysis instead of only internal integer
IDs.

### 4. [OPTIONAL] Offline analysis API

Loading a `Store` alone is technically sufficient, but users should not have to manually coordinate
several entity queries. Add a higher-level API, for example:

```python
analysis = EvaluationAnalysis.open("result.t4eval")

analysis.metrics()
analysis.metrics(metric="ap", class_name="car")
analysis.errors(status="false_positive")
analysis.frame(120)
analysis.object_history(instance="...")
analysis.filter_failures("/estimation/objects")
```

Useful first analyses include:

- Metric tables by class and threshold
- TP/FP/FN counts and corresponding objects
- Matching-score distributions
- Confidence-versus-error analysis
- Filter rejection reasons
- Per-frame and time-range summaries
- Worst-scoring frames and objects
- Tracking history by instance
- Export to Arrow, pandas, CSV, or JSON

`MatchJoin` can supply much of the matching-to-object reconstruction that these queries need.

### 5. Multi-scene organization

The design currently treats one scene as one `Store`. A dataset-level result therefore needs an
archive index:

```text
result.t4eval/
  manifest.json
  scenes/
    scene-001/...
    scene-002/...
  aggregate/...
```

This permits both scene-level investigation and dataset-level metric comparison without mixing
identical frame numbers from different scenes.

### 6. Compatibility and validation

The chunk schema is already versioned. The recording itself needs a separate format version and
migration policy.

Tests should verify:

- A complete store is identical after save/load
- Static and temporal data survive
- Empty frames and zero-row chunks survive
- Multiple timelines survive
- Equal-timestamp ordering survives
- Labels and instances survive
- Metrics before and after loading are identical
- Unknown or newer format versions produce clear errors
- Missing or corrupted chunk files are detected

## Analysis Levels to Support

It helps to state what "analyze later" promises:

1. **Inspect existing results without computation.** Metrics, TP/FP/FN, masks, scores, and objects.
2. **Compute new summaries without rematching.** New groupings, plots, slices, and aggregate
   statistics using persisted matching results.
3. **Recompute metrics without rerunning matching.** This is possible when matching results and
   source entities are preserved.
4. **Change filters or matching thresholds.** This requires rerunning those pipeline stages, but it
   can still operate entirely from persisted raw estimations and ground truth without reopening the
   original dataset.

## Recommended MVP

The first deliverable should include:

- `EvaluationRecording`
- `RunMetadata`
- `write_recording()` and `read_recording()`
- `InstanceRegistry` serialization
- Round-trip tests
- A small `EvaluationAnalysis` API for metrics and TP/FP/FN inspection
- A CLI:

```bash
t4perceval inspect result.t4eval
t4perceval metrics result.t4eval
t4perceval errors result.t4eval --status fp --class car
```

Visualization and run-to-run comparison can then build on this same persisted format.
