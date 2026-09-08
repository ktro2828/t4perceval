# Architecture

The internal view: how the layers fit, what each one owns, and where the boundaries are. For the
public mental model, read [Concepts](../concepts/overview.md) first.

## Layers

```text
                        ┌──────────────────────────────────────────┐
   t4perceval.system    │  System / Pipeline                       │  the "S" of ECS
                        │  filter · matching · metric              │
                        └────────────────┬─────────────────────────┘
                                         │ reads / writes Chunk
                        ┌────────────────▼─────────────────────────┐
   t4perceval.core      │  Store          queried along timelines  │
                        │   ├ latest_at(entity, at)   → EntityView │
                        │   ├ range(entity, range)    → EntityView │
                        │   └ static                               │
                        ├──────────────────────────────────────────┤
                        │  Chunk    entity_path + indexes + frame  │
                        ├──────────────────────────────────────────┤
                        │  Archetype     validated bundles         │
                        ├──────────────────────────────────────────┤
                        │  Component     one column, fixed shape   │
                        ├──────────────────────────────────────────┤
                        │  EntityPath / ComponentDescriptor        │
                        └──────────────────────────────────────────┘
```

## Module responsibilities

| Module                   | Owns                                                   | Must not know about |
| :----------------------- | :----------------------------------------------------- | :------------------ |
| `t4perceval.core`        | addressing, columns, bundles, the log, queries         | systems, importers  |
| `t4perceval.component`   | the concrete column types and their dtypes             | archetypes          |
| `t4perceval.archetype`   | which columns go together, and under which descriptors | systems             |
| `t4perceval.descriptors` | the canonical descriptor names                         | everything          |
| `t4perceval.geometry`    | vectorized pairwise box geometry                       | the store           |
| `t4perceval.system`      | filtering, matching, metrics, pipeline validation      | importers, io       |
| `t4perceval.transform`   | the frame graph, lookup policy, pose composition       | systems             |
| `t4perceval.label`       | class and instance registries                          | the store           |
| `t4perceval.recording`   | store + registries + provenance, read-only             | systems             |
| `t4perceval.evaluation`  | materializing recordings into a writable store         | importers           |
| `t4perceval.align`       | pairing two recordings' `FRAME` axes by timestamp      | systems             |
| `t4perceval.importer`    | external formats in                                    | systems, metrics    |
| `t4perceval.io`          | Arrow / Parquet encoding of a chunk                    | importers           |

Two boundaries are load-bearing:

- **`importer` versus `io`.** `importer` converts an external representation into this one; `io`
  moves an already-native recording to and from storage. Reading a saved recording is `io`; reading
  a dataset is `importer`.
- **The extras.** The evaluation core never imports `t4_devkit` or `mcap`, and keeping them out of
  the base install is what _mechanically_ enforces that: an accidental import outside
  `t4perceval.importer` fails for anyone who installed without the extra. `tests/test_importer_isolation.py`
  checks it.

## Data flow of an evaluation

```text
   T4 dataset ─▶ T4Importer ────┐
                                 ├─▶ Recording (read-only)
   MCAP bag   ─▶ RosbagImporter ┘        │
                                          │  align (optional)
                                          ▼
                              build_evaluation_store
                                          │  only the named entities move;
                                          │  chunks are shared, not copied
                                          ▼
                                  Store (writable)
                                          │
                              ┌───────────┼───────────┐
                              ▼           ▼           ▼
                           Filter      Matcher     Metric
                              │           │           │
                              └───────────┴───────────┘
                                          │ send_chunk
                                          ▼
                                  the same Store
                                          │
                                          ▼
                                  Recording (frozen)
```

Three things do **not** ride along with a chunk when an entity moves between stores, and
`build_evaluation_store` handles each explicitly: static columns, per-path log order, and the
registries.

## Invariants

These hold everywhere, and code may rely on them:

1. **Columns are read-only** and never alias a writable array a caller passed in.
2. **Chunks are frozen.** Moving one between stores copies nothing.
3. **A descriptor identifies a column by meaning.** The `archetype` and `component_type` fields are
   hints and never affect equality.
4. **Static beats temporal** for the same descriptor in a view.
5. **A batch of zero rows is legal** in every archetype, and an empty frame keeps its `frame_id`.
6. **`select()` produces independent data**; lazy narrowing is `EntityView`'s job.
7. **A system writes only its declared targets**, and always through `Chunk`.

## Where the design is written down

| Document                                                                     | What it covers                                         |
| :--------------------------------------------------------------------------- | :----------------------------------------------------- |
| [Design principles](design-principles.md)                                    | the rules the code is written to                       |
| [Data model design](design/en/data_model.md) ([ja](design/ja/data_model.md)) | the long-form rationale for the model                  |
| [System design](design/en/system.md) ([ja](design/ja/system.md))             | the system protocol and pipeline                       |
| [Migration guide](design/en/migration.md) ([ja](design/ja/migration.md))     | mapping from `autoware_perception_evaluation`          |
| [Design decisions](design-decisions/index.md)                                | ADRs: what was decided, and why                        |
| [Serialization format](serialization-format.md)                              | how a chunk maps onto Arrow                            |
| [Roadmap](roadmap.md)                                                        | what is done and what is next                          |
| [Metric divergences](metric-divergences.md)                                  | where the numbers differ from the official definitions |

## Relationship to Rerun

The vocabulary and semantics are [Rerun](https://github.com/rerun-io/rerun)'s, reimplemented here
rather than depended on. Evaluation needs three things a visualization log does not: archetypes that
_validate_, registries that give integer columns meaning, and systems that write results back into
the same store. See [ADR 0001](design-decisions/0001-ecs-data-model.md).

## Where to go next

- [Extending components](extending-components.md) · [archetypes](extending-archetypes.md) · [systems](extending-systems.md)
- [Contributing](contributing.md)
