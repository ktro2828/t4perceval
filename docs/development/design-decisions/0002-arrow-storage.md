# 0002: Arrow as the storage and interchange format

## Status

Accepted.

## Context

An evaluation produces data that has to outlive the process: the inputs, the filter masks, the
matching verdicts and the metric values. It also has to be readable by things that are not
`t4perceval` -- a notebook, a dashboard, a comparison script.

The [data model](0001-ecs-data-model.md) already stores each property as a homogeneous column with a
fixed per-row shape and dtype. The question is what to write those columns as.

## Decision

Encode a `Chunk` as a **PyArrow table**: one component becomes one Arrow field, named by its
descriptor, with a fixed-size list type for a nested row shape. Everything that is not row-shaped --
the entity path, `frame_id`, `is_static`, the partition offsets, the timeline index values and an
optional label registry -- goes into the schema metadata under the key `b"t4perceval"`, versioned by
`SCHEMA_VERSION`.

Parquet is then just `pq.write_table` on that.

`pyarrow` is a **direct dependency of the core**, not an extra.

## Rationale

The columns are already Arrow-shaped, so the encoding is close to a memory copy rather than a
serialization pass.

Arrow's type system happens to express exactly the two constraints the component model makes:

- **Fixed-size lists** carry the per-row shape. A variable-size list would lose the one guarantee a
  component type exists to make.
- **Non-nullable fields** carry the "components never contain nulls" invariant into the file, so a
  reader cannot silently accept a column with holes.

Parquet then comes free, and with it every tool that reads Parquet -- which matters because the
point of keeping intermediates is that somebody analyses them, and that somebody should not have to
import this package to do it.

The metadata split falls out of the shape of the data rather than being a design choice: `offsets`
has one entry per _partition_ and `indexes` one time per partition per timeline, both of which are
different lengths from the columns. They cannot be columns.

Carrying the **label registry** in the metadata is the one addition that is a choice. Without it the
integer columns are meaningless, and a file that cannot be interpreted without a second file that
nothing links to is a trap.

## Alternatives considered

**NumPy `.npz`.** Simplest possible thing, and it round-trips arrays exactly. But there is no schema,
no type-level record of the row shape, no metadata convention, and nothing outside Python reads it.
Any consumer would have to reimplement the layout.

**Pickle.** Fast to write, and unreadable by anything else, unsafe to load from an untrusted source,
and tied to the exact class layout at the time of writing -- so a refactor invalidates every stored
result.

**HDF5.** Handles nested shapes and metadata well. But it is a heavier dependency, its Python story
has historically been awkward around concurrent readers, and the ecosystem around Parquet for
columnar analytics is much larger.

**JSON or CSV.** Loses dtypes and shapes, which is precisely what the component model spends its
effort pinning.

**A custom binary format.** Would have to reinvent schema, versioning, compression and tooling, and
give nothing in return.

## Consequences

**Good.**

- A chunk round-trips losslessly: dtypes, row shapes, partition boundaries, timelines, `frame_id`,
  `is_static` and the registry.
- Any Arrow or Parquet tool can read the file. The metadata is what makes it a _chunk_ rather than a
  table, and a reader that ignores the metadata still gets the columns.
- Encoding cost is low enough that persisting intermediates is not a decision anyone has to weigh.

**Costs.**

- `pyarrow` is a large dependency for a library whose core is otherwise NumPy and SciPy. It is a
  direct dependency because IO is public API, not an add-on.
- A **new component type must be resolvable by name** on decode, via
  [`t4perceval.io.registry`](../../reference/api/io.md) -- so it has to be reachable from
  `t4perceval.component`. See [Extending components](../extending-components.md#register-it-for-io).
- The metadata is JSON inside a binary format, so it is versioned by hand (`SCHEMA_VERSION`) rather
  than by Arrow's own schema evolution.
- **One chunk at a time.** The format says nothing about log order across chunks, about which files
  belong to one entity, or about the instance registry -- all of which belong to the _store_. That
  gap is [ADR 0004](0004-persistent-recording.md).

## Where it is written down

[Serialization format](../serialization-format.md).
