# Serialization format

How a `Chunk` maps onto Arrow, and therefore onto Parquet.

## The rule

**One component becomes one Arrow field, named by its descriptor.** Everything that is _not_
row-shaped goes into the schema metadata, because it has a different length from the columns and so
cannot be a column itself.

```text
pyarrow.Table
├── fields                          one per component, named by descriptor.component
│   ├── position      fixed_size_list<double>[3]   nullable=False
│   ├── class_id      int32                        nullable=False
│   └── confidence    double                       nullable=False
└── schema.metadata[b"t4perceval"]  JSON
    ├── version
    ├── entity_path
    ├── frame_id
    ├── is_static
    ├── offsets           partition boundaries
    ├── indexes           one per timeline: name, kind, times
    ├── columns           per field: component, archetype, component_type, row_shape
    └── labels            the label registry, when one was given
```

The metadata key is `b"t4perceval"` and the layout is versioned by `SCHEMA_VERSION`, bumped whenever
it changes incompatibly.

## Field types

| Component row shape | Arrow type                                         |
| :------------------ | :------------------------------------------------- |
| `()`                | the scalar type: `int32`, `double`, `bool`, `int8` |
| `(3,)`              | `fixed_size_list<double>[3]`                       |
| `(M, T, 3)`         | nested fixed-size lists                            |
| a frame name        | `string`                                           |

Fixed-size lists rather than variable-size ones: a component's row shape is part of its schema, so
encoding it as variable-length would lose exactly the guarantee the type exists to make.

Every field is declared **non-nullable**. Components never contain nulls, and pinning that in the
schema keeps readers honest.

`BatchFrameId.to_arrow` pins `pa.string()` rather than letting Arrow infer it: inference on an
`object` array depends on the _values_, and a zero-row column would infer `null`, which the encoder
then rejects for declaring its fields non-nullable. A zero-row frame column is ordinary -- an entity
can exist before it has any edge -- so the schema must not depend on the row count.

## Why the metadata carries what it does

| Metadata field | Why it cannot be a column                                                |
| :------------- | :----------------------------------------------------------------------- |
| `entity_path`  | one value per chunk, not per row                                         |
| `frame_id`     | likewise -- and it is what the cross-frame guard reads                   |
| `is_static`    | likewise; a static chunk keeps its own `frame_id`                        |
| `offsets`      | one value per **partition**, a different length from the rows            |
| `indexes`      | one time per partition, per timeline -- again a different length         |
| `columns`      | the component type and row shape needed to reconstruct the right class   |
| `labels`       | a registry, not row data; without it the integer columns are meaningless |

`column_type` is resolved back to a class through
[`t4perceval.io.registry`](../reference/api/io.md), which is why a new component must be reachable
from `t4perceval.component` -- see [Extending components](extending-components.md#register-it-for-io).

## Round trip

```python
from t4perceval.io import chunk_from_table, chunk_to_table, read_parquet, write_parquet

table = chunk_to_table(chunk, labels=labels)
chunk, labels = chunk_from_table(table)

write_parquet(chunk, "matching.parquet", labels=labels)
chunk, labels = read_parquet("matching.parquet")
```

Lossless for one chunk: dtypes, row shapes, partition boundaries, timelines, `frame_id`, `is_static`
and the registry all survive.

## What one chunk does not carry

Three things belong to the **store**, not to a chunk, and so are not in the file:

1. **Log order.** `Store.latest_at` prefers the most recently logged chunk when two share a time.
   Writing chunks to separate files loses that ordering unless a manifest records it.
2. **Which entity a file belongs to** -- the metadata has the path, but nothing indexes files by it.
3. **The instance registry.** Only the label registry travels.

That is why saving a whole evaluation is not just "write every chunk". See
[Persistent recordings](persistent-recordings.md) for the design, and
[ADR 0004](design-decisions/0004-persistent-recording.md) for the decision.

## Compatibility

`SCHEMA_VERSION` is `1`. A reader rejects a table with no `t4perceval` metadata key rather than
guessing:

```text
ValueError: Table is missing the 't4perceval' schema metadata written by chunk_to_table()
```

Because the format is plain Arrow, a file is readable by any Arrow or Parquet tool -- the metadata is
what makes it a _chunk_ rather than a table.

## Where to go next

- [Persistence](../user-guide/persistence.md) -- using it.
- [ADR 0002: Arrow as the storage format](design-decisions/0002-arrow-storage.md).
