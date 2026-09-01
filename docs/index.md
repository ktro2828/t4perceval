# t4perceval

**Component-oriented perception evaluation for the T4 dataset.**

`t4perceval` is an experimental redesign of perception evaluation around columnar components,
archetypes, and ECS-style systems. Instead of representing every detected object as a large Python
object, it stores homogeneous NumPy columns that can be selected, serialized, and evaluated in
batches.

!!! warning "Project status"

    This project is under active development. APIs and data layouts may change without backward
    compatibility until the design stabilizes.

## Architecture at a glance

```text
T4 dataset / NumPy / Parquet
              │
              ▼
     Component columns
              │
              ▼
        Archetype bundles
              │
              ▼
    Chunk → Store → EntityView
              │
              ▼
   Filter / Matching / Metrics
```

The model is split into four layers:

| Layer                  | Responsibility                                                            |
| ---------------------- | ------------------------------------------------------------------------- |
| `t4perceval.component` | Typed, column-oriented values such as positions, labels, and trajectories |
| `t4perceval.archetype` | Validated bundles of related components                                   |
| `t4perceval.core`      | Entity paths, timelines, chunks, storage, and views                       |
| `t4perceval.system`    | Filtering, matching, metrics, and pass/fail evaluation                    |

## Quick example

```python
import numpy as np

from t4perceval.component import BatchPosition3D

positions = BatchPosition3D(
    np.array(
        [
            [1.0, 2.0, 0.0],
            [4.0, 5.0, 0.0],
        ]
    )
)

nearby = positions.select(np.array([True, False]))
```

Components and archetypes keep the batch dimension explicit. For example, a batch of predicted
trajectories uses an `[N, M, T, D]` layout: objects, modes, timesteps, and spatial dimensions.

## Design documents

The detailed design is available in Japanese and English.

| Topic                | 日本語                                  | English                                   |
| -------------------- | --------------------------------------- | ----------------------------------------- |
| Data model           | [データモデル](design/ja/data_model.md) | [Data model](design/en/data_model.md)     |
| Systems and pipeline | [システム設計](design/ja/system.md)     | [System design](design/en/system.md)      |
| Migration            | [移行ガイド](design/ja/migration.md)    | [Migration guide](design/en/migration.md) |

## Local preview

Install the development dependencies and start the documentation server:

```console
$ uv sync --group dev
$ uv run zensical serve
```

Then open the local URL printed by Zensical. To produce a static site instead, run:

```console
$ uv run zensical build --clean
```
