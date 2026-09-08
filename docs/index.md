# t4perceval

**Component-oriented perception evaluation for the [T4 dataset](https://github.com/tier4/t4-devkit).**

A redesign of [`autoware_perception_evaluation`](https://github.com/tier4/autoware_perception_evaluation)
around [Rerun](https://github.com/rerun-io/rerun)'s data model: data lives at an **entity path**, is
made of **component** columns, is bundled into **archetypes**, stored as **chunks**, indexed along
**timelines** inside a **store**, and transformed by **systems**.

The point of the redesign is that an evaluation task stops being a value to branch on. There is no
`EvaluationTask` enum and no single config dict: **a task is the set of components present plus the
pipeline you compose**.

!!! warning "Project status"

    Under active development. APIs and data layouts may change without backward compatibility until
    the design stabilizes. See the [roadmap](development/roadmap.md).

## Start here

<div class="grid cards" markdown>

- **New to t4perceval**

  [Installation](getting-started/installation.md) →
  [Quick start](getting-started/quickstart.md) →
  [First evaluation](getting-started/first-evaluation.md)

- **I want to understand it**

  [Concepts overview](concepts/overview.md), then
  [the data model](concepts/data-model.md) and
  [the evaluation pipeline](concepts/evaluation-pipeline.md)

- **I want to evaluate something**

  [Detection 3D](evaluation/detection-3d.md) ·
  [Detection 2D](evaluation/detection-2d.md) ·
  [Tracking](evaluation/tracking.md) ·
  [Prediction](evaluation/prediction.md)

- **I have a specific task**

  [Evaluate model output](recipes/evaluate-model-output.md) ·
  [a T4 dataset](recipes/evaluate-t4-dataset.md) ·
  [a ROS bag](recipes/evaluate-rosbag.md)

</div>

## In thirty seconds

```python
from t4perceval import FRAME, Detections3D, LabelRegistry, MetricValues, Store, TimePoint, TimeRange
from t4perceval.system import Pipeline, SystemContext, average_precision_sweep

labels = LabelRegistry.from_names(["car", "pedestrian"])
store = Store()

store.log(
    "/ground_truth/objects",
    Detections3D(
        position=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]] * 2,
        size=[[1.9, 4.5, 1.6]] * 2,
        class_id=labels.encode(["car", "car"]),
        confidence=[1.0, 1.0],
    ),
    at=TimePoint.at(frame=0),
    frame_id="base_link",
)
store.log("/estimation/objects", ..., at=TimePoint.at(frame=0), frame_id="base_link")

Pipeline(
    average_precision_sweep("/estimation/objects", "/ground_truth/objects", thresholds=[1.0, 2.0])
).run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())

store.range("/metrics/map", timeline=FRAME, time_range=TimeRange.everything()).materialize(
    MetricValues
).aggregate
```

One frame of objects is one **columnar batch**, not a list of objects. Every stage writes its result
back into the store, so the filter masks, the matching verdicts and the per-threshold AP are all
still there afterwards -- and asking about one frame costs another query, not another run.

## What it gives you

**Speed.** The [benchmark](development/benchmarks.md) runs both libraries over the same synthetic
scenes: 67x to 224x faster on the metric workloads, 16x smaller in retained memory. On the
unambiguous scene all 133 compared metric values agree with `perception_eval` to within `1e-9`.

**Answers, not just numbers.** Which rows a filter dropped and what a matcher scored stay in the
store as queryable entities, so [offline analysis](user-guide/offline-analysis.md) is a query rather
than a re-run with print statements.

**Composition.** A system declares the components it needs and runs against anything carrying them,
so `Detections3D`, `Trackings3D` and `Predictions3D` all work with the same filters and matchers.

**Failures that fail.** This domain's error mode is a _plausible number_. Comparing geometry across
coordinate frames raises; a pipeline that reads an entity a later system writes raises at
construction; two recordings that disagree about class ids raise.

## Supported evaluation tasks

| Task                                             | Matching                                  | Metrics                                   |
| :----------------------------------------------- | :---------------------------------------- | :---------------------------------------- |
| [Detection 3D](evaluation/detection-3d.md)       | centre distance, BEV, plane, IoU BEV / 3D | AP, APH, mAP, mAPH, confusion matrix      |
| [Detection 2D](evaluation/detection-2d.md)       | ROI IoU                                   | AP, mAP, classification, confusion matrix |
| [Tracking](evaluation/tracking.md)               | any 3D or 2D matcher                      | MOTA, MOTP, ID switches                   |
| [Prediction](evaluation/prediction.md)           | any 3D matcher, on the current pose       | ADE, FDE, miss rate                       |
| [Segmentation 3D](evaluation/segmentation-3d.md) | _not applicable_                          | _not implemented yet_                     |

Inputs come from your own arrays, from a [T4 dataset](user-guide/dataset-importers.md#t4-dataset),
or from an [MCAP ROS bag](user-guide/dataset-importers.md#mcap-ros-bag).

## How this documentation is organised

Not by Python package -- by what you are trying to do.

| Section                                                               | Answers                                              |
| :-------------------------------------------------------------------- | :--------------------------------------------------- |
| [Getting started](getting-started/installation.md)                    | "I want to try it."                                  |
| [Concepts](concepts/overview.md)                                      | "I want to understand it."                           |
| [User guide](user-guide/logging-data.md)                              | "I want to use feature X."                           |
| [Evaluation](evaluation/detection-3d.md)                              | "I want to evaluate Detection 3D."                   |
| [Recipes](recipes/evaluate-model-output.md)                           | "I want to accomplish X."                            |
| [Components](components/index.md) & [Archetypes](archetypes/index.md) | "What data types exist?"                             |
| [API reference](reference/api/index.md)                               | "What exactly does this API do?"                     |
| [Development](development/architecture.md)                            | "I want to understand or extend the library itself." |

Only the API reference follows the implementation structure, and it is generated from the docstrings.

## Design documents

The long-form design is available in Japanese and English.

| Topic                | 日本語                                              | English                                               |
| :------------------- | :-------------------------------------------------- | :---------------------------------------------------- |
| Data model           | [データモデル](development/design/ja/data_model.md) | [Data model](development/design/en/data_model.md)     |
| Systems and pipeline | [システム設計](development/design/ja/system.md)     | [System design](development/design/en/system.md)      |
| Migration            | [移行ガイド](development/design/ja/migration.md)    | [Migration guide](development/design/en/migration.md) |

Plus [Architecture decision records](development/design-decisions/index.md) for the individual
calls.

## Local preview

```console
uv sync --group dev
uv run zensical serve
```

To produce a static site instead:

```console
uv run zensical build --clean
```
