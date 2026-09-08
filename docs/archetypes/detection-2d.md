# Detections2D

2D regions of interest with a class and a confidence.

## Schema

| Component    | Requirement | Shape    | dtype | Description                     |
| :----------- | :---------- | :------- | :---- | :------------------------------ |
| `roi`        | Required    | `(N, 4)` | `i32` | `(x_min, y_min, height, width)` |
| `class_id`   | Required    | `(N,)`   | `i32` | semantic class                  |
| `confidence` | Required    | `(N,)`   | `f64` | in `[0, 1]`                     |
| `visibility` | Optional    | `(N,)`   | `i8`  | occlusion level                 |

Descriptors: `ROI`, `CLASS_ID`, `CONFIDENCE`, `VISIBILITY`.

## Constructing

```python
from t4perceval import Detections2D

detections = Detections2D(
    roi=[[100, 200, 50, 30]],  # (x_min, y_min, height, width)
    class_id=labels.encode(["car"]),
    confidence=[0.8],
)
```

!!! warning "ROI order and dtype"

    `(x_min, y_min, height, width)` -- height before width -- and an integer dtype. This is not
    `(x, y, w, h)` and not floats.

`BatchRoi` derives the rest:

```python
detections.roi.x_max  # x_min + width
detections.roi.y_max  # y_min + height
detections.roi.area()  # height * width
```

## One entity per camera

A 2D box is only meaningful in one image, so the chunk's `frame_id` is the **camera channel** and
two cameras are two entities:

```text
/ground_truth/CAM_FRONT/objects   frame_id="CAM_FRONT"
/ground_truth/CAM_BACK/objects    frame_id="CAM_BACK"
```

The [cross-frame guard](../concepts/coordinate-system.md#the-cross-frame-guard) then stops a
`CAM_FRONT` estimation from being matched against `CAM_BACK` ground truth.

## What it unlocks

| Component present | Enables                                              |
| :---------------- | :--------------------------------------------------- |
| `roi`             | `IoURoiMatchingSystem`                               |
| `class_id`        | every matcher, every metric, `FilterByLabelSystem`   |
| `confidence`      | `AveragePrecisionSystem`, `FilterByConfidenceSystem` |
| `visibility`      | `FilterByVisibilitySystem`                           |

The geometric filters need `position`, which a 2D detection does not carry, so they do not apply.

## Relationships

`Trackings2D` declares the same three required components plus `instance_id`.

## Where to go next

- [Detection 2D evaluation](../evaluation/detection-2d.md) -- the task.
- [Trackings2D](tracking.md)
