# 移行ガイド

`autoware_perception_evaluation` (`perception_eval`) から `t4perceval` への対応表。

## 型の対応

| 旧 (`perception_eval`)                                     | 新 (`t4perceval`)                                                                                                                                        | 備考                                                         |
| :--------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------- |
| `DynamicObject`                                            | `Detections3D` / `Trackings3D` / `Predictions3D` の 1 行                                                                                                 | 20+ フィールドの 1 クラスを、タスクごとの component 束に分解 |
| `DynamicObject2D`                                          | `Detections2D` / `Trackings2D` / `Classifications2D` の 1 行                                                                                             | `roi` は `BatchRoi`                                          |
| `Shape` / `ShapeType`                                      | `BatchSize3D`                                                                                                                                            | footprint 計算は matching system 側へ                        |
| `FrameGroundTruth`                                         | `/ground_truth/objects` の 1 partition                                                                                                                   | `unix_time` → `TIMESTAMP` timeline                           |
| `FrameGroundTruth.raw_data`                                | 別 entity path (`/sensor/<channel>`)                                                                                                                     | 今回は未実装                                                 |
| `FrameID` enum                                             | 文字列としての `Chunk.frame_id`                                                                                                                          | system は比較するだけで、分岐はしない                        |
| `HomogeneousMatrix` / 引数で渡す変換                       | `Transform3D` の行 + `TransformResolver.lookup()`                                                                                                        | 状態ではなく記録データ。親は `Chunk.frame_id`、子は列        |
| `Catalog` / `Scenario` / `Scene`                           | `Store` + timeline 上の `TimeRange`                                                                                                                      | list のネストを廃止                                          |
| `PerceptionFrameResult`                                    | `store.latest_at(...)` の結果 + `/matching/*` chunk                                                                                                      | frame ごとの再計算が不要                                     |
| `DynamicObjectWithPerceptionResult`                        | `MatchResults`                                                                                                                                           | 参照ではなく行 index。保存・再解析できる                     |
| `MatchingMode` enum                                        | マッチング system の種類 (`CenterDistance` / `CenterDistanceBEV` / `PlaneDistance` / `IoUBEV` / `IoU3D` / `IoURoi`) と `/matching/<mode>` の entity path | 6 モードすべて実装済み                                       |
| `MatchingMethod` (`CenterDistanceMatching` 等)             | `BatchMatchingScore` 列                                                                                                                                  | 値の意味は entity path が持つ                                |
| `EvaluationTask` enum                                      | 存在する component の集合 + `Pipeline` の構成                                                                                                            | enum 分岐が消える                                            |
| `PerceptionEvaluationConfig`                               | 各 system のパラメータ + `LabelRegistry`                                                                                                                 |                                                              |
| `evaluation_config_dict`                                   | system ごとの dataclass                                                                                                                                  | 構築時に検証される                                           |
| `center_distance_thresholds` 等の list of list             | `Thresholds(default, by_class=...)`                                                                                                                      | クラス順を知らなくても対象が分かる                           |
| `MetricsScoreConfig`                                       | 各 metric system のパラメータ                                                                                                                            |                                                              |
| `MetricsScore`                                             | `/metrics/*` chunk                                                                                                                                       |                                                              |
| `LabelConverter` / `label_prefix` / `merge_similar_labels` | `LabelRegistry` / `LabelRegistry.merged()`                                                                                                               | マージがデータとして見える                                   |
| `Visibility`                                               | `BatchVisibility` + `VisibilityLevel`                                                                                                                    | 文字列 enum から順序つき整数へ                               |
| `objects_filter` の各関数                                  | `Filter*System` 群                                                                                                                                       | 行を落とさず `BatchMask` を出力                              |
| `object_matching` の各関数                                 | `*MatchingSystem` 群                                                                                                                                     | 幾何は `t4perceval.geometry` にベクトル化して集約            |
| `PassFailResult` / `CriticalObjectFilterConfig`            | `PassFailSystem` + critical 用フィルタ system                                                                                                            |                                                              |
| `PerceptionEvaluationManager`                              | `Pipeline` + `SystemContext`                                                                                                                             |                                                              |
| `perception_analyzer3d` / `visualization/`                 | store へのクエリ (後続で可視化層)                                                                                                                        | 中間結果が残るので後付けできる                               |
| `class_to_dict` + `json.dump`                              | `write_parquet` / `chunk_to_table`                                                                                                                       | 型・shape が schema で固定される                             |

## 概念の対応

| 旧の概念                          | 新の概念                                              |
| :-------------------------------- | :---------------------------------------------------- |
| オブジェクトのフィールド          | component (列)                                        |
| タスクごとのクラス                | archetype (component の束)                            |
| est / gt の区別 (引数の順序)      | entity path (`/estimation/...` / `/ground_truth/...`) |
| `frame_id` (座標系)               | `Chunk.frame_id`                                      |
| ego pose / センサー外部パラメータ | `Transform3D` の行: `frame_id` -> `child_frame_id`    |
| `unix_time`                       | `TIMESTAMP` timeline 上の値                           |
| frame 番号                        | `FRAME` timeline 上の値                               |
| `List[FrameResult]` の走査        | `store.range(...)`                                    |
| 中間結果 (破棄)                   | `/`配下の chunk (保存)                                |

## コード例

### オブジェクトの構築

```python
# 旧
obj = DynamicObject(
    unix_time=1624164470849887,
    frame_id=FrameID.BASE_LINK,
    position=(1.0, 2.0, 3.0),
    orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
    shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(1.0, 4.0, 2.0)),
    velocity=(1.0, 0.0, 0.0),
    semantic_score=0.9,
    semantic_label=Label(AutowareLabel.CAR, "car"),
    uuid="c28556c19064ad491ff1dc438a38a3a7",
)
objects = [obj, ...]
```

```python
# 新 — フレーム内の全オブジェクトを 1 つの列指向 batch にする
labels = LabelRegistry.from_names(["car", "bicycle", "pedestrian", "motorbike"])
instances = InstanceRegistry()

detections = Trackings3D(
    position=[[1.0, 2.0, 3.0], ...],
    quaternion=[[0.0, 0.0, 0.0, 1.0], ...],  # xyzw
    size=[[1.0, 4.0, 2.0], ...],  # width, length, height
    class_id=labels.encode(["car", ...]),
    confidence=[0.9, ...],
    instance_id=instances.encode(["c28556c1...", ...]),
    velocity=[[1.0, 0.0, 0.0], ...],
)

store.log(
    "/estimation/objects",
    detections,
    at=TimePoint.at(frame=0, timestamp_ns=1_624_164_470_849_887_000),
    frame_id="base_link",
)
```

`orientation` は `wxyz` の `Quaternion` から **`xyzw` 順の配列**に変わる点に注意
(`BatchQuaternion` は `scipy` の規約に合わせている)。

### 評価の実行

```python
# 旧
manager = PerceptionEvaluationManager(evaluation_config=config)
for frame in frames:
    manager.add_frame_result(
        unix_time=frame.unix_time,
        ground_truth_now_frame=frame.gt,
        estimated_objects=frame.est,
        critical_object_filter_config=critical_config,
        frame_pass_fail_config=pass_fail_config,
    )
scene_score = manager.get_scene_result()
```

```python
# 新
pipeline = Pipeline(
    [
        FilterByDistanceSystem.on("/ground_truth/objects", max_distance=102.4),
        CenterDistanceMatchingSystem.between(
            "/estimation/objects",
            "/ground_truth/objects",
            threshold=1.0,
        ),
    ]
)
pipeline.run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())

# scene 全体
scene = store.range(
    "/matching/center_distance",
    timeline=FRAME,
    time_range=TimeRange.everything(),
).materialize(MatchResults)
scene.num_tp, scene.num_fp, scene.num_fn

# frame 単位 — 再計算なしで同じ store から取れる
frame_1 = store.range(
    "/matching/center_distance",
    timeline=FRAME,
    time_range=TimeRange.single(1),
).materialize(MatchResults)
```

### オブジェクトのフィルタ

```python
# 旧 — 行が消えるので、なぜ消えたかは残らない
filtered = filter_objects(objects, is_gt=True, max_distance=102.4, target_labels=labels)
```

```python
# 新 — 各フィルタの判定が entity の子パスに残る
src = "/ground_truth/objects"
region = FilterByRegionSystem.symmetric(src, max_xy=(102.4, 102.4))
label = FilterByLabelSystem.on(src, labels=["car", "bicycle", "pedestrian", "motorbike"])
points = FilterByNumPointsSystem.on(src, min_num_points=5)
keep = CombineMasksSystem.of(
    [region.target, label.target, points.target],
    f"{src}/filter/keep",
    mode="all",
)

ctx = SystemContext(store, FRAME, labels=labels, instances=instances)
Pipeline([region, label, points, keep]).run(ctx, TimeRange.everything())

# どのフィルタが落としたのかを個別に問える
store.range(region.target, timeline=FRAME, time_range=TimeRange.everything()).component(MASK)

# 通過した行だけを遅延 view で取る
passed = masked_view(
    store,
    src,
    keep.target,
    timeline=FRAME,
    time_range=TimeRange.everything(),
).materialize(Detections3D)
```

### 型判定

```python
# 旧
if isinstance(obj, DynamicObject) and obj.uuid is not None:
    ...  # tracking として扱う
```

```python
# 新 — 「その component を持つか」を問う
if batch.has(INSTANCE_ID):
    ...
# 「detection として扱えるか」
if batch.has(*Detections3D.required_descriptors()):
    ...
```

`isinstance(tracking, Detections3D)` は **False** になる (継承をやめたため)。
これは意図した変更で、詳細は [data_model.md](data_model.md) の「継承をやめた理由」を参照。

### 結果の保存

```python
# 旧
dict_result = class_to_dict(manager.frame_results)
json.dump(dict_result, f)
```

```python
# 新 — dtype と shape が schema で固定され、列指向で読み戻せる
chunk = store.range(
    "/matching/center_distance",
    timeline=FRAME,
    time_range=TimeRange.everything(),
).to_chunk()
write_parquet(chunk, "matching.parquet", labels=labels)

chunk, labels = read_parquet("matching.parquet")
result = MatchResults.from_chunk(chunk)
```

## 廃止された API

| 廃止                                                                | 代替                                                                                             |
| :------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------- |
| `t4perceval.dataclass` パッケージ                                   | `t4perceval.core` / `t4perceval.component` / `t4perceval.archetype`                              |
| `Header(timestamp_ns, frame_id)`                                    | `TimePoint` (時刻) + `Chunk.frame_id` (座標系)                                                   |
| `BatchDetection3D → BatchTracking3D → BatchPrediction3D` の継承     | 各 archetype が component を明示的に宣言                                                         |
| `BatchTrajectory3D` (3 配列を持つ component)                        | archetype に昇格。列は `BatchWaypoints3D` / `BatchModeConfidence` / `BatchTimeOffset` などに分解 |
| `BatchTrajectory3D.positions` / `.confidences` / `.time_offsets_ns` | `.waypoints` / `.mode_confidence` / `.time_offset`                                               |
| 各 component の `from_array()` / `as_array()`                       | 残置 (`ColumnarComponent` の基底実装)                                                            |
