# システム層設計

## 目的

元リポジトリの評価フローは `PerceptionEvaluationManager` に集約され、
`EvaluationTask` enum (`detection` / `tracking` / `prediction` / `detection2d` / `tracking2d` /
`classification2d`) による if 分岐が config・matching・metrics・visualization の全層に漏れていた。
さらに、閾値・ラベル設定・TP/FP 判定基準がすべて 1 つの `evaluation_config_dict` に混在し、
組み合わせの妥当性は実行時エラーでしか分からなかった。

システム層はこれを構造的に解消する。**評価タスクは enum の値ではなく、
「存在する component の集合」と「組んだパイプライン」である。**

## System protocol

```python
@runtime_checkable
class System(Protocol):
    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]]  # 必要な component
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]]  # 生成する component

    @property
    def sources(self) -> tuple[EntityPath, ...]: ...  # 読む entity
    @property
    def target(self) -> EntityPath: ...  # 書く entity

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]: ...
```

`SystemContext` は system 自身のパラメータ以外に必要なものを運ぶ。

```python
@define(frozen=True, slots=True)
class SystemContext:
    store: Store
    timeline: Timeline = FRAME
    labels: LabelRegistry | None = None
    instances: InstanceRegistry | None = None
```

2 つの registry があるおかげで、system はクラス名や UUID で設定できる。列に入っている生の整数を
呼び出し側が知っている必要はない。

`EntitySystem` が `sources` / `target` の配線を実装した基底クラスである。

### 中間結果を捨てない

すべての system は結果を `Chunk` として返し、`Pipeline` がそれを store に送る。
したがってフィルタが落とした理由やマッチングのスコアが store に残り、後段で再解析・可視化できる。
元リポジトリではこれらが破棄されていた。

フィルタは**行を削除せず boolean mask を出力する**。行が残るので、
「なぜこの物体が評価対象外になったか」を後から問える。

## Pipeline

```python
pipeline = Pipeline(
    [
        FilterByDistanceSystem.on("/estimation/objects", max_distance=102.4),
        CenterDistanceMatchingSystem.between(
            "/estimation/objects", "/ground_truth/objects", threshold=1.0
        ),
    ]
)
pipeline.run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())
```

### 構築時の検証

検証しているのは**順序**である。

- ある system が、より後ろの system の `target` を `sources` に持つ → エラー
  (実行時に空の結果が返るのではなく、その場で分かる)
- ある system の source が、より前の system の `target` である場合、
  その system の `PROVIDES` が `REQUIRES` を満たしているか検査する

store から来ることが期待される component は実行時に `require()` が検査する。
store の内容は構築時には分からないためである。

```python
# フィルタは mask しか書かないので、その entity を matching の source にはできない
Pipeline(
    [
        FilterByDistanceSystem.on("/estimation/objects"),
        CenterDistanceMatchingSystem.between(
            "/estimation/objects/filter/distance", "/ground_truth/objects"
        ),
    ]
)
# ValueError: ... reads /estimation/objects/filter/distance for component(s)
#             class_id, position, which no earlier system provides there
```

## フィルタ system

### `MaskSystem` — フィルタの共通基底

フィルタは述語だけを書く。クエリ・検証・空フレームの扱い・chunk 生成は `MaskSystem` が一度だけ実装する。

```python
@define(slots=True)
class FilterByConfidenceSystem(MaskSystem):
    REQUIRES = (CONFIDENCE,)
    FILTER_NAME = "confidence"  # 既定の target は <source>/filter/confidence

    min_confidence: float = field(default=0.0, kw_only=True)
    max_confidence: float = field(default=1.0, kw_only=True)

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        confidence = view.component(CONFIDENCE).values
        return (confidence >= self.min_confidence) & (confidence <= self.max_confidence)
```

`MaskSystem.on(source, *, name=None, **params)` が配線を担い、`target` は `<source>/filter/<name>`
になる。mask を source の子パスに置くので、prefix 検索でその entity と、それについて記録された
全判定がまとめて取れる。

### 一族で共通の規約

- **閾値は両端を含む (inclusive)**。したがって既定パラメータのフィルタは必ず全行を通す。
  元リポジトリは strict に比較していた (`score > threshold`、`abs(x) < max_x_position`) が、
  それだと `min_distance=0.0` が原点上の物体を落とす。浮動小数では境界値そのものは測度 0 であり、
  「既定値が確実に no-op」という性質のほうが価値が高いと判断した。
- **必要な列が無いのはエラーであり、通過ではない**。`t4_devkit` は velocity を持たない box を
  speed フィルタで素通しするが、ここでは「entity が持っていない component でフィルタしている」
  = パイプラインの配線ミスなので `require()` が報告する。ただし行数 0 の frame は
  ふつうの空フレームなので検査をスキップする。

### 実装済みフィルタ

| system                     | `REQUIRES`    | パラメータ                          | 旧 config                             |
| :------------------------- | :------------ | :---------------------------------- | :------------------------------------ |
| `FilterByDistanceSystem`   | `POSITION`    | `min_distance` `max_distance` `bev` | `max_distance` / `min_distance`       |
| `FilterByRegionSystem`     | `POSITION`    | `min_xy` `max_xy`                   | `max_x_position` / `max_y_position`   |
| `FilterByLabelSystem`      | `CLASS_ID`    | `labels` `exclude`                  | `target_labels` / `ignore_attributes` |
| `FilterByConfidenceSystem` | `CONFIDENCE`  | `min_confidence` `max_confidence`   | `confidence_threshold`                |
| `FilterByInstanceSystem`   | `INSTANCE_ID` | `instances` `exclude`               | `target_uuids`                        |
| `FilterBySpeedSystem`      | `VELOCITY`    | `min_speed` `max_speed`             | —                                     |
| `FilterByNumPointsSystem`  | `NUM_POINTS`  | `min_num_points` `max_num_points`   | `min_point_numbers`                   |
| `FilterByVisibilitySystem` | `VISIBILITY`  | `min_visibility`                    | —                                     |

個別の注意点:

- `FilterByDistanceSystem` の距離は source chunk が宣言する座標系で測る。position は測定基点
  (通常 `base_link`) を原点とする座標系に既に載っている必要がある。既定は 3D ノルム
  (`t4_devkit.filtering.FilterByDistance` と同じ) で、`bev=True` にすると xy 平面のみで測る —
  元リポジトリの `max_distance` はこちらだった。
- `FilterByRegionSystem.symmetric(source, max_xy=(102.4, 102.4))` が元 config の
  `max_x_position` / `max_y_position` にあたる対称領域を表す。上限だけを与える
  `on(max_xy=...)` とは違い、`min_xy` も `-max_xy` に固定される。
- `FilterByLabelSystem` / `FilterByInstanceSystem` はクラス名・UUID と生の id の両方を受ける。
  名前は `ctx.labels` / `ctx.instances` で解決する。**未知の名前は例外にする** —
  typo が黙って「何もマッチしないフィルタ」になると全下流指標が狂うため。
- `FilterByVisibilitySystem` は `UNAVAILABLE` の物体を常に通す
  (`t4_devkit.filtering.FilterByVisibility` と同じ)。落とす仕様にすると、visibility を
  付けていないデータセットが丸ごと空になる。

### `CombineMasksSystem` — mask の合成

```python
CombineMasksSystem.of([mask_a, mask_b], target, mode="all")  # 積 (AND)
CombineMasksSystem.of([mask_a, mask_b], target, mode="any")  # 和 (OR)
```

両方のモードを持つことで、**クラスごとの閾値をパラメータなしで表現できる**。
元 config の `min_point_numbers: [5, 0, 0]` (target_labels と対応した per-class 閾値) は
「クラス判定 AND 閾値判定」をクラスごとに作り、それらを OR する形になる。

```python
# car だけ 50 点以上を要求し、他クラスは無条件で通す
is_car = FilterByLabelSystem.on(src, labels=["car"], name="is_car")
not_car = FilterByLabelSystem.on(src, exclude=["car"], name="not_car")
many_points = FilterByNumPointsSystem.on(src, min_num_points=50, name="pts50")
car_ok = CombineMasksSystem.of(
    [is_car.target, many_points.target], f"{src}/filter/car_ok", mode="all"
)
keep = CombineMasksSystem.of([car_ok.target, not_car.target], f"{src}/filter/keep", mode="any")

Pipeline([is_car, not_car, many_points, car_ok, keep]).run(ctx, TimeRange.everything())
```

source の mask はすべて同じ行を記述していなければならない (通常は 1 つの source entity に
対するフィルタ群から来る)。これは仮定せず検査する。

### `masked_view()` — 通過した行だけを見る

```python
view = masked_view(
    store, "/estimation/objects", keep.target, timeline=FRAME, time_range=TimeRange.everything()
)
passed = view.materialize(Detections3D)
```

返るのは遅延 view であり、元の chunk を参照したままなので列を要求するまでコピーは起きない。

## マッチング system

### `MatchingSystem` — マッチングの共通基底

マッチャーはスコア行列だけを書く。frame ループ・可否判定・割当は `MatchingSystem` が一度だけ実装する。

```python
@define(slots=True)
class CenterDistanceMatchingSystem(MatchingSystem):
    REQUIRES = (POSITION, CLASS_ID)
    MATCHING_NAME = "center_distance"  # 既定の target は /matching/center_distance
    DEFAULT_THRESHOLD = 1.0
    # HIGHER_IS_BETTER = False            # 距離なので小さいほど良い (既定)

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        est = est_view.component(POSITION).values
        gt = gt_view.component(POSITION).values
        return np.linalg.norm(est[:, None, :] - gt[None, :, :], axis=-1)
```

`MatchingSystem.between(estimation, ground_truth, *, target=None, **params)` が配線を担う。

### 割当

frame ごとに `scipy.optimize.linear_sum_assignment` で**大域最適な 1 対 1 割当**を解く。
貪欲マッチングだと、先に選んだペアのせいで成立しうるペアを取り逃す。

割り当てないペア (可否判定に落ちたもの) は次の条件で決まる:

- スコアが閾値を満たさない。`HIGHER_IS_BETTER=False` なら `score <= threshold`、
  `True` なら `score >= threshold` が可の条件。
- スコアが有限でない (位置が NaN の推定など)。
- `class_agnostic=False` のときクラスが一致しない。

`linear_sum_assignment` は禁止ペアを表現できないため、実装は禁止ペアに可行なコストより大きい値を
入れ、割当後に棄却する。`HIGHER_IS_BETTER=True` のモードはコストを `-score` にして最大化に変える。

出力の `est_index` / `gt_index` は**そのframe内の行 index** であり、`-1` は「対応なし」を意味する。
`matching_score` にはコストではなく**そのモードの自然な値** (距離なら m、IoU なら比) が入り、
未マッチ行は `NaN` になる。

### 実装済みモード

| system                            | `REQUIRES`                                | 良い方向 | 既定閾値 | target                          | 旧 config                        |
| :-------------------------------- | :---------------------------------------- | :------- | :------- | :------------------------------ | :------------------------------- |
| `CenterDistanceMatchingSystem`    | `POSITION` `CLASS_ID`                     | 小       | 1.0      | `/matching/center_distance`     | `center_distance_thresholds`     |
| `CenterDistanceBEVMatchingSystem` | `POSITION` `CLASS_ID`                     | 小       | 1.0      | `/matching/center_distance_bev` | `center_distance_bev_thresholds` |
| `PlaneDistanceMatchingSystem`     | `POSITION` `QUATERNION` `SIZE` `CLASS_ID` | 小       | 2.0      | `/matching/plane_distance`      | `plane_distance_thresholds`      |
| `IoUBEVMatchingSystem`            | `POSITION` `QUATERNION` `SIZE` `CLASS_ID` | 大       | 0.5      | `/matching/iou_bev`             | `iou_2d_thresholds` (3D タスク)  |
| `IoU3DMatchingSystem`             | `POSITION` `QUATERNION` `SIZE` `CLASS_ID` | 大       | 0.5      | `/matching/iou_3d`              | `iou_3d_thresholds`              |
| `IoURoiMatchingSystem`            | `ROI` `CLASS_ID`                          | 大       | 0.5      | `/matching/iou_roi`             | `iou_2d_thresholds` (2D タスク)  |

モードごとに target が違うので、同じ frame に対して複数モードを同時に走らせて比較できる。

個別の注意点:

- **BEV 中心距離を別クラスにした理由**: マッチングモードは後段の指標が読む entity を指す名前なので、
  3D 距離と BEV 距離が同じ target を共有してはならない。フィルタ側の `bev` フラグと扱いが違うのは、
  フィルタの mask は「フィルタ対象 entity」の話であって、モードの同一性ではないため。
- **`PlaneDistanceMatchingSystem`** は 2 つの箱の**最も近い面**を比べる。自車から見て近い面の
  左右頂点の距離の二乗平均平方根であり、遠い面だけずれていてもスコアは 0 になる。
  センサが実際に観測できた面だけを評価するので、中心距離とは違う性質を持つ。
  position は原点からの距離を測る座標系 (通常 `base_link`) に載っている必要がある。
- **IoU を 2 クラスに分けた理由**: BEV IoU は `POSITION` / `QUATERNION` / `SIZE` を要求し、
  画像 IoU は `ROI` を要求する。要求する component が違えば別の system である。
  「どの component があるかを見て分岐する 1 クラス」は、この設計が取り除こうとしている
  隠れた分岐そのものなので採らない。
- **回転した footprint** は軸平行の重なりでは計算できない。`t4perceval.geometry` が
  shapely のベクトル化 API で多角形クリップを行う (直交する 2 つの箱を「完全一致」と
  誤判定しないため)。

### クラスごとの閾値

`threshold` は数値のほか、クラスをキーにした `Thresholds` を受ける。
マッチング system は元々 `CLASS_ID` を要求しているので、追加の component は要らない。

```python
CenterDistanceMatchingSystem.between(
    "/estimation/objects",
    "/ground_truth/objects",
    threshold=Thresholds(1.0, by_class={"car": 2.0, "pedestrian": 0.5}),
)
```

元 config の `center_distance_thresholds: [[1.0, 1.0, 1.0, 1.0]]` は `target_labels` の
順序を知らないと何に対する数値か分からなかった。`Thresholds` は明示的にクラスをキーにするので、
読んだだけで対象が分かる。

閾値は**正解側のクラス**で引く。`class_agnostic=True` のときは両者のクラスが食い違いうるため、
どちらかに決める必要があり、正解を権威とした。クラス名で書いた場合は `ctx.labels` で解決し、
未知の名前は例外にする (既定値が黙って使われると、per-class 閾値が効いたように見えてしまう)。

### 幾何

`t4perceval.geometry` が箱の幾何をベクトル化して持つ。列全体と、さらに**ペア**に対して働く —
マッチャーが必要とするのは `(N, M)` のスコア行列なので、ペアごとに Python から呼ぶのではなく
行列を直接組み立てる。

| 関数                                                 | 返り値                                  |
| :--------------------------------------------------- | :-------------------------------------- |
| `bev_corners(position, quaternion, size)`            | `(N, 4, 2)` footprint 頂点              |
| `bev_area(size)` / `volume(size)`                    | `(N,)`                                  |
| `pairwise_bev_iou(...)` / `pairwise_volume_iou(...)` | `(N, M)`                                |
| `pairwise_roi_iou(est_roi, gt_roi)`                  | `(N, M)` 軸平行なので多角形クリップ不要 |
| `pairwise_plane_distance(...)`                       | `(N, M)`                                |
| `canonical_bev_corners(corners)`                     | 重心まわりに反時計回りへ並べ替え        |

## 座標系

異なる座標系をまたいだ幾何計算には意味がなく、しかも下流の誰も気づかない。`map` の位置から
`base_link` の位置を引いてもエラーにはならず数値が出るだけで、その上に載る指標はもっともらしく
見える。そこで system 層はこの比較を拒否する。

```python
from t4perceval.system.base import require_same_frame, resolve_frame
```

| ヘルパー                     | 挙動                                                                 |
| :--------------------------- | :------------------------------------------------------------------- |
| `require_same_frame(*views)` | 宣言された座標系を返し、2 つの view が別の座標系を宣言していれば例外 |
| `resolve_frame(*views)`      | 検査せず最初に宣言された座標系を返す (入力が 1 つの system 用)       |

覆う経路は 2 箇所だけで足りる: `MatchingSystem` が呼ぶので 6 つのマッチャー全てが検査され、
`MatchJoin.of` が呼ぶので幾何を使う指標は全て、既に通っている join 1 箇所で検査される。

- **宣言されていない**座標系は不一致ではない。entity が範囲内に何も持たないとき、座標系なしで
  記録されたデータのとき、幾何ではなく指標を保持しているときに view は `None` を持つ。
  例外になるのは _異なる値が 2 つ宣言された_ 場合だけである。
- 行数は意図的に見ない。オブジェクト 0 個の frame も何らかの座標系で記録されており、それを飛ばすと
  出力の座標系が scene の中でちらつき、`concat_chunks` が結合を拒否する。
- 見るのは **temporal** chunk の座標系だけである。static 列の座標系は broadcast 先の行を説明すると
  は限らない — transform はそこに辺の「親」を書く — ので、`base_link` で log された static な
  `time_offset` が `map` での比較を例外にしてはいけない。
- `check_frames=False` で system ごとに無効化できる。store 組み立て側の `require_same_frame_id`
  と対になっており、これがないとそのフラグで組んだ store がマッチング不能になる。

これは対策ではなく検知である。transform の**解決**は実装済みで
(`TransformResolver.lookup(target_frame=..., source_frame=..., at=...)`、
[data_model.md](data_model.md) の「transform」を参照)、まだ無いのは entity の行を別座標系へ書き換える
system である — passthrough な system が自分の運ぶ列を宣言できないため。それまで、入力を 1 つの
座標系に揃えるのは呼び出し側の仕事であり、この検知がその代替案をもっともらしく見せないための砦である。

## 未実装 system の配置

以下は同じ protocol 上に載る。predicate やコスト計算だけが違う。指標 system は `HotaSystem` を
除いて実装済みで、`PassFailSystem` は未実装である。

### 指標 (`PROVIDES = /metrics/* の descriptors`)

| system                                                      | `REQUIRES`                                                 | source                           |
| :---------------------------------------------------------- | :--------------------------------------------------------- | :------------------------------- |
| `MeanAveragePrecisionSystem`                                | `MATCH_STATUS` `MATCHING_SCORE` `CONFIDENCE` `CLASS_ID`    | `/matching/<mode>` + 推定 entity |
| `ClearSystem` (MOTA / MOTP / IDSwitch)                      | `MATCH_STATUS` `EST_INDEX` `GT_INDEX` `INSTANCE_ID`        | `TimeRange` 全体                 |
| `HotaSystem`                                                | 同上                                                       | `TimeRange` 全体                 |
| `PathDisplacementSystem` (ADE / FDE)                        | `MATCH_STATUS` `WAYPOINTS` `MODE_CONFIDENCE` `TIME_OFFSET` | `TimeRange` 全体                 |
| `ClassificationSystem` (accuracy / precision / recall / F1) | `MATCH_STATUS` `CLASS_ID`                                  | frame or scene                   |

フレーム跨ぎの指標 (CLEAR / HOTA / ADE) は `at` に `TimeRange` を受け取り、
`store.range()` で scene 全体を 1 回のクエリで読む。旧構造の `List[PerceptionFrameResult]` を
Python で歩く必要はない。

### pass/fail

| system           | `REQUIRES`                                 |
| :--------------- | :----------------------------------------- |
| `PassFailSystem` | `MATCH_STATUS` + critical object の `MASK` |

`CriticalObjectFilterConfig` は「critical 判定用のフィルタ system 群」に、
`PerceptionPassFailConfig` は `PassFailSystem` のパラメータになる。

## `EvaluationTask` enum と `evaluation_config_dict` の解体

### enum の解体

| 旧 `evaluation_task` | 新しい表現                                         |
| :------------------- | :------------------------------------------------- |
| `detection`          | `Detections3D` の component + matching + mAP       |
| `tracking`           | 上記 + `INSTANCE_ID` + CLEAR / HOTA                |
| `prediction`         | 上記 + `WAYPOINTS` / `MODE_CONFIDENCE` + ADE / FDE |
| `detection2d`        | `Detections2D` の component + IoU2D matching + mAP |
| `tracking2d`         | 上記 + `INSTANCE_ID` + CLEAR                       |
| `classification2d`   | `Classifications2D` の component + classification  |

分岐が消えるのは、system が「必要な component があるか」だけを見るからである。
tracking データを detection として評価したければ、tracking entity をそのまま detection 用の
pipeline に通せばよい (`tracking.has(*Detections3D.required_descriptors())` が True)。

### config dict の解体

1 つの巨大 dict を、system ごとのパラメータ dataclass に分解する。

```python
# 旧
evaluation_config_dict = {
    "evaluation_task": "detection",
    "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
    "max_x_position": 102.4,
    "max_y_position": 102.4,
    "min_point_numbers": [0, 0, 0, 0],
    "label_prefix": "autoware",
    "merge_similar_labels": False,
    "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
    "iou_3d_thresholds": [0.5],
    ...
}

# 新
labels = LabelRegistry.from_names(["car", "bicycle", "pedestrian", "motorbike"])
pipeline = Pipeline([
    FilterByRegionSystem.symmetric("/ground_truth/objects", max_xy=(102.4, 102.4)),
    FilterByNumPointsSystem.on("/ground_truth/objects", min_num_points=0),
    CenterDistanceMatchingSystem.between("/estimation/objects", "/ground_truth/objects",
                                         threshold=1.0),
    IoU3DMatchingSystem.between("/estimation/objects", "/ground_truth/objects",
                                threshold=0.5),
    MeanAveragePrecisionSystem.on("/matching/center_distance"),
])
```

得られる性質:

- パラメータの妥当性は各 system の `__attrs_post_init__` が構築時に検査する
  (旧: `RuntimeError: Either max x/y position or max/min distance should be specified` を実行時に受ける)
- 「どのフィルタが効いているか」がパイプラインの記述から読める
- 同じ閾値の別モードや、別閾値の同モードを、target を変えて並列に走らせられる
