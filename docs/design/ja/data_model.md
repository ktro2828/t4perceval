# データモデル設計

## 背景と目的

元の [`autoware_perception_evaluation`](https://github.com/tier4/autoware_perception_evaluation) は、
1 オブジェクト = 1 Python オブジェクト (`DynamicObject`) というモデルを採っていた。この構造は次の問題を生む。

- `DynamicObject` が position / orientation / shape / velocity / tracked*\* / predicted*\* を 20 以上の
  フィールドとして 1 クラスに抱える。タスクによって使わないフィールドが `None` で埋まる。
- `List[DynamicObject]` を Python ループで回すため、ベクトル化できない。
- `EvaluationTask` enum が config・matching・metrics・visualization の全層に if 分岐として漏れる。
- `Catalog → Scenario → Scene → PerceptionFrameResult` が list のネストで表現され、
  フレーム跨ぎのクエリ (CLEAR / HOTA / ADE) が書きづらい。
- フィルタ結果やマッチングコストといった中間結果が破棄され、再解析できない。

`t4perceval` は [rerun](https://github.com/rerun-io/rerun) のデータモデルを**自前実装**することでこれを解消する。
rerun 自体は依存に入れない (後述の「rerun との関係」を参照)。

## レイヤ構成

```text
                        ┌─────────────────────────────────────────┐
   t4perceval.system    │  System / Pipeline                      │  ECS の "S"
                        │  filter · matching · metric · pass/fail │
                        └────────────────┬────────────────────────┘
                                         │ reads / writes Chunk
                        ┌────────────────▼────────────────────────┐
   t4perceval.core      │  Store          timeline ごとのクエリ    │
                        │   ├ latest_at(entity, at)   → EntityView │
                        │   ├ range(entity, range)    → EntityView │
                        │   └ static                               │
                        ├──────────────────────────────────────────┤
                        │  Chunk    entity_path + indexes          │
                        │           + offsets + columns            │
                        ├──────────────────────────────────────────┤
                        │  Archetype   component の束 (合成)        │
                        ├──────────────────────────────────────────┤
                        │  Component   1 列の NumPy 配列            │
                        │  ComponentDescriptor  列の同一性          │
                        │  EntityPath           ストリームの住所     │
                        │  Timeline             時間軸              │
                        └──────────────────────────────────────────┘
                        ┌──────────────────────────────────────────┐
   t4perceval.transform │  Transform3D の行 -> FrameGraph          │  座標系もデータ
                        │  transform_edges · TransformResolver     │
                        └──────────────────────────────────────────┘
                        ┌──────────────────────────────────────────┐
   t4perceval.io        │  Chunk ↔ pyarrow.Table ↔ Parquet         │
                        └──────────────────────────────────────────┘
```

## EntityPath — 「何のデータか」

`/`区切りの階層パス。元リポジトリで `frame_id` + task enum + est/gt の別を型やフィールドで表していたものを、
すべてパスの区別に移す。

```python
EntityPath.parse("/estimation/objects")  # 推定結果
EntityPath.parse("/ground_truth/objects")  # 正解
EntityPath.parse("/estimation/objects") / "filter" / "distance"
```

### 規約

| パス                                | 内容                                            |
| :---------------------------------- | :---------------------------------------------- |
| `/ground_truth/objects`             | 正解オブジェクト                                |
| `/estimation/objects`               | 推定オブジェクト                                |
| `/estimation/objects/<channel>`     | センサchannel別の推定                           |
| `/estimation/objects/filter/<name>` | そのentityに対するフィルタ判定 (mask)           |
| `/matching/<mode>`                  | マッチング結果 (`center_distance`, `iou_3d`, …) |
| `/metrics/<name>`                   | 指標値 (`map`, `clear`, …)                      |

フィルタ判定を**source の子パス**に置くのは意図的である。prefix 検索でそのentityと、
それについて記録された全判定がまとめて取れる。

## ComponentDescriptor — 列の同一性

```python
@define(frozen=True, slots=True)
class ComponentDescriptor:
    component: str  # 同一性はこれだけで決まる
    archetype: str | None  # eq=False。provenance のヒント
    component_type: str | None  # eq=False。component クラス名のヒント
```

**重要な設計判断**: descriptor 名は archetype 非依存にする。
`Detections3D` と `Trackings3D` は同じ 3D 中心を、どちらも `POSITION` (= `"position"`) として公開する。

```python
Detections3D.descriptor_of("position") == Trackings3D.descriptor_of("position")  # True
```

これにより System は `REQUIRES = (POSITION,)` と宣言するだけで、
3D 位置を持つ任意の entity に対して動く。rerun は `Points3D:positions` / `Boxes3D:positions` のように
archetype 名を含めるが、評価ツールでは「archetype 横断で同じ列を要求する」ほうが本質的なので、
そこは意図的に変えている。

正規の descriptor は `t4perceval/descriptors.py` に集約されている。

## Component — 1 列

`ColumnarComponent` が `values` フィールド・`__len__`・`select()`・Arrow 変換を**一度だけ**実装する。
サブクラスは列のレイアウトを ClassVar で宣言するだけでよい。

```python
@define(frozen=True, slots=True)
class BatchPosition3D(ColumnarComponent):
    SHAPE = (3,)  # 行あたりの形状。() はスカラ列。ANY はデータから推論
    DTYPE = np.float64  # すべての値をこの dtype に正規化
    # VALUE_RANGE = (0.0, 1.0)   # 任意。閉区間の値域
    # REQUIRE_FINITE = True      # 任意。NaN / Inf を拒否
```

### 不変条件

- `values` は常に **read-only** (`values.flags.writeable is False`)。frozen dataclass の不変性と整合する。
- 呼び出し側が渡した**書き込み可能な**配列とメモリを共有しない。呼び出し側のオブジェクトを勝手に
  read-only にすることはない。
- `select()` は常に独立したデータを返す (fancy index による copy)。遅延 view は `EntityView` の担当。
- 空入力 (`BatchPosition3D([])`) は 0 行として受理する。ただし `SHAPE` に `ANY` を含む場合は
  形状が一意に決まらないため `empty()` を使う。

### 列の一覧

| モジュール      | component                                                                                                                               | 形状 / dtype                   |
| :-------------- | :-------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------- |
| `vector.py`     | `BatchVector2D` / `BatchVector3D`                                                                                                       | `(2,)` / `(3,)` f64            |
| `geometry.py`   | `BatchPosition3D` `BatchPosition2D` `BatchQuaternion` `BatchVelocity` `BatchSize3D` `BatchSize2D`                                       | `BatchQuaternion` は `xyzw` 順 |
| `scalar.py`     | `BatchClassId` (i32) `BatchConfidence` (f64, `[0,1]`) `BatchInstanceId` (i64) `BatchNumPoints` (i32) `BatchVisibility` (i8)             | `()`                           |
| `image.py`      | `BatchRoi` (i32, `(4,)`, `(x_min, y_min, height, width)`) `BatchPixel` (i32)                                                            |                                |
| `mask.py`       | `BatchMask` (bool)                                                                                                                      | `()`                           |
| `trajectory.py` | `BatchWaypoints3D` `(M,T,3)` `BatchModeConfidence` `(M,)` `BatchModeValid` `(M,)` `BatchTimestepValid` `(M,T)` `BatchTimeOffset` `(T,)` |                                |
| `matching.py`   | `BatchRowIndex` (i64) `BatchMatchingScore` (f64) `BatchMatchStatus` (i8)                                                                | `()`                           |

`BatchVector3D` は `BatchVector2D` を**継承しない**。3D ベクトルは 2D ベクトルではないため、
`isinstance` が嘘をつかないようにしている。

## Archetype — component の束 (合成)

```python
@define(frozen=True, slots=True)
class Trackings3D(Archetype):
    position = component_field(POSITION, BatchPosition3D)
    quaternion = component_field(QUATERNION, BatchQuaternion)
    size = component_field(SIZE, BatchSize3D)
    class_id = component_field(CLASS_ID, BatchClassId)
    confidence = component_field(CONFIDENCE, BatchConfidence)
    instance_id = component_field(INSTANCE_ID, BatchInstanceId, kw_only=True)
    velocity = component_field(VELOCITY, BatchVelocity, optional=True, kw_only=True)
```

### 継承をやめた理由

以前は `BatchDetection3D → BatchTracking3D → BatchPrediction3D` の継承チェーンだった。これは ECS の合成と逆向きで、

- `select()` が 3 箇所でほぼ同一実装として重複する
- 「trajectory は持つが instance_id は持たない」といった組み合わせが表現できない
- `isinstance(tracking, Detections3D)` が「tracking は detection の一種」という嘘をつく

いまは `Trackings3D` が box 系 component を**明示的に再宣言**する。descriptor は同一なので:

```python
tracking.has(*Detections3D.required_descriptors())  # True
isinstance(tracking, Detections3D)  # False
```

`has()` は System が `REQUIRES` で問うのと同じ質問であり、`isinstance` の正しい置き換えである。

### 基底が提供するもの

`select()` は `attrs.fields()` を走査して各 component に委譲するため、**archetype ごとの実装が不要**。
`as_components()` / `from_components()` / `to_chunk()` / `from_chunk()` / `has()` /
`descriptors()` / `required_descriptors()` も基底の 1 実装で済む。

### archetype 一覧

| archetype                | components                                                                                    |
| :----------------------- | :-------------------------------------------------------------------------------------------- |
| `Detections3D`           | position, quaternion, size, class_id, confidence, [velocity], [num_points], [visibility]      |
| `Detections2D`           | roi, class_id, confidence, [visibility]                                                       |
| `Trackings3D`            | Detection3D の各列 + instance_id                                                              |
| `Trackings2D`            | Detection2D の各列 + instance_id                                                              |
| `Predictions3D`          | Tracking3D の各列 + waypoints, mode_confidence, [mode_valid], [timestep_valid], [time_offset] |
| `Classifications2D`      | class_id, confidence, [instance_id]                                                           |
| `SemanticSegmentation2D` | pixel, class_id                                                                               |
| `SemanticSegmentation3D` | point, class_id                                                                               |
| `Trajectories3D`         | waypoints, mode_confidence, [mode_valid], [timestep_valid], [time_offset]                     |
| `MatchResults`           | est_index, gt_index, matching_score, match_status                                             |

### trajectory の設計判断

- `M` (mode 数) と `T` (timestep 数) は **1 インスタンス内で固定**。可変長は
  `BatchModeValid` / `BatchTimestepValid` の padding + mask で表現する。
- `mode_confidence` は「mode 事後確率」と定義するが、**有効 mode 合計 1 を検証しない**。
  正規化しないモデルが実在し、ここで正規化すると指標が黙って変わる。値域 `[0,1]` と有限性のみ検証する。
- `waypoints` は有限値を要求する。無効 timestep は `timestep_valid=False` で表し、`NaN` を入れない。
- `time_offset` は `(N, T)` の列。全行共通なら `N=1` の **static** として log すればよい。

## Timeline — 時間軸

```python
class TimeKind(Enum):
    SEQUENCE  # frame index 等の単調カウンタ
    TIMESTAMP  # Unix epoch からの ns
    DURATION  # 経過 ns


FRAME = Timeline("frame", TimeKind.SEQUENCE)
TIMESTAMP = Timeline("timestamp_ns", TimeKind.TIMESTAMP)
```

1 つのデータは複数 timeline に同時に乗る。`TimePoint.at(frame=3, timestamp_ns=...)` で両方指定でき、
クエリ側は好きな軸を選べる。

### `Header` の解体

旧 `Header(timestamp_ns, frame_id)` は廃止した。

- `timestamp_ns` → `TIMESTAMP` timeline 上の値 (`Chunk.indexes`)
- `frame_id` → 「その chunk の全行が乗る座標系」として `Chunk.frame_id`

これにより archetype は純粋な component の束になり、構築時に header を引き回さなくなった。

> **決定済み**: rerun は座標系を entity path 階層 + `Transform3D` archetype で表す
> (例: `/base_link/estimation/objects`)。認識データについては `Chunk.frame_id` を維持する。
> entity path は「そのデータが何か」を表すものであり、座標系を畳み込むと、座標系に関心のない
> system から `/ground_truth/objects` を指定できなくなる。
>
> ~~transform の辺は逆で、path に置く: `/transforms/<parent>/<child>` に
> `translation` と `rotation` を持つ `Transform3D` を記録する。ここでは座標系の組そのものが
> データの同一性である。~~
>
> **撤回**: transform は **親**を `Chunk.frame_id` で (このフィールドが他の場所で持つ意味と同じ)、
> **子**を `child_frame_id` 列で表す。ROS が `TransformStamped` を分けているのと同じ形である。
> 決め手は、path に入れた版ではできなかった 2 点: 座標系名が path として安全でなければならず
> (`/robot1/base_link` が表現できない)、座標系グラフを別の場所に置き直すと名前まで変わってしまう。
> 詳細は後述の「transform」を参照。

## Chunk — 列指向テーブル

```python
@define(frozen=True, slots=True)
class Chunk:
    entity_path: EntityPath
    indexes: tuple[TimeColumn, ...]  # 長さ P (partition 数)
    offsets: NDArrayI64  # 長さ P+1、行境界
    columns: dict[ComponentDescriptor, Component]  # 各長さ N = offsets[-1]
    frame_id: str | None = None
    is_static: bool = False
```

### 行 = オブジェクト という選択

rerun の chunk は「行 = log 呼び出し、セル = 可変長 component batch」である。
`t4perceval` は**行 = オブジェクト 1 個**に平坦化し、フレーム境界を `offsets` で持つ。

理由: 評価計算はオブジェクト方向の要素ごと演算 (距離行列、IoU、TP/FP 判定) が中心であり、
平坦な contiguous 配列に対する NumPy 演算に直接落ちる。フレームごとに list offset を歩く必要がない。
`offsets` があるのでフレーム単位の集計 (`partition(i)`, `partition_ids()`) も O(1) で取れる。

```text
offsets = [0, 2, 5]        indexes[frame].times = [0, 1]
           │  │  └── frame 1 は行 2..4 (3 オブジェクト)
           │  └───── frame 0 は行 0..1 (2 オブジェクト)
           └──────── 常に 0
```

### 不変条件

- 全 `indexes` の長さ == `num_partitions`
- 全 `columns` の長さ == `num_rows` == `offsets[-1]`
- `offsets[0] == 0` かつ単調非減少
- timeline の重複禁止
- `is_static=True` の chunk は `indexes` が空で `num_partitions == 1`

### `select()` の契約

`Chunk.select()` は partition 構造を保つ。そのため **partition を跨いで行を並べ替える選択は拒否する**
(boolean mask と昇順 index は常に条件を満たす)。partition が空になっても index の項目は残すので、
時間軸は失われない。

## Store — recording

```python
store.send_chunk(chunk)
store.log(entity_path, archetype, at=TimePoint.at(frame=0), frame_id="base_link")
store.log_static(entity_path, archetype, frame_id="base_link")

store.latest_at(entity_path, timeline=FRAME, at=12)  # → EntityView
store.range(entity_path, timeline=FRAME, time_range=TimeRange(0, 99))  # → EntityView
```

`Catalog → Scenario → Scene → List[PerceptionFrameResult]` のネストを置き換える。

| 旧構造              | 新しい表現                                 |
| :------------------ | :----------------------------------------- |
| 1 frame             | `store.latest_at(...)`                     |
| 1 scene             | `store.range(..., TimeRange.everything())` |
| 全 frame 共通の情報 | `store.log_static(...)`                    |

`static` は「**時間**に依存しない」という主張であり、データの種類についての主張ではない。どの
archetype でもどちらでも log できるので、センサーのキャリブレーションは static な `Transform3D`、
ego pose は temporal な `Transform3D` であり、2 つ目の archetype は要らない。

### セマンティクス (rerun 準拠)

- **static データは全 timeline に属する**。同一 entity・同一 descriptor の時系列データより**優先**する。
  1 行の static 列は view の行数に broadcast される。
- static な書き込みは **chunk のまま**保持されるので `frame_id` が残る。`static_chunks()` と
  `static_frame_id()` がそこに届き、`static()` は従来どおり列だけを返す (descriptor 単位で後勝ち)。
- したがって **static だけの** entity は `latest_at` / `range` から _0 行_ として読み出される。view は
  「1 つの temporal chunk + broadcast される overlay」であり、broadcast する先の行数が存在しない。
  これは意図的である: 時刻クエリから static 行を出すと、オブジェクトのない frame にオブジェクトを
  でっちあげ、view に時刻を尋ねる system に index を持たない chunk を渡すことになる。static 行が
  必要な読み手は chunk を要求する。
- `EntityView.frame_id` は **temporal** chunk の座標系だけを返す。static 列の座標系は行を説明すると
  は限らない — transform の座標系はその辺の**親**である — ので、通してしまうと無関係な static 列が
  座標系チェックを踏む。
- `latest_at` は指定時刻**以下**の最新 partition を返す。同時刻が複数あれば**後に log したもの**が勝つ。
- `range` は partition を**時刻順**に並べる (同時刻は log 順)。
- 1 entity に異なる列構成の chunk を log してもよい。`range` がそれらを跨いだときに初めてエラーになる
  (`latest_at` は単一 chunk しか見ないので常に成立する)。

## EntityView — 遅延 view

```python
@define(frozen=True, slots=True)
class EntityView:
    chunk: Chunk
    indices: NDArrayI64  # chunk の行への正規化済み index
    static: dict[ComponentDescriptor, Component]
```

`select()` は index を合成するだけで**コピーしない**。実体化は `component()` / `materialize()` /
`to_chunk()` の時点で起きる。

```python
view.select(slice(None, None, 2)).select([1])  # コピーなし
view.component(POSITION)  # ここで 1 列だけコピー
view.materialize(Detections3D)  # archetype として実体化
```

### copy / view の契約

| API                   | 挙動                                          |
| :-------------------- | :-------------------------------------------- |
| `Component.select()`  | 独立したデータ (コピー)                       |
| `Archetype.select()`  | 独立したデータ (コピー)                       |
| `Chunk.select()`      | 独立したデータ (コピー)、partition 構造は保持 |
| `EntityView.select()` | index 合成のみ (コピーなし)                   |

TODO.md が要求していた `BatchDetection3DView` / `BatchTracking3DView` / `BatchPrediction3DView` の
3 クラスは不要になった。archetype 型は `materialize()` の引数なので、汎用の 1 クラスで足りる。

## LabelRegistry / InstanceRegistry — 数値の意味

`BatchClassId` (i32) と `BatchInstanceId` (i64) はただの整数列である。その意味を持つのは registry だけ。

```python
labels = LabelRegistry.from_names(["car", "truck", "pedestrian"])
labels.class_id("truck")  # 1
labels.encode(["car", "truck"])  # BatchClassId 用の i32 列

merged = labels.merged({"vehicle": ["car", "truck"]})
merged.class_id("car") == merged.class_id("vehicle")  # True
```

元リポジトリの `LabelConverter` + `label_prefix` + `merge_similar_labels` + `count_label_number` を置き換える。
`merge_similar_labels` フラグを match 時に読むのではなく、**merge 済みの新しい registry を作る**ため、
マージがデータとして見える。

registry は**列ではなくメタデータ**である (名前は数値でなく、registry は行ではなく recording 全体を記述する)。
そのため `SystemContext.labels` として、また Arrow schema metadata として運ばれる。
rerun が static `AnnotationContext` に与えている役割にあたる。

文字列 UUID ↔ `BatchInstanceId` は `InstanceRegistry` が担い、`SystemContext.instances` として
同じように運ばれる。id は 1 registry の生存期間で安定であり、これが tracking 指標のフレーム跨ぎ
同一性判定に必要な性質である。`intern()` は未知の UUID に新しい id を振るが、`instance_id()` は
振らずに例外を出す — フィルタが必要とするのは後者で、typo した UUID が黙って新しい identity に
なってはいけない。

## transform — 座標系をデータとして記録する

transform は service が抱える隠れた状態ではなく、他の観測と同じ 1 つのデータである。1 行 = 1 辺で、
ROS が `TransformStamped` を分けているのと同じように chunk と列に分かれる:

| ROS `TransformStamped`  | `t4perceval`                 |
| :---------------------- | :--------------------------- |
| `header.frame_id`       | `Chunk.frame_id` — 親        |
| `child_frame_id`        | `Transform3D.child_frame_id` |
| `transform.translation` | `Transform3D.translation`    |
| `transform.rotation`    | `Transform3D.rotation`       |

1 行は child 座標系の点を parent へ写す: `p_parent = R p_child + t`。

親が chunk 側にあるのは、`frame_id` がもともと「この行が乗っている座標系」を意味しており、それが
transform の行についてもそのまま成り立つからである。

### transform は 1 辺なので component は mono

`Transform3D` はこのパッケージで唯一、component が **mono** (`MonoComponent`) の archetype である。
他の archetype は _N_ 個のオブジェクトを表すので全列が batch だが、transform が表すのは 1 つの関係で
あり、1 entity が 1 時刻に持つのはちょうど 1 つである。したがって並進は `(3,)` の値、子座標系は
`str` になる:

```python
pose = Transform3D(
    translation=[1.2, 0.0, 1.8], rotation=[0.0, 0.0, 0.0, 1.0], child_frame_id="lidar"
)
pose.translation.value  # array([1.2, 0. , 1.8])
pose.child_frame_id.name  # 'lidar'
```

添字を取る行が存在せず、「3 行あったらどうなるか」は型に対して問えない — 構築時に例外になる。

**mono は境界の型であり、store は列指向のままである。** `as_components()` が chunk へ入る手前で
mono を `BATCH` の相手方に広げ (`Position3D` → `BatchPosition3D`、`FrameId` → `BatchFrameId`)、
archetype の field converter が戻すときに狭める。この分担は本質的である: `Store.range` は partition を
連結するので、1 つの辺の 3 サンプルにまたがるクエリは 3 行の列を返す — 「ちょうど 1 行」を要求する型
ではありえない。よって chunk が持つのは常に batch の列である:

```python
scene.range("/tf/base_link", timeline=FRAME, time_range=EVERYTHING).component(TRANSLATION)
# 3 行の BatchPosition3D — ego の軌跡

Transform3D.from_chunk(scene.static_chunks("/tf/LIDAR_TOP")[0]).translation.value
# array([0., 0., 2.]) — 1 行は 1 つの値に戻る
```

複数行の view を `Transform3D` として materialize するのは、先頭行を黙って採るのではなく例外になる。
系列が欲しいときは列を読み、辺が欲しいときに materialize する。

どちらの座標系も entity path には入れない。path は「どこに置いたか」であり、座標系は「グラフの
ノード名」である。混ぜると、座標系名が path として安全でなければならず (`/robot1/base_link` が
表現できない)、ツリーの置き場所を変えると名前まで変わってしまう。

```python
store.log_static(  # キャリブレーション: 固定
    "/tf/lidar",
    Transform3D(translation=[1.2, 0.0, 1.8], rotation=[0.0, 0.0, 0.0, 1.0], child_frame_id="lidar"),
    frame_id="base_link",
)
store.log(  # ego pose: frame ごと
    "/tf/base_link",
    Transform3D(
        translation=[10.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0, 1.0], child_frame_id="base_link"
    ),
    at=TimePoint.at(frame=1, timestamp_ns=...),
    frame_id="map",
)
```

### static / temporal は時間についての主張

上の 2 つは同じ archetype であり、どちらも特別扱いではない。`static` は「timeline に依存しない」
という意味なので、キャリブレーションは static、ego pose はそうではない。そして static な書き込みも
`frame_id` を保つ — これが、サンプル時刻をでっちあげずに辺を解釈可能にしている。

以前の設計ではこれができなかった。static ストレージが列以外を捨てていたため、固定の外部パラメータを
scene 先頭フレームの _temporal_ sample として置き、`latest_at` が過去に手を伸ばすことに頼るしかなかった。
代償は、そのサンプルより後から始まる窓付き `range` がそれを見ないことだった。static な辺には窓が
ないので、この代償は消える。

残る帰結は、static 行が `latest_at` / `range` から出てこないこと (「Store」参照) であり、固定の辺は
時刻クエリではなく `static_chunks()` で読む。

### `TRANSLATION` / `ROTATION` は独立した descriptor

`POSITION` / `QUATERNION` の再利用ではない。3D 位置を要求する system — 例えば距離フィルタ — に
transform の entity を渡したとき、動いてしまわないためである。

### 座標系名は文字列列

`FrameId` とその保存形 `BatchFrameId` は、モデル内で唯一の非数値 component である。それを許す規則は
「**文字列が許されるのは、オブジェクトごとではなく辺ごとに 1 つの値であるとき**」。クラス名やインスタンス名はオブジェクトごとなので
registry に intern してメタデータとして運ぶ。座標系名は辺ごとなので、registry にしても数百バイトの
節約にしかならず、`Chunk.frame_id` は文字列・列は整数という 1 つの概念に 2 つの符号化を残し (しかも
両者の一致を誰も検査しない)、`Store` / `SystemContext` / `Recording` / Arrow schema に registry を
通す必要が出る。

列の dtype は `object` で、固定幅の `<U*` は使わない: numpy が黙って切り詰めるため、長い名前が別の
短い座標系になり、2 つのセンサーが 1 つに潰れうる。Arrow 型は推論させず `string` に固定する。object
配列の推論は値に依存し、0 行の列は `null` と推論されて、全 field を non-nullable と宣言する schema に
拒否されるからである。

### グラフを読み直す

```python
from t4perceval.transform import FrameGraph, TransformResolver, transform_edges

transform_edges(recording)  # -> (TransformEdge(parent, child, entity_path, is_static), ...)
FrameGraph.of(recording).frames()  # -> ("map", "base_link", "LIDAR_TOP", ...)
```

探索は chunk を**読む**。以前の設計は entity path の一覧だけからグラフを列挙できたが、それはもう
できない — これが「座標系をデータにする」ことの代償である。読む量は小さい: `child_frame_id` を持たない
chunk は dict 参照 1 回で飛ばされ、持つ chunk は 1 行 1 辺である。

- `child_frame_id` 列を持たない chunk は**無視**する。近くに置かれた無関係な entity が探索を壊さない。
- 子を名乗るのに `frame_id` を持たない chunk は**例外**にする。親が不明では辺をまったく解釈できない。
  これは `require_same_frame` の「未宣言は不一致ではない」とは別である — あちらは 2 つを比較する話、
  こちらは 1 つを解釈する話である。
- 同じ `(親, 子)` が 2 か所に記録されていれば**例外**にする。選ぶ根拠がない。

### 連鎖を解決する

`TransformResolver` は store が取らない解釈のステップである。グラフを幅優先で辿り (合成ごとに誤差が
積み上がるので最小ホップ数)、記録された向きと逆に辿る辺は反転し (剛体変換なので厳密)、合成する:

```python
resolver = TransformResolver.of(recording, timeline=FRAME)
resolver.lookup(target_frame="map", source_frame="LIDAR_TOP", at=1)
# T_map_lidar(t) = T_map_base_link(t) @ T_base_link_lidar
```

static な辺と temporal な辺は 1 つのグラフに参加する。ROS が latched な transform と live な
transform を合成するのと同じである。temporal な辺は `LookupPolicy` でサンプルを選ぶ: `LATEST` /
`EXACT` / `NEAREST` / `INTERPOLATE` (並進は線形、回転は `Slerp`)。static な辺は policy を無視する —
変化しないものを補間するのは、エラーではなく答えが 1 つに決まる問いである。未知・非連結の座標系は
例外にするので、キャリブレーション漏れが黙って恒等変換になることはない。

これは `System` では**ない**: system は pipeline が保存する chunk を返すが、lookup は問いに答えて
何も書かない。_変換後の entity_ を materialize するのが system の形をした仕事で、そちらは
「passthrough な system が自分の運ぶ列を宣言できない」問題が未解決のままである。

### import した scene の座標系ツリー

`t4perceval.importer.t4` の `log_scene_transforms` は 2 種類の辺を記録する:

| 辺                       | 置き場所        | 記録の仕方           | 出所                |
| :----------------------- | :-------------- | :------------------- | :------------------ |
| `map -> base_link`       | `/tf/base_link` | keyframe ごとに 1 行 | `ego_pose`          |
| `base_link -> <channel>` | `/tf/<channel>` | static               | `calibrated_sensor` |

- 外部パラメータはセンサー 1 台 1 行の `calibrated_sensor` から読む。`sample_data` を歩くと、数個の値を
  得るために scene の長さに比例した仕事をすることになる。これらは frame ではなくデータセットの
  センサー構成の性質なので、この scene がデータを持たない channel も現れ、ツリーは import した frame 数に
  依存しない。
- 1 つの channel に 2 つのキャリブレーションがある場合、ポーズが一致しない限り例外を出す。`sensor`
  テーブルに無いセンサーを指すキャリブレーションも同様。黙ってどちらかを選ぶと、下流の誰も気づけない
  場所に誤った外部パラメータが入る。
- データセットは回転を `wxyz`、本パッケージは `xyzw` で持つ。どちらも 4 つの float なので、そのまま
  渡すとエラーではなく「もっともらしい回転」になる。並べ替えはこの境界だけで行う。
- ego pose は **keyframe** 単位で、両方の timeline に記録する。評価が frame index で走っても timestamp
  で走っても参照できる。frame 間の ego 運動は表現しないが、これは annotation 自身の解像度と同じである。

未実装: _変換後の entity_ を materialize する system。それまでは transform は記録・探索・解決できるが、
オブジェクトの chunk を別座標系へ書き換えることはしない — 代わりに system 層が座標系をまたぐ幾何比較を
拒否する。[system.md](system.md) の「座標系」を参照。

## IO — Arrow / Parquet

```python
table = chunk_to_table(chunk, labels=labels)
chunk, labels = chunk_from_table(table)

write_parquet(chunk, path, labels=labels)
chunk, labels = read_parquet(path)
```

### schema 設計

- component 1 つ = Arrow field 1 つ。field 名は `ComponentDescriptor.component`。
- ベクトルは `fixed_size_list<T, W>`、`BatchWaypoints3D` は入れ子 `fixed_size_list` で `(M,T,3)` を表現。
- 全 field は `nullable=False`。component は null を含まない (optional は「列の有無」で表す)。
- 行方向の長さを持たないもの — `entity_path` / `frame_id` / `is_static` / timeline 名と kind /
  index 時刻 / `offsets` / `labels` — は **schema metadata** (`b"t4perceval"` キー、JSON) に格納する。
  列にできないのは長さが違うためである。
- component クラスはクラス名で記録し、`t4perceval.io.registry` で解決する。
  モジュールパスをファイルに焼き込まない。

## rerun との関係

rerun のデータモデルを**設計として採用**し、SDK には依存しない。

| 項目                            | rerun                                  | t4perceval                                  |
| :------------------------------ | :------------------------------------- | :------------------------------------------ |
| Entity / EntityPath             | ○                                      | ○ (自前)                                    |
| Component / ComponentDescriptor | ○                                      | ○ (ただし descriptor 名は archetype 非依存) |
| Archetype                       | builder / convenience                  | 型としての component 束 + 汎用 `select()`   |
| Chunk                           | 行 = log 呼び出し、セル = 可変長 batch | 行 = オブジェクト、`offsets` = frame 境界   |
| Timeline / static / latest-at   | ○                                      | ○                                           |
| AnnotationContext               | static component                       | `LabelRegistry` (メタデータ)                |
| System                          | 形式化は Rust Viewer 内部に留まる      | `t4perceval.system` で第一級                |

### 依存にしない理由

- rerun 0.36 で読み出しクエリ API (`rr.dataframe`) が削除済みであり、
  `.rrd` を評価入力として読み戻す経路が安定していない。評価ツールは読み書き両方が必要である。
- 評価計算は「行 = オブジェクト」の平坦な列が有利で、rerun の chunk レイアウトとは前提が違う。
- 依存すると rerun のバージョン追従コストが評価結果の再現性に直接効く。

なお `rerun-sdk` は `t4-devkit` の依存として venv には入っているが、`t4perceval` は import しない。

## dataloader 設計

`t4_devkit.T4Devkit` から `Chunk` を作る経路。`tests/data/t4dataset` fixture に対して
`t4perceval.importer.t4` として実装済みで、以下はその設計そのものである。

```text
T4Devkit(data_root, revision)
  ├ get_box3ds(sample_data_token, future_seconds=...)  → list[Box3D]
  └ get_box2ds(sample_data_token)                      → list[Box2D]
        │
        ├─ SemanticLabel.name  ──→ LabelRegistry.encode()   → BatchClassId
        ├─ Box3D.uuid          ──→ InstanceRegistry.encode() → BatchInstanceId
        ├─ position / rotation / shape.size / velocity / num_points / visibility
        │                      ──→ BatchPosition3D / BatchQuaternion / BatchSize3D / …
        └─ Box3D.future (Future: timestamps (T,), confidences (M,), waypoints (M,T,3))
                               ──→ BatchWaypoints3D / BatchModeConfidence / BatchTimeOffset
        │
        └→ Detections3D / Trackings3D / Predictions3D
              .to_chunk(entity_path, at=TimePoint.at(frame=i, timestamp_ns=box.unix_time * 1000),
                        frame_id=box.frame_id)
```

考慮事項:

- 空 annotation は 0 行の batch として扱う (全 archetype で許可済み)。
- `velocity` は欠損値ではなく NaN ベクトルで返る。列を出すかどうかは frame ごとではなく
  **scene 全体で 1 回**決める: `concat_chunks` は列集合の異なる chunk を拒否するため、
  ある frame にあって次の frame にない列は `Store.range()` を失敗させる。
- `Box3D.unix_time` は μs なので `TIMESTAMP` timeline (ns) へ変換する。
- 1 scene = 1 `Store`、frame index を `FRAME` timeline に振る。
- `t4_devkit` への依存は dataloader モジュールに閉じる。`t4perceval.core` は `t4perceval/typing.py` の
  自前エイリアスだけを使う。

## 決定済みの未決事項

TODO.md に残っていた論点の結論。

| 論点                                    | 決定                                                                               |
| :-------------------------------------- | :--------------------------------------------------------------------------------- |
| trajectory の `M` / `T` を固定するか    | chunk 内で固定。可変長は validity mask                                             |
| `mode_confidence` の合計 1 を検証するか | しない。値域と有限性のみ                                                           |
| `waypoints` の有限値検証                | する。無効 timestep は mask で表現                                                 |
| component 内部配列を read-only にするか | する                                                                               |
| オブジェクト数 0 の batch を許可するか  | 全 archetype で許可                                                                |
| `Selection` が受理する入力              | slice / int 配列 / bool 配列 / int list / bool list。負index・重複・逆順可         |
| category ↔ `BatchClassId` の対応        | `LabelRegistry`。static なメタデータとして運ぶ                                     |
| view を archetype ごとに作るか          | 作らない。汎用 `EntityView` 1 つ                                                   |
| 座標系を path に入れるか                | 入れない。認識データは `Chunk.frame_id`、transform は親を `frame_id`・子を列で表す |
| transform を static データにするか      | 時間不変なら する。`static` は「timeline に乗らない」の意味で、frame_id も残る     |
| 座標系名を registry で intern するか    | しない。文字列列にする。frame の列は O(辺) で O(オブジェクト) ではない             |
