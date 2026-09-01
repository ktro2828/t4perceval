# データモデル設計

## 背景と目的

元の [`autoware_perception_evaluation`](https://github.com/tier4/autoware_perception_evaluation) は、
1 オブジェクト = 1 Python オブジェクト (`DynamicObject`) というモデルを採っていた。この構造は次の問題を生む。

- `DynamicObject` が position / orientation / shape / velocity / tracked_\* / predicted_\* を 20 以上の
  フィールドとして 1 クラスに抱える。タスクによって使わないフィールドが `None` で埋まる。
- `List[DynamicObject]` を Python ループで回すため、ベクトル化できない。
- `EvaluationTask` enum が config・matching・metrics・visualization の全層に if 分岐として漏れる。
- `Catalog → Scenario → Scene → PerceptionFrameResult` が list のネストで表現され、
  フレーム跨ぎのクエリ (CLEAR / HOTA / ADE) が書きづらい。
- フィルタ結果やマッチングコストといった中間結果が破棄され、再解析できない。

`t4perceval` は [rerun](https://github.com/rerun-io/rerun) のデータモデルを**自前実装**することでこれを解消する。
rerun 自体は依存に入れない (後述の「rerun との関係」を参照)。

## レイヤ構成

```
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
   t4perceval.io        │  Chunk ↔ pyarrow.Table ↔ Parquet         │
                        └──────────────────────────────────────────┘
```

## EntityPath — 「何のデータか」

`/`区切りの階層パス。元リポジトリで `frame_id` + task enum + est/gt の別を型やフィールドで表していたものを、
すべてパスの区別に移す。

```python
EntityPath.parse("/estimation/objects")          # 推定結果
EntityPath.parse("/ground_truth/objects")        # 正解
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
    component: str                    # 同一性はこれだけで決まる
    archetype: str | None             # eq=False。provenance のヒント
    component_type: str | None        # eq=False。component クラス名のヒント
```

**重要な設計判断**: descriptor 名は archetype 非依存にする。
`BatchDetection3D` と `BatchTracking3D` は同じ 3D 中心を、どちらも `POSITION` (= `"position"`) として公開する。

```python
BatchDetection3D.descriptor_of("position") == BatchTracking3D.descriptor_of("position")  # True
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
    SHAPE = (3,)            # 行あたりの形状。() はスカラ列。ANY はデータから推論
    DTYPE = np.float64      # すべての値をこの dtype に正規化
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
class BatchTracking3D(Archetype):
    position    = component_field(POSITION,    BatchPosition3D)
    quaternion  = component_field(QUATERNION,  BatchQuaternion)
    size        = component_field(SIZE,        BatchSize3D)
    class_id    = component_field(CLASS_ID,    BatchClassId)
    confidence  = component_field(CONFIDENCE,  BatchConfidence)
    instance_id = component_field(INSTANCE_ID, BatchInstanceId, kw_only=True)
    velocity    = component_field(VELOCITY,    BatchVelocity, optional=True, kw_only=True)
```

### 継承をやめた理由

以前は `BatchDetection3D → BatchTracking3D → BatchPrediction3D` の継承チェーンだった。これは ECS の合成と逆向きで、

- `select()` が 3 箇所でほぼ同一実装として重複する
- 「trajectory は持つが instance_id は持たない」といった組み合わせが表現できない
- `isinstance(tracking, BatchDetection3D)` が「tracking は detection の一種」という嘘をつく

いまは `BatchTracking3D` が box 系 component を**明示的に再宣言**する。descriptor は同一なので:

```python
tracking.has(*BatchDetection3D.required_descriptors())   # True
isinstance(tracking, BatchDetection3D)                   # False
```

`has()` は System が `REQUIRES` で問うのと同じ質問であり、`isinstance` の正しい置き換えである。

### 基底が提供するもの

`select()` は `attrs.fields()` を走査して各 component に委譲するため、**archetype ごとの実装が不要**。
`as_components()` / `from_components()` / `to_chunk()` / `from_chunk()` / `has()` /
`descriptors()` / `required_descriptors()` も基底の 1 実装で済む。

### archetype 一覧

| archetype                     | components                                                                                    |
| :---------------------------- | :-------------------------------------------------------------------------------------------- |
| `BatchDetection3D`            | position, quaternion, size, class_id, confidence, [velocity], [num_points], [visibility]      |
| `BatchDetection2D`            | roi, class_id, confidence, [visibility]                                                       |
| `BatchTracking3D`             | Detection3D の各列 + instance_id                                                              |
| `BatchTracking2D`             | Detection2D の各列 + instance_id                                                              |
| `BatchPrediction3D`           | Tracking3D の各列 + waypoints, mode_confidence, [mode_valid], [timestep_valid], [time_offset] |
| `BatchClassification2D`       | class_id, confidence, [instance_id]                                                           |
| `BatchSemanticSegmentation2D` | pixel, class_id                                                                               |
| `BatchSemanticSegmentation3D` | point, class_id                                                                               |
| `BatchTrajectory3D`           | waypoints, mode_confidence, [mode_valid], [timestep_valid], [time_offset]                     |
| `BatchMatchResult`            | est_index, gt_index, matching_score, match_status                                             |

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
    SEQUENCE   # frame index 等の単調カウンタ
    TIMESTAMP  # Unix epoch からの ns
    DURATION   # 経過 ns

FRAME     = Timeline("frame", TimeKind.SEQUENCE)
TIMESTAMP = Timeline("timestamp_ns", TimeKind.TIMESTAMP)
```

1 つのデータは複数 timeline に同時に乗る。`TimePoint.at(frame=3, timestamp_ns=...)` で両方指定でき、
クエリ側は好きな軸を選べる。

### `Header` の解体

旧 `Header(timestamp_ns, frame_id)` は廃止した。

- `timestamp_ns` → `TIMESTAMP` timeline 上の値 (`Chunk.indexes`)
- `frame_id` → 「その chunk の全行が乗る座標系」として `Chunk.frame_id`

これにより archetype は純粋な component の束になり、構築時に header を引き回さなくなった。

> **将来案**: rerun は座標系を entity path 階層 + `Transform3D` archetype で表す
> (例: `/base_link/estimation/objects`)。transform system を導入する段階で再検討する。
> 現時点では `Chunk.frame_id` を採用している。

## Chunk — 列指向テーブル

```python
@define(frozen=True, slots=True)
class Chunk:
    entity_path: EntityPath
    indexes: tuple[TimeColumn, ...]                    # 長さ P (partition 数)
    offsets: NDArrayI64                                # 長さ P+1、行境界
    columns: dict[ComponentDescriptor, Component]      # 各長さ N = offsets[-1]
    frame_id: str | None = None
    is_static: bool = False
```

### 行 = オブジェクト という選択

rerun の chunk は「行 = log 呼び出し、セル = 可変長 component batch」である。
`t4perceval` は**行 = オブジェクト 1 個**に平坦化し、フレーム境界を `offsets` で持つ。

理由: 評価計算はオブジェクト方向の要素ごと演算 (距離行列、IoU、TP/FP 判定) が中心であり、
平坦な contiguous 配列に対する NumPy 演算に直接落ちる。フレームごとに list offset を歩く必要がない。
`offsets` があるのでフレーム単位の集計 (`partition(i)`, `partition_ids()`) も O(1) で取れる。

```
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
store.log_static(entity_path, archetype)

store.latest_at(entity_path, timeline=FRAME, at=12)                  # → EntityView
store.range(entity_path, timeline=FRAME, time_range=TimeRange(0, 99))  # → EntityView
```

`Catalog → Scenario → Scene → List[PerceptionFrameResult]` のネストを置き換える。

| 旧構造              | 新しい表現                                 |
| :------------------ | :----------------------------------------- |
| 1 frame             | `store.latest_at(...)`                     |
| 1 scene             | `store.range(..., TimeRange.everything())` |
| 全 frame 共通の情報 | `store.log_static(...)`                    |

### セマンティクス (rerun 準拠)

- **static データは全 timeline に属する**。同一 entity・同一 descriptor の時系列データより**優先**する。
  1 行の static 列は view の行数に broadcast される。
- `latest_at` は指定時刻**以下**の最新 partition を返す。同時刻が複数あれば**後に log したもの**が勝つ。
- `range` は partition を**時刻順**に並べる (同時刻は log 順)。
- 1 entity に異なる列構成の chunk を log してもよい。`range` がそれらを跨いだときに初めてエラーになる
  (`latest_at` は単一 chunk しか見ないので常に成立する)。

## EntityView — 遅延 view

```python
@define(frozen=True, slots=True)
class EntityView:
    chunk: Chunk
    indices: NDArrayI64                              # chunk の行への正規化済み index
    static: dict[ComponentDescriptor, Component]
```

`select()` は index を合成するだけで**コピーしない**。実体化は `component()` / `materialize()` /
`to_chunk()` の時点で起きる。

```python
view.select(slice(None, None, 2)).select([1])   # コピーなし
view.component(POSITION)                        # ここで 1 列だけコピー
view.materialize(BatchDetection3D)              # archetype として実体化
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
labels.class_id("truck")            # 1
labels.encode(["car", "truck"])     # BatchClassId 用の i32 列

merged = labels.merged({"vehicle": ["car", "truck"]})
merged.class_id("car") == merged.class_id("vehicle")   # True
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

## dataloader 設計 (後続ステップ)

`t4_devkit.Tier4` から `Chunk` を作る経路。実装は最小の T4 dataset fixture を用意できる段階で行う。

```
Tier4(data_root, version)
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
        └→ BatchDetection3D / BatchTracking3D / BatchPrediction3D
              .to_chunk(entity_path, at=TimePoint.at(frame=i, timestamp_ns=box.unix_time * 1000),
                        frame_id=box.frame_id)
```

考慮事項:

- 空 annotation は 0 行の batch として扱う (全 archetype で許可済み)。
- `velocity` が欠損する sample は optional component を省略する。
- `Box3D.unix_time` は μs なので `TIMESTAMP` timeline (ns) へ変換する。
- 1 scene = 1 `Store`、frame index を `FRAME` timeline に振る。
- `t4_devkit` への依存は dataloader モジュールに閉じる。`t4perceval.core` は `t4perceval/typing.py` の
  自前エイリアスだけを使う。

## 決定済みの未決事項

TODO.md に残っていた論点の結論。

| 論点                                    | 決定                                                                       |
| :-------------------------------------- | :------------------------------------------------------------------------- |
| trajectory の `M` / `T` を固定するか    | chunk 内で固定。可変長は validity mask                                     |
| `mode_confidence` の合計 1 を検証するか | しない。値域と有限性のみ                                                   |
| `waypoints` の有限値検証                | する。無効 timestep は mask で表現                                         |
| component 内部配列を read-only にするか | する                                                                       |
| オブジェクト数 0 の batch を許可するか  | 全 archetype で許可                                                        |
| `Selection` が受理する入力              | slice / int 配列 / bool 配列 / int list / bool list。負index・重複・逆順可 |
| category ↔ `BatchClassId` の対応        | `LabelRegistry`。static なメタデータとして運ぶ                             |
| view を archetype ごとに作るか          | 作らない。汎用 `EntityView` 1 つ                                           |
