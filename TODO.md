# TODO

データモデルとシステム層の設計・実装は完了した。設計は [docs/design/](docs/design/) を参照。

## 前提

- 複数行を保持する component には `Batch` を付け、archetype には意味に基づく複数形の名前を付ける。
- descriptor 名は archetype 非依存 (`POSITION` は Detection3D でも Tracking3D でも `"position"`)。
- `Component.select()` / `Archetype.select()` / `Chunk.select()` は独立したデータを生成する。
  遅延 view は `EntityView` が担う。
- component 内部の NumPy 配列は read-only。
- オブジェクト数 0 の batch は全 archetype で許可する。

## 完了

- [x] `core` の実装 — `EntityPath` / `ComponentDescriptor` / `ColumnarComponent` / `Archetype` /
      `Timeline` / `Chunk` / `Store` / `EntityView` / `normalize_selection`
- [x] archetype を継承から合成へ移行。`select()` は基底の 1 実装のみ
- [x] `Header` を解体 (`TimePoint` + `Chunk.frame_id`)
- [x] `Trajectories3D` を component から archetype へ昇格し、列を分解
- [x] `BatchModeValid` / `BatchTimestepValid` / `BatchNumPoints` / `BatchVisibility` /
      `BatchRoi` / `BatchPixel` / `BatchMask` / matching 系 component を追加
- [x] `LabelRegistry` / `InstanceRegistry` (category ↔ `BatchClassId` の対応規則)
- [x] Arrow IO を公開 API として実装。nested vector は fixed-size list、
      非行方向の情報は schema metadata。Parquet round-trip 検証済み
- [x] `pyarrow` を直接依存へ追加
- [x] System protocol / `SystemContext` / `Pipeline` (順序検証)
- [x] フィルタ system 一式 — 共通基底 `MaskSystem` + 8 種
      (`FilterByDistance` / `Region` / `Label` / `Confidence` / `Instance` / `Speed` /
      `NumPoints` / `Visibility`)
- [x] `CombineMasksSystem` (`mode="all"` / `"any"`)
- [x] `masked_view()` — mask を通過した行の遅延 view
- [x] `SystemContext.instances` と `InstanceRegistry.instance_id()` (intern しない参照)
- [x] マッチング system 一式 — 共通基底 `MatchingSystem` + 6 モード
      (`CenterDistance` / `CenterDistanceBEV` / `PlaneDistance` / `IoUBEV` / `IoU3D` / `IoURoi`)
- [x] `t4perceval.geometry` — ベクトル化した箱の幾何 (footprint 頂点、BEV/3D IoU、
      ROI IoU、plane distance)。`shapely` を直接依存へ追加
- [x] クラスごとの閾値 — マッチングは `Thresholds(default, by_class=...)` (正解側のクラスで引く)。
      フィルタは `CLASS_ID` を要求しない system もあるため、`CombineMasksSystem` による合成で表現する
- [x] 設計ドキュメント (ja / en) — data_model / system / migration
- [x] `README.md` に目的・データshape・使用例を記載
- [x] `pyproject.toml` の description を更新

## P0: dataloader

- [ ] `t4_devkit.Tier4` を利用する dataloader を実装する。設計は
      [data_model.md](docs/design/ja/data_model.md) の「dataloader 設計」を参照。
  - [ ] dataset root / revision を指定してロードできる。
  - [ ] scene、sample、sensor channel で絞り込める。
  - [ ] `Box3D` / `Box2D` を `Detections3D` / `Trackings3D` / `Predictions3D` へ変換する。
  - [ ] `Box3D.unix_time` (μs) を `TIMESTAMP` timeline (ns) へ変換する。
  - [ ] 空annotation、欠損velocity、無効sample dataを処理する。
  - [ ] `t4_devkit` への依存を dataloader モジュールに閉じる。
- [ ] 最小構成のT4 dataset fixtureを用意し、dataloaderを実データ形式で検証する。

## P1: 指標 system

- [ ] `MeanAveragePrecisionSystem` (mAP / APH)
- [ ] `ClearSystem` (MOTA / MOTP / IDSwitch)
- [ ] `PathDisplacementSystem` (ADE / FDE / MissRate)
- [ ] `ClassificationSystem` (accuracy / precision / recall / F1)
- [ ] `HotaSystem` / `PassFailSystem` (critical object 判定を含む)は保留
- [ ] `/metrics/*` chunk のスキーマ (指標名 × クラス × 閾値をどう列にするか) を決める

## P1: 座標変換

- [ ] `Chunk.frame_id` を使う transform system を設計する。
  - [ ] `HomogeneousMatrix` (`t4_devkit.dataclass`) 相当を component / static データとして持つか決める。
  - [ ] 座標系を `EntityPath` 階層に埋める案 (rerun の `Transform3D` 流儀) を再検討する。

## P1: オフライン後解析

- [ ] `Store`全体とその他評価メタデータを保存し、後解析・可視化できるようにする。
  - [ ] [docs/design/en/offline_analysis.md](./docs/design/en/offline_analysis.md)を参照して実装方針を決める。
  - [ ] `Store`の保存時のフォルダ・ファイル構成を決める。

## P2: 可視化

- [ ] store へのクエリを入力とする可視化層を設計する。
  - [ ] rerun を optional な出力 sink として使うか、matplotlib で自前実装するか決める。 ->> `t4_devkit.viewer.RerunViewer`を使う。
  - [ ] [OPTIONAL] 元リポジトリの `perception_analyzer3d` / `eda_tool` / `field_analyzer` に相当する解析を、store のクエリとして書き直せるか確認する。

## P2: 品質

- [ ] Ruffに加えてPyrightまたはMypyをCIで実行する。
- [ ] Python 3.10以降のテストマトリクスを用意する。
- [ ] `Store` のクエリに chunk 単位のキャッシュが必要か、実データ規模で測ってから決める。
- [ ] 大規模 scene での `Store.range()` の性能を測る (現状は partition ごとに小さな chunk を作って concat する)。
- [ ] changelogを追加し、rerunベースのデータモデルへの移行を記載する。
