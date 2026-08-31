# TODO

## 前提

- 複数行を保持する component / archetype には `Batch` を付ける。
- `BatchTrajectory3D` は dense 形式を使用する。
  - `positions`: `(N, M, T, 3)`
  - `confidences`: `(N, M)`
  - `time_offsets_ns`: `(T,)`
- `select()` は独立したデータを生成し、遅延 view は別 API とする。

## P0: IO とデータロード

- [ ] Arrow IO をテスト内の helper ではなく公開 API として実装する。
  - [ ] `BatchDetection3D` / `BatchTracking3D` の `to_arrow()` と `from_arrow()` を定義する。
  - [ ] `Header` の保存方法を schema metadata または明示列として決定する。
  - [ ] nested vector を Arrow の fixed-size list として保存する。
  - [ ] dtype、shape、nullability を schema で固定する。
  - [ ] Parquet round-trip を検証する。
- [ ] `pyarrow` を適切な依存グループへ明示的に追加する。
  - 現在のテストは `pyarrow` を import するが、`pyproject.toml` に直接依存がない。
- [ ] `t4_devkit.T4Devkit` を利用する dataloader を実装する。
  - [ ] dataset root / revision を指定してロードできる。
  - [ ] scene、sample、sensor channel で絞り込める。
  - [ ] `t4_devkit` の Box を `BatchDetection3D` / `BatchTracking3D` に変換する。
  - [ ] category と `BatchClassId` の対応規則を定義する。
  - [ ] 空annotation、欠損velocity、無効sample dataを処理する。

## P0: BatchTrajectory3D

- [ ] `M` と `T` を常に固定するか、padding + mask を許可するか決定する。
- [ ] 可変長を許可する場合は以下を追加する。
  - [ ] `mode_valid`: `(N, M)`
  - [ ] `timestep_valid`: `(N, M, T)`
- [ ] objectごとの `confidences` の意味を確定する。
  - 確率として扱う場合は、各objectで合計が1になることを検証する。
- [ ] positions の有限値検証方針を決定する。
- [ ] yaw、velocity、covarianceなど追加stateの必要性を決定する。
- [ ] `from_modes()` の空batch生成 API を追加する。

## P1: View とメモリ管理

- [ ] `BatchDetection3DView`、`BatchTracking3DView`、`BatchPrediction3DView` を設計する。
  - 元データと正規化済みobject indexを保持する遅延viewとする。
  - viewへの `select()` はindexを合成する。
  - `materialize()` でBatch型を生成する。
- [ ] component内部のNumPy配列をread-onlyにするか決定する。
- [ ] slice、boolean mask、integer indexでcopy/viewの挙動をテストする。

## P1: Component と Archetype

- [ ] `SemanticSegmentation2D/3D` の `pixel` / `point` を専用componentにする。
- [ ] semantic segmentationにもselection APIが必要か決定する。
- [ ] 2D tracking / prediction archetypeの必要性を確認する。
- [ ] `Selection` が受理する入力を明確化する。
  - 現在の型はsliceとNumPy配列だが、実行時にはlistも使用できる。
- [ ] NaN / Inf、空batch、重複index、逆順indexの共通テストを追加する。

## P1: テスト

- [ ] 最小構成のT4 dataset fixtureを用意し、dataloaderを実データ形式で検証する。
- [ ] Arrow / ParquetのファイルIOを`tmp_path`でround-trip検証する。
- [ ] `BatchTrajectory3D.from_modes()` のmode数・時刻軸不一致を網羅する。
- [ ] object数0のBatch型を許可するか決定し、全archetypeで統一する。
- [ ] 公開importと旧型名が混在しないことをテストする。

## P2: 品質とドキュメント

- [ ] `README.md` に目的、データshape、使用例を記載する。
- [ ] `pyproject.toml` のdescriptionを正式な内容に変更する。
- [ ] Ruffに加えてPyrightまたはMypyをCIで実行する。
- [ ] Python 3.10以降のテストマトリクスを用意する。
- [ ] Batch型へのrenameとdense trajectory形式をchangelogに記載する。
