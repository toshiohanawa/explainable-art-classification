# タイムスタンプ/成果物管理の新方針

## 概要

`TimestampManager` はデフォルトで **ステージごとの「最新」ディレクトリ**を維持します。処理を再実行しても、常に同じ場所（例: `data/artifacts/features/latest`）が上書きされるため、Git で追跡したい成果物を簡単に更新できます。必要なときだけスナップショットを作成し、従来の `analysis_YYMMDDHHMM` 構造もオプションで利用できます。

## ディレクトリ構成

```
data/
├── artifacts/
│   ├── features/
│   │   ├── latest/               # 特徴量の正規保存先
│   │   └── snapshot_2502011200/  # 任意スナップショット（必要な場合のみ）
│   ├── models/
│   │   └── latest/
│   ├── results/
│   │   └── latest/
│   ├── shap_explanations/
│   │   └── latest/
│   ├── visualizations/
│   │   └── latest/
│   └── gestalt/
│       └── latest/               # ゲシュタルト原則スコアの唯一の保存場所
└── raw/
    ├── raw_data/                 # Met APIメタデータ・チェックポイント
    └── raw_images/               # 元画像
```

- `TimestampManager.create_directories()` が各ステージの `latest` を自動生成
- `get_features_dir()` などのアクセサは、`use_timestamp=false` の場合に自動で `…/latest` を返す
- 生データ (`data/raw/raw_data`, `data/raw/raw_images`) はこれまで通り固定のベースディレクトリ

## タイムスタンプ付きスナップショットを使いたい場合

`config.yaml` で `data.use_timestamp: true` を指定すると、`data/experiments/analysis_<timestamp>/…` に成果物が保存されます（段階実行や検証用途向け）。

また、`use_timestamp=false` のままでも `snapshot_stage()` を利用すると任意ステージの `latest` 内容を `snapshot_<timestamp>` として凍結できます。

```python
tm = TimestampManager(config)
tm.create_directories()

# 最新の特徴量をスナップショット保存
snapshot_path = tm.snapshot_stage('features')
print(f"Snapshot saved to {snapshot_path}")
```

## ゲシュタルト原則スコアの再利用

- すべてのスコアは `data/artifacts/gestalt/latest/gestalt_scores_<モデル名>.csv` に集約
- `config.gestalt_scoring.reuse_existing_results` が `true`（デフォルト）の場合、ファイルが存在すれば **再スコアリングを自動でスキップ**
- 再計算が必要なら `config.gestalt_scoring.force_run=true` または `scripts/score_gestalt_principles.py --force`

これにより「一度重い処理を走らせたら、その後は常に既存結果を利用する」という運用が可能です。

## 推奨設定

```yaml
data:
  use_timestamp: false      # デフォルト。latest を常に上書き
  latest_label: "latest"    # 任意で変更可能
  snapshot_prefix: "snapshot"

gestalt_scoring:
  reuse_existing_results: true
  force_run: false
```

開発中に履歴を残したい場合だけ `use_timestamp: true` を有効化してください。`latest` 構造はそのまま残るため、スナップショットと最新版を並行で運用できます。

## クリーンアップ

旧 `analysis_*` ディレクトリが不要になった場合は、従来どおり手動で削除できます。

```bash
find data/experiments -type d -name "analysis_*" -mtime +7 -exec rm -rf {} \;
```

`latest` 以下を削除しない限り、現在の成果物は維持されます。Git からは `data/*/latest` のみを追跡対象にすることで、常に最新のアウトプットを共有できます。
