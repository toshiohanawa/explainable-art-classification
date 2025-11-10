# data/ ディレクトリガイド

`data/` 配下は「データのライフサイクル」に沿って5つのカテゴリに整理されています。生データと成果物を分離しつつ、タイムスタンプ付き実験結果も簡単に参照できる構成です。

## 全体構成

```
data/
├── raw/            # 取得直後のデータと画像
├── curated/        # フィルタ済みメタデータと画像
├── artifacts/      # 最新の特徴量・モデル・可視化など
├── experiments/    # use_timestamp=true 時のスナップショット
└── external/       # 外部公開データ（WikiArt_VLM など）
```

### raw/
| パス | 内容 |
| --- | --- |
| `raw/raw_data/` | Met API由来のメタデータ、`checkpoint.json`、`qa_report.txt` など |
| `raw/raw_images/` | ダウンロード済みの元画像（大容量のためGit管理外） |

### curated/
| パス | 内容 |
| --- | --- |
| `curated/filtered_data/paintings_metadata.csv` | フィルタ後のメタデータ |
| `curated/filtered_data/paintings_images/` | カリキュラム用に整理した画像 |

### external/
| パス | 内容 |
| --- | --- |
| `external/WikiArt_VLM-main/` | 公式WikiArt_VLMデータセット（Original/Generated/Prompts） |
| `external/WikiArt/` | Hugging Face配布の補助CSV |

### artifacts/
`TimestampManager` が扱うステージごとの最新成果物です。`config.yaml` の `data.latest_label`（デフォルト: `latest`）を使って常に最新の成果だけを参照できます。

| パス | 主なファイル |
| --- | --- |
| `artifacts/features/<label>/` | `wikiart_vlm_features_*.csv` |
| `artifacts/models/<label>/` | `random_forest_model.pkl`, `scaler.pkl` |
| `artifacts/results/<label>/` | `training_results.txt`, `bias_feature_summary.csv` |
| `artifacts/visualizations/<label>/` | `confusion_matrix.png`, `shap_summary_plot.png` など |
| `artifacts/shap_explanations/<label>/` | `shap_values.csv`, `feature_importance_shap.csv` |
| `artifacts/gestalt/<label>/` | `gestalt_scores_<model>.csv` |

### experiments/
`config.data.use_timestamp` を `true` にすると、実行ごとに `experiments/analysis_<YYMMDDHHMM>/` が作成され、各ステージの成果物がタイムスタンプ付きで保存されます。`snapshot_stage('<stage>')` を使えば、`use_timestamp=false` のまま任意のステージを `snapshot_<timestamp>` として凍結できます。

## 運用ヒント

- 生データ (`raw/*`) と外部データ (`external/*`) は常に固定ディレクトリで共有します。
- モデル・可視化などの成果物だけを `artifacts/` にまとめることで、学習者は「最新成果物」と「スナップショット」を簡単に行き来できます。
- Git では `artifacts/*/latest/` だけを追跡対象にし、`raw/raw_images/` や `experiments/analysis_*` は `.gitignore` で除外しています。
