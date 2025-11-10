# 説明可能AIによる絵画様式分類プロジェクト

## プロジェクト概要

このプロジェクトは、WikiArt_VLMデータセットに含まれる本物の絵画画像と生成AI（Stable Diffusion / FLUX / F-Lite）による模倣画像を対象に、ランダムフォレストとSHAPで「本物 vs フェイク」を説明可能に分類する講義教材です。

## 特徴

- **説明可能AI**: SHAPを用いてモデルの判断根拠を可視化
- **芸術とテクノロジーの融合**: 機械学習と芸術史の学際的アプローチ
- **教育的価値**: データサイエンス学習者向けの実践的教材
- **公開データ**: WikiArt_VLMデータセットとGestaltスコアを活用

## 技術スタック

- **Python 3.9+**
- **機械学習**: scikit-learn, SHAP
- **画像処理**: OpenCV, PIL
- **可視化**: matplotlib, seaborn, plotly
- **データソース**: WikiArt_VLMデータセット（Original + Generatedペア）

## プロジェクト構造

```
explainable-art-classification/
├── src/                    # ソースコード
│   ├── data_collection/    # データ収集
│   ├── feature_extraction/  # 特徴量抽出
│   ├── model_training/     # モデル訓練
│   ├── explainability/     # SHAP説明
│   ├── visualization/      # 可視化
│   └── utils/             # ユーティリティ
│       ├── config_manager.py   # 設定管理
│       ├── logger.py            # 共通ログ設定
│       ├── status_checker.py    # ステータスチェック
│       └── timestamp_manager.py # タイムスタンプ管理
├── scripts/               # 実行スクリプト
│   ├── train_wikiart_vlm.py        # WikiArt_VLMパイプライン（特徴量抽出～学習）
│   ├── score_gestalt_principles.py # MLLMによるゲシュタルト原則スコア評価
│   ├── compare_generation_models.py# 生成モデル間の性能比較
│   ├── run_all_phases.py           # 現行フェーズ一括実行ラッパー
│   └── legacy_*                    # 初期の美術館API向けスクリプト（現在は使用しない）
├── data/                  # データファイル
│   ├── raw/               # 生データ
│   │   ├── raw_data/      # Met APIメタデータ＆チェックポイント
│   │   └── raw_images/    # 元画像
│   ├── curated/           # フィルタ済みデータ
│   │   └── filtered_data/ # paintings_metadata.csv, paintings_images/
│   ├── artifacts/         # 最新成果物（latest/配下がデフォルト）
│   │   ├── features/          # 特徴量CSV（例: latest/wikiart_vlm_features_with_gestalt_*.csv）
│   │   ├── models/            # 学習済みモデル・スケーラー
│   │   ├── results/           # 訓練ログ・分析CSV
│   │   ├── visualizations/    # 可視化（confusion_matrix.png, dependence_plot_*.png など）
│   │   ├── shap_explanations/ # SHAP解析結果
│   │   └── gestalt/           # LLaVAによるゲシュタルト原則スコア
│   ├── experiments/       # タイムスタンプ付きスナップショット
│   │   └── analysis_<ts>/ # 各ステージ成果物をまとめて保存
│   └── external/          # 外部公開データ
│       ├── WikiArt_VLM-main/  # 公式配布データセット（Original/Generated画像・CSV）
│       └── WikiArt/           # Hugging Face由来の補助CSV
├── notebooks/             # Jupyterノートブック
│   ├── eda_paintings_complete_dataset.ipynb # EDAノートブック
│   └── prepare_data_for_shap_class.ipynb # shap-class用データ準備ノートブック
├── docs/                  # ドキュメント群
│   ├── history/           # 開発ログ・コミット履歴
│   ├── reports/           # 実装レポートや演習レポート
│   ├── guides/            # 手順書・チートシート
│   └── plans/             # 計画書・提案資料
├── tests/                 # テストファイル
├── logs/                  # ログファイル
├── config.yaml           # 設定ファイル
├── requirements.txt      # 依存関係
└── main.py              # メイン実行ファイル
```

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd explainable-art-classification
```

### 2. 仮想環境の作成

#### Windows環境

```bash
# uvを使用（推奨）
uv venv
uv pip install -r requirements.txt

# または pipを使用
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### macOS環境

```bash
# uvを使用（推奨）
uv venv
uv pip install -r requirements.txt

# または pipを使用
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 環境構築の注意点

- **Python 3.9以上**が必要です
- **uv**を使用することを強く推奨（依存関係の解決が高速）
- **OpenCV**のインストールで問題が発生する場合は、以下を試してください：
  ```bash
  # macOSでOpenCVの問題が発生した場合
  brew install opencv
  pip install opencv-python-headless
  ```

### 3. 設定ファイルの確認

`config.yaml`ファイルで設定を確認・調整してください。

## 使用方法

### 現行パイプラインの実行

```bash
# 特徴量抽出 → Gestaltスコア統合 → ランダムフォレスト学習 → SHAP解析
python scripts/train_wikiart_vlm.py --mode all --generation-model Stable-Diffusion --include-gestalt
```

### 主なサブモード

```bash
# 特徴量抽出のみ（generation-modelは Stable-Diffusion / FLUX / F-Lite から選択）
python scripts/train_wikiart_vlm.py --mode features --generation-model Stable-Diffusion

# ゲシュタルト原則スコア取得のみ（LLaVA等のMLLMが必要）
python scripts/score_gestalt_principles.py --generation-model Stable-Diffusion

# モデル訓練 + SHAPのみ（抽出済み特徴量を再利用）
python scripts/train_wikiart_vlm.py --mode train --generation-model Stable-Diffusion --include-gestalt

# 生成モデルごとの性能比較
python scripts/compare_generation_models.py --generation-models Stable-Diffusion FLUX F-Lite
```

`run_all_phases.py` からも同様のフェーズ実行が可能です。`main.py` にある旧モード類は保守目的で残っていますが、現行カリキュラムでは `scripts/train_wikiart_vlm.py` 系を利用してください。

### バイアス可視化ツール

```bash
# ラベル別の主要特徴量統計と分布図を出力
python scripts/analyze_bias.py --generation-model Stable-Diffusion
```

- 出力: `data/artifacts/results/latest/bias_feature_summary.csv`
- 可視化: `data/artifacts/visualizations/latest/bias_feature_mean_diff.png`, `bias_feature_distributions.png`
- 生成モデル固有の色彩・テクスチャ差異を学習者に提示する補助教材として利用できます。

### 特徴量抽出の可視化ツール

```bash
# 実際の画像を使って特徴量抽出の過程を可視化
python scripts/visualize_feature_extraction.py --image-id 9733
```

- 出力: `data/artifacts/visualizations/latest/feature_visualization_color_*.png`, `feature_visualization_edge_*.png`, `feature_visualization_texture_*.png`
- 色彩特徴量（HSV色空間）、エッジ特徴量（Canny検出）、テクスチャ特徴量（LBP）の可視化を生成
- 本物とフェイクの画像を並べて比較し、特徴量の違いを視覚的に理解できます

### Jupyterノートブック

```bash
# Jupyter Notebookの起動
jupyter notebook notebooks/

# または JupyterLabを使用
jupyter lab notebooks/
```

#### 利用可能なノートブック

- **`notebooks/prepare_data_for_shap_class.ipynb`**: shap-class用データ準備
  - WikiArt_VLMデータセットの読み込み
  - 特徴量カラムの抽出（label以外の数値カラム、説明文カラムを除外）
  - train/validation分割（70/30、stratifyを使用してラベル分布を同数に保持）
  - CSVファイルとして保存（`shap_class_rf/dataset/train_num.csv`, `validation_num.csv`）
  - shap-classリポジトリのセットアップとモデル訓練
  - Random Forestモデルの訓練と評価
  - SHAP説明可能性の可視化

#### ノートブックの実行環境

```bash
# Jupyterカーネルの確認
jupyter kernelspec list

# プロジェクトの仮想環境を使用するカーネルのセットアップ
# カーネル名: explainable-art-classification
source .venv/bin/activate  # 仮想環境を有効化
pip install ipykernel jupyter
python -m ipykernel install --user --name explainable-art-classification --display-name "Python (explainable-art-classification)"
```

#### shap-class用データ準備ノートブックの使用方法

1. **Notebookを開く**:
   ```bash
   jupyter notebook notebooks/prepare_data_for_shap_class.ipynb
   ```

2. **カーネルの選択**:
   - Notebookを開いたら、右上のカーネル名をクリック
   - `Python (explainable-art-classification)`を選択

3. **セルの実行**:
   - 各セルを順番に実行（Shift+Enter）
   - または「Run All」で全セルを実行

4. **処理内容**:
   - セル1-13: データ準備（読み込み、特徴量抽出、分割、保存）
   - セル14-26: shap-classリポジトリのセットアップとモデル訓練
     - shap-classリポジトリのクローン
     - 必要なパッケージのインストール
     - データファイルのコピー
     - データファイルの検証
     - Random Forestモデルの訓練と評価
     - 混同行列の可視化

5. **出力ファイル**:
   - `shap_class_rf/dataset/train_num.csv` (70%)
   - `shap_class_rf/dataset/validation_num.csv` (30%)
   - `shap-class/results/classification/` (モデル訓練結果)

## 実行フロー

### 基本ワークフロー

1. **データ同期**: `data/external/WikiArt_VLM-main/` を取得（配布リポジトリをサブモジュールとして追加するか、アーカイブを展開）
2. **（任意）ゲシュタルト評価**: `scripts/score_gestalt_principles.py` でLLaVA等のMLLMを用いて視覚的原則スコアを付与
3. **特徴量抽出**: `scripts/train_wikiart_vlm.py --mode features ...` で色彩・テクスチャ・エッジ特徴量をCSV化
4. **特徴量統合**: 抽出特徴量とゲシュタルトスコアをマージし、タイムスタンプ付き`data/artifacts/features/<ts>/`に保存
5. **モデル訓練**: ランダムフォレストによる学習・交差検証・グリッドサーチ、`data/artifacts/models/<ts>/`へ成果物保存
6. **SHAP解析**: `scripts/train_wikiart_vlm.py --mode explain ...` または `src/explainability/shap_explainer.py` を経由してTreeExplainerを生成
7. **可視化/レポート**: `data/artifacts/visualizations/<ts>/`, `data/artifacts/results/<ts>/` に混同行列やSHAP summary plotを出力

### データセット準備フロー

```bash
# shap-class向けにtrain/validation CSVを書き出す
jupyter notebook notebooks/prepare_data_for_shap_class.ipynb
```

## 出力ファイル

プロジェクト実行後は以下の構造に成果物がまとまります。

```
data/
├── raw/
│   ├── raw_data/                    # Met APIメタデータ、checkpoint.json、QAログ
│   └── raw_images/                  # 生画像（ダウンロード済み）
├── curated/
│   └── filtered_data/
│       ├── paintings_metadata.csv
│       └── paintings_images/
├── artifacts/                       # 最新成果物（latest/ 配下）
│   ├── features/latest/
│   ├── models/latest/
│   ├── results/latest/
│   ├── visualizations/latest/
│   ├── shap_explanations/latest/
│   └── gestalt/latest/
├── experiments/
│   └── analysis_<ts>/               # use_timestamp=true 時のスナップショット
└── external/
    ├── WikiArt_VLM-main/            # 公式配布データセット（Original/Generated/Prompts）
    └── WikiArt/                     # Hugging Face由来の補助CSV

docs/
├── history/                         # cursor_chat_history.md など
├── reports/                         # exercise_report.*, implementation_status_report.md など
├── guides/                          # feature_columns_definition.md など
└── plans/                           # roadmapや提案資料

logs/
├── project.log
└── *.log
```

詳細な配置と活用ルールは [`data/README.md`](data/README.md) と [`docs/README.md`](docs/README.md) を参照してください。

## 設定オプション

`config.yaml` の主な項目は次の通りです。

- `data`: 出力ディレクトリ、タイムスタンプ管理、スナップショットプレフィックス
- `features`: 抽出する色彩・エッジ・テクスチャ特徴量の有効/無効
- `classification`: RandomForestのデフォルトパラメータ、分割比率、CV設定
- `shap`: TreeExplainerのサンプリング数、背景データ数
- `wikiart_vlm`: 生成モデルやsplit比率、データセットパス
- `gestalt_scoring`: LLaVAエンドポイント、スコア範囲、再試行設定

### タイムスタンプ / 最新ディレクトリ機能

- `data.use_timestamp: false`（デフォルト）は `data/artifacts/<stage>/latest/` を常に更新します。教育用途ではこの設定で十分です。
- `data.use_timestamp: true` にすると `data/experiments/analysis_<YYMMDDHHMM>/` が生成され、実験ごとに成果物をアーカイブできます。

ベースデータ（`data/raw/…`, `data/external/WikiArt_VLM-main/` など）は設定に関わらず固定ディレクトリに配置されます。最新版を凍結したい場合は `TimestampManager.snapshot_stage('<stage>')` を使って `snapshot_<timestamp>` を作成してください。

## 教育的活用

このプロジェクトは以下の学習に活用できます：

- 機械学習の基礎（分類、特徴量エンジニアリング）
- 説明可能AI（SHAP、モデル解釈）
- 画像処理（OpenCV、色彩分析）
- データサイエンス（探索的データ分析、可視化）
- 芸術とテクノロジーの融合

## ライセンス

このプロジェクトは教育目的で作成されています。WikiArt_VLMデータセットおよび付随する生成画像は配布元のライセンスと利用規約に従ってください（研究・教育目的での利用を想定しています）。本リポジトリのコードは各ファイルヘッダーで示すライセンスに従います。

## 貢献

プルリクエストやイシューの報告を歓迎します。

## プロジェクトの最新状況

### 実装状況（2025年11月7日更新）

| 項目 | 状況 |
| --- | --- |
| データ取得 | WikiArt_VLM-main一式をローカルに同期済み |
| 特徴量抽出 | 色彩 / テクスチャ / エッジ / ゲシュタルトをCSV化済み |
| モデル訓練 | RandomForest + GridSearch + Stratified split 完了 |
| 説明可能性 | SHAP summary / dependence plot / feature importance 完了 |
| shap-class連携 | Notebookでtrain/validation CSV書き出し済み |
| ログ / 設定 | 共通Config・Logger・TimestampManagerで管理 |

### 主要な更新内容

1. **WikiArt_VLMパイプラインの安定化**
   - `scripts/train_wikiart_vlm.py` による end-to-end 実行
   - Stable-Diffusion / FLUX / F-Lite ごとの特徴量抽出と性能比較
2. **ゲシュタルト原則スコアリング**
   - `scripts/score_gestalt_principles.py` でLLaVA 7Bを利用した自動スコア化
   - スコアを特徴量CSVへ統合し、解釈性を強化
3. **shap-class 連携教材**
   - `notebooks/prepare_data_for_shap_class.ipynb` でtrain/validation CSVを生成
   - そのまま `shap-class` リポジトリに投入して授業用の結果を再現可能

### 実装済み機能（✓）

- WikiArt_VLMデータローダーおよび特徴量抽出
- Gestaltスコア統合と再利用可能な特徴量CSV
- RandomForest + GridSearchCV + Stratified split
- SHAP summary / dependence plot / feature importance
- shap-class用データ出力ノートブック
- Config / Logger / TimestampManager などの共通基盤

### 部分実装・未実装

- 構図特徴量の細分化（3×3領域解析など）は未着手
- 誤分類作品ビューア / インタラクティブレポートは未実装
- 多クラス（様式ごと）や回帰タスクは今後の検討項目

### 今後の改善予定

1. ペア単位でのデータ分割（Original/Generatedが異なるfoldに入らないよう調整）
2. 追加の説明可視化（force plot、誤分類サマリ）  
3. WikiArt_VLM以外の生成モデルを追加して汎化性能を確認

## 参考資料

- [WikiArt_VLM (HuggingFace)](https://huggingface.co/datasets/keremberke/wikiart-vlm-dataset)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)

## ドキュメント

プロジェクトの詳細な情報は`docs/`ディレクトリを参照してください：

- [実装状況詳細レポート](docs/reports/implementation_status_report.md)
- [演習・評価レポート（HTML版）](docs/reports/exercise_report.html)
- [特徴量定義ガイド](docs/guides/feature_columns_definition.md)
- [開発履歴](docs/history/cursor_chat_history.md)

## トラブルシューティング

### ログファイルの確認

各スクリプトの実行ログは`logs/`ディレクトリに保存されます：

```bash
# ログファイルの一覧確認
ls logs/

# 特定のログファイルを確認
cat logs/project.log
```

### データセットの問題

- `data/external/WikiArt_VLM-main/` が存在するか確認してください（git submodule もしくはZIP展開が必要です）。
- 生成モデル別ディレクトリ（`images/Original`, `images/Stable-Diffusion` など）が揃っているか確認してください。
- `scripts/train_wikiart_vlm.py --mode features ...` を再実行すると不足している特徴量CSVが再生成されます。

### Jupyter Notebookの問題

Jupyter Notebookが起動しない場合は、カーネルが正しく設定されているか確認してください：

```bash
# カーネルの一覧確認
jupyter kernelspec list

# カーネルの再インストール（explainable-art-classification）
source .venv/bin/activate  # 仮想環境を有効化
pip install ipykernel jupyter
python -m ipykernel install --user --name explainable-art-classification --display-name "Python (explainable-art-classification)"
```

### shap-class用データ準備ノートブックの問題

- **ファイルが見つからないエラー**: Notebookの最初のセル（セル1）を実行して、プロジェクトルートが正しく設定されているか確認してください。
- **GitHub TOKENエラー**: セル16でGitHub TOKENが必要です。環境変数`GITHUB_TOKEN`を設定するか、コード内のTOKENを更新してください。
- **shap-classリポジトリのクローンエラー**: 既に`shap-class`ディレクトリが存在する場合は、クローンをスキップします。必要に応じて既存のディレクトリを削除してください。
