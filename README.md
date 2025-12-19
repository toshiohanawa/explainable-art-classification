# 説明可能AIによる絵画様式分類プロジェクト

## プロジェクト概要

このプロジェクトは、WikiArt_VLMデータセットに含まれる本物の絵画画像と生成AI（Stable Diffusion / FLUX / F-Lite）による模倣画像を対象に、ランダムフォレストとSHAPで「本物 vs フェイク」を説明可能に分類する講義教材です。

## 特徴

- **説明可能AI**: SHAPを用いてモデルの判断根拠を可視化
- **芸術とテクノロジーの融合**: 機械学習と芸術史の学際的アプローチ
- **教育的価値**: データサイエンス学習者向けの実践的教材
- **公開データ**: WikiArt_VLMデータセットとGestaltスコアを活用

## 技術スタック

### バックエンド
- **Python 3.9+**
- **機械学習**: scikit-learn, SHAP
- **画像処理**: OpenCV, PIL, scikit-image
- **API**: FastAPI, uvicorn
- **可視化**: matplotlib, seaborn, plotly
- **データ管理**: pandas, numpy
- **データソース**: WikiArt_VLMデータセット（Original + Generatedペア）

### フロントエンド
- **React 19** + **TypeScript**
- **Vite**: ビルドツール・開発サーバ
- **Tailwind CSS 3**: スタイリング
- **環境変数**: `.env` でAPIエンドポイント設定

### 外部サービス（オプション）
- **Ollama / LLaVA**: ゲシュタルト原則スコアリング用（MLLM）

## プロジェクト構造

```
explainable-art-classification/
├── src/                    # ソースコード
│   ├── api/                # FastAPI推論エンドポイント
│   │   └── app.py          # FastAPIアプリケーション（/health, /predict）
│   ├── inference/          # 推論サービス
│   │   └── service.py      # InferenceService（画像→特徴量→推論→結果）
│   ├── data_collection/    # データ収集
│   │   ├── wikiart_vlm_loader.py  # WikiArt_VLMデータローダー
│   │   ├── met_api_client.py      # Met APIクライアント（レガシー）
│   │   └── hybrid_collector.py    # ハイブリッドデータ収集
│   ├── feature_extraction/  # 特徴量抽出
│   │   ├── color_extractor.py      # 色彩・テクスチャ・エッジ特徴量抽出
│   │   └── gestalt_scorer.py       # ゲシュタルト原則スコアリング（LLaVA連携）
│   ├── model_training/     # モデル訓練
│   │   └── random_forest_trainer.py # RandomForest訓練・評価・保存
│   ├── explainability/     # SHAP説明
│   │   └── shap_explainer.py       # SHAP TreeExplainer・可視化
│   ├── analysis/           # 分析ツール
│   │   └── authenticity_analyzer.py # 誤分類分析・生成モデル比較
│   ├── visualization/      # 可視化
│   │   └── result_visualizer.py    # 結果可視化ユーティリティ
│   └── utils/             # ユーティリティ
│       ├── config_manager.py   # 設定管理（YAML読み込み）
│       ├── logger.py            # 共通ログ設定
│       ├── status_checker.py    # ステータスチェック
│       └── timestamp_manager.py # タイムスタンプ管理（latest/experiments切り替え）
├── frontend/              # React + TypeScript + Vite フロントエンド
│   ├── src/
│   │   ├── App.tsx         # メインUIコンポーネント（アップロード・結果表示）
│   │   ├── main.tsx        # エントリーポイント
│   │   └── index.css       # Tailwind CSS設定
│   ├── package.json        # npm依存関係
│   ├── tailwind.config.js  # Tailwind設定
│   ├── vite.config.ts      # Vite設定
│   └── tsconfig.*.json    # TypeScript設定
├── scripts/               # 実行スクリプト
│   ├── train_wikiart_vlm.py        # WikiArt_VLMパイプライン（特徴量抽出～学習）
│   ├── score_gestalt_principles.py  # MLLMによるゲシュタルト原則スコア評価
│   ├── compare_generation_models.py# 生成モデル間の性能比較
│   ├── run_api.py                  # FastAPIサーバ起動スクリプト
│   ├── run_all_phases.py           # 現行フェーズ一括実行ラッパー
│   ├── analyze_bias.py              # バイアス可視化
│   ├── visualize_feature_extraction.py # 特徴量抽出可視化
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
│   │   │   └── latest/
│   │   │       ├── random_forest_model.pkl  # 学習済みRFモデル
│   │   │       └── scaler.pkl              # StandardScaler
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
├── requirements.txt      # Python依存関係
└── main.py              # メイン実行ファイル（レガシー）
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

### 4. フロントエンド環境のセットアップ

```bash
cd frontend
npm install      # 初回のみ
```

## クイックスタート

### 最短パス: 推論UIの起動（既存モデル使用）

既に学習済みモデルがある場合、以下の手順で推論UIを起動できます：

```bash
# 1. バックエンドAPI起動
ENABLE_GESTALT=true MAX_UPLOAD_BYTES=31457280 uv run python scripts/run_api.py --host 0.0.0.0 --port 8000 --reload

# 2. 別ターミナルでフロントエンド起動
cd frontend
npm run dev      # http://localhost:5173 でアクセス可能
```

ブラウザで http://localhost:5173 を開き、画像をアップロードして推論結果を確認できます。

### フルパイプライン: モデル訓練から推論まで

```bash
# 1. 特徴量抽出 + ゲシュタルトスコア統合
python scripts/train_wikiart_vlm.py --mode features --generation-model Stable-Diffusion --include-gestalt

# 2. モデル訓練
python scripts/train_wikiart_vlm.py --mode train --generation-model Stable-Diffusion --include-gestalt

# 3. 推論API起動
ENABLE_GESTALT=true uv run python scripts/run_api.py --host 0.0.0.0 --port 8000

# 4. フロントエンド起動（別ターミナル）
cd frontend && npm run dev
```

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

### 学習済みモデルを使った推論API (FastAPI)

学習済みランダムフォレスト (`data/artifacts/models/latest/random_forest_model.pkl`, `scaler.pkl`) を読み込むAPIを起動できます。

#### 起動方法

```bash
# 推奨: uv経由で起動（リロード機能付き）
uv run python scripts/run_api.py --host 0.0.0.0 --port 8000 --reload

# 直接uvicornを使う場合
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# 環境変数付き起動
ENABLE_GESTALT=true MAX_UPLOAD_BYTES=31457280 uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

#### 環境変数

- `MAX_UPLOAD_BYTES`: アップロード上限（デフォルト: 8MB = 8388608）
- `ENABLE_GESTALT=true`: ゲシュタルト指標計算を有効化（Ollama / LLaVA が必要）

#### APIエンドポイント

**GET /health**
- 用途: ヘルスチェック・モデル状態確認
- レスポンス例:
  ```json
  {
    "status": "ok",
    "model_version": "2025-11-07 06:25:23",
    "feature_count": 44,
    "gestalt_enabled": false,
    "model_error": null,
    "model_loaded": true
  }
  ```

**POST /predict**
- 用途: 画像ファイルから本物/偽物判定
- パラメータ:
  - `file`: 画像ファイル（multipart/form-data、.jpg/.jpeg/.png）
  - `include_gestalt`: boolean（クエリパラメータ、デフォルト: false）
- レスポンス例:
  ```json
  {
    "prediction": {
      "label": "fake",
      "probabilities": {"authentic": 0.404, "fake": 0.596},
      "confidence": 0.596,
      "feature_importances": [
        {"feature": "texture_lbp_std", "importance": 0.172, "value": 2.957},
        {"feature": "texture_lbp_histogram_entropy", "importance": 0.109, "value": 3.162},
        ...
      ]
    },
    "features": {
      "mean_hue": 50.825,
      "mean_saturation": 72.712,
      ...
    },
    "metadata": {
      "width": 1080,
      "height": 1392,
      "format": "JPEG",
      "file_size_bytes": 389017
    },
    "gestalt_status": "skipped",
    "model_version": "2025-11-07 06:25:23",
    "file_name": "example.jpg"
  }
  ```

#### エラーハンドリング

- `400 Bad Request`: ファイル名が空、画像が空、未対応拡張子
- `413 Payload Too Large`: ファイルサイズが上限超過
- `500 Internal Server Error`: 推論処理中のエラー（モデル未読み込み、特徴量抽出失敗など）

### Reactベースの推論UI (Vite + Tailwind)

アップロード〜推論結果表示までを行うモダンUIを提供します。

#### セットアップ

```bash
cd frontend
npm install      # 初回のみ
npm run dev      # http://localhost:5173 で起動
npm run build    # プロダクションビルド
```

#### 環境変数設定

`.env` ファイル（`frontend/.env`）を作成してAPIエンドポイントを設定：

```bash
VITE_API_BASE_URL=http://localhost:8000
```

#### UI機能詳細

**ヘッダー**
- プロジェクトタイトル・説明
- 画像ファイル選択ボタン
- ゲシュタルト計算チェックボックス
- API URL入力フィールド
- 推論実行ボタン

**メインエリア（3カラムレイアウト）**

1. **左カラム（画像プレビュー + メタ情報）**
   - アップロード画像のプレビュー
   - 寸法（width × height）
   - 形式（JPEG/PNG等）
   - ファイルサイズ
   - ゲシュタルト計算ステータス

2. **中央カラム（推論結果 + 重要度特徴）**
   - 本物/偽物ラベル（バッジ表示）
   - 確信度パーセンテージ
   - 確率バー（Authentic/Fake）
   - **重要度の高い特徴（RF feature_importances）**
     - ランク番号
     - 横バーチャート（重要度の視覚化）
     - 特徴量名（完全表示）
     - 実際の値
     - 重要度パーセンテージ

3. **右カラム（色・テクスチャ + ゲシュタルト指標）**
   - **色・テクスチャ特徴**
     - mean_hue, mean_saturation, mean_value
     - hue_std, saturation_std
     - color_diversity
     - dominant_color_ratio_max
   - **ゲシュタルト指標**（計算済みの場合）
     - simplicity, proximity, similarity
     - continuity, closure, figure_ground
   - モデルバージョン表示

**履歴セクション**
- セッション内の推論履歴（最大8件）
- 各履歴アイテム:
  - サムネイル画像
  - ラベル（Authentic/Fake）
  - 確信度
  - タイムスタンプ
  - ファイル名

#### メモリ管理

- `URL.createObjectURL()` で生成したオブジェクトURLは適切に `URL.revokeObjectURL()` で解放
- 履歴アイテムは独立したURLを保持（現在のプレビューURLとは分離）
- コンポーネントアンマウント時に全URLをクリーンアップ

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

## システムアーキテクチャ

### 全体構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                        ユーザー（ブラウザ）                        │
└────────────────────────────┬──────────────────────────────────┘
                              │ HTTP/HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  React Frontend (Vite + Tailwind)                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  App.tsx                                                   │   │
│  │  - 画像アップロード（ドラッグ&ドロップ）                    │   │
│  │  - 推論結果表示（ラベル・確率・特徴量重要度）              │   │
│  │  - 色・テクスチャ・ゲシュタルト指標表示                    │   │
│  │  - 推論履歴（セッション内）                                │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬──────────────────────────────────┘
                              │ REST API (POST /predict)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Backend (uvicorn)                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  src/api/app.py                                           │   │
│  │  - GET /health: ヘルスチェック                             │   │
│  │  - POST /predict: 画像→推論結果                           │   │
│  └───────────────────────┬──────────────────────────────────┘   │
└───────────────────────────┼──────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  InferenceService (src/inference/service.py)                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. 画像前処理 (ColorFeatureExtractor.preprocess_image)  │   │
│  │  2. 特徴量抽出 (extract_all_features)                     │   │
│  │     - 色彩: HSV統計、支配色、色多様性                     │   │
│  │     - テクスチャ: GLCM、LBP                                │   │
│  │     - エッジ: Canny、曲率、方向性                         │   │
│  │  3. ゲシュタルト指標（オプション）                         │   │
│  │     - GestaltScorer.score_single_image()                  │   │
│  │     - Ollama/LLaVA API呼び出し                            │   │
│  │  4. 特徴量ベクトル構築・スケーリング                       │   │
│  │  5. モデル推論 (RandomForest.predict_proba)               │   │
│  │  6. 特徴量重要度計算 (feature_importances_)                │   │
│  │  7. メタデータ抽出 (EXIF、サイズ)                         │   │
│  └───────────────────────┬──────────────────────────────────┘   │
└───────────────────────────┼──────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  学習済みモデル (data/artifacts/models/latest/)                   │
│  - random_forest_model.pkl (RandomForestClassifier)              │
│  - scaler.pkl (StandardScaler)                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 処理フロー

### 1. モデル訓練パイプライン

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: データ準備                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  WikiArt_VLMデータセット読み込み      │
        │  (Original + Generated ペア)         │
        └─────────────────┬───────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: 特徴量抽出                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                             │
        ▼                                             ▼
┌──────────────────┐                      ┌──────────────────┐
│ 色彩・テクスチャ・  │                      │  ゲシュタルト原則  │
│ エッジ特徴量抽出   │                      │  スコアリング      │
│                    │                      │  (LLaVA/Ollama)  │
│ - HSV統計          │                      │                  │
│ - GLCM/LBP         │                      │ - Simplicity     │
│ - Cannyエッジ      │                      │ - Proximity      │
│                    │                      │ - Similarity     │
└────────┬──────────┘                      │ - Continuity     │
         │                                  │ - Closure        │
         │                                  │ - Figure-Ground  │
         └──────────┬───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  特徴量CSV統合          │
        │  (wikiart_vlm_features │
        │   _with_gestalt_*.csv) │
        └───────────┬───────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: モデル訓練                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                             │
        ▼                                             ▼
┌──────────────────┐                      ┌──────────────────┐
│  データ分割       │                      │  ハイパーパラメータ│
│  (train/test)    │                      │  チューニング      │
│  Stratified      │                      │  (GridSearchCV)   │
└────────┬─────────┘                      └────────┬──────────┘
         │                                           │
         └───────────────┬───────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────┐
        │  RandomForest訓練         │
        │  - n_estimators: 100     │
        │  - max_depth: 10         │
        │  - 交差検証 (5-fold)      │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │  モデル・スケーラー保存   │
        │  - random_forest_model.pkl│
        │  - scaler.pkl             │
        └──────────┬───────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: 説明可能性解析                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                             │
        ▼                                             ▼
┌──────────────────┐                      ┌──────────────────┐
│  SHAP TreeExplainer│                      │  可視化生成       │
│  作成              │                      │                  │
│  - 背景データ準備   │                      │ - Summary plot   │
│  - SHAP値計算      │                      │ - Dependence plot│
│                    │                      │ - Feature imp.   │
└────────┬───────────┘                      └────────┬──────────┘
         │                                           │
         └───────────────┬───────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────┐
        │  結果保存                 │
        │  - shap_values.csv        │
        │  - feature_importance.png │
        │  - confusion_matrix.png   │
        └──────────────────────────┘
```

### 2. 推論APIフロー

```
ユーザー画像アップロード
        │
        ▼
┌───────────────────────────────────┐
│  FastAPI /predict エンドポイント   │
│  - ファイル検証（拡張子・サイズ）  │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  InferenceService.predict_image_bytes()│
└───────────────┬───────────────────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
┌──────────────┐  ┌──────────────┐
│ 画像前処理    │  │ 特徴量抽出   │
│ (512x512)    │  │              │
│              │  │ - 色彩       │
│              │  │ - テクスチャ  │
│              │  │ - エッジ     │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
┌──────────────┐  ┌──────────────┐
│ ゲシュタルト  │  │ 特徴量ベクトル│
│ スコア計算    │  │ 構築          │
│ (オプション)  │  │              │
│              │  │ - 44次元     │
│ - LLaVA API  │  │ - スケーリング│
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                │
                ▼
┌───────────────────────────────────┐
│  モデル推論                        │
│  - RandomForest.predict_proba()    │
│  - ラベル判定 (authentic/fake)     │
│  - 確率計算                        │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  特徴量重要度計算                  │
│  - feature_importances_ から      │
│  - トップ10特徴量を抽出            │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  レスポンス構築                    │
│  {                                 │
│    prediction: {                   │
│      label, probabilities,        │
│      confidence,                  │
│      feature_importances           │
│    },                              │
│    features: {...},               │
│    metadata: {...},               │
│    gestalt_status,                │
│    model_version                   │
│  }                                 │
└───────────────┬───────────────────┘
                │
                ▼
         JSONレスポンス返却
```

### 3. フロントエンド処理フロー

```
ユーザー操作
        │
        ▼
┌───────────────────────────────────┐
│  画像ファイル選択                   │
│  - ドラッグ&ドロップ or ファイル選択│
│  - URL.createObjectURL() でプレビュー│
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  推論実行ボタンクリック            │
│  - FormData作成                    │
│  - fetch() で POST /predict       │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  APIレスポンス受信                 │
│  - 結果をstateに保存               │
│  - 履歴に追加（独立URL生成）       │
└───────────────┬───────────────────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
┌──────────────┐  ┌──────────────┐
│ 結果表示      │  │ 履歴表示     │
│              │  │              │
│ - ラベル      │  │ - サムネイル │
│ - 確信度バー  │  │ - ラベル     │
│ - 確率        │  │ - 確信度     │
│ - 重要度特徴  │  │ - タイムスタンプ│
│ - 色特徴      │  │              │
│ - ゲシュタルト│  │              │
│ - EXIF        │  │              │
└──────────────┘  └──────────────┘
```

### 4. データフロー図

```
WikiArt_VLMデータセット
        │
        ├─ Original画像 (本物)
        └─ Generated画像 (フェイク: Stable-Diffusion/FLUX/F-Lite)
                │
                ▼
┌──────────────────────────────────────────────┐
│  特徴量抽出パイプライン                        │
│  (ColorFeatureExtractor.extract_all_features)│
└───────────────────┬──────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ 色彩特徴  │  │ テクスチャ│  │ エッジ特徴│
│          │  │ 特徴      │  │          │
│ - HSV    │  │ - GLCM    │  │ - Canny  │
│ - 支配色  │  │ - LBP     │  │ - 曲率   │
│ - 多様性  │  │           │  │ - 方向性 │
└────┬─────┘  └────┬──────┘  └────┬─────┘
     │             │              │
     └──────┬──────┴──────┬───────┘
            │             │
            ▼             ▼
    ┌──────────────┐  ┌──────────────┐
    │ 統合特徴量CSV │  │ ゲシュタルト  │
    │              │  │ スコア        │
    │ 38特徴量      │  │ (6原則)      │
    └──────┬───────┘  └──────┬───────┘
           │                 │
           └────────┬────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  統合特徴量CSV         │
        │  (44特徴量)            │
        │  - 色彩: 19            │
        │  - テクスチャ: 9        │
        │  - エッジ: 10           │
        │  - ゲシュタルト: 6      │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  データ分割             │
        │  - Train: 70%          │
        │  - Test: 20%          │
        │  - Stratified split   │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  標準化 (StandardScaler)│
        │  - fit on train       │
        │  - transform test     │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  RandomForest訓練      │
        │  - GridSearchCV       │
        │  - Cross-validation   │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  モデル保存             │
        │  - model.pkl          │
        │  - scaler.pkl         │
        └───────────────────────┘
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

## 主要モジュール詳細

### 推論システム

#### InferenceService (`src/inference/service.py`)

推論処理の中核となるサービスクラス。画像アップロードから特徴量抽出、モデル推論、結果返却までを統合的に処理します。

**主要メソッド:**
- `__init__()`: モデル・スケーラーの読み込み（エラー時はグレースフルに継続）
- `predict_image_bytes()`: 画像バイト列から推論を実行
- `_load_model()`: モデルファイルの読み込み（FileNotFoundErrorをキャッチ）
- `_top_feature_importances()`: RF feature_importances_ からトップN特徴量を抽出
- `_compute_gestalt_scores()`: ゲシュタルト原則スコア計算（Ollama/LLaVA連携）
- `_extract_metadata()`: EXIF情報・画像サイズの抽出

**特徴:**
- モデル未読み込み時も起動可能（`/health`で状態確認）
- ゲシュタルト計算はオプション（`ENABLE_GESTALT`環境変数で制御）
- 一時ファイルの自動クリーンアップ

#### FastAPI App (`src/api/app.py`)

RESTful APIエンドポイントを提供するFastAPIアプリケーション。

**エンドポイント:**
- `GET /health`: モデル状態・バージョン・特徴量数の確認
- `POST /predict`: 画像ファイルアップロード→推論結果返却

**セキュリティ:**
- CORS設定（開発環境では全オリジン許可）
- ファイルサイズ制限（`MAX_UPLOAD_BYTES`環境変数）
- 拡張子バリデーション（`.jpg`, `.jpeg`, `.png`のみ）

### 特徴量抽出

#### ColorFeatureExtractor (`src/feature_extraction/color_extractor.py`)

画像から多様な特徴量を抽出するクラス。

**抽出特徴量:**

1. **色彩特徴量（19次元）**
   - HSV統計: mean_hue, mean_saturation, mean_value, hue_std, saturation_std, value_std
   - 色多様性: color_diversity
   - 支配色: dominant_color_hue_mean/std, dominant_color_saturation_mean/std, dominant_color_value_mean/std, dominant_color_ratio_max/mean/std
   - 明度分布: brightness_skewness, brightness_kurtosis, brightness_entropy

2. **テクスチャ特徴量（9次元）**
   - GLCM: texture_glcm_contrast, texture_glcm_dissimilarity, texture_glcm_homogeneity, texture_glcm_energy, texture_glcm_correlation
   - LBP: texture_lbp_mean, texture_lbp_std, texture_lbp_histogram_entropy, texture_lbp_histogram_skewness, texture_lbp_histogram_kurtosis

3. **エッジ特徴量（10次元）**
   - 基本: edge_density, edge_mean_length, edge_std_length, edge_count
   - 方向性: edge_orientation_mean, edge_orientation_std
   - 形状: edge_smoothness, edge_curvature_mean, edge_curvature_std

4. **ゲシュタルト原則スコア（6次元、オプション）**
   - simplicity_score, proximity_score, similarity_score, continuity_score, closure_score, figure_ground_score

**処理フロー:**
```
画像読み込み → RGB変換 → リサイズ(512x512)
    ↓
HSV変換 → 色彩統計計算
    ↓
グレースケール変換 → GLCM/LBP計算
    ↓
Cannyエッジ検出 → エッジ特徴量計算
    ↓
特徴量辞書統合 → 返却
```

### モデル訓練

#### RandomForestTrainer (`src/model_training/random_forest_trainer.py`)

ランダムフォレストモデルの訓練・評価・保存を行うクラス。

**訓練プロセス:**
1. 特徴量データ読み込み
2. ラベル準備（authenticity: 0=本物, 1=フェイク）
3. Train/Test分割（Stratified split）
4. StandardScalerで標準化（訓練データでfit、テストデータに適用）
5. GridSearchCVでハイパーパラメータチューニング
6. 最終モデル訓練
7. 評価（精度、混同行列、交差検証）
8. モデル・スケーラー保存

**保存ファイル:**
- `random_forest_model.pkl`: モデル、label_encoder、feature_columnsを含む辞書
- `scaler.pkl`: StandardScalerオブジェクト

### ゲシュタルト原則スコアリング

#### GestaltScorer (`src/feature_extraction/gestalt_scorer.py`)

Ollama/LLaVAを使用してゲシュタルト原則スコアを計算するクラス。

**評価原則:**
- Simplicity（簡潔性）
- Proximity（近接性）
- Similarity（類同）
- Continuity（連続性）
- Closure（閉合性）
- Figure-Ground Separation（図地分離）

**処理フロー:**
```
画像 → base64エンコード
    ↓
Ollama API呼び出し (LLaVA)
    ↓
JSONレスポンス解析
    ↓
各原則のスコア（1-5）抽出
    ↓
スコア辞書返却
```

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

### 実装状況（2025年12月7日更新）

| 項目 | 状況 |
| --- | --- |
| データ取得 | WikiArt_VLM-main一式をローカルに同期済み |
| 特徴量抽出 | 色彩 / テクスチャ / エッジ / ゲシュタルトをCSV化済み |
| モデル訓練 | RandomForest + GridSearch + Stratified split 完了 |
| 説明可能性 | SHAP summary / dependence plot / feature importance 完了 |
| shap-class連携 | Notebookでtrain/validation CSV書き出し済み |
| ログ / 設定 | 共通Config・Logger・TimestampManagerで管理 |
| **推論API** | **FastAPI + InferenceService 実装完了** |
| **Web UI** | **React + TypeScript + Tailwind 実装完了** |
| **エラーハンドリング** | **モデル未読み込み時のグレースフル処理** |
| **メモリ管理** | **オブジェクトURLの適切な解放** |

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

**データ処理・訓練**
- WikiArt_VLMデータローダーおよび特徴量抽出
- Gestaltスコア統合と再利用可能な特徴量CSV
- RandomForest + GridSearchCV + Stratified split
- SHAP summary / dependence plot / feature importance
- shap-class用データ出力ノートブック
- Config / Logger / TimestampManager などの共通基盤

**推論システム（新規追加）**
- FastAPI推論エンドポイント（`/health`, `/predict`）
- InferenceService（画像→特徴量→推論の統合処理）
- モデル未読み込み時のグレースフルエラーハンドリング
- ファイル名・拡張子・サイズのバリデーション
- ゲシュタルト指標のオプション計算（Ollama/LLaVA連携）
- RF feature_importances ベースの特徴量重要度表示

**Web UI（新規追加）**
- React + TypeScript + Vite によるモダンUI
- Tailwind CSS による洗練されたデザイン
- 画像アップロード（ドラッグ&ドロップ対応）
- 推論結果の包括的表示（ラベル・確率・重要度・特徴量）
- 推論履歴管理（セッション内、最大8件）
- メモリリーク防止（オブジェクトURLの適切な解放）
- レスポンシブデザイン（3カラムレイアウト）

### 部分実装・未実装

- 構図特徴量の細分化（3×3領域解析など）は未着手
- 誤分類作品ビューア / インタラクティブレポートは未実装
- 多クラス（様式ごと）や回帰タスクは今後の検討項目
- SHAP背景データの自動読み込み（現在は警告のみ、推論は動作）

### 今後の改善予定

1. ペア単位でのデータ分割（Original/Generatedが異なるfoldに入らないよう調整）
2. 追加の説明可視化（force plot、誤分類サマリ）  
3. WikiArt_VLM以外の生成モデルを追加して汎化性能を確認
4. SHAP背景データの自動生成・統合
5. 推論履歴の永続化（ローカルストレージ）
6. バッチ推論機能（複数画像の一括処理）

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

### 推論APIの問題

- **モデルファイルが見つからない**: `data/artifacts/models/latest/random_forest_model.pkl` が存在するか確認してください。存在しない場合、APIは起動しますが `/predict` でエラーが返ります。`/health` で `model_loaded: false` を確認できます。
- **背景データ警告**: 「背景データに必要な列が不足しています」という警告はSHAP用の背景サンプル読み込み時のものです。推論自体は正常に動作します。
- **ゲシュタルト計算が失敗**: Ollamaが起動していない、またはLLaVAモデルがインストールされていない場合、ゲシュタルト計算はスキップされます。`gestalt_status` にエラー内容が表示されます。

### フロントエンドの問題

- **API接続エラー**: `.env` ファイルで `VITE_API_BASE_URL` が正しく設定されているか確認してください。デフォルトは `http://localhost:8000` です。
- **ビルドエラー**: `npm install` を再実行して依存関係を再インストールしてください。
- **画像が表示されない**: ブラウザのコンソールでエラーを確認してください。CORS設定やAPIエンドポイントの確認が必要な場合があります。
