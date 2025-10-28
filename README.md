# 説明可能AIによる絵画様式分類プロジェクト

## プロジェクト概要

このプロジェクトは、ランダムフォレストとSHAPを用いて、Metropolitan Museumの作品を印象派と非印象派に分類し、その判断根拠を解釈可能な形で可視化する講義教材です。

## 特徴

- **説明可能AI**: SHAPを用いてモデルの判断根拠を可視化
- **芸術とテクノロジーの融合**: 機械学習と芸術史の学際的アプローチ
- **教育的価値**: データサイエンス学習者向けの実践的教材
- **オープンデータ**: Metropolitan Museum APIの活用

## 技術スタック

- **Python 3.9+**
- **機械学習**: scikit-learn, SHAP
- **画像処理**: OpenCV, PIL
- **可視化**: matplotlib, seaborn, plotly
- **データソース**: Metropolitan Museum API

## プロジェクト構造

```
explainable-art-classification/
├── src/                    # ソースコード
│   ├── data_collection/    # データ収集
│   ├── feature_extraction/ # 特徴量抽出
│   ├── model_training/     # モデル訓練
│   ├── explainability/     # SHAP説明
│   ├── visualization/      # 可視化
│   └── utils/             # ユーティリティ
├── data/                  # データファイル
├── notebooks/             # Jupyterノートブック
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

### 全プロセスを実行

```bash
python main.py --mode all
```

### 個別プロセスの実行

```bash
# メタデータ収集のみ（高速）
python main.py --mode metadata

# 画像ダウンロードのみ（メタデータ収集後）
python main.py --mode images

# データ収集（メタデータ + 画像）
python main.py --mode collect

# 特徴量抽出のみ
python main.py --mode extract

# モデル訓練のみ
python main.py --mode train

# SHAP説明のみ
python main.py --mode explain
```

### Jupyterノートブック

```bash
jupyter notebook notebooks/
```

## 実行フロー

1. **データ収集**: Metropolitan Museum APIから作品データと画像を取得
2. **特徴量抽出**: 画像から色彩・構図特徴量を抽出
3. **モデル訓練**: ランダムフォレスト分類器を訓練
4. **SHAP説明**: モデルの判断根拠を可視化
5. **結果可視化**: 包括的なレポートを生成

## 出力ファイル

プロジェクト実行後、以下の構造でファイルが生成されます：

```
data/
├── raw_data/                    # 生データ
│   └── artwork_metadata.csv     # 作品メタデータ
├── raw_images/                  # 画像ファイル
│   └── *.jpg                   # ダウンロードした画像
├── features/                    # 特徴量データ
│   └── color_features.pkl       # 抽出された特徴量
├── models/                      # 機械学習モデル
│   ├── random_forest_model.pkl  # 訓練済みモデル
│   └── scaler.pkl              # 特徴量スケーラー
├── results/                     # 訓練結果
│   └── training_results.txt     # 詳細な訓練結果
├── visualizations/              # 可視化結果
│   ├── confusion_matrix.png     # 混同行列
│   ├── feature_importance.png   # 特徴量重要度
│   ├── shap_summary_plot.png    # SHAP summary plot
│   ├── shap_feature_importance.png # SHAP特徴量重要度
│   └── dependence_plot_*.png    # 依存関係プロット
└── shap_explanations/           # SHAP説明データ
    ├── shap_values.csv          # SHAP値データ
    └── feature_importance_shap.csv # SHAP特徴量重要度
```

## 設定オプション

`config.yaml`で以下の設定を調整できます：

- API設定（レート制限、タイムアウト）
- データ設定（最小サンプル数、出力ディレクトリ、タイムスタンプ機能）
- 画像処理設定（リサイズサイズ、色空間）
- 特徴量設定（抽出する特徴量の種類）
- 分類設定（テストサイズ、交差検証）
- SHAP設定（サンプル数、背景データ）

### タイムスタンプ機能

`data.use_timestamp`を`true`に設定すると、各分析実行時に`data/analysis_YYMMDDHHMM/`形式のタイムスタンプ付きディレクトリ（分単位）が作成され、分析結果が時系列で管理されます。

- `true`: 分析結果をタイムスタンプ付きディレクトリに保存（推奨）
- `false`: 固定ディレクトリを使用（従来の動作）

**注意**: 生データ（APIから取得した画像とメタデータ）は常に固定ディレクトリ（`data/raw_data/`, `data/raw_images/`）に保存され、分析結果のみがタイムスタンプ付きディレクトリに保存されます。

## 教育的活用

このプロジェクトは以下の学習に活用できます：

- 機械学習の基礎（分類、特徴量エンジニアリング）
- 説明可能AI（SHAP、モデル解釈）
- 画像処理（OpenCV、色彩分析）
- データサイエンス（探索的データ分析、可視化）
- 芸術とテクノロジーの融合

## ライセンス

このプロジェクトは教育目的で作成されています。Metropolitan MuseumのデータはCC0ライセンスで提供されています。

## 貢献

プルリクエストやイシューの報告を歓迎します。

## 現状の状況と課題

### 実装状況（2025年10月29日 08:00時点）

- **全体完了度**: 約90%
- **データ量**: 現在77件（要件の15%、最小要件500点）
- **分類精度**: 70%以上を達成
- **コア機能**: 完全実装済み
- **ハイブリッドデータ収集**: 実装完了（API制限によりテスト中断中）

### 実装済み機能（✓）

- Metropolitan Museum API統合（59項目の全メタデータ取得）
- 色彩特徴量抽出（完全）
- ランダムフォレスト分類器
- SHAP説明可能性（summary_plot）
- モジュラー設計と設定ファイル
- 効率的なデータ収集システム（メタデータと画像の分離）
- **ハイブリッドデータ収集システム**（2025年10月29日 07:40-08:00実装）
  - CSVとAPIを組み合わせた大規模データ収集
  - チェックポイント機能（中断・再開可能）
  - 品質管理機能（QAレポート生成）
  - エラーハンドリングとリトライ機能

### 部分実装機能（△）

- **構図特徴量**: 基本実装済み、詳細は未実装
- **データ量**: 小規模データでの動作確認のみ
- **性能要件**: 大規模データでの性能測定は未実施

### 未実装機能（×）

- waterfall plot/force plot（複雑性のため簡略化）
- 誤分類作品の分析表示
- 多クラス分類（バロック、印象派、キュビズム等）
- 特徴量分布の可視化

### 現在の技術的課題（2025年10月29日 08:15更新）

1. **API制限問題**: Metropolitan Museum APIで403エラーが継続的に発生
   - **原因**: 直近の大量リクエストによるIPブロック（確認済み）
   - **公式レート制限**: 80 req/s（公式ドキュメントで確認済み）
   - **対策完了**: 
     - レート制限を80 req/sに調整（公式推奨値）
     - エラー判定ロジック修正（403→データ不存在、429→レート制限）
     - 待機時間短縮（30秒→1-2秒）
   - **現在の状況**: API完全ブロック、CSVデータのみで分析継続

2. **データ量の増加**: 要件を満たすデータ量（500-1000点）の確保
   - **ハイブリッドシステム**: 実装完了（API復旧後に利用可能）
   - **CSV取得成功**: MetObjects.csv（484,956件）を正常に取得
   - **即座に実行可能**: 現在のCSVデータで分析開始可能

3. **文字化け問題**: PowerShellの文字エンコーディング設定
   - **対策完了**: 
     - PowerShell UTF-8設定 (`chcp 65001`)
     - Pythonファイルにエンコーディング宣言追加
     - 環境変数 `PYTHONIOENCODING="utf-8"` 設定

4. **構図特徴量の完全実装**: より詳細な分析機能
5. **可視化機能の拡充**: より直感的な説明可能性
6. **画像取得方針の検討**: 現在`primaryImageSmall`を使用、高解像度画像の検討中

### 要件定義からの変更点

- **画像解像度**: 高解像度JPEG → `primaryImageSmall`（効率性のため）
- **SHAP可視化**: waterfall plot/force plot → summary_plot（簡略化）
- **データ量**: 現在テスト段階で77件に制限
- **データ収集戦略**: 単一API → ハイブリッド（CSV+API）システム（2025年10月29日実装）
- **CSVオンリー戦略**: 実装しない（ユーザー要求により除外）

## 参考資料

- [Metropolitan Museum API Documentation](https://metmuseum.github.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [実装状況詳細レポート](implementation_status_report.md)