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
│   ├── collect_all_paintings.py   # 全件データ収集
│   ├── download_painting_images.py # 画像ダウンロード
│   ├── filter_paintings_only.py    # 絵画フィルタリング
│   ├── filter_european_paintings.py # European Paintingsフィルタリング
│   ├── create_dataset_from_images.py # 画像からデータセット作成
│   ├── verify_filtering_results.py   # フィルタリング結果確認
│   ├── test_painting_collection.py  # テスト収集
│   ├── check_status.py             # ステータスチェック
│   └── analyze_csv_for_paintings.py # CSV分析
├── data/                  # データファイル
│   ├── raw_data/          # 生データ
│   │   ├── MetObjects.csv          # MetObjects全データ（484,956件）
│   │   ├── paintings_metadata.csv  # 絵画メタデータ
│   │   └── checkpoint.json         # データ収集チェックポイント
│   └── filtered_data/     # フィルタリング済みデータ
│       ├── paintings_complete_dataset.csv  # 完全データセット
│       ├── paintings_metadata.csv          # メタデータ
│       ├── dataset_from_images.csv         # 画像から作成したデータセット
│       └── paintings_images/              # ダウンロード済み画像
├── notebooks/             # Jupyterノートブック
│   ├── eda_paintings_complete_dataset.ipynb # EDAノートブック
│   └── prepare_data_for_shap_class.ipynb # shap-class用データ準備ノートブック
├── docs/                  # ドキュメントとレポート
│   ├── cursor_chat_history.md              # 開発履歴
│   ├── implementation_status_report.md      # 実装状況レポート
│   ├── project_status_20251029.md         # プロジェクト状況
│   ├── requirements_specification.md       # 要件定義
│   ├── painting_analysis_results.txt      # CSV分析結果
│   └── european_paintings_filter_report.txt # フィルタリングレポート
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

### データ収集スクリプト

```bash
# 絵画データのフィルタリング
python scripts/filter_paintings_only.py

# 全件データ収集（9,005件）
python scripts/collect_all_paintings.py

# 画像ダウンロード
python scripts/download_painting_images.py

# テスト収集（200件）
python scripts/test_painting_collection.py

# ステータスチェック
python scripts/check_status.py --type all
python scripts/check_status.py --type collection  # データ収集のみ
python scripts/check_status.py --type download    # ダウンロードのみ
python scripts/check_status.py --type test        # テスト結果のみ

# CSV分析
python scripts/analyze_csv_for_paintings.py

# European Paintingsフィルタリング
python scripts/filter_european_paintings.py

# フィルタリング結果確認
python scripts/verify_filtering_results.py

# 画像からデータセット作成
# paintings_imagesディレクトリの画像ファイル名（Object_ID）から
# MetObjects.csvから対応するデータを抽出してデータセットを作成
python scripts/create_dataset_from_images.py
```

### Jupyterノートブック

```bash
# Jupyter Notebookの起動
jupyter notebook notebooks/

# または JupyterLabを使用
jupyter lab notebooks/
```

#### 利用可能なノートブック

- **`notebooks/eda_paintings_complete_dataset.ipynb`**: 探索的データ分析（EDA）
  - データセットの概要分析
  - 文化圏、年代、メディア、アーティストの分析
  - JSONデータ（タグ、constituents、測定値）の解析
  - 画像データの分析
  - 各種可視化と統計分析

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

1. **データ収集**: 
   - Metropolitan Museum APIから作品データと画像を取得
   - または `MetObjects.csv`からデータを抽出
2. **データ前処理**: 
   - 絵画データのフィルタリング
   - 画像ファイル名からデータセット作成
3. **探索的データ分析（EDA）**: 
   - Jupyter Notebookでデータの概要を把握
   - 統計分析と可視化
4. **特徴量抽出**: 画像から色彩・構図特徴量を抽出
5. **モデル訓練**: ランダムフォレスト分類器を訓練
6. **SHAP説明**: モデルの判断根拠を可視化
7. **結果可視化**: 包括的なレポートを生成

### データセット作成フロー

```bash
# 1. 画像ファイル名からデータセットを作成
python scripts/create_dataset_from_images.py

# 2. 作成されたデータセットでEDAを実行
jupyter notebook notebooks/eda_paintings_complete_dataset.ipynb
```

## 出力ファイル

プロジェクト実行後、以下の構造でファイルが生成されます：

```
data/
├── raw_data/                    # 生データ
│   ├── MetObjects.csv           # MetObjects全データ
│   ├── paintings_metadata.csv  # 作品メタデータ
│   ├── checkpoint.json          # データ収集チェックポイント
│   └── api_data.json           # API取得データ（JSON形式）
├── filtered_data/               # フィルタリング済みデータ
│   ├── paintings_complete_dataset.csv  # 完全データセット
│   ├── paintings_metadata.csv         # メタデータ
│   ├── dataset_from_images.csv        # 画像から作成したデータセット
│   └── paintings_images/              # ダウンロード済み画像
│       └── *.jpg                       # 画像ファイル（Object_ID.jpg形式）
├── features/                    # 特徴量データ（タイムスタンプ付きディレクトリ）
│   └── color_features.pkl       # 抽出された特徴量
├── models/                      # 機械学習モデル（タイムスタンプ付きディレクトリ）
│   ├── random_forest_model.pkl  # 訓練済みモデル
│   └── scaler.pkl              # 特徴量スケーラー
├── results/                     # 訓練結果（タイムスタンプ付きディレクトリ）
│   └── training_results.txt     # 詳細な訓練結果
├── visualizations/              # 可視化結果（タイムスタンプ付きディレクトリ）
│   ├── confusion_matrix.png     # 混同行列
│   ├── feature_importance.png   # 特徴量重要度
│   ├── shap_summary_plot.png    # SHAP summary plot
│   ├── shap_feature_importance.png # SHAP特徴量重要度
│   └── dependence_plot_*.png    # 依存関係プロット
└── shap_explanations/           # SHAP説明データ（タイムスタンプ付きディレクトリ）
    ├── shap_values.csv          # SHAP値データ
    └── feature_importance_shap.csv # SHAP特徴量重要度

docs/                            # ドキュメントとレポート
├── painting_analysis_results.txt      # CSV分析結果
└── european_paintings_filter_report.txt # フィルタリングレポート

logs/                            # ログファイル
├── project.log                  # メインログ
├── collect_all_paintings.log    # データ収集ログ
├── download_images.log          # 画像ダウンロードログ
└── *.log                        # その他のスクリプトログ
```

## 設定オプション

`config.yaml`で以下の設定を調整できます：

- API設定（レート制限、タイムアウト）
- データ設定（最小サンプル数、出力ディレクトリ、タイムスタンプ機能）
- 画像処理設定（リサイズサイズ、色空間）
- 特徴量設定（抽出する特徴量の種類）
- 分類設定（テストサイズ、交差検証）
- SHAP設定（サンプル数、背景データ）

### タイムスタンプ / 最新ディレクトリ機能

- `data.use_timestamp: false`（デフォルト）では、各ステージの成果物が `data/<stage>/latest/` に常に上書きされます。Git からは `latest` 以下だけを追跡すれば「常に最新」を共有できます。
- `data.use_timestamp: true` を指定すると、従来通り `data/analysis_YYMMDDHHMM/` に成果物を時系列保存します。履歴を残したい検証時向けです。

生データ（`data/raw_data/`, `data/raw_images/`）はどちらの設定でも固定ディレクトリに保存されます。最新版を凍結したい場合は `TimestampManager.snapshot_stage('<stage>')` を呼び出し、`snapshot_<timestamp>` ディレクトリを作成できます。

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

## プロジェクトの最新状況

### 実装状況（2025年11月7日更新）

- **全体完了度**: 約98%
- **データ収集システム**: 完全実装済み
- **EDA環境**: Jupyter Notebook環境構築済み
- **shap-class統合**: データ準備ノートブック実装済み
- **Jupyterカーネル**: `explainable-art-classification`カーネルセットアップ済み
- **コード整理**: リファクタリング完了
- **ファイル整理**: ディレクトリ構造の最適化完了

### 主要な更新内容

#### データ収集と整理
- **ハイブリッドデータ収集システム**: CSVとAPIを組み合わせた効率的なデータ収集
- **画像からデータセット作成**: `create_dataset_from_images.py`で画像ファイル名からデータセットを自動生成
- **データセット**: `dataset_from_images.csv`（5,396行）を作成可能

#### 探索的データ分析（EDA）
- **Jupyter Notebook**: 包括的なEDAノートブックを作成
  - 文化圏、年代、メディア、アーティストの分析
  - JSONデータ（タグ、constituents、測定値）の解析
  - 各種可視化と統計分析
  - 共通ユーティリティ関数でコード再利用性向上

#### shap-class統合
- **データ準備ノートブック**: `notebooks/prepare_data_for_shap_class.ipynb`を作成
  - WikiArt_VLMデータセットの読み込みと前処理
  - train/validation分割（70/30、stratifyを使用）
  - shap-classリポジトリのセットアップとモデル訓練
  - Random Forestモデルの訓練と評価
  - SHAP説明可能性の可視化
- **Jupyterカーネル**: `explainable-art-classification`カーネルをセットアップ
  - プロジェクトの仮想環境（`.venv`）を使用
  - Python 3.12.11環境

#### コードとファイルの整理
- **スクリプトの整理**: 全ての実行スクリプトを`scripts/`ディレクトリに統一
- **ドキュメントの整理**: 開発ドキュメントを`docs/`ディレクトリに集約
- **ログの整理**: 全てのログファイルを`logs/`ディレクトリに集約
- **共通機能の抽出**: ログ設定、ステータスチェックなどの共通機能を`src/utils/`に統一

### 実装済み機能（✓）

- **Metropolitan Museum API統合**: 59項目の全メタデータ取得
- **ハイブリッドデータ収集システム**: CSVとAPIを組み合わせた大規模データ収集
  - チェックポイント機能（中断・再開可能）
  - 品質管理機能（QAレポート生成）
  - エラーハンドリングとリトライ機能
- **画像からデータセット作成**: 画像ファイル名から自動的にデータセットを生成
- **探索的データ分析（EDA）**: 包括的なJupyter Notebook環境
  - 共通ユーティリティ関数によるコード再利用
  - JSONデータの自動解析
  - 各種可視化と統計分析
- **shap-class統合**: データ準備ノートブックとJupyterカーネルセットアップ
  - WikiArt_VLMデータセットの前処理
  - train/validation分割とデータ保存
  - shap-classリポジトリとの統合
  - Random Forestモデルの訓練と評価
- **色彩特徴量抽出**: 完全実装
- **ランダムフォレスト分類器**: 実装済み
- **SHAP説明可能性**: summary_plot実装済み
- **モジュラー設計と設定ファイル**: 完全実装
- **効率的なデータ収集システム**: メタデータと画像の分離
- **プロジェクト構造の最適化**: 
  - スクリプトの整理（`scripts/`ディレクトリ）
  - ドキュメントの整理（`docs/`ディレクトリ）
  - ログの整理（`logs/`ディレクトリ）
  - 共通機能の抽出（`src/utils/`）

### 部分実装機能（△）

- **構図特徴量**: 基本実装済み、詳細は未実装
- **データ量**: 小規模データでの動作確認のみ
- **性能要件**: 大規模データでの性能測定は未実施

### 未実装機能（×）

- waterfall plot/force plot（複雑性のため簡略化）
- 誤分類作品の分析表示
- 多クラス分類（バロック、印象派、キュビズム等）
- 特徴量分布の可視化

### 今後の改善点

1. **構図特徴量の完全実装**: より詳細な分析機能の追加
2. **可視化機能の拡充**: より直感的な説明可能性の向上
3. **多クラス分類の実装**: バロック、印象派、キュビズムなどの様式分類
4. **特徴量分布の可視化**: より詳細なデータ分析
5. **高解像度画像の検討**: 現在`primaryImageSmall`を使用、必要に応じて高解像度画像の取得を検討

### データセットについて

プロジェクトでは以下のデータセットが利用可能です：

1. **`paintings_complete_dataset.csv`**: 
   - APIから取得した完全な絵画データセット
   - JSON形式の詳細情報（tags, constituents, measurements）を含む

2. **`dataset_from_images.csv`**: 
   - `paintings_images`ディレクトリの画像ファイル名から自動生成
   - MetObjects.csvから対応するデータを抽出
   - ダウンロード済み画像に対応するデータのみを含む

3. **`MetObjects.csv`**: 
   - Metropolitan Museumの全作品データ（484,956件）
   - 大容量ファイルのため、`.gitignore`に含まれています

### 要件定義からの変更点

- **画像解像度**: 高解像度JPEG → `primaryImageSmall`（効率性のため）
- **SHAP可視化**: waterfall plot/force plot → summary_plot（簡略化）
- **データ収集戦略**: 単一API → ハイブリッド（CSV+API）システム
- **プロジェクト構造**: フラット構造 → 整理されたディレクトリ構造

## 参考資料

- [Metropolitan Museum API Documentation](https://metmuseum.github.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)

## ドキュメント

プロジェクトの詳細な情報は`docs/`ディレクトリを参照してください：

- [実装状況詳細レポート](docs/implementation_status_report.md)
- [プロジェクト状況](docs/project_status_20251029.md)
- [要件定義](docs/requirements_specification.md)
- [開発履歴](docs/cursor_chat_history.md)
- [CSV分析結果](docs/painting_analysis_results.txt)

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

データセットが見つからない場合は、以下のスクリプトを実行してください：

```bash
# 画像からデータセットを作成
python scripts/create_dataset_from_images.py

# フィルタリング結果を確認
python scripts/verify_filtering_results.py
```

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
