# タイムスタンプ管理システムの説明

## 概要

`analysis_`フォルダは、`TimestampManager`が初期化されるたびに、**その時刻（分単位）**に基づいて自動的に生成されます。

## 生成タイミング

### 1. TimestampManagerの初期化時

```python
# src/utils/timestamp_manager.py (30-32行目)
if self.use_timestamp:
    self.timestamp = datetime.now().strftime("%y%m%d%H%M")  # 例: 2511031906
    self.output_dir = self.base_output_dir / f"analysis_{self.timestamp}"
```

**タイムスタンプ形式**: `yymmddHHMM`（年/月/日/時/分）
- 例: `2511031906` = 2025年11月3日 19時6分

### 2. 各クラスの初期化時に自動生成

以下のクラスが初期化されると、その時点で`TimestampManager`が作成され、新しい`analysis_`フォルダが生成されます：

#### 特徴量抽出時
- **`ColorFeatureExtractor`**: 特徴量抽出クラスの初期化時
  ```python
  # src/feature_extraction/color_extractor.py (44行目)
  self.timestamp_manager = TimestampManager(config)
  self.timestamp_manager.create_directories()  # ディレクトリ作成
  ```

#### ゲシュタルト原則スコアリング時
- **`GestaltScorer`**: ゲシュタルト原則スコアリングクラスの初期化時
  ```python
  # src/feature_extraction/gestalt_scorer.py (75行目)
  self.timestamp_manager = TimestampManager(config)
  self.timestamp_manager.create_directories()  # ディレクトリ作成
  ```

#### モデル訓練時
- **`RandomForestTrainer`**: モデル訓練クラスの初期化時
- **`SHAPExplainer`**: SHAP説明クラスの初期化時

## 実際の生成パターン

実行ログから見ると、以下のように生成されています：

```
analysis_2511031830  # 18:30 - Phase 3テスト（10サンプル）
analysis_2511031831  # 18:31 - Phase 3テスト（10サンプル、SHAP説明）
analysis_2511031850  # 18:50 - Phase 4テスト（10サンプル、ゲシュタルトスコアリング）
analysis_2511031851  # 18:51 - Phase 4テスト（100サンプル、色彩特徴量抽出）
analysis_2511031852  # 18:52 - Phase 4テスト（100サンプル、ゲシュタルトスコアリング）
analysis_2511031906  # 19:06 - Phase 4テスト（モデル訓練）
```

## 問題点と解決策

### 問題点

異なるクラスが初期化されるたびに、**異なるタイムスタンプ**で新しいフォルダが生成されるため：

1. **`ColorFeatureExtractor`**が18:51に初期化 → `analysis_2511031851`生成
2. **`GestaltScorer`**が18:52に初期化 → `analysis_2511031852`生成
3. 結果として、特徴量ファイルとゲシュタルトスコアファイルが**異なるフォルダ**に保存される

### 解決策

#### オプション1: use_timestampをfalseに設定

`config.yaml`で設定を変更：

```yaml
data:
  use_timestamp: false  # 固定ディレクトリを使用
```

**メリット**:
- 常に同じディレクトリに保存される
- ファイルの関連付けが簡単

**デメリット**:
- 実行履歴が上書きされる
- 過去の結果を保持できない

#### オプション2: 実行単位でタイムスタンプを統一 ✅ 実装済み

スクリプトの最初で`TimestampManager`を一度だけ作成し、それを各クラスに渡すように修正しました。

**実装内容**:
- `TimestampManager`に`timestamp`引数を追加（既存のタイムスタンプを指定可能）
- 各クラス（`ColorFeatureExtractor`, `GestaltScorer`, `RandomForestTrainer`, `SHAPExplainer`）に`timestamp_manager`引数を追加
- スクリプト（`train_wikiart_vlm.py`, `compare_generation_models.py`）で最初に`TimestampManager`を作成し、各クラスに渡す

**効果**:
- 同一実行内で同じタイムスタンプが使用される
- 関連ファイルが同じフォルダに保存される
- 段階的実行でも正しいファイルを参照できる

#### オプション3: 最新のフォルダを自動検出（以前の実装）

以前は`merge_gestalt_scores`メソッドで最新のゲシュタルトスコアファイルを検出していましたが、タイムスタンプ統一により不要になりました。

## 推奨設定

### 開発・テスト時
```yaml
use_timestamp: true  # 実行履歴を保持
```

### 本番運用時
```yaml
use_timestamp: false  # 固定ディレクトリを使用
```

## 現在の状態

- **`use_timestamp: true`** が設定されているため、実行のたびに新しいフォルダが生成されます
- これは**意図された動作**で、実行履歴を保持するための機能です
- 不要なフォルダは手動で削除できます

## フォルダの整理方法

```bash
# 古いフォルダを確認
ls -ld data/analysis_*

# 特定の日付以前のフォルダを削除（例: 1週間以上古い）
find data -type d -name "analysis_*" -mtime +7 -exec rm -rf {} \;
```

