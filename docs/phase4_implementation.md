# Phase 4: LLMを用いたゲシュタルト原則スコアリング - 実装ガイド

## 概要

Phase 4では、Ollamaを使用したローカルVLM（Vision Language Model）を活用して、画像のゲシュタルト原則を評価する機能を実装しました。

## 推奨VLMモデル

### PCスペック
- **モデル**: MacBook Pro (Mac16,5)
- **CPU**: 16コア（12パフォーマンス + 4効率）
- **メモリ**: 128GB RAM

### 推奨モデル: `llava:13b` または `llava:latest` (7B)

詳細は `docs/vlm_model_recommendations.md` を参照してください。

## インストール手順

### 1. Ollamaのインストール

Ollamaが既にインストールされていることを確認：

```bash
which ollama
```

未インストールの場合は、[公式サイト](https://ollama.org/)からインストールしてください。

### 2. VLMモデルのダウンロード

```bash
# 最推奨モデル（13Bモデル）
ollama pull llava:13b

# または、標準的な7Bモデル（LLaVA 1.6バージョン）
ollama pull llava:latest
# または
ollama pull llava:7b

# 高精度オプション（34Bモデル）
ollama pull llava:34b
```

## 実装内容

### 1. ゲシュタルト原則スコアリングクラス

**ファイル**: `src/feature_extraction/gestalt_scorer.py`

主な機能：
- Ollama APIを使用した画像のゲシュタルト原則評価
- 6つのゲシュタルト原則のスコアリング（1-5スケール）
  - Simplicity (簡潔性)
  - Proximity (近接性)
  - Similarity (類同)
  - Continuity (連続性)
  - Closure (閉合性)
  - Figure-Ground Separation (図地分離)
- バッチ処理対応
- JSON形式の応答パース
- エラーハンドリングとリトライ機能

### 2. ゲシュタルト原則評価用プロンプト

arXiv 2504.12511論文のアプローチを参考に、英語でプロンプトを設計：

- 明確なタスク定義
- 各ゲシュタルト原則の詳細な説明
- 構造化されたJSON出力フォーマット
- スコアリング基準の明示

### 3. 特徴量統合機能

**ファイル**: `src/feature_extraction/color_extractor.py`

新規メソッド：
- `merge_gestalt_scores()`: 色彩特徴量とゲシュタルト原則スコアを統合
- `extract_features_with_gestalt_scores()`: 特徴量抽出とゲシュタルト原則スコアリングを統合実行

### 4. 実行スクリプト

**ファイル**: `scripts/score_gestalt_principles.py`

使用方法：

```bash
# 単一画像のスコアリング
python scripts/score_gestalt_principles.py --mode single --image path/to/image.jpg

# 複数画像のバッチスコアリング
python scripts/score_gestalt_principles.py --mode batch --image-dir path/to/images/

# WikiArt_VLMデータセットのスコアリング
python scripts/score_gestalt_principles.py --mode wikiart --generation-model Stable-Diffusion --max-samples 100

# カスタムモデルを使用
python scripts/score_gestalt_principles.py --mode wikiart --model llava:13b
```

### 5. 統合訓練スクリプトの拡張

**ファイル**: `scripts/train_wikiart_vlm.py`

新規オプション：
- `--include-gestalt`: ゲシュタルト原則スコアを含めて特徴量抽出
- `--gestalt-model`: ゲシュタルトスコアリング用のOllamaモデル名を指定

使用方法：

```bash
# ゲシュタルト原則スコアを含めた特徴量抽出と訓練
python scripts/train_wikiart_vlm.py --mode all --include-gestalt --generation-model Stable-Diffusion

# カスタムモデルを使用
python scripts/train_wikiart_vlm.py --mode all --include-gestalt --gestalt-model llava:13b
```

## 設定ファイル

**ファイル**: `config.yaml`

新規セクション：

```yaml
# ゲシュタルト原則スコアリング設定
gestalt_scoring:
  model_name: "llava:13b"  # Ollamaモデル名 (llava:latest, llava:7b, llava:13b, llava:34b)
  ollama_base_url: "http://localhost:11434"
  score_scale: [1, 5]  # スコアリングスケール
  max_retries: 3  # 最大リトライ回数
  retry_delay: 1.0  # リトライ間隔（秒）
  batch_size: 10  # チェックポイント保存間隔
```

## 出力ファイル

### ゲシュタルト原則スコアファイル

**形式**: CSV

**列**:
- `image_path`: 画像パス
- `image_name`: 画像名
- `simplicity_score`: 簡潔性スコア（1-5）
- `simplicity_explanation`: 簡潔性の説明
- `proximity_score`: 近接性スコア（1-5）
- `proximity_explanation`: 近接性の説明
- ... (他の原則も同様)

### 統合特徴量ファイル

**ファイル名**: `wikiart_vlm_features_with_gestalt_{generation_model}.csv`

色彩特徴量とゲシュタルト原則スコアが統合された特徴量ファイル。分類モデルの訓練に使用可能。

## 使用例

### 例1: 単一画像の評価

```bash
python scripts/score_gestalt_principles.py \
    --mode single \
    --image data/WikiArt_VLM-main/images/Original/0.jpg \
    --model llava:13b
```

### 例2: WikiArt_VLMデータセットの評価（小規模）

```bash
python scripts/score_gestalt_principles.py \
    --mode wikiart \
    --generation-model Stable-Diffusion \
    --max-samples 10 \
    --model llava:13b
```

### 例3: ゲシュタルト原則スコアを含めた分類モデル訓練

```bash
# 特徴量抽出（ゲシュタルト原則スコア含む）
python scripts/train_wikiart_vlm.py \
    --mode extract \
    --generation-model Stable-Diffusion \
    --include-gestalt \
    --max-samples 100

# モデル訓練（統合特徴量を使用）
python scripts/train_wikiart_vlm.py \
    --mode train \
    --generation-model Stable-Diffusion
```

## 注意事項

1. **Ollamaサーバーの起動**: Ollamaが起動していることを確認してください
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **メモリ使用量**: `llava:13b`は約16-20GBのメモリを使用します。128GB RAMがあれば問題ありません。`llava:latest`（7B）は約10-12GBです。

3. **処理時間**: 画像1枚あたり約5-10秒かかります。大量の画像を処理する場合は、`--max-samples`オプションでテストを推奨。

4. **エラーハンドリング**: API呼び出しエラー時は自動的にリトライしますが、Ollamaサーバーが応答しない場合は手動で確認してください。

## 次のステップ

- Phase 5: 分析と可視化の拡充
- ゲシュタルト原則スコアの可視化機能の追加
- 本物 vs フェイク分類におけるゲシュタルト原則スコアの重要性分析

