# データリーク修正の完了と次のステップ

## 修正内容の確認

### ✅ 修正完了項目

1. **`src/feature_extraction/color_extractor.py`**:
   - `extract_features_from_wikiart_vlm()`: スケーリング処理を削除（生の特徴量を保存）
   - `extract_features_with_gestalt_scores()`: スケーリング処理を削除（生の特徴量を保存）
   - 変数名の修正: `features_df_scaled` → `features_df`

2. **`src/model_training/random_forest_trainer.py`**:
   - データ分割後に、訓練データのみで`StandardScaler`をfit
   - 訓練データでfitした`StandardScaler`でテストデータをtransform
   - スケーラーを`models/scaler.pkl`として保存

3. **`src/explainability/shap_explainer.py`**:
   - 保存されたスケーラーを読み込んで使用
   - スケーラーが見つからない場合は警告を出力

## 古いスケーラーファイルのクリーンアップ

以下の古いスケーラーファイルが`features`ディレクトリに残っています：

```
data/analysis_2511052310/features/wikiart_vlm_scaler_with_gestalt_Stable-Diffusion.pkl
data/analysis_2511052310/features/wikiart_vlm_scaler_Stable-Diffusion.pkl
data/analysis_2511032213/features/wikiart_vlm_scaler_Stable-Diffusion.pkl
data/analysis_2511052259/features/wikiart_vlm_scaler_Stable-Diffusion.pkl
data/analysis_2511050614/features/wikiart_vlm_scaler_Stable-Diffusion.pkl
data/analysis_2511050822/features/wikiart_vlm_scaler_with_gestalt_Stable-Diffusion.pkl
data/analysis_2511050822/features/wikiart_vlm_scaler_Stable-Diffusion.pkl
data/analysis_2511052300/features/wikiart_vlm_scaler_with_gestalt_Stable-Diffusion.pkl
data/analysis_2511052300/features/wikiart_vlm_scaler_Stable-Diffusion.pkl
data/analysis_2511031928/features/wikiart_vlm_scaler_with_gestalt_Stable-Diffusion.pkl
data/analysis_2511031928/features/wikiart_vlm_scaler_Stable-Diffusion.pkl
```

**注意**: これらのファイルは古い実装で生成されたもので、データリークを含んでいます。新しい実装では、スケーラーは`models/scaler.pkl`として保存されます。

## 次のステップ

### 1. 特徴量抽出の再実行（生の特徴量を保存）

既存のスケーリング済み特徴量ファイルを削除または無視し、生の特徴量を再抽出する必要があります。

```bash
# Phase2: 特徴量抽出（生の特徴量を保存）
python scripts/train_wikiart_vlm.py --mode features --generation-model Stable-Diffusion
```

または、`extract_features_with_gestalt_scores()`を使用：

```bash
python scripts/train_wikiart_vlm.py --mode all --include-gestalt --generation-model Stable-Diffusion
```

### 2. モデルの再訓練（新しいスケーラーを生成）

生の特徴量を使用してモデルを再訓練し、訓練データのみでfitしたスケーラーを生成します。

```bash
# Phase3: モデル訓練（新しいスケーラーを生成）
python scripts/train_wikiart_vlm.py --mode train --generation-model Stable-Diffusion
```

または、全フェーズを実行：

```bash
python scripts/train_wikiart_vlm.py --mode all --include-gestalt --generation-model Stable-Diffusion
```

### 3. 古いスケーラーファイルのクリーンアップ（オプション）

混乱を避けるため、古いスケーラーファイルを削除することを推奨します：

```bash
# 古いスケーラーファイルを削除（注意: バックアップを取ってから実行）
find data -name "*scaler*.pkl" -path "*/features/*" -type f -delete
```

**注意**: このコマンドは`features`ディレクトリ内のスケーラーファイルのみを削除します。`models`ディレクトリ内の新しいスケーラーファイルは保持されます。

## 検証方法

### 1. 特徴量ファイルの確認

生の特徴量ファイルがスケーリングされていないことを確認：

```python
import pandas as pd
import numpy as np

# 特徴量ファイルを読み込み
df = pd.read_csv('data/analysis_XXXXXX/features/wikiart_vlm_features_with_gestalt_Stable-Diffusion.csv')

# 数値特徴量を抽出
numeric_cols = df.select_dtypes(include=[np.number]).columns
feature_cols = [col for col in numeric_cols if col not in ['image_id', 'label']]

# 特徴量の統計を確認（スケーリングされていない値の範囲を確認）
print(df[feature_cols].describe())
```

### 2. スケーラーファイルの確認

新しいスケーラーファイルが`models`ディレクトリに保存されていることを確認：

```bash
ls -la data/analysis_XXXXXX/models/scaler.pkl
```

### 3. モデル性能の確認

データリーク修正後、モデルの性能が適切に評価されることを確認：

- 訓練精度とテスト精度の差が適切であること
- 過学習が発生していないこと
- SHAP値が正しく計算されること

## 注意事項

1. **既存の分析結果**: 既存の分析結果（`analysis_*`ディレクトリ）は、データリークを含んでいる可能性があります。新しい分析を実行する際は、新しいタイムスタンプで実行してください。

2. **後方互換性**: 既存のスケーリング済み特徴量ファイルを使用しているコードがある場合は、生の特徴量を使用するように修正する必要があります。

3. **SHAP説明**: SHAP説明を生成する際は、保存されたスケーラーを使用する必要があります。スケーラーが見つからない場合は警告が出力されます。

## まとめ

データリークの問題は修正されました。次のステップとして：

1. ✅ コード修正完了
2. ⏳ 特徴量抽出の再実行（生の特徴量を保存）
3. ⏳ モデルの再訓練（新しいスケーラーを生成）
4. ⏳ 古いスケーラーファイルのクリーンアップ（オプション）
5. ⏳ 検証（特徴量ファイル、スケーラーファイル、モデル性能の確認）

