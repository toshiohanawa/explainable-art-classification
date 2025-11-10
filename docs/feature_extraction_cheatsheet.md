# 説明変数（特徴量）抽出処理の詳細チートシート

質疑応答で適切に回答できるよう、特徴量抽出の具体的な処理手順をまとめました。

---

## 目次

1. [画像前処理](#1-画像前処理)
2. [色彩特徴量の抽出](#2-色彩特徴量の抽出)
3. [テクスチャ特徴量の抽出](#3-テクスチャ特徴量の抽出)
4. [エッジ特徴量の抽出](#4-エッジ特徴量の抽出)
5. [ゲシュタルト原則スコアの評価](#5-ゲシュタルト原則スコアの評価)
6. [データ前処理（標準化・分割）](#6-データ前処理標準化分割)
7. [よくある質問と回答](#7-よくある質問と回答)

---

## 1. 画像前処理

### 処理手順

1. **画像の読み込み**
   - OpenCV (`cv2.imread`)を使用
   - BGR形式で読み込まれる

2. **色空間の変換**
   - BGR → RGBに変換 (`cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`)
   - 理由: 標準的なRGB形式に統一

3. **リサイズ**
   - ターゲットサイズ: `(512, 512)`（設定ファイルで指定）
   - 補間方法: `cv2.INTER_LINEAR`（双線形補間）
   - 理由: 計算効率と特徴量の一貫性を確保

### パラメータ

- **リサイズサイズ**: 512×512ピクセル
- **補間方法**: 双線形補間（INTER_LINEAR）

---

## 2. 色彩特徴量の抽出

### 2.1 HSV色空間への変換

**処理**:
```python
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv)
```

**HSV色空間の範囲**:
- **H (色相)**: 0-179（OpenCVの範囲）
- **S (彩度)**: 0-255
- **V (明度)**: 0-255

### 2.2 HSV基本統計量（6カラム）

**抽出方法**:
- `mean_hue`: `np.mean(h)` - 色相の平均値
- `mean_saturation`: `np.mean(s)` - 彩度の平均値
- `mean_value`: `np.mean(v)` - 明度の平均値
- `hue_std`: `np.std(h)` - 色相の標準偏差
- `saturation_std`: `np.std(s)` - 彩度の標準偏差
- `value_std`: `np.std(v)` - 明度の標準偏差

**色の多様性**:
- `color_diversity`: `np.var(h)` - 色相の分散

### 2.3 支配色分析（9カラム）

**処理手順**:

1. **K-meansクラスタリング**
   - クラスタ数: `k=5`
   - 初期化: `random_state=42`（再現性のため）
   - 実行回数: `n_init=10`
   - 入力: 画像を1次元に変換した画素配列 `pixels = image.reshape(-1, 3)`

2. **支配色の抽出**
   - 各クラスタの中心色（RGB値）を取得
   - 各クラスタのサイズ（支配度）を計算

3. **HSV統計量の計算**
   - RGB値をHSVに変換
   - 5つの支配色のHSV統計量を計算:
     - `dominant_color_hue_mean`: 色相の平均
     - `dominant_color_hue_std`: 色相の標準偏差
     - `dominant_color_saturation_mean`: 彩度の平均
     - `dominant_color_saturation_std`: 彩度の標準偏差
     - `dominant_color_value_mean`: 明度の平均
     - `dominant_color_value_std`: 明度の標準偏差

4. **占有率の統計量**
   - `dominant_color_ratio_max`: 最大占有率 `np.max(ratios)`
   - `dominant_color_ratio_mean`: 平均占有率 `np.mean(ratios)`
   - `dominant_color_ratio_std`: 占有率の標準偏差 `np.std(ratios)`
   - ここで `ratios = cluster_sizes / len(pixels)`

**なぜRGB値ではなくHSV統計量を使用するか？**
- 解釈性が高い（色相、彩度、明度は直感的に理解しやすい）
- 特徴量の次元数を削減（RGB値だと15カラム、HSV統計量だと9カラム）

### 2.4 明度分布の特徴量（3カラム）

**処理手順**:

1. **ヒストグラムの計算**
   - ビン数: `10`
   - 範囲: `(0, 256)`
   - `hist, bins = np.histogram(v_channel, bins=10, range=(0, 256))`

2. **歪度（Skewness）の計算**
   ```python
   mean = np.mean(v_channel)
   std = np.std(v_channel)
   skewness = np.mean(((v_channel - mean) / std) ** 3)
   ```
   - `brightness_skewness`: 明度分布の非対称性
   - 正の値: 右に歪む（暗い画素が多い）
   - 負の値: 左に歪む（明るい画素が多い）

3. **尖度（Kurtosis）の計算**
   ```python
   kurtosis = np.mean(((v_channel - mean) / std) ** 4) - 3
   ```
   - `brightness_kurtosis`: 明度分布の尖り具合
   - 正の値: 尖った分布（極端な値が多い）
   - 負の値: 平坦な分布（極端な値が少ない）

4. **エントロピーの計算**
   ```python
   hist_norm = hist / np.sum(hist)
   hist_norm = hist_norm[hist_norm > 0]  # 0を除外
   entropy = -np.sum(hist_norm * np.log2(hist_norm))
   ```
   - `brightness_entropy`: 明度分布の情報量
   - 値が大きい: 分布が均等（多様な明度が含まれる）
   - 値が小さい: 分布が偏っている（特定の明度に集中）

**注意**: `brightness_mean`と`brightness_std`は使用しない（`mean_value`と`value_std`と重複するため）

---

## 3. テクスチャ特徴量の抽出

### 3.1 グレースケール変換

**処理**:
```python
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```

### 3.2 GLCM特徴量（5カラム）

**GLCM（Gray-Level Co-occurrence Matrix）とは？**
- グレースケール画像の画素間の共起関係を分析
- 画素ペアの出現頻度を表す行列

**処理手順**:

1. **GLCMの計算**
   ```python
   distances = [1]  # 距離=1
   angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0, 45, 90, 135度
   glcm = graycomatrix(gray, distances=distances, angles=angles, 
                       levels=256, symmetric=True, normed=True)
   ```
   - `levels=256`: グレーレベル数
   - `symmetric=True`: 対称行列にする
   - `normed=True`: 正規化する

2. **特徴量の計算（全角度の平均）**
   - `texture_glcm_contrast`: `graycoprops(glcm, 'contrast').mean()`
     - 局所的な明度の変化（値が大きい=テクスチャが粗い）
   - `texture_glcm_dissimilarity`: `graycoprops(glcm, 'dissimilarity').mean()`
     - 画素間の明度の差（値が大きい=テクスチャが不均一）
   - `texture_glcm_homogeneity`: `graycoprops(glcm, 'homogeneity').mean()`
     - 画素間の類似度（値が大きい=テクスチャが均一）
   - `texture_glcm_energy`: `graycoprops(glcm, 'energy').mean()`
     - 規則性（値が大きい=繰り返しパターンが多い）
   - `texture_glcm_correlation`: `graycoprops(glcm, 'correlation').mean()`
     - 画素間の線形関係（値が大きい=テクスチャが規則的）

**なぜ全角度の平均を取るか？**
- 方向に依存しない特徴量を抽出するため
- 4方向（0, 45, 90, 135度）の平均を計算

### 3.3 LBP特徴量（5カラム）

**LBP（Local Binary Pattern）とは？**
- 局所的なテクスチャパターンを分析
- 各画素とその近傍画素の明度関係を符号化

**処理手順**:

1. **LBPの計算**
   ```python
   radius = 1
   n_points = 8 * radius  # 8
   lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
   ```
   - `radius=1`: 半径1ピクセル
   - `n_points=8`: 近傍点数8
   - `method='uniform'`: uniformパターンを使用

2. **基本統計量**
   - `texture_lbp_mean`: `np.mean(lbp)` - LBP値の平均
   - `texture_lbp_std`: `np.std(lbp)` - LBP値の標準偏差

3. **LBPヒストグラムの統計量**
   ```python
   hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, n_points + 2))
   hist = hist.astype(float)
   hist /= (hist.sum() + 1e-7)  # 正規化
   ```
   - ビン数: `10`
   - 範囲: `(0, n_points + 2)` = `(0, 10)`

4. **ヒストグラムの統計量を計算**
   - `texture_lbp_histogram_entropy`: `-np.sum(hist_nonzero * np.log2(hist_nonzero))`
     - テクスチャパターンの多様性
   - `texture_lbp_histogram_skewness`: `np.mean((hist_centered / hist_std) ** 3)`
     - 分布の非対称性
   - `texture_lbp_histogram_kurtosis`: `np.mean((hist_centered / hist_std) ** 4) - 3`
     - 分布の尖り具合

**なぜLBPヒストグラムのビン特徴量を使用しないか？**
- 解釈性を重視（統計量の方が理解しやすい）
- 特徴量の次元数を削減（ビン特徴量だと10カラム、統計量だと3カラム）

---

## 4. エッジ特徴量の抽出

### 4.1 Cannyエッジ検出

**処理**:
```python
edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)
```
- `threshold1=50`: 低い閾値（弱いエッジ）
- `threshold2=150`: 高い閾値（強いエッジ）
- `apertureSize=3`: Sobel演算子のカーネルサイズ

### 4.2 エッジ密度・カウント（2カラム）

**処理**:
- `edge_density`: `np.sum(edges > 0) / edges.size`
  - エッジ画素の割合（0-1の範囲）
- `edge_count`: `cv2.connectedComponentsWithStats(edges, connectivity=8)`
  - エッジコンポーネントの数（背景を除く）

### 4.3 エッジ長さ（2カラム）

**処理**:
```python
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
component_sizes = stats[1:, cv2.CC_STAT_AREA]  # 背景を除外
```
- `edge_mean_length`: `np.mean(component_sizes)` - 平均長さ（面積）
- `edge_std_length`: `np.std(component_sizes)` - 標準偏差

### 4.4 エッジ方向性（2カラム）

**処理**:
```python
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
angles = lines[:, 0, 1]  # 角度（ラジアン）
```
- `edge_orientation_mean`: `np.mean(angles)` - エッジ方向の平均
- `edge_orientation_std`: `np.std(angles)` - エッジ方向の標準偏差

**Hough変換のパラメータ**:
- `rho=1`: 距離の分解能（ピクセル）
- `theta=np.pi/180`: 角度の分解能（ラジアン）
- `threshold=100`: 投票数の閾値

### 4.5 エッジ滑らかさ・曲率（3カラム）

**処理手順**:

1. **輪郭の検出**
   ```python
   contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   ```

2. **曲率の計算**
   - 各輪郭点で、3点間の角度変化を計算
   - ベクトル間の角度を計算（曲率の近似）
   ```python
   for i in range(1, len(contour_points) - 1):
       p1 = contour_points[i - 1]
       p2 = contour_points[i]
       p3 = contour_points[i + 1]
       v1 = p2 - p1
       v2 = p3 - p2
       # 角度の変化を計算
       angle = np.arccos(np.dot(v1_norm, v2_norm))
   ```

3. **統計量の計算**
   - `edge_smoothness`: `1.0 / (np.mean(curvatures) + 1e-7)` - 滑らかさ（曲率の逆数）
   - `edge_curvature_mean`: `np.mean(curvatures)` - 曲率の平均
   - `edge_curvature_std`: `np.std(curvatures)` - 曲率の標準偏差

---

## 5. ゲシュタルト原則スコアの評価

### 5.1 評価方法

**使用ツール**:
- Ollama（ローカルLLMサーバー）
- LLaVA 7Bモデル（Vision-Language Model）

**処理手順**:

1. **画像のエンコード**
   ```python
   with open(image_path, 'rb') as image_file:
       encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
   ```

2. **プロンプトの作成**
   - arXiv 2504.12511論文のアプローチを参考
   - 6つのゲシュタルト原則の説明を含む
   - JSON形式の出力を要求

3. **LLMへのリクエスト**
   - Ollama APIを使用
   - 画像（base64エンコード）とプロンプトを送信

4. **レスポンスのパース**
   - JSON形式のレスポンスを解析
   - 各原則のスコア（1-5）と説明文を抽出

### 5.2 ゲシュタルト原則の定義

1. **Simplicity（簡潔性）**
   - 最も単純な解釈を選ぶ傾向
   - スコア: 1-5（1=複雑、5=簡潔）

2. **Proximity（近接性）**
   - 近くの要素をグループとして認識
   - スコア: 1-5（1=分散、5=近接）

3. **Similarity（類同）**
   - 類似した要素をグループとして認識
   - スコア: 1-5（1=多様、5=類似）

4. **Continuity（連続性）**
   - 滑らかな線やパスを認識
   - スコア: 1-5（1=断続的、5=連続的）

5. **Closure（閉合性）**
   - 不完全な形状を完全な形状として認識
   - スコア: 1-5（1=開いた形式、5=閉じた形式）

6. **Figure-Ground Separation（図地分離）**
   - 主観と背景を区別
   - スコア: 1-5（1=混在、5=明確な分離）

### 5.3 スコアの保存形式

- **生の値**: 1-5の整数値で保存
- **標準化**: モデル訓練時に訓練データのみでStandardScalerを適用（データリーク防止）
- **説明文**: 各スコアに対して、LLMが生成した説明文も保存

---

## 6. データ前処理（標準化・分割）

### 6.1 データ分割

**処理**:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # クラス分布を保持
)
```

**パラメータ**:
- `test_size=0.2`: テストデータ20%、訓練データ80%
- `random_state=42`: 再現性のため
- `stratify=y`: クラス分布を保持（本物:フェイク = 50:50）

### 6.2 標準化（Standardization）

**処理手順**:

1. **訓練データのみでStandardScalerをfit**
   ```python
   from sklearn.preprocessing import StandardScaler
   
   scaler = StandardScaler()
   scaler.fit(X_train)  # 訓練データのみでfit
   ```

2. **訓練データとテストデータをtransform**
   ```python
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

**なぜ訓練データのみでfitするか？**
- **データリーク防止**: テストデータの情報が訓練に漏れないようにするため
- テストデータの平均・標準偏差は未知のデータとして扱う

**標準化の効果**:
- 平均値: 約0（訓練データの平均で中心化）
- 標準偏差: 約1（訓練データの標準偏差で正規化）
- 異なるスケールの特徴量を同じスケールに統一

**保存**:
- スケーラーを`models/scaler.pkl`として保存
- SHAP説明生成時に使用

### 6.3 特徴量の保存形式

**生の値で保存**:
- 特徴量ファイルには標準化前の生の値を保存
- 理由: データリーク防止、再現性の確保

**標準化はモデル訓練時に適用**:
- 特徴量抽出時には標準化を行わない
- モデル訓練時に訓練データのみでStandardScalerをfit

---

## 7. よくある質問と回答

### Q1: なぜHSV色空間を使用したのですか？

**A**: 
- RGB色空間は直感的でない（色相、彩度、明度が分離されていない）
- HSV色空間は人間の知覚に近い（色相、彩度、明度が独立）
- 解釈性が高い（「赤っぽい」「鮮やか」「明るい」など直感的に理解できる）

### Q2: なぜK-meansクラスタリングでk=5にしたのですか？

**A**: 
- 画像の主要な色を適切に表現するため
- k=3だと少なすぎ、k=10だと多すぎる
- k=5は、多くの画像で主要な色の数を適切に表現できる

### Q3: なぜGLCMで全角度の平均を取るのですか？

**A**: 
- 方向に依存しない特徴量を抽出するため
- 画像の回転に対してロバストな特徴量になる
- 4方向（0, 45, 90, 135度）の平均を計算することで、方向の影響を平均化

### Q4: なぜLBPヒストグラムのビン特徴量を使用しないのですか？

**A**: 
- 解釈性を重視（統計量の方が理解しやすい）
- 特徴量の次元数を削減（ビン特徴量だと10カラム、統計量だと3カラム）
- エントロピー、歪度、尖度で十分にテクスチャの特徴を表現できる

### Q5: なぜCannyエッジ検出の閾値を50と150に設定したのですか？

**A**: 
- 一般的な推奨値（低い閾値: 高い閾値 = 1:3の比率）
- 弱いエッジと強いエッジを適切に検出するため
- 画像の種類によって最適値は異なるが、一般的な値として設定

### Q6: なぜゲシュタルト原則スコアをLLMで評価したのですか？

**A**: 
- 人間の視覚的知覚を数値化するため
- ゲシュタルト原則は主観的な評価が必要
- LLaVA（Vision-Language Model）は画像を理解し、言語で説明できる
- arXiv 2504.12511論文のアプローチを参考

### Q7: データリークを防ぐためにどのような対策を取ったのですか？

**A**: 
1. **特徴量抽出時**: 標準化を行わない（生の値を保存）
2. **データ分割後**: 訓練データのみでStandardScalerをfit
3. **テストデータ**: 訓練データでfitしたStandardScalerでtransform
4. **スケーラーの保存**: `models/scaler.pkl`として保存し、SHAP説明生成時に使用

### Q8: 特徴量の総数はいくつですか？

**A**: 
- **色彩特徴量**: 19カラム
- **テクスチャ特徴量**: 10カラム
- **エッジ特徴量**: 9カラム
- **ゲシュタルト原則スコア**: 6カラム
- **合計**: 44カラムの数値特徴量

### Q9: なぜ512×512ピクセルにリサイズしたのですか？

**A**: 
- 計算効率と特徴量の一貫性を確保するため
- 512×512は、多くの画像処理タスクで標準的なサイズ
- より大きなサイズにすると計算コストが増加
- より小さなサイズにすると情報が失われる

### Q10: エラーハンドリングはどのように行っていますか？

**A**: 
- 各特徴量抽出処理でtry-exceptブロックを使用
- エラー時はデフォルト値（0.0）を設定
- ログに警告を記録
- エラーが発生しても処理を継続（一部の特徴量が欠損しても他の特徴量は使用可能）

---

## 参考資料

- **OpenCV**: 画像処理ライブラリ
- **scikit-image**: GLCM、LBPの実装
- **scikit-learn**: K-meansクラスタリング、StandardScaler
- **Ollama**: ローカルLLMサーバー
- **LLaVA**: Vision-Language Model
- **arXiv 2504.12511**: ゲシュタルト原則評価のアプローチ

---

*このチートシートは、質疑応答で適切に回答できるよう、特徴量抽出の具体的な処理手順をまとめたものです。*

