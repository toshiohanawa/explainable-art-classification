# ループ防止リファクタリング

## 問題の概要

Phase 4（ゲシュタルト原則スコアリング）の実行時に、既存の完了済みファイルをチェックせずに常に新規処理を開始していたため、同じデータセットに対して複数回処理が実行される問題が発生しました。

## ループした原因

### 1. 既存ファイルチェック機能の欠如

**問題点**:
- `gestalt_scorer.py`の`score_wikiart_vlm_images`メソッドが、既存のゲシュタルトスコアファイルの存在をチェックしていなかった
- タイムスタンプディレクトリが異なる場合、既存ファイルが見つからない
- 同じデータセットに対して複数回処理が実行される

**影響**:
- 処理時間の無駄（17,328画像 × 約6秒/画像 ≈ 約29時間）
- リソースの無駄（GPU/CPU、メモリ）
- 重複した処理履歴の生成

### 2. run_all_phases.pyでのチェック機能の欠如

**問題点**:
- `run_all_phases.py`がPhase 4の完了状況をチェックしていなかった
- 既存の完了済みファイルを再利用する仕組みがなかった

## 実装した解決策

### 1. gestalt_scorer.pyへの既存ファイルチェック機能追加

**追加した機能**:
- `_check_existing_scores()`メソッドを追加
  - 現在のタイムスタンプディレクトリ内のファイルをチェック
  - 他の`analysis_*`ディレクトリから既存ファイルを検索
  - 既存ファイルが見つかった場合、現在のディレクトリにコピー
  - 期待されるペア数と実際の行数を比較して検証

**実装内容**:
```python
def _check_existing_scores(self, generation_model: str, expected_pairs: int) -> Optional[Path]:
    """
    既存のゲシュタルトスコアファイルをチェック
    
    - 現在のディレクトリ内のファイルをチェック
    - 他のanalysis_ディレクトリから既存ファイルを検索
    - 期待されるペア数と一致する場合、ファイルを再利用
    """
    # 1. 現在のディレクトリ内のファイルをチェック
    # 2. 他のanalysis_ディレクトリから既存ファイルを検索
    # 3. 既存ファイルが見つかった場合、現在のディレクトリにコピー
    # 4. 期待されるペア数と実際の行数を比較して検証
```

**score_wikiart_vlm_images()メソッドの修正**:
```python
def score_wikiart_vlm_images(self, generation_model: str = 'Stable-Diffusion',
                             max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    # 期待されるペア数（Original画像数）
    expected_pairs = len(image_pairs_df[image_pairs_df['label'] == 0])
    
    # 既存ファイルをチェック
    existing_file = self._check_existing_scores(generation_model, expected_pairs)
    
    if existing_file is not None:
        self.logger.info(f"既存のゲシュタルトスコアファイルを再利用します: {existing_file}")
        scores_df = pd.read_csv(existing_file)
        return scores_df, str(existing_file)
    
    # 既存ファイルがない場合、新規にスコアリング
    # ...
```

### 2. run_all_phases.pyへのPhase 4完了チェック機能追加

**追加した機能**:
- `check_phase4_completed()`メソッドを追加
  - データディレクトリ内のすべての`analysis_*`ディレクトリを検索
  - 最新の完了済みファイルを検出
  - 最低限の行数チェック（17,000行以上）

**実装内容**:
```python
def check_phase4_completed(self):
    """Phase 4の完了状況をチェック"""
    # データディレクトリから最新のanalysis_ディレクトリを検索
    # 最新の完了済みファイルを検出
    # 最低限の行数チェック（17,000行以上）
    return True if completed else False

def run_phase4(self):
    """Phase 4: ゲシュタルト原則スコアリング"""
    # 既存ファイルをチェック
    if self.check_phase4_completed():
        self.logger.info("Phase 4は既に完了しています。スキップします。")
        return True
    # 既存ファイルがない場合、新規に実行
    # ...
```

## 検証方法

### 既存ファイルの検証基準

1. **行数チェック**:
   - 期待される行数 = 期待されるペア数 × 2（Original + Generated）
   - 実際の行数が期待値と一致するか（10行の誤差を許容）

2. **画像パス分布チェック**:
   - Original画像数 = Generated画像数
   - 画像パスに`Original`または生成モデル名が含まれている

3. **ファイルサイズチェック**:
   - 最低限の行数（17,000行以上）を満たしている

## 効果

### 処理時間の削減

- **Before**: 17,328画像 × 約6秒/画像 ≈ **約29時間**
- **After**: 既存ファイルを再利用 → **数秒**

### リソースの節約

- GPU/CPUの使用時間削減
- メモリ使用量の削減
- ディスクI/Oの削減

### ユーザー体験の向上

- 不要な待機時間の削減
- 明確なログメッセージによる状況把握
- 既存ファイルの自動検出と再利用

## 今後の改善点

1. **キャッシュ機能の拡充**:
   - フィルタリング条件やデータセットサイズに基づくキャッシュ管理
   - キャッシュの有効期限やクリア機能

2. **進捗状況の表示**:
   - 既存ファイル検出時の明確なメッセージ
   - 処理スキップ時のログ出力

3. **エラーハンドリングの強化**:
   - 既存ファイルの破損検出
   - 不完全なファイルの自動再処理

## 変更ファイル一覧

1. `src/feature_extraction/gestalt_scorer.py`
   - `_check_existing_scores()`メソッド追加
   - `score_wikiart_vlm_images()`メソッド修正

2. `scripts/run_all_phases.py`
   - `check_phase4_completed()`メソッド追加
   - `run_phase4()`メソッド修正

