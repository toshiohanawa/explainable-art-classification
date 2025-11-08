"""
ゲシュタルト原則スコアリングクラス
Ollamaを使用したLLM/VLMによるゲシュタルト原則の評価
"""

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import requests
from PIL import Image
import io

from ..utils.config_manager import ConfigManager
from ..utils.timestamp_manager import TimestampManager


class GestaltScorer:
    """ゲシュタルト原則スコアリングクラス"""
    
    # ゲシュタルト原則の定義
    GESTALT_PRINCIPLES = {
        'simplicity': {
            'name': 'Simplicity (簡潔性)',
            'description': 'The tendency to perceive the simplest interpretation of visual elements. Simpler compositions are easier to understand and remember.'
        },
        'proximity': {
            'name': 'Proximity (近接性)',
            'description': 'Elements that are close together are perceived as related or grouped. Spatial distance affects perceived relationships.'
        },
        'similarity': {
            'name': 'Similarity (類同)',
            'description': 'Similar elements (in color, shape, size, texture, etc.) are perceived as belonging together or forming a group.'
        },
        'continuity': {
            'name': 'Continuity (連続性)',
            'description': 'The tendency to perceive continuous, smooth lines or paths rather than disconnected segments. The eye follows smooth curves.'
        },
        'closure': {
            'name': 'Closure (閉合性)',
            'description': 'The tendency to perceive incomplete shapes as complete by mentally filling in gaps. Closed forms are perceived as more stable.'
        },
        'figure_ground': {
            'name': 'Figure-Ground Separation (図地分離)',
            'description': 'The ability to distinguish the main subject (figure) from the background (ground). Clear separation improves visual clarity.'
        }
    }
    
    def __init__(self, config: Dict[str, Any], model_name: str = 'llava:7b', 
                 ollama_base_url: str = 'http://localhost:11434',
                 timestamp_manager: Optional[TimestampManager] = None,
                 force_run: bool = False):
        """
        初期化
        
        Args:
            config: 設定辞書
            model_name: Ollamaモデル名（例: 'llava:latest', 'llava:7b', 'llava:13b', 'llava:34b'）
            ollama_base_url: OllamaサーバーのベースURL
            timestamp_manager: タイムスタンプ管理オブジェクト（Noneの場合は新規作成）
                             段階的実行で同じタイムスタンプを使用する場合に指定
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.gestalt_config = config.get('gestalt_scoring', {})
        
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.api_url = f"{ollama_base_url}/api/generate"
        self.chat_url = f"{ollama_base_url}/api/chat"
        
        self.logger = logging.getLogger(__name__)

        # タイムスタンプ管理（共有または新規作成）
        if timestamp_manager is not None:
            self.timestamp_manager = timestamp_manager
        else:
            self.timestamp_manager = TimestampManager(config)
        self.output_dir = self.timestamp_manager.get_output_dir()
        self.features_dir = self.timestamp_manager.get_features_dir()
        self.gestalt_dir = self.timestamp_manager.get_gestalt_dir()
        self.data_root = Path(self.data_config.get('output_dir', 'data'))

        # ディレクトリ作成
        self.timestamp_manager.create_directories()

        # スコアリング設定
        self.score_scale = self.gestalt_config.get('score_scale', (1, 5))  # 1-5スケール
        self.max_retries = self.gestalt_config.get('max_retries', 3)
        self.retry_delay = self.gestalt_config.get('retry_delay', 1.0)  # 秒
        self.reuse_existing = self.gestalt_config.get('reuse_existing_results', True)
        self.force_run = force_run or self.gestalt_config.get('force_run', False)
        
        # プロンプトテンプレート
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> str:
        """
        ゲシュタルト原則評価用プロンプトテンプレートを作成
        arXiv 2504.12511論文のアプローチを参考
        
        Returns:
            プロンプトテンプレート文字列
        """
        principles_desc = "\n".join([
            f"- **{principle['name']}**: {principle['description']}"
            for principle in self.GESTALT_PRINCIPLES.values()
        ])
        
        template = f"""You are an expert in visual perception and Gestalt psychology. Evaluate the provided image based on the following Gestalt principles:

{principles_desc}

**Task**: For each Gestalt principle, provide:
1. A numerical score from {self.score_scale[0]} to {self.score_scale[1]} (where {self.score_scale[0]} = poor adherence, {self.score_scale[1]} = excellent adherence)
2. A brief explanation (1-2 sentences) justifying the score

**Output Format** (strictly follow this JSON structure):
{{
    "simplicity": {{
        "score": <number 1-5>,
        "explanation": "<brief explanation>"
    }},
    "proximity": {{
        "score": <number 1-5>,
        "explanation": "<brief explanation>"
    }},
    "similarity": {{
        "score": <number 1-5>,
        "explanation": "<brief explanation>"
    }},
    "continuity": {{
        "score": <number 1-5>,
        "explanation": "<brief explanation>"
    }},
    "closure": {{
        "score": <number 1-5>,
        "explanation": "<brief explanation>"
    }},
    "figure_ground": {{
        "score": <number 1-5>,
        "explanation": "<brief explanation>"
    }}
}}

**Important**: 
- Provide scores based on how well the image demonstrates each principle
- Be objective and focus on visual composition, not subjective aesthetic preferences
- Return ONLY valid JSON, no additional text before or after

Now evaluate the following image:"""
        
        return template
    
    def _encode_image_to_base64(self, image_path: Path) -> str:
        """
        画像をbase64エンコード
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            base64エンコードされた画像文字列
        """
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        LLMの応答をパースしてゲシュタルト原則スコアを抽出
        
        Args:
            response_text: LLMの応答テキスト
            
        Returns:
            パースされたスコア辞書
        """
        scores = {}
        
        # JSON形式の応答を抽出（ネストされたJSONを正しく抽出）
        # まず、コードブロック内のJSONを探す
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if code_block_match:
            json_text = code_block_match.group(1)
            try:
                scores = json.loads(json_text)
                return scores
            except json.JSONDecodeError:
                self.logger.warning("コードブロック内のJSONパースエラー")
        
        # コードブロックがない場合、全体からJSONを抽出
        # ネストされたJSONをマッチするパターン（各原則のオブジェクトを含む）
        json_match = re.search(r'\{[^{}]*"simplicity"\s*:\s*\{[^}]+\}[^}]*"proximity"\s*:\s*\{[^}]+\}[^}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                json_text = json_match.group(0)
                scores = json.loads(json_text)
                return scores
            except json.JSONDecodeError:
                self.logger.warning("JSONパースエラー、個別抽出を試行")
        
        # JSONが見つからない場合、正規表現でスコアを抽出
        if not scores:
            for principle in self.GESTALT_PRINCIPLES.keys():
                # スコアを抽出（例: "simplicity": {"score": 4, ...}）
                pattern = f'"{principle}"' + r'\s*:\s*\{[^}]*"score"\s*:\s*(\d+)'
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    scores[principle] = {
                        'score': int(match.group(1)),
                        'explanation': ''
                    }
                else:
                    # デフォルト値
                    scores[principle] = {
                        'score': (self.score_scale[0] + self.score_scale[1]) // 2,
                        'explanation': 'Could not extract score'
                    }
        
        # すべての原則が含まれているか確認
        for principle in self.GESTALT_PRINCIPLES.keys():
            if principle not in scores:
                scores[principle] = {
                    'score': (self.score_scale[0] + self.score_scale[1]) // 2,
                    'explanation': 'Not evaluated'
                }
        
        return scores
    
    def _call_ollama_api(self, image_path: Path) -> str:
        """
        Ollama APIを呼び出してゲシュタルト原則スコアを取得
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            LLMの応答テキスト
        """
        # 画像をbase64エンコード
        encoded_image = self._encode_image_to_base64(image_path)
        
        # APIリクエスト用のペイロード
        # Ollama APIでは、画像を直接base64文字列として渡す
        payload = {
            'model': self.model_name,
            'prompt': self.prompt_template,
            'images': [encoded_image],  # base64エンコードされた画像のリスト
            'stream': False,
            'options': {
                'temperature': 0.3,  # より一貫性のある出力
                'top_p': 0.9,
                'num_predict': 1000  # 応答の最大トークン数（JSON形式のため多めに）
            }
        }
        
        # リトライロジック
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=120  # 2分のタイムアウト
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get('response', '')
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # 指数バックオフ
                    self.logger.warning(f"API呼び出しエラー (試行 {attempt + 1}/{self.max_retries}): {e}. {wait_time:.1f}秒待機して再試行...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"API呼び出しが失敗しました: {e}")
                    raise
        
        return ''
    
    def score_single_image(self, image_path: Path) -> Dict[str, Any]:
        """
        単一画像のゲシュタルト原則スコアを取得
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            ゲシュタルト原則スコア辞書
        """
        if not image_path.exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
        
        self.logger.info(f"ゲシュタルト原則スコアリング: {image_path.name}")
        
        try:
            # Ollama APIを呼び出し
            response_text = self._call_ollama_api(image_path)
            
            if not response_text:
                raise ValueError("LLMからの応答が空です")
            
            # 応答をパース
            scores = self._parse_llm_response(response_text)
            
            # 画像パス情報を追加
            result = {
                'image_path': str(image_path),
                'image_name': image_path.name,
                **scores
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"スコアリングエラー ({image_path.name}): {e}")
            # エラー時はデフォルト値を返す
            default_scores = {
                principle: {
                    'score': (self.score_scale[0] + self.score_scale[1]) // 2,
                    'explanation': f'Error: {str(e)}'
                }
                for principle in self.GESTALT_PRINCIPLES.keys()
            }
            return {
                'image_path': str(image_path),
                'image_name': image_path.name,
                **default_scores
            }
    
    def score_images_batch(self, image_paths: List[Path], 
                          batch_size: int = 10, 
                          save_checkpoint: bool = True) -> pd.DataFrame:
        """
        複数画像のゲシュタルト原則スコアをバッチ処理
        
        Args:
            image_paths: 画像ファイルのパスリスト
            batch_size: チェックポイント保存間隔
            save_checkpoint: チェックポイント保存を行うか
            
        Returns:
            スコア結果のDataFrame
        """
        self.logger.info(f"バッチスコアリング開始: {len(image_paths)}件の画像")
        
        results = []
        failed_count = 0
        
        for idx, image_path in enumerate(image_paths):
            if (idx + 1) % 10 == 0:
                self.logger.info(f"進捗: {idx + 1}/{len(image_paths)} (成功: {len(results)}, 失敗: {failed_count})")
            
            try:
                score_result = self.score_single_image(image_path)
                
                # スコアをフラット化（CSV保存用）
                flat_result = {
                    'image_path': score_result['image_path'],
                    'image_name': score_result['image_name']
                }
                
                for principle in self.GESTALT_PRINCIPLES.keys():
                    if principle in score_result:
                        if isinstance(score_result[principle], dict):
                            flat_result[f'{principle}_score'] = score_result[principle].get('score', 0)
                            flat_result[f'{principle}_explanation'] = score_result[principle].get('explanation', '')
                        else:
                            flat_result[f'{principle}_score'] = score_result[principle]
                    else:
                        flat_result[f'{principle}_score'] = 0
                        flat_result[f'{principle}_explanation'] = ''
                
                results.append(flat_result)
                
                # チェックポイント保存
                if save_checkpoint and (idx + 1) % batch_size == 0:
                    self._save_checkpoint(results, idx + 1)
                
            except Exception as e:
                self.logger.error(f"スコアリングエラー ({image_path.name}): {e}")
                failed_count += 1
                continue
        
        # 結果をDataFrameに変換
        df = pd.DataFrame(results)
        
        self.logger.info(f"バッチスコアリング完了: 成功 {len(results)}件, 失敗 {failed_count}件")
        
        return df
    
    def _save_checkpoint(self, results: List[Dict], current_count: int) -> None:
        """チェックポイントを保存"""
        df = pd.DataFrame(results)
        checkpoint_dir = self.gestalt_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f'gestalt_scores_checkpoint_{current_count}.csv'
        df.to_csv(checkpoint_file, index=False)
        self.logger.info(f"チェックポイント保存: {checkpoint_file}")
    
    def _check_existing_scores(self, generation_model: str, expected_pairs: int) -> Optional[Path]:
        """
        既存のゲシュタルトスコアファイルをチェック
        
        Args:
            generation_model: 生成モデル名
            expected_pairs: 期待されるペア数
            
        Returns:
            既存ファイルのPath（存在する場合）、None（存在しない場合）
        """
        if self.force_run or not self.reuse_existing:
            return None

        output_file = self.gestalt_dir / f'gestalt_scores_{generation_model}.csv'
        
        if output_file.exists():
            try:
                df = pd.read_csv(output_file)
                
                # 行数をチェック（期待されるペア数 × 2）
                expected_rows = expected_pairs * 2
                actual_rows = len(df)
                
                # 画像パスの分布をチェック
                if 'image_path' in df.columns:
                    original_count = df[df['image_path'].str.contains('Original')].shape[0]
                    generated_count = df[df['image_path'].str.contains(generation_model) | 
                                        df['image_path'].str.contains('Stable-Diffusion') | 
                                        df['image_path'].str.contains('FLUX') | 
                                        df['image_path'].str.contains('F-Lite')].shape[0]
                    
                    # 期待されるペア数と一致するかチェック（10行の誤差を許容）
                    if abs(actual_rows - expected_rows) <= 10 and original_count == generated_count:
                        self.logger.info(f"既存のゲシュタルトスコアファイルが見つかりました: {output_file}")
                        self.logger.info(f"  行数: {actual_rows:,} (期待: {expected_rows:,})")
                        self.logger.info(f"  Original: {original_count:,}, Generated: {generated_count:,}")
                        return output_file
                    else:
                        self.logger.warning(f"既存ファイルの行数が期待値と一致しません")
                        self.logger.warning(f"  実際: {actual_rows:,}, 期待: {expected_rows:,}")
                        self.logger.warning(f"  Original: {original_count:,}, Generated: {generated_count:,}")
                else:
                    self.logger.warning(f"既存ファイルに'image_path'カラムがありません")
            except Exception as e:
                self.logger.warning(f"既存ファイルの読み込みエラー: {e}")
        
        legacy_dirs = sorted(self.data_root.glob("analysis_*"), reverse=True)
        for analysis_dir in legacy_dirs:
            candidate_file = analysis_dir / "features" / f'gestalt_scores_{generation_model}.csv'
            if not candidate_file.exists():
                continue
            try:
                df = pd.read_csv(candidate_file)
                expected_rows = expected_pairs * 2
                actual_rows = len(df)
                
                if 'image_path' in df.columns:
                    original_count = df[df['image_path'].str.contains('Original')].shape[0]
                    generated_count = df[df['image_path'].str.contains(generation_model) | 
                                        df['image_path'].str.contains('Stable-Diffusion') | 
                                        df['image_path'].str.contains('FLUX') | 
                                        df['image_path'].str.contains('F-Lite')].shape[0]
                    
                    if abs(actual_rows - expected_rows) <= 10 and original_count == generated_count:
                        self.logger.info(f"既存のゲシュタルトスコアファイルが見つかりました: {candidate_file}")
                        self.logger.info(f"  行数: {actual_rows:,} (期待: {expected_rows:,})")
                        self.logger.info(f"  Original: {original_count:,}, Generated: {generated_count:,}")
                        import shutil
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(candidate_file, output_file)
                        self.logger.info(f"  最新ディレクトリにコピーしました: {output_file}")
                        return output_file
            except Exception as e:
                self.logger.warning(f"候補ファイルの読み込みエラー ({candidate_file}): {e}")
                continue
        return None
    
    def score_wikiart_vlm_images(self, generation_model: str = 'Stable-Diffusion',
                                 max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
        """
        WikiArt_VLMデータセットの画像に対してゲシュタルト原則スコアリング
        
        Args:
            generation_model: 生成モデル名 ('Stable-Diffusion', 'FLUX', 'F-Lite')
            max_samples: 最大サンプル数（Noneの場合は全データ）
            
        Returns:
            (スコア結果のDataFrame, ファイルパス)のタプル
        """
        from ..data_collection.wikiart_vlm_loader import WikiArtVLMDataLoader
        
        self.logger.info(f"WikiArt_VLM画像のゲシュタルト原則スコアリング開始 (モデル: {generation_model})")
        
        # WikiArt_VLMデータローダーを初期化
        loader = WikiArtVLMDataLoader(self.config)
        
        # 画像ペアを読み込み
        image_pairs_df = loader.load_image_pairs(generation_model)
        
        # サンプル数を制限
        if max_samples is not None:
            image_pairs_df = image_pairs_df.head(max_samples)
            self.logger.info(f"サンプル数を制限: {len(image_pairs_df)}件")
        
        # 期待されるペア数（Original画像数）
        expected_pairs = len(image_pairs_df[image_pairs_df['label'] == 0])
        
        # 既存ファイルをチェック
        existing_file = self._check_existing_scores(generation_model, expected_pairs)
        
        if existing_file is not None:
            self.logger.info(f"既存のゲシュタルトスコアファイルを再利用します: {existing_file}")
            scores_df = pd.read_csv(existing_file)
            return scores_df, str(existing_file)
        
        # 既存ファイルがない場合、新規にスコアリング
        self.logger.info(f"新規にゲシュタルト原則スコアリングを開始します")
        
        # 画像パスを取得
        image_paths = [Path(row['image_path']) for _, row in image_pairs_df.iterrows()]
        
        # バッチスコアリング
        scores_df = self.score_images_batch(image_paths)
        
        # 結果を保存
        output_file = self.gestalt_dir / f'gestalt_scores_{generation_model}.csv'
        scores_df.to_csv(output_file, index=False)
        
        self.logger.info(f"ゲシュタルト原則スコアを保存: {output_file}")
        
        # ファイルパスも返す（統合時の参照用）
        return scores_df, str(output_file)
