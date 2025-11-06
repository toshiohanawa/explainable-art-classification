"""
WikiArt_VLMデータセットを使用した本物 vs フェイク分類の訓練スクリプト
"""

import sys
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.wikiart_vlm_loader import WikiArtVLMDataLoader
from src.feature_extraction.color_extractor import ColorFeatureExtractor
from src.model_training.random_forest_trainer import RandomForestTrainer
from src.explainability.shap_explainer import SHAPExplainer
from src.utils.config_manager import ConfigManager
from src.utils.timestamp_manager import TimestampManager
from src.utils.logger import setup_logging
import logging


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description='WikiArt_VLMデータセットを使用した本物 vs フェイク分類の訓練'
    )
    parser.add_argument(
        '--mode',
        choices=['extract', 'train', 'all'],
        default='all',
        help='実行モード: extract=特徴量抽出のみ, train=訓練のみ, all=両方実行'
    )
    parser.add_argument(
        '--generation-model',
        choices=['Stable-Diffusion', 'FLUX', 'F-Lite'],
        default='Stable-Diffusion',
        help='使用する生成モデル'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='特徴量抽出時の最大サンプル数（テスト用、Noneの場合は全データ）'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--features-file',
        default=None,
        help='特徴量ファイルのパス（trainモード時のみ使用）'
    )
    parser.add_argument(
        '--include-gestalt',
        action='store_true',
        help='ゲシュタルト原則スコアを含める（extractモード時）'
    )
    parser.add_argument(
        '--gestalt-model',
        type=str,
        default=None,
        help='Ollamaモデル名（ゲシュタルトスコアリング用、デフォルト: config.yamlから取得）'
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # ログ設定
    setup_logging(config=config)
    logger = logging.getLogger(__name__)
    
    # ✅ スクリプトの最初で一度だけTimestampManagerを作成（段階的実行対応）
    timestamp_manager = TimestampManager(config)
    timestamp = timestamp_manager.get_timestamp()
    
    logger.info("=" * 60)
    logger.info("WikiArt_VLMデータセット: 本物 vs フェイク分類の訓練")
    logger.info("=" * 60)
    logger.info(f"タイムスタンプ: {timestamp}")
    logger.info(f"出力ディレクトリ: {timestamp_manager.get_output_dir()}")
    logger.info(f"生成モデル: {args.generation_model}")
    logger.info(f"実行モード: {args.mode}")
    if args.max_samples:
        logger.info(f"最大サンプル数: {args.max_samples}")
    
    try:
        # 特徴量抽出
        if args.mode in ['extract', 'all']:
            logger.info("\n" + "=" * 60)
            logger.info("特徴量抽出を開始...")
            logger.info("=" * 60)
            
            # ✅ 作成したTimestampManagerを渡す
            feature_extractor = ColorFeatureExtractor(config, timestamp_manager=timestamp_manager)
            
            # ゲシュタルト原則スコアを含めるか
            if args.include_gestalt:
                logger.info("ゲシュタルト原則スコアを含めて特徴量抽出")
                gestalt_model = args.gestalt_model or config.get('gestalt_scoring', {}).get('model_name', 'llava:7b')
                logger.info(f"ゲシュタルトスコアリングモデル: {gestalt_model}")
                
                feature_extractor.extract_features_with_gestalt_scores(
                    generation_model=args.generation_model,
                    max_samples=args.max_samples,
                    include_gestalt=True,
                    gestalt_model=gestalt_model
                )
                
                # 統合された特徴量ファイルを使用するよう設定
                features_dir = feature_extractor.timestamp_manager.get_features_dir()
                features_file = features_dir / f"wikiart_vlm_features_with_gestalt_{args.generation_model}.csv"
            else:
                feature_extractor.extract_features_from_wikiart_vlm(
                    generation_model=args.generation_model,
                    max_samples=args.max_samples
                )
                
                # 通常の特徴量ファイルを使用
                features_dir = feature_extractor.timestamp_manager.get_features_dir()
                features_file = features_dir / f"wikiart_vlm_features_{args.generation_model}.csv"
            
            logger.info("特徴量抽出完了")
        
        # モデル訓練
        if args.mode in ['train', 'all']:
            logger.info("\n" + "=" * 60)
            logger.info("モデル訓練を開始...")
            logger.info("=" * 60)
            
            # 特徴量ファイルを取得
            if args.features_file is None:
                # ✅ 同じTimestampManagerを使用（段階的実行対応）
                features_dir = timestamp_manager.get_features_dir()
                
                # 統合された特徴量ファイルを優先的に探す
                merged_features_file = features_dir / f"wikiart_vlm_features_with_gestalt_{args.generation_model}.csv"
                if merged_features_file.exists():
                    features_file = merged_features_file
                    logger.info(f"統合特徴量ファイル（ゲシュタルト原則スコア含む）を使用: {features_file}")
                else:
                    # 通常の特徴量ファイルを使用
                    features_file = features_dir / f"wikiart_vlm_features_{args.generation_model}.csv"
                    if features_file.exists():
                        logger.info(f"通常の特徴量ファイルを使用: {features_file}")
                    else:
                        logger.error(f"特徴量ファイルが見つかりません: {features_file}")
                        logger.error("先にextractモードで特徴量抽出を実行してください")
                        return
                
                features_file_path = str(features_file)
            else:
                features_file_path = args.features_file
            
            # ✅ 同じTimestampManagerを使用（段階的実行対応）
            trainer = RandomForestTrainer(
                config,
                task_type='authenticity',  # 本物 vs フェイク分類
                features_file=features_file_path,
                timestamp_manager=timestamp_manager
            )
            trainer.train_model()
            
            logger.info("モデル訓練完了")
            
            # SHAP説明を生成（オプション）
            logger.info("\n" + "=" * 60)
            logger.info("SHAP説明を生成中...")
            logger.info("=" * 60)
            
            try:
                # ✅ 同じTimestampManagerを使用（段階的実行対応）
                explainer = SHAPExplainer(
                    config,
                    task_type='authenticity',
                    features_file=features_file_path,
                    timestamp_manager=timestamp_manager
                )
                explainer.generate_explanations()
                logger.info("SHAP説明完了")
            except Exception as e:
                logger.warning(f"SHAP説明生成でエラー: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info("すべての処理が完了しました")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

