"""
ゲシュタルト原則スコアリングスクリプト
Ollamaを使用して画像のゲシュタルト原則を評価
"""

import sys
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_extraction.gestalt_scorer import GestaltScorer
from src.data_collection.wikiart_vlm_loader import WikiArtVLMDataLoader
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging
import logging


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description='ゲシュタルト原則スコアリングスクリプト'
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'batch', 'wikiart'],
        default='wikiart',
        help='実行モード: single=単一画像, batch=複数画像, wikiart=WikiArt_VLMデータセット'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='単一画像のパス（singleモード時）'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help='画像ディレクトリのパス（batchモード時）'
    )
    parser.add_argument(
        '--generation-model',
        choices=['Stable-Diffusion', 'FLUX', 'F-Lite'],
        default='Stable-Diffusion',
        help='WikiArt_VLMデータセットの生成モデル（wikiartモード時）'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大サンプル数（wikiartモード時、Noneの場合は全データ）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Ollamaモデル名（例: llava:latest, llava:7b, llava:13b, llava:34b）'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='既存のゲシュタルトスコアがあっても再スコアリングを強制する'
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # ログ設定
    setup_logging(config=config)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("ゲシュタルト原則スコアリング")
    logger.info("=" * 60)
    
    # モデル名を決定
    model_name = args.model or config.get('gestalt_scoring', {}).get('model_name', 'llava:7b')
    logger.info(f"使用モデル: {model_name}")
    
    try:
        # GestaltScorerを初期化
        scorer = GestaltScorer(config, model_name=model_name, force_run=args.force)
        
        if args.mode == 'single':
            # 単一画像のスコアリング
            if not args.image:
                logger.error("単一画像モードでは --image オプションが必要です")
                return
            
            logger.info(f"\n画像: {args.image}")
            result = scorer.score_single_image(Path(args.image))
            
            logger.info("\n" + "=" * 60)
            logger.info("スコアリング結果")
            logger.info("=" * 60)
            for principle, data in result.items():
                if principle not in ['image_path', 'image_name']:
                    if isinstance(data, dict):
                        logger.info(f"{principle}: {data.get('score', 'N/A')}")
                        logger.info(f"  説明: {data.get('explanation', '')}")
        
        elif args.mode == 'batch':
            # バッチスコアリング
            if not args.image_dir:
                logger.error("バッチモードでは --image-dir オプションが必要です")
                return
            
            image_dir = Path(args.image_dir)
            if not image_dir.exists():
                logger.error(f"画像ディレクトリが見つかりません: {image_dir}")
                return
            
            # 画像ファイルを取得
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(image_dir.glob(f'*{ext}'))
            
            logger.info(f"対象画像数: {len(image_paths)}")
            
            # バッチスコアリング
            scores_df = scorer.score_images_batch(image_paths)
            
            # 結果を保存
            output_file = scorer.gestalt_dir / 'gestalt_scores_batch.csv'
            scores_df.to_csv(output_file, index=False)
            
            logger.info(f"\nスコアリング結果を保存: {output_file}")
            logger.info(f"  成功: {len(scores_df)}件")
        
        elif args.mode == 'wikiart':
            # WikiArt_VLMデータセットのスコアリング
            logger.info(f"生成モデル: {args.generation_model}")
            if args.max_samples:
                logger.info(f"最大サンプル数: {args.max_samples}")
            
            # WikiArt_VLMデータセットのスコアリング
            scores_df, output_path = scorer.score_wikiart_vlm_images(
                generation_model=args.generation_model,
                max_samples=args.max_samples
            )
            
            logger.info(f"\nスコアリング結果: {len(scores_df)}件 (保存先: {output_path})")
        
        logger.info("\n" + "=" * 60)
        logger.info("すべての処理が完了しました")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
