"""
Phase 5: 分析と可視化の拡充スクリプト
誤分類作品の詳細分析、生成モデル間の比較、教育的価値の可視化
"""

import sys
import argparse
from pathlib import Path
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.authenticity_analyzer import AuthenticityAnalyzer
from src.utils.config_manager import ConfigManager
from src.utils.timestamp_manager import TimestampManager
from src.utils.logger import setup_logging


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description='Phase 5: 分析と可視化の拡充 - 誤分類作品の詳細分析、生成モデル間の比較、教育的価値の可視化'
    )
    parser.add_argument(
        '--mode',
        choices=['misclassification', 'comparison', 'authenticity', 'educational', 'all'],
        default='all',
        help='実行モード'
    )
    parser.add_argument(
        '--generation-model',
        choices=['Stable-Diffusion', 'FLUX', 'F-Lite'],
        default='Stable-Diffusion',
        help='分析する生成モデル（misclassification, authenticity, educationalモード時）'
    )
    parser.add_argument(
        '--generation-models',
        nargs='+',
        choices=['Stable-Diffusion', 'FLUX', 'F-Lite'],
        default=['Stable-Diffusion', 'FLUX', 'F-Lite'],
        help='比較する生成モデル（comparisonモード時）'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help='分析対象のタイムスタンプ（Noneの場合は最新）'
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # ログ設定
    setup_logging(config=config)
    logger = logging.getLogger(__name__)
    
    # タイムスタンプ管理
    if args.timestamp:
        timestamp_manager = TimestampManager(config, timestamp=args.timestamp)
    else:
        timestamp_manager = TimestampManager(config)
    
    timestamp = timestamp_manager.get_timestamp()
    
    logger.info("=" * 60)
    logger.info("Phase 5: 分析と可視化の拡充")
    logger.info("=" * 60)
    logger.info(f"タイムスタンプ: {timestamp}")
    logger.info(f"出力ディレクトリ: {timestamp_manager.get_output_dir()}")
    logger.info(f"実行モード: {args.mode}")
    
    try:
        # AuthenticityAnalyzerを初期化
        analyzer = AuthenticityAnalyzer(config, timestamp_manager=timestamp_manager)
        
        # 誤分類作品の詳細分析
        if args.mode in ['misclassification', 'all']:
            logger.info("\n" + "=" * 60)
            logger.info("誤分類作品の詳細分析を開始...")
            logger.info("=" * 60)
            
            misclass_df = analyzer.analyze_misclassifications(args.generation_model)
            
            if len(misclass_df) > 0:
                logger.info(f"誤分類分析完了: {len(misclass_df)}件の誤分類を分析")
            else:
                logger.warning("誤分類作品が見つかりませんでした")
            
            logger.info("誤分類分析完了")
        
        # 生成モデル間の詳細比較
        if args.mode in ['comparison', 'all']:
            logger.info("\n" + "=" * 60)
            logger.info("生成モデル間の詳細比較を開始...")
            logger.info("=" * 60)
            
            comparison_df = analyzer.compare_generation_models(args.generation_models)
            
            if len(comparison_df) > 0:
                logger.info(f"生成モデル比較完了: {len(comparison_df)}モデルを比較")
                logger.info("\n比較結果:")
                print(comparison_df.to_string(index=False))
            else:
                logger.warning("比較結果が見つかりませんでした")
            
            logger.info("生成モデル比較完了")
        
        # 「本物らしさ」の可視化
        if args.mode in ['authenticity', 'all']:
            logger.info("\n" + "=" * 60)
            logger.info("「本物らしさ」の可視化を開始...")
            logger.info("=" * 60)
            
            analyzer.visualize_authenticity_factors(args.generation_model)
            
            logger.info("「本物らしさ」の可視化完了")
        
        # 教育的レポートの生成
        if args.mode in ['educational', 'all']:
            logger.info("\n" + "=" * 60)
            logger.info("教育的レポートを生成中...")
            logger.info("=" * 60)
            
            analyzer.generate_educational_report(args.generation_model)
            
            logger.info("教育的レポート生成完了")
        
        logger.info("\n" + "=" * 60)
        logger.info("すべての処理が完了しました")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

