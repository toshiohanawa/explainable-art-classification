#!/usr/bin/env python3
"""
説明可能AIによる絵画様式分類プロジェクト
メイン実行ファイル

使用方法:
    python main.py --mode [collect|extract|train|explain|all]
"""

import argparse
import logging
import yaml
from pathlib import Path

from src.data_collection.met_api_client import MetAPIClient
from src.feature_extraction.color_extractor import ColorFeatureExtractor
from src.model_training.random_forest_trainer import RandomForestTrainer
from src.explainability.shap_explainer import SHAPExplainer
from src.visualization.result_visualizer import ResultVisualizer
from src.utils.config_manager import ConfigManager
from src.utils.timestamp_manager import TimestampManager


def setup_logging(config):
    """ログ設定を初期化"""
    log_dir = Path(config['logging']['file']).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file'], encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='説明可能AIによる絵画様式分類')
    parser.add_argument('--mode', choices=['metadata', 'images', 'collect', 'extract', 'train', 'explain', 'all'],
                       default='all', help='実行モード')
    parser.add_argument('--config', default='config.yaml', help='設定ファイルのパス')
    
    args = parser.parse_args()
    
    # 設定読み込み
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # タイムスタンプ管理
    timestamp_manager = TimestampManager(config)
    timestamp = timestamp_manager.get_timestamp()
    
    # ログ設定
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info(f"プロジェクト開始: モード={args.mode}, タイムスタンプ={timestamp}")
    logger.info(f"出力ディレクトリ: {timestamp_manager.get_output_dir()}")
    
    try:
        if args.mode in ['metadata', 'all']:
            logger.info("メタデータ収集を開始...")
            api_client = MetAPIClient(config)
            # テスト用にサンプル数を制限
            api_client.collect_metadata(max_objects=100)
            
        if args.mode in ['images', 'all']:
            logger.info("画像ダウンロードを開始...")
            api_client = MetAPIClient(config)
            # 既存のメタデータを読み込んで画像をダウンロード
            import pandas as pd
            metadata_file = timestamp_manager.get_metadata_file()
            if metadata_file.exists():
                df = pd.read_csv(metadata_file)
                api_client.download_images(df, max_images=50)  # テスト用に制限
            else:
                logger.error("メタデータファイルが見つかりません。先にmetadataモードを実行してください。")
                
        if args.mode in ['collect', 'all']:
            logger.info("データ収集を開始...")
            api_client = MetAPIClient(config)
            # テスト用にサンプル数を制限
            api_client.collect_artwork_data(max_objects=100, max_images=50)
            
        if args.mode in ['extract', 'all']:
            logger.info("特徴量抽出を開始...")
            feature_extractor = ColorFeatureExtractor(config)
            feature_extractor.extract_features()
            
        if args.mode in ['train', 'all']:
            logger.info("モデル訓練を開始...")
            trainer = RandomForestTrainer(config)
            trainer.train_model()
            
        if args.mode in ['explain', 'all']:
            logger.info("SHAP説明を開始...")
            explainer = SHAPExplainer(config)
            explainer.generate_explanations()
            
        logger.info("プロジェクト完了")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
