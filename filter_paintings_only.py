# -*- coding: utf-8 -*-
"""
絵画データのフィルタリングのみを実行するスクリプト
"""

import sys
import os
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.config_manager import ConfigManager
from src.data_collection.hybrid_collector import HybridCollector

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('filter_paintings.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def filter_paintings():
    """絵画データのフィルタリングを実行"""
    logger = logging.getLogger(__name__)
    logger.info("=== 絵画データフィルタリング開始 ===")
    
    try:
        # 設定読み込み
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # HybridCollector初期化
        collector = HybridCollector(config)
        
        # CSVデータの読み込みと前処理
        logger.info("CSVデータを読み込み中...")
        if collector.csv_file.exists():
            import pandas as pd
            csv_df = pd.read_csv(collector.csv_file, low_memory=False)
            logger.info(f"CSVデータ読み込み完了: {len(csv_df):,}件")
            
            # 前処理
            csv_df = collector.preprocess_csv(csv_df)
            
            # 絵画データのフィルタリング
            filtered_df = collector.filter_painting_data(csv_df)
            
            # フィルタリング済みデータを保存
            collector.save_filtered_data(filtered_df)
            
            logger.info("=== フィルタリング完了 ===")
            logger.info(f"元データ: {len(csv_df):,}件")
            logger.info(f"フィルタリング後: {len(filtered_df):,}件")
            logger.info(f"削減率: {((len(csv_df) - len(filtered_df)) / len(csv_df)) * 100:.1f}%")
            
            return True
        else:
            logger.error(f"CSVファイルが見つかりません: {collector.csv_file}")
            return False
            
    except Exception as e:
        logger.error(f"フィルタリング中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """メイン関数"""
    setup_logging()
    success = filter_paintings()
    
    if success:
        print("\nフィルタリングが正常に完了しました。")
        print("結果ファイル: data/filtered_data/paintings_metadata.csv")
    else:
        print("\nフィルタリング中にエラーが発生しました。ログを確認してください。")

if __name__ == "__main__":
    main()
