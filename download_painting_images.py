# -*- coding: utf-8 -*-
"""
絵画画像のダウンロードスクリプト
既存のHybridCollectorを活用
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
            logging.FileHandler('download_images.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def download_images():
    """絵画画像のダウンロードを実行"""
    logger = logging.getLogger(__name__)
    logger.info("=== 絵画画像ダウンロード開始 ===")
    
    try:
        # 設定読み込み
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # レート制限を画像ダウンロード用に調整
        config['api']['rate_limit'] = 10  # 10 req/s（画像ダウンロード用）
        
        # HybridCollector初期化
        collector = HybridCollector(config)
        
        # 統合データファイルを確認
        complete_dataset_file = collector.filtered_data_dir / 'paintings_complete_dataset.csv'
        
        if not complete_dataset_file.exists():
            logger.error("統合データファイルが見つかりません。先に全件データ収集を実行してください。")
            return False
        
        # 統合データを読み込み
        import pandas as pd
        df = pd.read_csv(complete_dataset_file, low_memory=False)
        logger.info(f"統合データ読み込み完了: {len(df):,}件")
        
        # 画像URLが存在するレコード数を確認
        has_image = df['primaryImageSmall'].notna().sum()
        logger.info(f"画像URL有り: {has_image:,}件")
        
        if has_image == 0:
            logger.warning("ダウンロード対象の画像がありません。")
            return False
        
        # 画像ダウンロード実行
        logger.info("画像ダウンロードを開始...")
        updated_df = collector.download_painting_images(df)
        
        # 更新されたデータを保存
        updated_df.to_csv(complete_dataset_file, index=False, encoding='utf-8')
        logger.info(f"更新されたデータを保存: {complete_dataset_file}")
        
        # 結果サマリー
        downloaded_count = updated_df['image_downloaded'].sum()
        logger.info("=== 画像ダウンロード結果サマリー ===")
        logger.info(f"総データ数: {len(updated_df):,}件")
        logger.info(f"画像URL有り: {has_image:,}件")
        logger.info(f"ダウンロード成功: {downloaded_count:,}件")
        logger.info(f"ダウンロード率: {(downloaded_count / has_image) * 100:.1f}%")
        
        logger.info("=== 画像ダウンロード完了 ===")
        return True
        
    except Exception as e:
        logger.error(f"画像ダウンロード中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """メイン関数"""
    setup_logging()
    success = download_images()
    
    if success:
        print("\n画像ダウンロードが正常に完了しました。")
        print("画像保存先: data/filtered_data/paintings_images/")
    else:
        print("\n画像ダウンロード中にエラーが発生しました。ログを確認してください。")

if __name__ == "__main__":
    main()
