# -*- coding: utf-8 -*-
"""
絵画データ収集の小規模テスト（200件）
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging, get_logger
from src.data_collection.hybrid_collector import HybridCollector


def test_painting_collection():
    """絵画データ収集のテスト実行"""
    # 設定読み込み
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # ログ設定
    setup_logging(
        log_file='test_painting_collection.log',
        config=config
    )
    logger = get_logger(__name__)
    
    logger.info("=== 絵画データ収集テスト開始（200件） ===")
    
    try:
        # テスト用にレート制限を調整（安全のため）
        config['api']['rate_limit'] = 10  # 10 req/s
        
        # HybridCollector初期化
        collector = HybridCollector(config)
        
        # フィルタリング済みデータが存在するかチェック
        if collector.paintings_metadata_file.exists():
            logger.info("既存のフィルタリング済みデータを使用")
            import pandas as pd
            paintings_df = pd.read_csv(collector.paintings_metadata_file, low_memory=False)
            logger.info(f"フィルタリング済みデータ: {len(paintings_df):,}件")
            
            # 最初の200件をテスト対象とする
            test_df = paintings_df.head(200)
            logger.info(f"テスト対象: {len(test_df):,}件")
            
            # テスト用のObject IDリストを作成
            test_object_ids = test_df['Object_ID'].dropna().astype(int).tolist()
            logger.info(f"テスト用Object ID数: {len(test_object_ids)}件")
            
            # API詳細データ収集（200件のみ）
            logger.info("API詳細データ収集を開始...")
            collector.collect_api_data(test_object_ids, resume=False)
            
            # 失敗したIDの再取得
            collector.retry_failed_ids(max_retries=1)
            
            # データ統合
            logger.info("データ統合を開始...")
            merged_df = collector.merge_data(test_df)
            
            # 統合データ保存
            test_output_file = collector.filtered_data_dir / 'test_paintings_200.csv'
            merged_df.to_csv(test_output_file, index=False, encoding='utf-8')
            logger.info(f"テスト結果保存完了: {test_output_file}")
            
            # 結果サマリー
            logger.info("=== テスト結果サマリー ===")
            logger.info(f"元データ: {len(test_df):,}件")
            logger.info(f"API取得成功: {len(collector.api_data):,}件")
            logger.info(f"API取得失敗: {len(collector.failed_ids):,}件")
            logger.info(f"成功率: {(len(collector.api_data) / len(test_object_ids)) * 100:.1f}%")
            logger.info(f"統合後データ: {len(merged_df):,}件")
            
            # 画像URLの有無をチェック
            image_columns = ['primaryImage', 'primaryImageSmall', 'additionalImages']
            for col in image_columns:
                if col in merged_df.columns:
                    has_image = merged_df[col].notna().sum()
                    percentage = (has_image / len(merged_df)) * 100
                    logger.info(f"{col}: {has_image:,}件 ({percentage:.1f}%)")
            
            logger.info("=== テスト完了 ===")
            return True
            
        else:
            logger.error("フィルタリング済みデータが見つかりません。先にフィルタリングを実行してください。")
            return False
            
    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """メイン関数"""
    success = test_painting_collection()
    
    if success:
        print("\nテストが正常に完了しました。")
        print("結果ファイル: data/filtered_data/test_paintings_200.csv")
    else:
        print("\nテスト中にエラーが発生しました。ログを確認してください。")


if __name__ == "__main__":
    main()

