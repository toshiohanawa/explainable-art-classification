# -*- coding: utf-8 -*-
"""
全9,005件の絵画データ収集と画像ダウンロード
既存のHybridCollectorを活用
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging, get_logger
from src.data_collection.hybrid_collector import HybridCollector


def collect_all_paintings():
    """全9,005件の絵画データ収集を実行"""
    # 設定読み込み
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # ログ設定
    setup_logging(
        log_file='collect_all_paintings.log',
        config=config
    )
    logger = get_logger(__name__)
    
    logger.info("=== 全絵画データ収集開始（9,005件） ===")
    
    try:
        # レート制限を安全な値に設定（全件処理のため）
        config['api']['rate_limit'] = 20  # 20 req/s
        
        # HybridCollector初期化
        collector = HybridCollector(config)
        
        # フィルタリング済みデータが存在するかチェック
        if not collector.paintings_metadata_file.exists():
            logger.error("フィルタリング済みデータが見つかりません。先にフィルタリングを実行してください。")
            return False
        
        # フィルタリング済みデータを読み込み
        import pandas as pd
        paintings_df = pd.read_csv(collector.paintings_metadata_file, low_memory=False)
        logger.info(f"フィルタリング済みデータ読み込み完了: {len(paintings_df):,}件")
        
        # 全Object IDリストを作成
        all_object_ids = paintings_df['Object_ID'].dropna().astype(int).tolist()
        logger.info(f"対象Object ID数: {len(all_object_ids):,}件")
        
        # 全件API詳細データ収集
        logger.info("全件API詳細データ収集を開始...")
        collector.collect_api_data(all_object_ids, resume=False)
        
        # 失敗したIDの再取得
        collector.retry_failed_ids(max_retries=2)
        
        # データ統合
        logger.info("データ統合を開始...")
        merged_df = collector.merge_data(paintings_df)
        
        # 統合データ保存
        final_output_file = collector.filtered_data_dir / 'paintings_complete_dataset.csv'
        merged_df.to_csv(final_output_file, index=False, encoding='utf-8')
        logger.info(f"統合データ保存完了: {final_output_file}")
        
        # 結果サマリー
        logger.info("=== 全件収集結果サマリー ===")
        logger.info(f"元データ: {len(paintings_df):,}件")
        logger.info(f"API取得成功: {len(collector.api_data):,}件")
        logger.info(f"API取得失敗: {len(collector.failed_ids):,}件")
        logger.info(f"成功率: {(len(collector.api_data) / len(all_object_ids)) * 100:.1f}%")
        logger.info(f"統合後データ: {len(merged_df):,}件")
        
        # 画像URLの有無をチェック
        image_columns = ['primaryImage', 'primaryImageSmall', 'additionalImages']
        for col in image_columns:
            if col in merged_df.columns:
                has_image = merged_df[col].notna().sum()
                percentage = (has_image / len(merged_df)) * 100
                logger.info(f"{col}: {has_image:,}件 ({percentage:.1f}%)")
        
        logger.info("=== 全件収集完了 ===")
        return True
        
    except Exception as e:
        logger.error(f"全件収集中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """メイン関数"""
    success = collect_all_paintings()
    
    if success:
        print("\n全件データ収集が正常に完了しました。")
        print("結果ファイル: data/filtered_data/paintings_complete_dataset.csv")
    else:
        print("\n全件データ収集中にエラーが発生しました。ログを確認してください。")


if __name__ == "__main__":
    main()

