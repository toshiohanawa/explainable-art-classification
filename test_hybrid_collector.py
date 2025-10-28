"""
ハイブリッドデータ収集のテストスクリプト
少数件（100件）で動作確認を行う
"""

import logging
import yaml
from pathlib import Path
from src.data_collection.hybrid_collector import HybridCollector

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_hybrid.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def test_hybrid_collector():
    """ハイブリッドデータ収集のテスト"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 設定読み込み
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # テスト用に設定を調整
        config['hybrid_collection']['checkpoint_interval'] = 10  # 10件ごとにチェックポイント
        
        logger.info("ハイブリッドデータ収集テストを開始...")
        
        # ハイブリッドコレクターを初期化
        collector = HybridCollector(config)
        
        # 1. CSVダウンロードテスト
        logger.info("=== CSVダウンロードテスト ===")
        csv_df = collector.download_csv()
        logger.info(f"CSVダウンロード成功: {len(csv_df)}件")
        
        # 2. CSV前処理テスト
        logger.info("=== CSV前処理テスト ===")
        processed_df = collector.preprocess_csv(csv_df)
        logger.info(f"前処理完了: {len(processed_df)}件")
        
        # 3. API全ID取得テスト
        logger.info("=== API全ID取得テスト ===")
        api_ids = collector.get_all_object_ids()
        if api_ids:
            logger.info(f"API全ID取得成功: {len(api_ids)}件")
        else:
            logger.info("API全ID取得失敗: CSVデータのみを使用")
        
        # 4. 少数件でのAPI詳細取得テスト
        logger.info("=== 少数件API詳細取得テスト ===")
        if api_ids:
            test_ids = api_ids[:100]  # 最初の100件をテスト
        else:
            # CSVデータから最初の100件を使用
            test_ids = processed_df['Object_ID'].dropna().astype(int).head(100).tolist()
        collector.collect_api_data(test_ids, resume=False)
        
        logger.info(f"API詳細取得完了: 成功 {len(collector.api_data)}件, 失敗 {len(collector.failed_ids)}件")
        
        # 5. データ統合テスト
        logger.info("=== データ統合テスト ===")
        merged_df = collector.merge_data(processed_df)
        logger.info(f"データ統合完了: {len(merged_df)}件")
        
        # 6. 品質管理レポート生成テスト
        logger.info("=== 品質管理レポート生成テスト ===")
        collector.generate_qa_report(merged_df)
        
        logger.info("ハイブリッドデータ収集テスト完了")
        
        # 結果サマリー
        print("\n=== テスト結果サマリー ===")
        print(f"CSV件数: {len(processed_df):,}")
        print(f"API全ID数: {len(api_ids) if api_ids else 0:,}")
        print(f"テスト対象ID数: {len(test_ids)}")
        print(f"API詳細取得成功: {len(collector.api_data)}")
        print(f"API詳細取得失敗: {len(collector.failed_ids)}")
        print(f"統合後件数: {len(merged_df):,}")
        
        # 生成されたファイルを確認
        print("\n=== 生成されたファイル ===")
        files_to_check = [
            'data/raw_data/MetObjects.csv',
            'data/raw_data/artwork_metadata_full.csv',
            'data/raw_data/checkpoint.json',
            'data/raw_data/failed_ids.json',
            'data/raw_data/qa_report.txt',
            'data/raw_data/qa_samples.csv'
        ]
        
        for file_path in files_to_check:
            path = Path(file_path)
            if path.exists():
                size = path.stat().st_size
                print(f"✓ {file_path} ({size:,} bytes)")
            else:
                print(f"✗ {file_path} (存在しません)")
        
    except Exception as e:
        logger.error(f"テストエラー: {e}")
        raise

if __name__ == "__main__":
    test_hybrid_collector()
