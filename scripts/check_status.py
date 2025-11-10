# -*- coding: utf-8 -*-
"""
ステータスチェックスクリプト
データ収集、画像ダウンロードなどの進捗状況を確認
"""

import sys
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_manager import ConfigManager
from src.utils.status_checker import StatusChecker


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='ステータスチェックスクリプト')
    parser.add_argument(
        '--type',
        choices=['collection', 'download', 'test', 'all'],
        default='all',
        help='チェックするタイプ'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='詳細情報を表示'
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # StatusChecker初期化
    data_config = config['data']
    checker = StatusChecker(
        data_dir=data_config['output_dir'],
        raw_data_dir=data_config.get('raw_data_dir'),
        filtered_data_dir=data_config.get('filtered_data_dir'),
        paintings_images_subdir=data_config.get('paintings_images_dir', 'paintings_images')
    )
    
    # チェックタイプに応じて実行
    if args.type in ['collection', 'all']:
        checker.print_collection_status()
        if args.type == 'all':
            print("\n" + "="*50 + "\n")
    
    if args.type in ['download', 'all']:
        checker.print_download_status(detailed=args.detailed)
        if args.type == 'all':
            print("\n" + "="*50 + "\n")
    
    if args.type in ['test', 'all']:
        checker.print_test_results()
    
    # 次のステップを表示
    if args.type in ['collection', 'all']:
        collection_status = checker.check_data_collection_status()
        if collection_status.get('is_complete'):
            print("\n次のステップ: python scripts/download_painting_images.py")
    
    if args.type in ['download', 'all']:
        download_status = checker.check_download_status()
        if download_status.get('is_complete'):
            print("\n次のステップ: 機械学習パイプラインの実行準備が完了しました！")
            print("実行可能なコマンド:")
            print("  python main.py --mode extract   # 特徴量抽出")
            print("  python main.py --mode train     # モデル学習")
            print("  python main.py --mode explain   # SHAP説明")


if __name__ == "__main__":
    main()
