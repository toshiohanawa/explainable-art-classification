# -*- coding: utf-8 -*-
"""
画像ダウンロード進捗確認スクリプト
"""

import os
from pathlib import Path

def check_download_progress():
    """ダウンロード進捗を確認"""
    print("=== 画像ダウンロード進捗確認 ===")
    
    # 画像ディレクトリの確認
    images_dir = Path("data/filtered_data/paintings_images")
    
    if not images_dir.exists():
        print("❌ 画像ディレクトリがまだ作成されていません")
        return
    
    # ダウンロード済み画像数
    downloaded_count = len([f for f in images_dir.iterdir() if f.is_file()])
    
    # 総数（ログから取得）
    total_count = 5402
    
    # 進捗計算
    progress_percent = (downloaded_count / total_count) * 100
    
    # 予想完了時間（10 req/s）
    remaining = total_count - downloaded_count
    estimated_minutes = remaining / 10 / 60
    
    print(f"総数: {total_count:,} 件")
    print(f"ダウンロード済み: {downloaded_count:,} 件")
    print(f"残り: {remaining:,} 件")
    print(f"進捗率: {progress_percent:.1f}%")
    print(f"予想完了時間: {estimated_minutes:.1f} 分")
    
    if downloaded_count > 0:
        print(f"画像保存先: {images_dir.absolute()}")
        
        # サンプルファイル名を表示
        sample_files = list(images_dir.iterdir())[:5]
        if sample_files:
            print("\nサンプルファイル:")
            for f in sample_files:
                print(f"  {f.name}")

def main():
    """メイン関数"""
    check_download_progress()

if __name__ == "__main__":
    main()
