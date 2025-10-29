# -*- coding: utf-8 -*-
"""
画像ダウンロード完了確認スクリプト
"""

import os
import json
from pathlib import Path

def check_download_completion():
    """ダウンロード完了を確認"""
    print("=== 画像ダウンロード完了確認 ===")
    
    # 画像ディレクトリの確認
    images_dir = Path("data/filtered_data/paintings_images")
    
    if not images_dir.exists():
        print("❌ 画像ディレクトリが存在しません")
        return False
    
    # ダウンロード済み画像数
    downloaded_count = len([f for f in images_dir.iterdir() if f.is_file()])
    
    # 総数
    total_count = 5402
    
    # 進捗計算
    progress_percent = (downloaded_count / total_count) * 100
    
    print(f"総数: {total_count:,} 件")
    print(f"ダウンロード済み: {downloaded_count:,} 件")
    print(f"進捗率: {progress_percent:.1f}%")
    
    if downloaded_count >= total_count:
        print("\n✅ 画像ダウンロードが完了しました！")
        
        # ファイルサイズの確認
        total_size = sum(f.stat().st_size for f in images_dir.iterdir() if f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        print(f"総ファイルサイズ: {total_size_mb:.1f} MB")
        
        # ファイル形式の確認
        extensions = {}
        for f in images_dir.iterdir():
            if f.is_file():
                ext = f.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
        
        print("\nファイル形式分布:")
        for ext, count in sorted(extensions.items()):
            print(f"  {ext}: {count:,} 件")
        
        return True
    else:
        remaining = total_count - downloaded_count
        print(f"\n⏳ 残り {remaining:,} 件 ダウンロード中...")
        return False

def main():
    """メイン関数"""
    if check_download_completion():
        print("\n次のステップ: 機械学習パイプラインの実行準備が完了しました！")
        print("実行可能なコマンド:")
        print("  python main.py --mode features  # 特徴量抽出")
        print("  python main.py --mode train     # モデル学習")
        print("  python main.py --mode explain   # SHAP説明")
    else:
        print("\n画像ダウンロードがまだ完了していません。しばらく待ってから再実行してください。")

if __name__ == "__main__":
    main()
