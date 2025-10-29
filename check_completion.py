# -*- coding: utf-8 -*-
"""
全件データ収集の完了確認スクリプト
"""

import sys
import os
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def check_completion():
    """完了状況を確認"""
    print("=== 全件データ収集完了確認 ===")
    
    # チェックポイントファイルの確認
    checkpoint_file = Path("data/raw_data/checkpoint.json")
    if checkpoint_file.exists():
        import json
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        
        processed = len(checkpoint.get('processed_ids', []))
        failed = len(checkpoint.get('failed_ids', []))
        success = checkpoint.get('api_data_count', 0)
        
        print(f"処理済み: {processed:,}件")
        print(f"失敗: {failed:,}件")
        print(f"成功: {success:,}件")
        print(f"進捗率: {(processed / 9005) * 100:.1f}%")
        print(f"成功率: {(success / (processed + failed)) * 100:.1f}%")
        
        if processed >= 9005:
            print("\n✅ 全件データ収集が完了しました！")
            return True
        else:
            print(f"\n⏳ 残り {9005 - processed:,}件 処理中...")
            return False
    else:
        print("❌ チェックポイントファイルが見つかりません")
        return False

def main():
    """メイン関数"""
    if check_completion():
        print("\n次のステップ: 画像ダウンロードを実行してください")
        print("コマンド: python download_painting_images.py")
    else:
        print("\n全件データ収集がまだ完了していません。しばらく待ってから再実行してください。")

if __name__ == "__main__":
    main()
