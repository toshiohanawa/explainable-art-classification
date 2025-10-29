# -*- coding: utf-8 -*-
"""
WAF遮断回復テストスクリプト
安全なレート制限でAPI接続をテスト
"""

import requests
import time
import json
from datetime import datetime

def test_api_connection():
    """API接続を安全にテスト"""
    base_url = "https://collectionapi.metmuseum.org/public/collection/v1"
    
    # ブラウザに近いヘッダー
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9,ja;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'cross-site',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"'
    }
    
    # セッション管理
    session = requests.Session()
    
    print(f"=== WAF回復テスト開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # 1. 部門一覧をテスト（軽量なエンドポイント）
    print("\n1. 部門一覧をテスト中...")
    try:
        time.sleep(2)  # 安全な間隔
        response = session.get(f"{base_url}/departments", headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 部門一覧取得成功: {len(data.get('departments', []))}部門")
            for dept in data.get('departments', [])[:5]:
                print(f"  - {dept.get('departmentId')}: {dept.get('displayName')}")
        else:
            print(f"✗ 部門一覧取得失敗: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ 部門一覧取得エラー: {e}")
        return False
    
    # 2. Object ID一覧をテスト（制限されたレートで）
    print("\n2. Object ID一覧をテスト中...")
    try:
        time.sleep(5)  # より長い間隔
        response = session.get(f"{base_url}/objects", headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Object ID一覧取得成功: {len(data.get('objectIDs', []))}件")
            print(f"  総数: {data.get('total', 0):,}件")
        elif response.status_code == 403:
            print("✗ まだWAF遮断中 (Error 15)")
            return False
        else:
            print(f"✗ Object ID一覧取得失敗: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Object ID一覧取得エラー: {e}")
        return False
    
    # 3. サンプルObject詳細をテスト
    print("\n3. サンプルObject詳細をテスト中...")
    try:
        time.sleep(10)  # さらに長い間隔
        
        # 小さいIDからテスト（存在する可能性が高い）
        test_ids = [1, 2, 3, 4, 5]
        
        for object_id in test_ids:
            print(f"  Object ID {object_id} をテスト中...")
            time.sleep(2)  # 各リクエスト間に2秒間隔
            
            response = session.get(f"{base_url}/objects/{object_id}", headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"    ✓ 成功: {data.get('title', 'N/A')[:30]}...")
                break
            elif response.status_code == 404:
                print(f"    - 404: データが存在しません")
            elif response.status_code == 403:
                print(f"    ✗ 403: まだWAF遮断中")
                return False
            else:
                print(f"    ✗ HTTP {response.status_code}")
                
    except Exception as e:
        print(f"✗ Object詳細取得エラー: {e}")
        return False
    
    print(f"\n✓ WAF回復テスト完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return True

def main():
    """メイン実行"""
    print("WAF遮断回復テストを開始します...")
    print("注意: 安全なレート制限（10 req/s以下）で実行します")
    
    # 現在の遮断時刻から経過時間を計算
    block_time = datetime(2025, 10, 29, 13, 34, 40)  # 遮断時刻
    current_time = datetime.now()
    elapsed = (current_time - block_time).total_seconds() / 60
    
    print(f"遮断からの経過時間: {elapsed:.1f}分")
    
    if elapsed < 10:
        print("⚠️  まだクールダウン期間中です。10分以上経過してから再試行してください。")
        return
    
    success = test_api_connection()
    
    if success:
        print("\n🎉 API接続が回復しました！HybridCollectorを実行できます。")
    else:
        print("\n❌ まだWAF遮断中です。さらに時間を置いてから再試行してください。")

if __name__ == "__main__":
    main()
