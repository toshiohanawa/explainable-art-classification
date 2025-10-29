# -*- coding: utf-8 -*-
"""
ブラウザで成功したURLと全く同じURLでテスト
"""

import requests
import time
import json
from datetime import datetime

def test_exact_urls():
    """ブラウザで成功したURLを直接テスト"""
    
    # ブラウザに近いヘッダー
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9,ja;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Referer': 'https://www.metmuseum.org/',
        'Origin': 'https://www.metmuseum.org',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'cross-site',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"'
    }
    
    # セッション管理
    session = requests.Session()
    
    print(f"=== ブラウザ成功URLテスト開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    # テスト1: /objects/1（ブラウザで成功したURL）
    print("1. Object ID 1の詳細をテスト中...")
    print("   URL: https://collectionapi.metmuseum.org/public/collection/v1/objects/1")
    try:
        start_time = time.time()
        response = session.get(
            "https://collectionapi.metmuseum.org/public/collection/v1/objects/1",
            headers=headers,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        print(f"   レスポンス時間: {elapsed:.2f}秒")
        print(f"   ステータスコード: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ 成功!")
            print(f"   Object ID: {data.get('objectID')}")
            print(f"   Title: {data.get('title')}")
            print(f"   Department: {data.get('department')}")
            print(f"   Artist: {data.get('artistDisplayName')}")
        else:
            print(f"   ✗ 失敗: HTTP {response.status_code}")
            print(f"   レスポンス: {response.text[:200]}")
    except Exception as e:
        print(f"   ✗ エラー: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # テスト2: /objects（ブラウザで成功したURL）
    print("2. 全Object ID一覧をテスト中...")
    print("   URL: https://collectionapi.metmuseum.org/public/collection/v1/objects")
    print("   注意: このエンドポイントは時間がかかります（ブラウザでも確認済み）")
    try:
        time.sleep(5)  # 5秒待機
        start_time = time.time()
        response = session.get(
            "https://collectionapi.metmuseum.org/public/collection/v1/objects",
            headers=headers,
            timeout=120  # タイムアウトを120秒に延長
        )
        elapsed = time.time() - start_time
        
        print(f"   レスポンス時間: {elapsed:.2f}秒")
        print(f"   ステータスコード: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ 成功!")
            print(f"   総件数: {data.get('total', 0):,}件")
            print(f"   Object ID数: {len(data.get('objectIDs', [])):,}件")
            print(f"   最初の10件: {data.get('objectIDs', [])[:10]}")
        else:
            print(f"   ✗ 失敗: HTTP {response.status_code}")
            print(f"   レスポンス: {response.text[:200]}")
    except requests.exceptions.Timeout:
        print(f"   ✗ タイムアウト: 120秒以内にレスポンスが返ってきませんでした")
    except Exception as e:
        print(f"   ✗ エラー: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # 結果サマリー
    print("=== テスト完了 ===")
    print("ブラウザで成功したURLと全く同じURLでテストしました。")
    print("403エラーが継続する場合、以下の可能性があります：")
    print("1. IPアドレス単位でのWAFブロック（時間経過で自動解除）")
    print("2. ブラウザとプログラムの微妙なヘッダー差異")
    print("3. セッションCookieの有無")

if __name__ == "__main__":
    test_exact_urls()

