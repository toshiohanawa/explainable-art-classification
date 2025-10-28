# -*- coding: utf-8 -*-
"""
API Object ID取得のテストスクリプト
"""

import requests
import json
import time

def test_api_objects():
    """API Object ID取得をテスト"""
    base_url = "https://collectionapi.metmuseum.org/public/collection/v1"
    
    print("API Object ID取得をテスト中...")
    
    try:
        # レート制限を考慮して1秒待機
        time.sleep(1)
        
        # ヘッダーを追加してリクエスト
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(f"{base_url}/objects", headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        object_ids = data.get('objectIDs', [])
        
        print(f"✓ API Object ID取得成功")
        print(f"  総数: {data.get('total', 0):,}件")
        print(f"  取得ID数: {len(object_ids):,}件")
        print(f"  最初の10個: {object_ids[:10]}")
        print(f"  最後の10個: {object_ids[-10:]}")
        
        # サンプルIDで詳細取得をテスト
        if object_ids:
            sample_id = object_ids[100]  # 100番目のIDをテスト
            print(f"\nサンプルID {sample_id} の詳細を取得中...")
            
            time.sleep(1)  # レート制限
            
            detail_response = requests.get(f"{base_url}/objects/{sample_id}", timeout=30)
            if detail_response.status_code == 200:
                detail_data = detail_response.json()
                print(f"✓ 詳細取得成功")
                print(f"  Title: {detail_data.get('title', 'N/A')}")
                print(f"  Artist: {detail_data.get('artistDisplayName', 'N/A')}")
                print(f"  Department: {detail_data.get('department', 'N/A')}")
                print(f"  Primary Image: {'あり' if detail_data.get('primaryImage') else 'なし'}")
            else:
                print(f"✗ 詳細取得失敗: {detail_response.status_code}")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTPエラー: {e}")
        print(f"  ステータスコード: {e.response.status_code}")
        return False
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False

if __name__ == "__main__":
    test_api_objects()
