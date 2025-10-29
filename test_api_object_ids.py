# -*- coding: utf-8 -*-
"""
API Object ID一覧取得のテストスクリプト
HybridCollectorで使用するIDが正しいか確認
"""

import requests
import json
import time
from typing import List, Optional

def get_api_object_ids() -> Optional[List[int]]:
    """APIから全Object ID一覧を取得"""
    base_url = "https://collectionapi.metmuseum.org/public/collection/v1"
    
    print("API Object ID一覧を取得中...")
    
    try:
        # レート制限を考慮して待機
        time.sleep(1)
        
        response = requests.get(f"{base_url}/objects", timeout=30)
        response.raise_for_status()
        
        data = response.json()
        object_ids = data.get('objectIDs', [])
        
        print(f"✓ API Object ID取得成功")
        print(f"  総数: {data.get('total', 0):,}件")
        print(f"  取得ID数: {len(object_ids):,}件")
        
        return object_ids
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTPエラー: {e}")
        print(f"  ステータスコード: {e.response.status_code}")
        return None
    except Exception as e:
        print(f"✗ エラー: {e}")
        return None

def get_csv_object_ids() -> List[int]:
    """CSVからObject ID一覧を取得"""
    import pandas as pd
    
    print("CSV Object ID一覧を取得中...")
    
    try:
        df = pd.read_csv('data/raw_data/MetObjects.csv', low_memory=False)
        
        # Object ID列を探す（複数の可能性）
        object_id_columns = ['Object ID', 'Object_ID', 'objectID', 'object_id']
        object_id_col = None
        
        for col in object_id_columns:
            if col in df.columns:
                object_id_col = col
                break
        
        if object_id_col is None:
            print("✗ Object ID列が見つかりません")
            print(f"  利用可能な列: {list(df.columns)[:10]}...")
            return []
        
        object_ids = df[object_id_col].dropna().astype(int).tolist()
        
        print(f"✓ CSV Object ID取得成功")
        print(f"  列名: {object_id_col}")
        print(f"  総数: {len(df):,}件")
        print(f"  Object ID数: {len(object_ids):,}件")
        print(f"  最初の10個: {object_ids[:10]}")
        print(f"  最後の10個: {object_ids[-10:]}")
        
        return object_ids
        
    except Exception as e:
        print(f"✗ CSV読み込みエラー: {e}")
        return []

def compare_object_ids(api_ids: List[int], csv_ids: List[int]):
    """APIとCSVのObject IDを比較"""
    print("\n=== Object ID比較分析 ===")
    
    api_set = set(api_ids)
    csv_set = set(csv_ids)
    
    print(f"API Object ID数: {len(api_set):,}")
    print(f"CSV Object ID数: {len(csv_set):,}")
    
    # 共通部分
    common_ids = api_set & csv_set
    print(f"共通ID数: {len(common_ids):,}")
    
    # APIのみにあるID
    api_only = api_set - csv_set
    print(f"APIのみのID数: {len(api_only):,}")
    if api_only:
        print(f"  APIのみの例: {sorted(list(api_only))[:10]}")
    
    # CSVのみにあるID
    csv_only = csv_set - api_set
    print(f"CSVのみのID数: {len(csv_only):,}")
    if csv_only:
        print(f"  CSVのみの例: {sorted(list(csv_only))[:10]}")
    
    # 統合ID（和集合）
    all_ids = api_set | csv_set
    print(f"統合ID数: {len(all_ids):,}")
    
    return {
        'api_ids': api_ids,
        'csv_ids': csv_ids,
        'common_ids': list(common_ids),
        'api_only': list(api_only),
        'csv_only': list(csv_only),
        'all_ids': list(all_ids)
    }

def test_sample_object_details(sample_ids: List[int], max_samples: int = 5):
    """サンプルObject IDの詳細を取得してテスト"""
    base_url = "https://collectionapi.metmuseum.org/public/collection/v1"
    
    print(f"\n=== サンプルObject詳細テスト（{max_samples}件） ===")
    
    success_count = 0
    failed_count = 0
    
    for i, object_id in enumerate(sample_ids[:max_samples]):
        print(f"\n{i+1}. Object ID {object_id} をテスト中...")
        
        try:
            time.sleep(0.1)  # レート制限
            
            response = requests.get(f"{base_url}/objects/{object_id}", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ 成功")
                print(f"    Title: {data.get('title', 'N/A')[:50]}...")
                print(f"    Artist: {data.get('artistDisplayName', 'N/A')}")
                print(f"    Department: {data.get('department', 'N/A')}")
                print(f"    Primary Image: {'あり' if data.get('primaryImage') else 'なし'}")
                success_count += 1
            elif response.status_code == 404:
                print(f"  ✗ 404 - データが存在しません")
                failed_count += 1
            elif response.status_code == 403:
                print(f"  ✗ 403 - アクセス拒否")
                failed_count += 1
            else:
                print(f"  ✗ HTTP {response.status_code}")
                failed_count += 1
                
        except Exception as e:
            print(f"  ✗ エラー: {e}")
            failed_count += 1
    
    print(f"\nサンプルテスト結果:")
    print(f"  成功: {success_count}件")
    print(f"  失敗: {failed_count}件")
    print(f"  成功率: {success_count/(success_count+failed_count)*100:.1f}%")

def main():
    """メイン実行関数"""
    print("=== API Object ID取得テスト ===\n")
    
    # 1. APIからObject ID一覧を取得
    api_ids = get_api_object_ids()
    
    # 2. CSVからObject ID一覧を取得
    csv_ids = get_csv_object_ids()
    
    if not api_ids and not csv_ids:
        print("✗ APIとCSVの両方でエラーが発生しました")
        return
    
    # 3. Object IDを比較
    if api_ids and csv_ids:
        comparison = compare_object_ids(api_ids, csv_ids)
        
        # 4. サンプルObject詳細をテスト
        if comparison['all_ids']:
            test_sample_object_details(comparison['all_ids'])
        
        # 5. 結果をファイルに保存
        with open('object_id_analysis.json', 'w', encoding='utf-8') as f:
            json.dump({
                'api_total': len(api_ids) if api_ids else 0,
                'csv_total': len(csv_ids) if csv_ids else 0,
                'common_total': len(comparison['common_ids']),
                'api_only_total': len(comparison['api_only']),
                'csv_only_total': len(comparison['csv_only']),
                'all_total': len(comparison['all_ids']),
                'sample_api_ids': api_ids[:100] if api_ids else [],
                'sample_csv_ids': csv_ids[:100] if csv_ids else [],
                'sample_common_ids': comparison['common_ids'][:100],
                'sample_api_only': comparison['api_only'][:100],
                'sample_csv_only': comparison['csv_only'][:100]
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 分析結果を object_id_analysis.json に保存しました")

if __name__ == "__main__":
    main()
