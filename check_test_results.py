# -*- coding: utf-8 -*-
"""
テスト結果の確認スクリプト
"""

import pandas as pd

def check_test_results():
    """テスト結果を確認"""
    df = pd.read_csv('data/filtered_data/test_paintings_200.csv')
    
    print("=== テスト結果確認 ===")
    print(f"総行数: {len(df)}")
    print(f"画像URL有り: {df['primaryImage'].notna().sum()}")
    print(f"タグ有り: {df['tags'].notna().sum()}")
    print(f"測定値有り: {df['measurements'].notna().sum()}")
    print(f"constituents有り: {df['constituents'].notna().sum()}")
    
    # 文化圏の分布
    print("\n=== 文化圏分布 ===")
    culture_counts = df['Culture'].value_counts()
    for culture, count in culture_counts.head(10).items():
        print(f"{culture}: {count}件")
    
    # Medium分布
    print("\n=== Medium分布（上位10件） ===")
    medium_counts = df['Medium'].value_counts()
    for medium, count in medium_counts.head(10).items():
        print(f"{medium}: {count}件")

if __name__ == "__main__":
    check_test_results()
