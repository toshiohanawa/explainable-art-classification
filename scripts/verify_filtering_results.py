# -*- coding: utf-8 -*-
"""
フィルタリング結果の確認スクリプト
"""

import pandas as pd
from pathlib import Path

def verify_filtering_results():
    """フィルタリング結果を確認"""
    print("=== フィルタリング結果確認 ===\n")
    
    # データファイル確認
    data_file = Path("data/filtered_data/paintings_complete_dataset.csv")
    if data_file.exists():
        df = pd.read_csv(data_file, encoding='utf-8')
        print(f"📊 データファイル: {len(df):,}件")
        
        # Department分布確認
        if 'Department' in df.columns:
            dept_counts = df['Department'].value_counts()
            print("\nDepartment別件数:")
            for dept, count in dept_counts.items():
                print(f"  {dept}: {count:,}件")
            
            # European Paintingsのみか確認
            if len(dept_counts) == 1 and dept_counts.index[0] == 'European Paintings':
                print("\n✅ Department='European Paintings'のみです")
            else:
                print("\n⚠️  他のDepartmentも含まれています")
        else:
            print("⚠️  Department列が見つかりません")
    else:
        print("❌ データファイルが見つかりません")
    
    # 画像ファイル確認
    images_dir = Path("data/filtered_data/paintings_images")
    if images_dir.exists():
        image_files = [f for f in images_dir.iterdir() if f.is_file()]
        print(f"\n🖼️  画像ファイル: {len(image_files):,}件")
    else:
        print("\n❌ 画像ディレクトリが見つかりません")
    
    # バックアップ確認
    backup_file = Path("data/filtered_data/paintings_complete_dataset_backup.csv")
    if backup_file.exists():
        print(f"\n💾 バックアップファイル: 存在します")
        df_backup = pd.read_csv(backup_file, encoding='utf-8')
        print(f"   バックアップ件数: {len(df_backup):,}件")
    
    # レポート確認
    report_file = Path("docs") / "european_paintings_filter_report.txt"
    if report_file.exists():
        print(f"\n📄 フィルタリングレポート: {report_file}")
        print("   詳細はレポートファイルを確認してください")

def main():
    """メイン関数"""
    verify_filtering_results()

if __name__ == "__main__":
    main()

