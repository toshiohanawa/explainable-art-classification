import requests
import pandas as pd

def check_image_data():
    """MetObjects.csvの画像関連データを確認"""
    url = "https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv"
    
    print("画像関連データを確認中...")
    
    try:
        # サンプルデータを読み込み
        df_sample = pd.read_csv(url, nrows=1000)
        
        # 画像関連の列を探す
        image_related_cols = [col for col in df_sample.columns if 'image' in col.lower() or 'photo' in col.lower() or 'url' in col.lower()]
        
        print("画像関連の列:")
        for col in image_related_cols:
            print(f"  - {col}")
        
        # Object IDがある行のサンプルを確認
        sample_with_id = df_sample[df_sample['Object ID'].notna()].head(5)
        
        print(f"\nObject IDがある作品のサンプル:")
        print(sample_with_id[['Object ID', 'Object Name', 'Title', 'Artist Display Name', 'Is Public Domain']])
        
        # 絵画作品を確認
        painting_keywords = ['painting', 'canvas', 'oil', 'watercolor', 'tempera', 'acrylic']
        painting_mask = df_sample['Object Name'].str.contains('|'.join(painting_keywords), case=False, na=False)
        
        paintings = df_sample[painting_mask]
        print(f"\n絵画作品数（サンプル1000件中）: {len(paintings)}")
        
        if len(paintings) > 0:
            print("絵画作品のサンプル:")
            print(paintings[['Object ID', 'Object Name', 'Title', 'Artist Display Name', 'Is Public Domain']].head(3))
        
        # パブリックドメインの絵画
        public_paintings = paintings[paintings['Is Public Domain'] == True]
        print(f"パブリックドメインの絵画: {len(public_paintings)}")
        
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    check_image_data()
