import requests
import pandas as pd

def check_metobjects_columns():
    """MetObjects.csvの列名を確認"""
    url = "https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv"
    
    print("MetObjects.csvの列名を確認中...")
    
    try:
        # 最初の数行だけ読み込んで列名を確認
        df_sample = pd.read_csv(url, nrows=5)
        
        print("利用可能な列名:")
        for i, col in enumerate(df_sample.columns):
            print(f"{i+1:2d}. {col}")
        
        print(f"\n総列数: {len(df_sample.columns)}")
        
        # サンプルデータを表示
        print("\nサンプルデータ:")
        print(df_sample.head(2))
        
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    check_metobjects_columns()
