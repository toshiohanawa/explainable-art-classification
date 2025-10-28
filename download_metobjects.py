import requests
import pandas as pd
from pathlib import Path
import logging

def download_metobjects_csv():
    """
    GitHubからMetObjects.csvをダウンロードして、絵画作品のみをフィルタリング
    """
    # GitHubのMetObjects.csvのURL（Metropolitan Museum公式）
    url = "https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv"
    
    print("MetObjects.csvをダウンロード中...")
    
    try:
        # CSVファイルをダウンロード
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # データフレームに読み込み
        print("CSVファイルを読み込み中...")
        df = pd.read_csv(url, low_memory=False)
        
        print(f"総作品数: {len(df)}")
        
        # 絵画作品のみをフィルタリング
        print("絵画作品をフィルタリング中...")
        painting_keywords = ['painting', 'canvas', 'oil', 'watercolor', 'tempera', 'acrylic']
        
        # Object Nameまたはmediumに絵画関連のキーワードが含まれる作品を抽出
        painting_mask = df['Object Name'].str.contains('|'.join(painting_keywords), case=False, na=False) | \
                       df['Medium'].str.contains('|'.join(painting_keywords), case=False, na=False)
        
        paintings_df = df[painting_mask].copy()
        
        print(f"絵画作品数: {len(paintings_df)}")
        
        # パブリックドメインの作品のみを抽出
        public_domain_df = paintings_df[paintings_df['Is Public Domain'] == True].copy()
        
        print(f"パブリックドメインの絵画作品数: {len(public_domain_df)}")
        
        # 画像URLがある作品のみを抽出
        with_images_df = public_domain_df[public_domain_df['Object Primary Image'].notna()].copy()
        
        print(f"画像付きパブリックドメイン絵画作品数: {len(with_images_df)}")
        
        # 出力ディレクトリを作成
        output_dir = Path("data/raw_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSVファイルとして保存
        output_file = output_dir / "artwork_metadata.csv"
        
        # 必要な列のみを選択してリネーム（既存のコードとの互換性のため）
        columns_mapping = {
            'Object ID': 'object_id',
            'Is Highlight': 'isHighlight',
            'Accession Number': 'accessionNumber',
            'Accession Year': 'accessionYear',
            'Is Public Domain': 'isPublicDomain',
            'Object Primary Image': 'primaryImage',
            'Object Name': 'objectName',
            'Title': 'title',
            'Culture': 'culture',
            'Period': 'period',
            'Dynasty': 'dynasty',
            'Reign': 'reign',
            'Portfolio': 'portfolio',
            'Artist Role': 'artistRole',
            'Artist Prefix': 'artistPrefix',
            'Artist Display Name': 'artistDisplayName',
            'Artist Display Bio': 'artistDisplayBio',
            'Artist Suffix': 'artistSuffix',
            'Artist Alpha Sort': 'artistAlphaSort',
            'Artist Nationality': 'artistNationality',
            'Artist Begin Date': 'artistBeginDate',
            'Artist End Date': 'artistEndDate',
            'Artist Gender': 'artistGender',
            'Artist ULAN URL': 'artistULAN_URL',
            'Artist Wikidata URL': 'artistWikidata_URL',
            'Object Date': 'objectDate',
            'Object Begin Date': 'objectBeginDate',
            'Object End Date': 'objectEndDate',
            'Medium': 'medium',
            'Dimensions': 'dimensions',
            'Department': 'department',
            'Classification': 'classification',
            'Credit Line': 'creditLine',
            'Geography Type': 'geographyType',
            'City': 'city',
            'State': 'state',
            'County': 'county',
            'Country': 'country',
            'Region': 'region',
            'Subregion': 'subregion',
            'Locale': 'locale',
            'Locus': 'locus',
            'Excavation': 'excavation',
            'River': 'river',
            'Link Resource': 'linkResource',
            'Object URL': 'objectURL',
            'Object Wikidata URL': 'objectWikidata_URL',
            'Metadata Date': 'metadataDate',
            'Repository': 'repository',
            'Tags': 'tags'
        }
        
        # 利用可能な列のみを選択
        available_columns = {k: v for k, v in columns_mapping.items() if k in with_images_df.columns}
        
        final_df = with_images_df[list(available_columns.keys())].rename(columns=available_columns)
        
        # primaryImageSmallを追加（primaryImageと同じ値）
        final_df['primaryImageSmall'] = final_df['primaryImage']
        
        # image_downloadedフラグを追加
        final_df['image_downloaded'] = False
        
        # 保存
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ 成功: {len(final_df)}件の絵画作品データを保存しました")
        print(f"📁 保存先: {output_file}")
        
        # サンプルデータを表示
        print("\n📊 サンプルデータ:")
        print(final_df[['object_id', 'title', 'artistDisplayName', 'objectDate', 'department']].head())
        
        return final_df
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        raise

if __name__ == "__main__":
    download_metobjects_csv()
