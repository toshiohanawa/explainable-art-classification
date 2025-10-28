"""
Metropolitan Museum API クライアント
APIからの作品データと画像の取得を行う
"""

import requests
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from PIL import Image
import io

from ..utils.config_manager import ConfigManager
from ..utils.timestamp_manager import TimestampManager


class MetAPIClient:
    """Metropolitan Museum API クライアント"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.api_config = config['api']
        self.data_config = config['data']
        self.base_url = self.api_config['base_url']
        self.rate_limit = self.api_config['rate_limit']
        self.timeout = self.api_config['timeout']
        
        # 生データは固定ディレクトリに保存
        self.output_dir = Path(self.data_config['output_dir'])
        self.images_dir = self.output_dir / self.data_config['images_dir']
        self.raw_data_dir = self.output_dir / 'raw_data'
        
        # ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # レート制限用
        self.last_request_time = 0
    
    def _rate_limit_delay(self):
        """レート制限を考慮した待機"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def get_objects(self, department_ids: Optional[List[int]] = None) -> List[int]:
        """
        作品IDのリストを取得
        
        Args:
            department_ids: 部門IDのリスト（Noneの場合は全部門）
            
        Returns:
            作品IDのリスト
        """
        self._rate_limit_delay()
        
        url = f"{self.base_url}/objects"
        params = {}
        
        if department_ids:
            params['departmentIds'] = '|'.join(map(str, department_ids))
        
        max_retries = 3
        retry_delay = 5  # 秒
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                return data.get('objectIDs', [])
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"403エラー発生、{retry_delay}秒待機後に再試行... (試行 {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数バックオフ
                        continue
                    else:
                        self.logger.error(f"作品ID取得エラー: 最大再試行回数に達しました - {e}")
                        raise
                else:
                    self.logger.error(f"作品ID取得エラー: {e}")
                    raise
            except requests.RequestException as e:
                self.logger.error(f"作品ID取得エラー: {e}")
                raise
    
    def get_object_details(self, object_id: int) -> Optional[Dict[str, Any]]:
        """
        特定作品の詳細情報を取得
        
        Args:
            object_id: 作品ID
            
        Returns:
            作品詳細情報の辞書（取得失敗時はNone）
        """
        self._rate_limit_delay()
        
        url = f"{self.base_url}/objects/{object_id}"
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.warning(f"作品詳細取得エラー (ID: {object_id}): {e}")
            return None
    
    def download_image(self, image_url: str, object_id: int) -> bool:
        """
        画像をダウンロード
        
        Args:
            image_url: 画像URL
            object_id: 作品ID
            
        Returns:
            ダウンロード成功時True
        """
        if not image_url:
            return False
        
        self._rate_limit_delay()
        
        try:
            response = requests.get(image_url, timeout=self.timeout)
            response.raise_for_status()
            
            # 画像を開いて保存
            image = Image.open(io.BytesIO(response.content))
            
            # ファイル名を生成
            filename = f"{object_id}.jpg"
            filepath = self.images_dir / filename
            
            # 画像を保存
            image.save(filepath, 'JPEG', quality=95)
            
            self.logger.debug(f"画像保存完了: {filepath}")
            return True
            
        except Exception as e:
            self.logger.warning(f"画像ダウンロードエラー (ID: {object_id}): {e}")
            return False
    
    def collect_metadata(self, 
                        department_ids: Optional[List[int]] = None,
                        max_objects: Optional[int] = None) -> pd.DataFrame:
        """
        作品メタデータを収集（画像ダウンロードなし）
        
        Args:
            department_ids: 対象部門IDのリスト（絵画部門を指定）
            max_objects: 最大収集作品数
            
        Returns:
            メタデータのDataFrame
        """
        self.logger.info("作品メタデータ収集を開始...")
        
        # 全部門から取得（絵画作品は後でフィルタリング）
        if department_ids is None:
            department_ids = None  # 全部門から取得
        
        # 作品IDを取得
        object_ids = self.get_objects(department_ids)
        self.logger.info(f"取得作品数: {len(object_ids)}")
        
        if max_objects:
            object_ids = object_ids[:max_objects]
            self.logger.info(f"制限後作品数: {len(object_ids)}")
        
        # 作品詳細を取得
        artworks_data = []
        filtered_count = 0
        
        for i, object_id in enumerate(object_ids):
            if i % 100 == 0:
                self.logger.info(f"進捗: {i}/{len(object_ids)}")
            
            # 作品詳細を取得
            artwork = self.get_object_details(object_id)
            if not artwork:
                continue
            
            # 絵画作品のみをフィルタリング
            object_name = artwork.get('objectName', '').lower()
            if not any(painting_type in object_name for painting_type in ['painting', 'canvas', 'oil', 'watercolor', 'tempera']):
                filtered_count += 1
                continue
            
            # すべての利用可能な項目を抽出
            artwork_info = {
                # 基本識別情報
                'object_id': object_id,
                'objectID': artwork.get('objectID', ''),
                'isHighlight': artwork.get('isHighlight', False),
                'accessionNumber': artwork.get('accessionNumber', ''),
                'accessionYear': artwork.get('accessionYear', ''),
                'isPublicDomain': artwork.get('isPublicDomain', False),
                
                # 画像情報
                'primaryImage': artwork.get('primaryImage', ''),
                'primaryImageSmall': artwork.get('primaryImageSmall', ''),
                'additionalImages': artwork.get('additionalImages', []),
                
                # 作品基本情報
                'objectName': artwork.get('objectName', ''),
                'title': artwork.get('title', ''),
                'culture': artwork.get('culture', ''),
                'period': artwork.get('period', ''),
                'dynasty': artwork.get('dynasty', ''),
                'reign': artwork.get('reign', ''),
                'portfolio': artwork.get('portfolio', ''),
                
                # アーティスト情報
                'constituents': artwork.get('constituents', []),
                'artistRole': artwork.get('artistRole', ''),
                'artistPrefix': artwork.get('artistPrefix', ''),
                'artistDisplayName': artwork.get('artistDisplayName', ''),
                'artistDisplayBio': artwork.get('artistDisplayBio', ''),
                'artistSuffix': artwork.get('artistSuffix', ''),
                'artistAlphaSort': artwork.get('artistAlphaSort', ''),
                'artistNationality': artwork.get('artistNationality', ''),
                'artistBeginDate': artwork.get('artistBeginDate', ''),
                'artistEndDate': artwork.get('artistEndDate', ''),
                'artistGender': artwork.get('artistGender', ''),
                'artistWikidata_URL': artwork.get('artistWikidata_URL', ''),
                'artistULAN_URL': artwork.get('artistULAN_URL', ''),
                
                # 作品日付情報
                'objectDate': artwork.get('objectDate', ''),
                'objectBeginDate': artwork.get('objectBeginDate', ''),
                'objectEndDate': artwork.get('objectEndDate', ''),
                
                # 物理情報
                'medium': artwork.get('medium', ''),
                'dimensions': artwork.get('dimensions', ''),
                'measurements': artwork.get('measurements', []),
                
                # 部門・分類情報
                'department': artwork.get('department', ''),
                'classification': artwork.get('classification', ''),
                
                # クレジット・権利情報
                'creditLine': artwork.get('creditLine', ''),
                'rightsAndReproduction': artwork.get('rightsAndReproduction', ''),
                
                # 地理情報
                'geographyType': artwork.get('geographyType', ''),
                'city': artwork.get('city', ''),
                'state': artwork.get('state', ''),
                'county': artwork.get('county', ''),
                'country': artwork.get('country', ''),
                'region': artwork.get('region', ''),
                'subregion': artwork.get('subregion', ''),
                'locale': artwork.get('locale', ''),
                'locus': artwork.get('locus', ''),
                'excavation': artwork.get('excavation', ''),
                'river': artwork.get('river', ''),
                
                # リンク・リソース情報
                'linkResource': artwork.get('linkResource', ''),
                'objectURL': artwork.get('objectURL', ''),
                'objectWikidata_URL': artwork.get('objectWikidata_URL', ''),
                
                # メタデータ
                'metadataDate': artwork.get('metadataDate', ''),
                'repository': artwork.get('repository', ''),
                
                # タグ・その他
                'tags': [tag.get('term', '') for tag in (artwork.get('tags') or [])],
                'isTimelineWork': artwork.get('isTimelineWork', False),
                'GalleryNumber': artwork.get('GalleryNumber', ''),
                
                # 画像ダウンロード状況（初期値）
                'image_downloaded': False
            }
            
            artworks_data.append(artwork_info)
        
        # データフレームに変換
        df = pd.DataFrame(artworks_data)
        
        # リストや辞書のデータを文字列に変換（CSV保存用）
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
        
        # メタデータを保存
        metadata_file = self.raw_data_dir / self.data_config['metadata_file']
        df.to_csv(metadata_file, index=False, encoding='utf-8')
        
        self.logger.info(f"メタデータ収集完了:")
        self.logger.info(f"  - 総作品数: {len(artworks_data)}")
        self.logger.info(f"  - フィルタリング除外: {filtered_count}")
        self.logger.info(f"  - メタデータ保存先: {metadata_file}")
        
        return df
    
    def download_images(self, 
                       df: pd.DataFrame,
                       max_images: Optional[int] = None) -> pd.DataFrame:
        """
        メタデータから対象を絞り込んで画像をダウンロード
        
        Args:
            df: メタデータのDataFrame
            max_images: 最大ダウンロード画像数
            
        Returns:
            画像ダウンロード状況を更新したDataFrame
        """
        self.logger.info("画像ダウンロードを開始...")
        
        # 画像ダウンロード対象を絞り込み（primaryImageSmallを使用）
        target_df = df[
            (df['primaryImageSmall'] != '') & 
            (df['isPublicDomain'] == True)
        ].copy()
        
        if max_images:
            target_df = target_df.head(max_images)
        
        self.logger.info(f"画像ダウンロード対象: {len(target_df)}件")
        
        successful_downloads = 0
        
        for i, (_, row) in enumerate(target_df.iterrows()):
            if i % 50 == 0:
                self.logger.info(f"画像ダウンロード進捗: {i}/{len(target_df)}")
            
            object_id = row['object_id']
            image_url = row['primaryImageSmall']
            
            if self.download_image(image_url, object_id):
                successful_downloads += 1
                # DataFrameの該当行を更新
                df.loc[df['object_id'] == object_id, 'image_downloaded'] = True
        
        # 更新されたDataFrameを保存
        metadata_file = self.raw_data_dir / self.data_config['metadata_file']
        df.to_csv(metadata_file, index=False, encoding='utf-8')
        
        self.logger.info(f"画像ダウンロード完了:")
        self.logger.info(f"  - 対象作品数: {len(target_df)}")
        self.logger.info(f"  - 成功ダウンロード: {successful_downloads}")
        self.logger.info(f"  - 成功率: {successful_downloads/len(target_df)*100:.1f}%")
        
        return df
    
    def collect_artwork_data(self, 
                           department_ids: Optional[List[int]] = None,
                           max_objects: Optional[int] = None,
                           max_images: Optional[int] = None) -> None:
        """
        作品データを収集（メタデータ + 画像）
        
        Args:
            department_ids: 対象部門IDのリスト（絵画部門を指定）
            max_objects: 最大収集作品数
            max_images: 最大ダウンロード画像数
        """
        # 1. メタデータを収集
        df = self.collect_metadata(department_ids, max_objects)
        
        # 2. 画像をダウンロード
        self.download_images(df, max_images)
    
    def get_departments(self) -> List[Dict[str, Any]]:
        """
        部門一覧を取得
        
        Returns:
            部門情報のリスト
        """
        self._rate_limit_delay()
        
        url = f"{self.base_url}/departments"
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('departments', [])
        except requests.RequestException as e:
            self.logger.error(f"部門一覧取得エラー: {e}")
            raise
