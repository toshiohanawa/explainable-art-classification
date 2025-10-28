# -*- coding: utf-8 -*-
"""
ハイブリッドデータ収集クラス
MetObjects.csvとAPIを組み合わせて最大限のメタデータを取得
"""

import requests
import pandas as pd
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import hashlib

from ..utils.config_manager import ConfigManager


class HybridCollector:
    """ハイブリッドデータ収集クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.api_config = config['api']
        self.data_config = config['data']
        self.hybrid_config = config.get('hybrid_collection', {})
        
        self.base_url = self.api_config['base_url']
        self.rate_limit = self.api_config['rate_limit']
        self.timeout = self.api_config['timeout']
        
        # ディレクトリ設定
        self.raw_data_dir = Path(self.data_config['output_dir']) / 'raw_data'
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイルパス
        self.csv_file = self.raw_data_dir / 'MetObjects.csv'
        self.output_file = self.raw_data_dir / 'artwork_metadata_full.csv'
        self.checkpoint_file = self.raw_data_dir / 'checkpoint.json'
        self.failed_ids_file = self.raw_data_dir / 'failed_ids.json'
        self.qa_report_file = self.raw_data_dir / 'qa_report.txt'
        self.qa_samples_file = self.raw_data_dir / 'qa_samples.csv'
        
        # 設定値
        self.checkpoint_interval = self.hybrid_config.get('checkpoint_interval', 1000)
        self.max_retries = self.hybrid_config.get('max_retries', 3)
        self.retry_base_delay = self.hybrid_config.get('retry_base_delay', 5)
        self.api_fields = self.hybrid_config.get('api_fields', [
            'primaryImage', 'primaryImageSmall', 'additionalImages',
            'constituents', 'tags', 'measurements', 'galleryNumber', 'objectWikidata_URL'
        ])
        
        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0
        
        # 進捗管理
        self.processed_ids: Set[int] = set()
        self.failed_ids: Set[int] = set()
        self.api_data: Dict[int, Dict[str, Any]] = {}
        
    def _rate_limit_delay(self):
        """レート制限を考慮した待機"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def download_csv(self) -> pd.DataFrame:
        """
        GitHubからMetObjects.csvをダウンロード
        
        Returns:
            ダウンロードしたCSVのDataFrame
        """
        csv_url = self.hybrid_config.get('csv_url', 
            'https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv')
        
        self.logger.info(f"MetObjects.csvをダウンロード中: {csv_url}")
        
        try:
            response = requests.get(csv_url, stream=True)
            response.raise_for_status()
            
            # CSVを読み込み
            df = pd.read_csv(csv_url, low_memory=False)
            
            # ファイルに保存
            df.to_csv(self.csv_file, index=False, encoding='utf-8')
            
            self.logger.info(f"CSVダウンロード完了: {len(df)}件")
            return df
            
        except Exception as e:
            self.logger.error(f"CSVダウンロードエラー: {e}")
            raise
    
    def preprocess_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CSVデータの前処理
        
        Args:
            df: 元のDataFrame
            
        Returns:
            前処理済みDataFrame
        """
        self.logger.info("CSVデータの前処理を開始...")
        
        # 列名の正規化（スペースをアンダースコアに）
        df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        
        # Object IDを整数に変換
        df['Object_ID'] = pd.to_numeric(df['Object_ID'], errors='coerce')
        
        # 重複除去
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Object_ID'])
        final_count = len(df)
        
        if initial_count != final_count:
            self.logger.info(f"重複除去: {initial_count - final_count}件")
        
        # 欠損値の確認
        missing_counts = df.isnull().sum()
        self.logger.info(f"欠損値の確認: {missing_counts[missing_counts > 0].to_dict()}")
        
        self.logger.info(f"前処理完了: {len(df)}件")
        return df
    
    def get_all_object_ids(self) -> Optional[List[int]]:
        """
        APIから全Object ID一覧を取得
        
        Returns:
            Object IDのリスト（取得失敗時はNone）
        """
        self.logger.info("APIから全Object ID一覧を取得中...")
        
        self._rate_limit_delay()
        
        url = f"{self.base_url}/objects"
        
        max_retries = 3
        retry_delay = 2  # 2秒から開始（通常のエラー処理）
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                object_ids = data.get('objectIDs', [])
                
                self.logger.info(f"API Object ID取得完了: {len(object_ids)}件")
                return object_ids
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # 429: Too Many Requests - レート制限
                    if attempt < max_retries - 1:
                        self.logger.warning(f"レート制限エラー、{retry_delay}秒待機後に再試行... (試行 {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        self.logger.warning(f"レート制限エラー: 最大再試行回数に達しました - {e}")
                        self.logger.info("CSVデータのみを使用して処理を継続します")
                        return None
                elif e.response.status_code == 403:
                    # 403: Forbidden - アクセス拒否
                    self.logger.warning(f"アクセス拒否エラー: {e}")
                    self.logger.info("CSVデータのみを使用して処理を継続します")
                    return None
                else:
                    self.logger.warning(f"HTTPエラー: {e.response.status_code} - {e}")
                    return None
            except Exception as e:
                self.logger.warning(f"API Object ID取得エラー: {e}")
                return None
    
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
        
        max_retries = self.max_retries
        retry_delay = 1  # 1秒から開始（高速化）
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    self.logger.debug(f"作品が見つかりません (ID: {object_id})")
                    return None
                elif e.response.status_code == 429:
                    # 429: Too Many Requests - レート制限
                    if attempt < max_retries - 1:
                        self.logger.warning(f"レート制限エラー (ID: {object_id}), {retry_delay}秒待機後に再試行... (試行 {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        self.logger.error(f"レート制限エラー (ID: {object_id}): 最大再試行回数に達しました - {e}")
                        return None
                elif e.response.status_code == 403:
                    # 403: Forbidden - アクセス拒否（データが存在しない可能性）
                    self.logger.debug(f"アクセス拒否 (ID: {object_id}) - データが存在しない可能性")
                    return None
                else:
                    self.logger.warning(f"HTTPエラー (ID: {object_id}): {e.response.status_code} - {e}")
                    return None
            except Exception as e:
                self.logger.warning(f"作品詳細取得エラー (ID: {object_id}): {e}")
                return None
        
        return None
    
    def extract_api_fields(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        APIデータから必要なフィールドを抽出
        
        Args:
            api_data: APIから取得した生データ
            
        Returns:
            抽出されたフィールドの辞書
        """
        extracted = {}
        
        for field in self.api_fields:
            if field in api_data:
                value = api_data[field]
                
                # リストや辞書を文字列に変換（CSV互換性のため）
                if isinstance(value, (list, dict)):
                    extracted[field] = json.dumps(value, ensure_ascii=False)
                else:
                    extracted[field] = value
            else:
                extracted[field] = None
        
        return extracted
    
    def save_checkpoint(self):
        """チェックポイントを保存"""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'processed_ids': list(self.processed_ids),
            'failed_ids': list(self.failed_ids),
            'api_data_count': len(self.api_data)
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # 失敗したIDも別途保存
        with open(self.failed_ids_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.failed_ids), f, indent=2)
        
        self.logger.info(f"チェックポイント保存: 処理済み {len(self.processed_ids)}件, 失敗 {len(self.failed_ids)}件")
    
    def load_checkpoint(self) -> bool:
        """
        チェックポイントを読み込み
        
        Returns:
            チェックポイントが存在する場合True
        """
        if not self.checkpoint_file.exists():
            return False
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            self.processed_ids = set(checkpoint_data.get('processed_ids', []))
            self.failed_ids = set(checkpoint_data.get('failed_ids', []))
            
            # APIデータも読み込み（部分的な復元）
            self.logger.info(f"チェックポイント読み込み: 処理済み {len(self.processed_ids)}件, 失敗 {len(self.failed_ids)}件")
            return True
            
        except Exception as e:
            self.logger.warning(f"チェックポイント読み込みエラー: {e}")
            return False
    
    def collect_api_data(self, object_ids: List[int], resume: bool = True):
        """
        APIから詳細データを収集
        
        Args:
            object_ids: 対象Object IDのリスト
            resume: チェックポイントから再開するか
        """
        if resume:
            self.load_checkpoint()
        
        # 未処理のIDのみを対象
        remaining_ids = [oid for oid in object_ids if oid not in self.processed_ids and oid not in self.failed_ids]
        
        self.logger.info(f"APIデータ収集開始: 残り {len(remaining_ids)}件")
        
        for i, object_id in enumerate(remaining_ids):
            if i % 100 == 0:
                self.logger.info(f"進捗: {i}/{len(remaining_ids)} ({i/len(remaining_ids)*100:.1f}%)")
            
            api_data = self.get_object_details(object_id)
            
            if api_data:
                # 必要なフィールドのみを抽出
                extracted_data = self.extract_api_fields(api_data)
                self.api_data[object_id] = extracted_data
                self.processed_ids.add(object_id)
            else:
                self.failed_ids.add(object_id)
            
            # チェックポイント保存
            if (i + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint()
        
        # 最終チェックポイント保存
        self.save_checkpoint()
        
        self.logger.info(f"APIデータ収集完了: 成功 {len(self.api_data)}件, 失敗 {len(self.failed_ids)}件")
    
    def merge_data(self, csv_df: pd.DataFrame) -> pd.DataFrame:
        """
        CSVデータとAPIデータを統合
        
        Args:
            csv_df: CSVのDataFrame
            
        Returns:
            統合されたDataFrame
        """
        self.logger.info("データ統合を開始...")
        
        # APIデータをDataFrameに変換
        if self.api_data:
            api_df = pd.DataFrame.from_dict(self.api_data, orient='index')
            api_df.index.name = 'Object_ID'
            api_df = api_df.reset_index()
        else:
            # APIデータがない場合は空のDataFrameを作成
            api_df = pd.DataFrame(columns=['Object_ID'] + self.api_fields)
        
        # CSVデータとAPIデータをマージ
        merged_df = csv_df.merge(api_df, on='Object_ID', how='left')
        
        # 画像URL列を追加（既存の列名に合わせる）
        if 'primaryImage' in merged_df.columns:
            merged_df['primaryImageSmall'] = merged_df['primaryImageSmall'].fillna(merged_df['primaryImage'])
        
        # image_downloadedフラグを追加
        merged_df['image_downloaded'] = False
        
        self.logger.info(f"データ統合完了: {len(merged_df)}件")
        return merged_df
    
    def generate_qa_report(self, df: pd.DataFrame):
        """
        品質管理レポートを生成
        
        Args:
            df: 統合されたDataFrame
        """
        self.logger.info("品質管理レポートを生成中...")
        
        report_lines = []
        report_lines.append("=== ハイブリッドデータ収集 品質管理レポート ===")
        report_lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 件数検証
        report_lines.append("## 件数検証")
        report_lines.append(f"総作品数: {len(df):,}")
        report_lines.append(f"APIデータ取得成功: {len(self.api_data):,}")
        report_lines.append(f"APIデータ取得失敗: {len(self.failed_ids):,}")
        report_lines.append("")
        
        # 欠損率レポート
        report_lines.append("## 主要列の欠損率")
        key_columns = ['Title', 'Artist_Display_Name', 'Object_Date', 'Medium', 'Department']
        for col in key_columns:
            if col in df.columns:
                missing_rate = df[col].isnull().sum() / len(df) * 100
                report_lines.append(f"{col}: {missing_rate:.1f}%")
        report_lines.append("")
        
        # API列の欠損率
        report_lines.append("## API列の欠損率")
        for field in self.api_fields:
            if field in df.columns:
                missing_rate = df[field].isnull().sum() / len(df) * 100
                report_lines.append(f"{field}: {missing_rate:.1f}%")
        report_lines.append("")
        
        # 統計サマリー
        report_lines.append("## 統計サマリー")
        if 'Department' in df.columns:
            dept_counts = df['Department'].value_counts().head(10)
            report_lines.append("### 部門別作品数（上位10）")
            for dept, count in dept_counts.items():
                report_lines.append(f"{dept}: {count:,}")
        report_lines.append("")
        
        if 'Object_Date' in df.columns:
            # 年代分布（数値として解釈可能なもののみ）
            date_numeric = pd.to_numeric(df['Object_Date'], errors='coerce')
            valid_dates = date_numeric.dropna()
            if len(valid_dates) > 0:
                report_lines.append("### 年代分布")
                report_lines.append(f"最古: {valid_dates.min():.0f}")
                report_lines.append(f"最新: {valid_dates.max():.0f}")
                report_lines.append(f"平均: {valid_dates.mean():.0f}")
        
        # レポートを保存
        with open(self.qa_report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # サンプルデータを保存
        sample_df = df.sample(min(10, len(df)))[['Object_ID', 'Title', 'Artist_Display_Name', 'Object_Date', 'Department']]
        sample_df.to_csv(self.qa_samples_file, index=False, encoding='utf-8')
        
        self.logger.info(f"品質管理レポート保存完了: {self.qa_report_file}")
    
    def collect_all_data(self, resume: bool = True):
        """
        全データ収集を実行
        
        Args:
            resume: チェックポイントから再開するか
        """
        self.logger.info("ハイブリッドデータ収集を開始...")
        
        try:
            # 1. CSVダウンロード
            if not self.csv_file.exists() or not resume:
                csv_df = self.download_csv()
                csv_df = self.preprocess_csv(csv_df)
            else:
                self.logger.info("既存のCSVファイルを使用")
                csv_df = pd.read_csv(self.csv_file, low_memory=False)
                csv_df = self.preprocess_csv(csv_df)
            
            # 2. API全ID取得
            api_object_ids = self.get_all_object_ids()
            
            # 3. CSVとAPIのID和集合
            csv_object_ids = csv_df['Object_ID'].dropna().astype(int).tolist()
            
            if api_object_ids:
                all_object_ids = list(set(csv_object_ids + api_object_ids))
                self.logger.info(f"統合Object ID数: {len(all_object_ids)} (CSV: {len(csv_object_ids)}, API: {len(api_object_ids)})")
            else:
                all_object_ids = csv_object_ids
                self.logger.info(f"CSVデータのみ使用: {len(all_object_ids)}件 (API取得失敗のため)")
            
            # 4. API詳細データ収集
            self.collect_api_data(all_object_ids, resume=resume)
            
            # 5. データ統合
            merged_df = self.merge_data(csv_df)
            
            # 6. 統合データ保存
            merged_df.to_csv(self.output_file, index=False, encoding='utf-8')
            self.logger.info(f"統合データ保存完了: {self.output_file}")
            
            # 7. 品質管理レポート生成
            self.generate_qa_report(merged_df)
            
            self.logger.info("ハイブリッドデータ収集完了")
            
        except Exception as e:
            self.logger.error(f"ハイブリッドデータ収集エラー: {e}")
            raise
