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
import random
import pickle
import numpy as np
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
        self.painting_filters = config.get('painting_filters', {})
        
        self.base_url = self.api_config['base_url']
        self.rate_limit = self.api_config['rate_limit']
        self.timeout = self.api_config['timeout']
        
        # ディレクトリ設定
        self.raw_data_dir = Path(self.data_config['raw_data_dir'])
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.filtered_data_dir = Path(self.data_config['filtered_data_dir'])
        self.filtered_data_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイルパス
        self.csv_file = self.raw_data_dir / 'MetObjects.csv'
        self.output_file = self.raw_data_dir / 'artwork_metadata_full.csv'
        
        # フィルタリング済みデータ用パス
        self.paintings_metadata_file = self.filtered_data_dir / self.data_config['paintings_metadata_file']
        self.paintings_images_dir = self.filtered_data_dir / self.data_config['paintings_images_dir']
        self.filter_report_file = self.filtered_data_dir / 'filter_report.txt'
        self.checkpoint_file = self.raw_data_dir / 'checkpoint.json'
        self.failed_ids_file = self.raw_data_dir / 'failed_ids.json'
        self.api_data_file = self.raw_data_dir / 'api_data.json'
        self.cookies_file = self.raw_data_dir / 'session_cookies.pkl'
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
        
        # セッション管理（Cookieサポート）
        self.session = requests.Session()
        self.load_cookies()  # 既存のCookieを読み込み
        
        # 進捗管理
        self.processed_ids: Set[int] = set()
        self.failed_ids: Set[int] = set()
        self.api_data: Dict[int, Dict[str, Any]] = {}
        
        # 人間的な操作をシミュレート
        self.request_count = 0
        self.batch_size = 20  # 20リクエストごとに長い休憩（24時間完結用）
        self.session_refresh_interval = 100  # 50リクエストごとにセッション再初期化
        self.last_session_refresh = 0
        self.batch_break_range = (1.0, 2.0)  # バッチ休憩時間の範囲（秒）
        
        # User-Agentローテーション（人間的なアクセスをシミュレート）
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        ]
        
    def _rate_limit_delay(self):
        """レート制限を考慮した待機（1秒あたり7 req/s以下に制御、24時間完結用）"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # 1秒あたり7 req/s以下に制御（24時間完結用）
        base_interval = 1.0 / 7  # 0.143秒（143ms）
        
        # ランダムジッタ（±20%の揺らぎ）
        jitter = random.uniform(0.8, 1.2)
        min_interval = base_interval * jitter
        
        # 最小0.1秒、最大0.3秒の制限
        min_interval = max(0.1, min(min_interval, 0.3))
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        # バッチごとの長い休憩（人間的な操作をシミュレート）
        self.request_count += 1
        
        # セッション再初期化のチェック
        self.refresh_session_if_needed()
        
        if self.request_count % self.batch_size == 0:
            # バッチサイズに応じた休憩時間を計算
            # 動的に設定された範囲で休憩時間を決定
            batch_break = random.uniform(self.batch_break_range[0], self.batch_break_range[1])
            self.logger.info(f"バッチ休憩: {batch_break:.1f}秒待機中...")
            time.sleep(batch_break)
            # Cookie保存（定期的に更新）
            self.save_cookies()
        else:
            # バッチ内でもランダムな短い休憩（人間的な操作をシミュレート）
            if random.random() < 0.3:  # 30%の確率で追加休憩
                micro_break = random.uniform(0.5, 2.0)
                self.logger.debug(f"マイクロ休憩: {micro_break:.1f}秒待機中...")
                time.sleep(micro_break)
        
        self.last_request_time = time.time()
    
    def save_cookies(self):
        """セッションのCookieを保存"""
        try:
            with open(self.cookies_file, 'wb') as f:
                pickle.dump(self.session.cookies, f)
            self.logger.debug("Cookieを保存しました")
        except Exception as e:
            self.logger.warning(f"Cookie保存エラー: {e}")
    
    def load_cookies(self):
        """保存されたCookieを読み込み"""
        if self.cookies_file.exists():
            try:
                with open(self.cookies_file, 'rb') as f:
                    self.session.cookies.update(pickle.load(f))
                self.logger.debug("Cookieを読み込みました")
            except Exception as e:
                self.logger.warning(f"Cookie読み込みエラー: {e}")
    
    def initialize_session(self):
        """
        セッションを初期化（ブラウザアクセスを模倣）
        最初に軽量なエンドポイントにアクセスしてCookieをセット
        """
        self.logger.info("セッション初期化中...")
        
        headers = {
            'User-Agent': self.get_random_user_agent(),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9,ja;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Referer': 'https://metmuseum.github.io/',
            'Origin': 'https://metmuseum.github.io'
        }
        
        try:
            # 軽量なdepartmentsエンドポイントにアクセス
            response = self.session.get(
                f"{self.base_url}/departments",
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                self.logger.info("セッション初期化成功")
                self.save_cookies()
                self.last_session_refresh = self.request_count
                return True
            else:
                self.logger.warning(f"セッション初期化失敗: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.warning(f"セッション初期化エラー: {e}")
            return False

    def get_random_user_agent(self):
        """
        ランダムなUser-Agentを取得
        """
        return random.choice(self.user_agents)

    def refresh_session_if_needed(self):
        """
        必要に応じてセッションを再初期化
        """
        if self.request_count - self.last_session_refresh >= self.session_refresh_interval:
            self.logger.info(f"セッション再初期化実行 (リクエスト数: {self.request_count})")
            
            # 現在のセッションをクリア
            self.session.close()
            self.session = requests.Session()
            
            # 新しいセッションで初期化
            if self.initialize_session():
                self.logger.info("セッション再初期化成功")
            else:
                self.logger.warning("セッション再初期化失敗、既存セッションで継続")
    
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
            response = self.session.get(csv_url, stream=True)
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

    def filter_painting_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        絵画関連のデータのみをフィルタリング
        
        Args:
            df: フィルタリング前のDataFrame
            
        Returns:
            フィルタリング後のDataFrame
        """
        self.logger.info("絵画データのフィルタリングを開始...")
        
        initial_count = len(df)
        self.logger.info(f"フィルタリング前: {initial_count:,}件")
        
        # フィルター条件を取得
        departments = self.painting_filters.get('departments', [])
        classifications = self.painting_filters.get('classifications', [])
        
        # フィルター条件を適用
        filtered_df = df.copy()
        
        # Departmentフィルター
        if departments:
            dept_mask = filtered_df['Department'].isin(departments)
            filtered_df = filtered_df[dept_mask]
            self.logger.info(f"Departmentフィルター後: {len(filtered_df):,}件")
        
        # Classificationフィルター
        if classifications:
            class_mask = filtered_df['Classification'].isin(classifications)
            filtered_df = filtered_df[class_mask]
            self.logger.info(f"Classificationフィルター後: {len(filtered_df):,}件")
        
        final_count = len(filtered_df)
        reduction_rate = ((initial_count - final_count) / initial_count) * 100
        
        self.logger.info(f"フィルタリング完了: {final_count:,}件 ({reduction_rate:.1f}%削減)")
        
        # フィルター結果の詳細をログ出力
        if 'Department' in filtered_df.columns:
            dept_counts = filtered_df['Department'].value_counts()
            self.logger.info("フィルター後のDepartment分布:")
            for dept, count in dept_counts.head(10).items():
                self.logger.info(f"  {dept}: {count:,}件")
        
        if 'Classification' in filtered_df.columns:
            class_counts = filtered_df['Classification'].value_counts()
            self.logger.info("フィルター後のClassification分布:")
            for classification, count in class_counts.head(10).items():
                self.logger.info(f"  {classification}: {count:,}件")
        
        return filtered_df

    def save_filtered_data(self, filtered_df: pd.DataFrame):
        """
        フィルタリング済みデータを保存
        
        Args:
            filtered_df: フィルタリング済みのDataFrame
        """
        self.logger.info("フィルタリング済みデータを保存中...")
        
        # 絵画画像ディレクトリを作成
        self.paintings_images_dir.mkdir(parents=True, exist_ok=True)
        
        # メタデータを保存
        filtered_df.to_csv(self.paintings_metadata_file, index=False, encoding='utf-8')
        self.logger.info(f"絵画メタデータ保存完了: {self.paintings_metadata_file}")
        
        # フィルタリングレポートを生成
        self.generate_filter_report(filtered_df)
        
        self.logger.info(f"フィルタリング済みデータ保存完了: {len(filtered_df):,}件")

    def generate_filter_report(self, filtered_df: pd.DataFrame):
        """
        フィルタリング結果のレポートを生成
        
        Args:
            filtered_df: フィルタリング済みのDataFrame
        """
        self.logger.info("フィルタリングレポートを生成中...")
        
        report_lines = []
        report_lines.append("=== 絵画データフィルタリングレポート ===")
        report_lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # フィルター条件
        report_lines.append("## 適用されたフィルター条件")
        departments = self.painting_filters.get('departments', [])
        classifications = self.painting_filters.get('classifications', [])
        
        report_lines.append(f"Department: {departments}")
        report_lines.append(f"Classification: {classifications}")
        report_lines.append("")
        
        # フィルタリング結果
        report_lines.append("## フィルタリング結果")
        report_lines.append(f"フィルタリング後件数: {len(filtered_df):,}件")
        report_lines.append("")
        
        # Department分布
        if 'Department' in filtered_df.columns:
            dept_counts = filtered_df['Department'].value_counts()
            report_lines.append("### Department分布")
            for dept, count in dept_counts.head(10).items():
                percentage = (count / len(filtered_df)) * 100
                report_lines.append(f"{dept}: {count:,}件 ({percentage:.1f}%)")
            report_lines.append("")
        
        # Classification分布
        if 'Classification' in filtered_df.columns:
            class_counts = filtered_df['Classification'].value_counts()
            report_lines.append("### Classification分布")
            for classification, count in class_counts.head(10).items():
                percentage = (count / len(filtered_df)) * 100
                report_lines.append(f"{classification}: {count:,}件 ({percentage:.1f}%)")
            report_lines.append("")
        
        # Medium分布（上位10件）
        if 'Medium' in filtered_df.columns:
            medium_counts = filtered_df['Medium'].value_counts()
            report_lines.append("### Medium分布（上位10件）")
            for medium, count in medium_counts.head(10).items():
                percentage = (count / len(filtered_df)) * 100
                report_lines.append(f"{medium}: {count:,}件 ({percentage:.1f}%)")
            report_lines.append("")
        
        # Culture分布（上位10件）
        if 'Culture' in filtered_df.columns:
            culture_counts = filtered_df['Culture'].value_counts()
            report_lines.append("### Culture分布（上位10件）")
            for culture, count in culture_counts.head(10).items():
                percentage = (count / len(filtered_df)) * 100
                report_lines.append(f"{culture}: {count:,}件 ({percentage:.1f}%)")
            report_lines.append("")
        
        # 画像URLの有無
        image_columns = ['primaryImage', 'primaryImageSmall', 'additionalImages']
        for col in image_columns:
            if col in filtered_df.columns:
                has_image = filtered_df[col].notna().sum()
                percentage = (has_image / len(filtered_df)) * 100
                report_lines.append(f"{col}: {has_image:,}件 ({percentage:.1f}%)")
        
        with open(self.filter_report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        self.logger.info(f"フィルタリングレポート生成完了: {self.filter_report_file}")

    def download_painting_images(self, df: pd.DataFrame, max_images: int = None):
        """
        絵画画像をダウンロード
        
        Args:
            df: 画像URLが含まれるDataFrame
            max_images: 最大ダウンロード件数（Noneの場合は全件）
        """
        self.logger.info("絵画画像のダウンロードを開始...")
        
        # 画像URLが存在するレコードをフィルタリング
        image_df = df[df['primaryImageSmall'].notna()].copy()
        
        if max_images:
            image_df = image_df.head(max_images)
        
        total_images = len(image_df)
        self.logger.info(f"ダウンロード対象: {total_images:,}件")
        
        # 画像ディレクトリを作成
        self.paintings_images_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_count = 0
        failed_count = 0
        
        for idx, row in image_df.iterrows():
            try:
                object_id = row['Object_ID']
                image_url = row['primaryImageSmall']
                
                if pd.isna(image_url) or not image_url:
                    continue
                
                # ファイル拡張子を取得
                file_extension = Path(image_url).suffix
                if not file_extension:
                    file_extension = '.jpg'  # デフォルト
                
                # ファイル名を生成
                image_filename = f"{object_id}{file_extension}"
                image_path = self.paintings_images_dir / image_filename
                
                # 既に存在する場合はスキップ
                if image_path.exists():
                    downloaded_count += 1
                    continue
                
                # レート制限を考慮した待機
                self._rate_limit_delay()
                
                # 画像をダウンロード
                response = self.session.get(image_url, timeout=self.timeout)
                response.raise_for_status()
                
                # ファイルに保存
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded_count += 1
                
                # 進捗表示
                if (downloaded_count + failed_count) % 100 == 0:
                    progress = (downloaded_count + failed_count) / total_images * 100
                    self.logger.info(f"画像ダウンロード進捗: {downloaded_count + failed_count}/{total_images} ({progress:.1f}%)")
                
            except Exception as e:
                failed_count += 1
                self.logger.warning(f"画像ダウンロード失敗 (ID: {object_id}): {e}")
                continue
        
        # 最終結果
        self.logger.info(f"画像ダウンロード完了: 成功 {downloaded_count:,}件, 失敗 {failed_count:,}件")
        
        # DataFrameのimage_downloadedフラグを更新
        for idx, row in image_df.iterrows():
            object_id = row['Object_ID']
            image_filename = f"{object_id}{Path(row['primaryImageSmall']).suffix or '.jpg'}"
            image_path = self.paintings_images_dir / image_filename
            
            if image_path.exists():
                df.loc[df['Object_ID'] == object_id, 'image_downloaded'] = True
        
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
        
        # ブラウザに近いヘッダーを設定
        headers = {
            'User-Agent': self.get_random_user_agent(),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9,ja;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Referer': 'https://metmuseum.github.io/',
            'Origin': 'https://metmuseum.github.io',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"'
        }
        
        max_retries = 3
        retry_delay = 60  # WAF遮断後のクールダウン期間（60秒）
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, headers=headers, timeout=self.timeout)
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
                    # 403: Forbidden - WAF遮断の可能性
                    if attempt < max_retries - 1:
                        self.logger.warning(f"WAF遮断検知 (Error 15), {retry_delay}秒クールダウン後に再試行... (試行 {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数バックオフ
                        continue
                    else:
                        self.logger.error(f"WAF遮断エラー: 最大再試行回数に達しました - {e}")
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
        
        # ブラウザに近いヘッダーを設定
        headers = {
            'User-Agent': self.get_random_user_agent(),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9,ja;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Referer': 'https://metmuseum.github.io/',
            'Origin': 'https://metmuseum.github.io',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"'
        }
        
        max_retries = self.max_retries
        retry_delay = 5  # WAF対策のため5秒から開始
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, headers=headers, timeout=self.timeout)
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
        
        # APIデータを別ファイルに保存（ファイルサイズを抑えるため）
        with open(self.api_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.api_data, f, indent=2)
        
        self.logger.info(f"チェックポイント保存: 処理済み {len(self.processed_ids)}件, 失敗 {len(self.failed_ids)}件, 成功 {len(self.api_data)}件")
    
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
            
            # APIデータを別ファイルから読み込み
            if self.api_data_file.exists():
                with open(self.api_data_file, 'r', encoding='utf-8') as f:
                    self.api_data = json.load(f)
            else:
                self.api_data = {}
            
            self.logger.info(f"チェックポイント読み込み: 処理済み {len(self.processed_ids)}件, 失敗 {len(self.failed_ids)}件, 成功 {len(self.api_data)}件")
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
    
    def retry_failed_ids(self, max_retries: int = 2):
        """
        失敗したIDの再取得を実行
        
        Args:
            max_retries: 最大再試行回数
        """
        if not self.failed_ids:
            self.logger.info("再試行対象の失敗IDがありません")
            return
        
        self.logger.info(f"失敗したIDの再取得を開始: {len(self.failed_ids)}件")
        
        retry_count = 0
        while self.failed_ids and retry_count < max_retries:
            retry_count += 1
            self.logger.info(f"再試行 {retry_count}/{max_retries}: {len(self.failed_ids)}件")
            
            # 失敗したIDのリストをコピー（ループ中に変更されるため）
            failed_ids_to_retry = list(self.failed_ids)
            self.failed_ids.clear()  # 一時的にクリア
            
            # 再試行実行
            for object_id in failed_ids_to_retry:
                api_data = self.get_object_details(object_id)
                if api_data:
                    # 成功した場合
                    extracted_data = self.extract_api_fields(api_data)
                    self.api_data[object_id] = extracted_data
                    self.processed_ids.add(object_id)
                    self.logger.debug(f"再試行成功: ID {object_id}")
                else:
                    # 再び失敗した場合
                    self.failed_ids.add(object_id)
                    self.logger.debug(f"再試行失敗: ID {object_id}")
            
            # 進捗表示
            success_count = len(self.api_data)
            failed_count = len(self.failed_ids)
            self.logger.info(f"再試行 {retry_count} 完了: 成功 {success_count}件, 失敗 {failed_count}件")
            
            # チェックポイント保存
            self.save_checkpoint()
            
            # 再試行間のクールダウン
            if self.failed_ids and retry_count < max_retries:
                self.logger.info("次の再試行まで30秒待機...")
                time.sleep(30)
        
        self.logger.info(f"失敗ID再取得完了: 最終成功 {len(self.api_data)}件, 最終失敗 {len(self.failed_ids)}件")

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
        
        # セッション初期化（Cookieセット）
        self.initialize_session()
        time.sleep(2)  # 初期化後の待機
        
        try:
            # 1. CSVダウンロード
            if not self.csv_file.exists() or not resume:
                csv_df = self.download_csv()
                csv_df = self.preprocess_csv(csv_df)
            else:
                self.logger.info("既存のCSVファイルを使用")
                csv_df = pd.read_csv(self.csv_file, low_memory=False)
                csv_df = self.preprocess_csv(csv_df)
            
            # 2. 絵画データのフィルタリング
            csv_df = self.filter_painting_data(csv_df)
            
            # 2.1 フィルタリング済みデータを保存
            self.save_filtered_data(csv_df)
            
            # 3. API全ID取得
            api_object_ids = self.get_all_object_ids()
            
            # 4. CSVとAPIのID和集合
            csv_object_ids = csv_df['Object_ID'].dropna().astype(int).tolist()
            
            if api_object_ids:
                all_object_ids = list(set(csv_object_ids + api_object_ids))
                self.logger.info(f"統合Object ID数: {len(all_object_ids)} (CSV: {len(csv_object_ids)}, API: {len(api_object_ids)})")
            else:
                all_object_ids = csv_object_ids
                self.logger.info(f"CSVデータのみ使用: {len(all_object_ids)}件 (API取得失敗のため)")
            
            # 5. API詳細データ収集
            self.collect_api_data(all_object_ids, resume=resume)
            
            # 6. 失敗したIDの再取得
            self.retry_failed_ids(max_retries=2)
            
            # 7. データ統合
            merged_df = self.merge_data(csv_df)
            
            # 8. 統合データ保存
            merged_df.to_csv(self.output_file, index=False, encoding='utf-8')
            self.logger.info(f"統合データ保存完了: {self.output_file}")
            
            # 9. 品質管理レポート生成
            self.generate_qa_report(merged_df)
            
            self.logger.info("ハイブリッドデータ収集完了")
            
        except Exception as e:
            self.logger.error(f"ハイブリッドデータ収集エラー: {e}")
            raise
