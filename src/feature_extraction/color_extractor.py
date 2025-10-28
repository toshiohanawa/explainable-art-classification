"""
色彩特徴量抽出クラス
画像から色彩に関する数値特徴量を抽出する
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from PIL import Image
from sklearn.preprocessing import StandardScaler
import pickle

from ..utils.config_manager import ConfigManager
from ..utils.timestamp_manager import TimestampManager


class ColorFeatureExtractor:
    """色彩特徴量抽出クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.data_config = config['data']
        self.image_config = config['image_processing']
        self.feature_config = config['features']
        
        self.logger = logging.getLogger(__name__)
        
        # 生データは固定ディレクトリから読み込み、分析結果はタイムスタンプ付きディレクトリに保存
        self.base_output_dir = Path(self.data_config['output_dir'])
        self.images_dir = self.base_output_dir / self.data_config['images_dir']
        
        # タイムスタンプ管理（分析結果用）
        self.timestamp_manager = TimestampManager(config)
        self.output_dir = self.timestamp_manager.get_output_dir()
        self.features_dir = self.timestamp_manager.get_features_dir()
        
        # ディレクトリ作成
        self.timestamp_manager.create_directories()
        
        # ファイルパス
        self.features_file = self.timestamp_manager.get_features_file()
        
        # 特徴量スケーラー
        self.scaler = StandardScaler()
    
    def load_metadata(self) -> pd.DataFrame:
        """メタデータを読み込み"""
        metadata_file = self.base_output_dir / 'raw_data' / self.data_config['metadata_file']
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"メタデータファイルが見つかりません: {metadata_file}")
        
        df = pd.read_csv(metadata_file)
        self.logger.info(f"メタデータ読み込み完了: {len(df)}件")
        return df
    
    def preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        画像の前処理
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            前処理済み画像配列（失敗時はNone）
        """
        try:
            # 画像を読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # BGRからRGBに変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # リサイズ
            target_size = tuple(self.image_config['target_size'])
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
            
            return image
        except Exception as e:
            self.logger.warning(f"画像前処理エラー ({image_path}): {e}")
            return None
    
    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        色彩特徴量を抽出
        
        Args:
            image: RGB画像配列
            
        Returns:
            色彩特徴量の辞書
        """
        features = {}
        
        # HSV色空間に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # 基本統計量
        features['mean_hue'] = np.mean(h)
        features['mean_saturation'] = np.mean(s)
        features['mean_value'] = np.mean(v)
        
        features['hue_std'] = np.std(h)
        features['saturation_std'] = np.std(s)
        features['value_std'] = np.std(v)
        
        # コントラスト（標準偏差）
        features['contrast'] = np.std(v)
        
        # 色の多様性（色相の分散）
        features['color_diversity'] = np.var(h)
        
        # 支配色の分析
        features.update(self._extract_dominant_colors(image))
        
        # 明度分布
        features.update(self._extract_brightness_distribution(v))
        
        return features
    
    def _extract_dominant_colors(self, image: np.ndarray) -> Dict[str, float]:
        """支配色の特徴量を抽出"""
        features = {}
        
        # 画像を1次元に変換
        pixels = image.reshape(-1, 3)
        
        # K-meansで色をクラスタリング
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # 各クラスタの中心色
            colors = kmeans.cluster_centers_
            
            # 各クラスタのサイズ（支配度）
            labels = kmeans.labels_
            cluster_sizes = np.bincount(labels)
            
            # 支配色の特徴量
            for i in range(5):
                if i < len(colors):
                    features[f'dominant_color_{i+1}_r'] = colors[i][0]
                    features[f'dominant_color_{i+1}_g'] = colors[i][1]
                    features[f'dominant_color_{i+1}_b'] = colors[i][2]
                    features[f'dominant_color_{i+1}_ratio'] = cluster_sizes[i] / len(pixels)
                else:
                    features[f'dominant_color_{i+1}_r'] = 0.0
                    features[f'dominant_color_{i+1}_g'] = 0.0
                    features[f'dominant_color_{i+1}_b'] = 0.0
                    features[f'dominant_color_{i+1}_ratio'] = 0.0
                    
        except Exception as e:
            self.logger.warning(f"支配色抽出エラー: {e}")
            # エラー時はデフォルト値
            for i in range(5):
                features[f'dominant_color_{i+1}_r'] = 0.0
                features[f'dominant_color_{i+1}_g'] = 0.0
                features[f'dominant_color_{i+1}_b'] = 0.0
                features[f'dominant_color_{i+1}_ratio'] = 0.0
        
        return features
    
    def _extract_brightness_distribution(self, v_channel: np.ndarray) -> Dict[str, float]:
        """明度分布の特徴量を抽出"""
        features = {}
        
        # ヒストグラムを計算
        hist, bins = np.histogram(v_channel, bins=10, range=(0, 256))
        
        # 分布の特徴量
        features['brightness_mean'] = np.mean(v_channel)
        features['brightness_std'] = np.std(v_channel)
        features['brightness_skewness'] = self._calculate_skewness(v_channel)
        features['brightness_kurtosis'] = self._calculate_kurtosis(v_channel)
        
        # 明度の集中度（エントロピー）
        hist_norm = hist / np.sum(hist)
        hist_norm = hist_norm[hist_norm > 0]  # 0を除外
        features['brightness_entropy'] = -np.sum(hist_norm * np.log2(hist_norm))
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """歪度を計算"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """尖度を計算"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def extract_features(self) -> None:
        """全画像の特徴量を抽出"""
        self.logger.info("特徴量抽出を開始...")
        
        # メタデータを読み込み
        metadata = self.load_metadata()
        
        # 画像がダウンロード済みの作品のみを対象
        valid_artworks = metadata[metadata['image_downloaded'] == True]
        self.logger.info(f"対象作品数: {len(valid_artworks)}")
        
        features_list = []
        failed_count = 0
        
        for idx, row in valid_artworks.iterrows():
            if idx % 50 == 0:
                self.logger.info(f"進捗: {idx}/{len(valid_artworks)}")
            
            object_id = row['object_id']
            image_path = self.images_dir / f"{object_id}.jpg"
            
            if not image_path.exists():
                failed_count += 1
                continue
            
            # 画像を前処理
            image = self.preprocess_image(image_path)
            if image is None:
                failed_count += 1
                continue
            
            # 特徴量を抽出
            try:
                features = self.extract_color_features(image)
                features['object_id'] = object_id
                features['title'] = row['title']
                features['artist'] = row['artist']
                features['period'] = row['period']
                features['culture'] = row['culture']
                features['department'] = row['department']
                
                features_list.append(features)
            except Exception as e:
                self.logger.warning(f"特徴量抽出エラー (ID: {object_id}): {e}")
                failed_count += 1
                continue
        
        # データフレームに変換
        features_df = pd.DataFrame(features_list)
        
        if len(features_df) == 0:
            raise ValueError("有効な特徴量が抽出できませんでした")
        
        # 数値特徴量のみを抽出してスケーリング
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['object_id']]
        
        # 特徴量をスケーリング
        features_scaled = self.scaler.fit_transform(features_df[feature_columns])
        features_df_scaled = features_df.copy()
        features_df_scaled[feature_columns] = features_scaled
        
        # 結果を保存
        features_df_scaled.to_csv(self.features_file.with_suffix('.csv'), index=False)
        
        # スケーラーも保存
        scaler_file = self.timestamp_manager.get_scaler_file()
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        self.logger.info(f"特徴量抽出完了:")
        self.logger.info(f"  - 成功: {len(features_df)}")
        self.logger.info(f"  - 失敗: {failed_count}")
        self.logger.info(f"  - 特徴量数: {len(feature_columns)}")
        self.logger.info(f"  - 保存先: {self.features_file}")
    
    def load_features(self) -> pd.DataFrame:
        """保存された特徴量を読み込み"""
        if not self.features_file.with_suffix('.csv').exists():
            raise FileNotFoundError(f"特徴量ファイルが見つかりません: {self.features_file}")
        
        return pd.read_csv(self.features_file.with_suffix('.csv'))
