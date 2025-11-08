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
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from ..utils.config_manager import ConfigManager
from ..utils.timestamp_manager import TimestampManager
from ..data_collection.wikiart_vlm_loader import WikiArtVLMDataLoader
from .gestalt_scorer import GestaltScorer


class ColorFeatureExtractor:
    """色彩特徴量抽出クラス"""
    
    def __init__(self, config: Dict[str, Any], timestamp_manager: Optional[TimestampManager] = None):
        """
        初期化
        
        Args:
            config: 設定辞書
            timestamp_manager: タイムスタンプ管理オブジェクト（Noneの場合は新規作成）
                             段階的実行で同じタイムスタンプを使用する場合に指定
        """
        self.config = config
        self.data_config = config['data']
        self.image_config = config['image_processing']
        self.feature_config = config['features']
        
        self.logger = logging.getLogger(__name__)
        
        # 生データは固定ディレクトリから読み込み、分析結果はタイムスタンプ付きディレクトリに保存
        self.base_output_dir = Path(self.data_config['output_dir'])
        self.images_dir = self.base_output_dir / self.data_config['images_dir']
        
        # タイムスタンプ管理（共有または新規作成）
        if timestamp_manager is not None:
            self.timestamp_manager = timestamp_manager
        else:
            self.timestamp_manager = TimestampManager(config)
        self.output_dir = self.timestamp_manager.get_output_dir()
        self.features_dir = self.timestamp_manager.get_features_dir()
        self.gestalt_dir = self.timestamp_manager.get_gestalt_dir()
        
        # ディレクトリ作成
        self.timestamp_manager.create_directories()
        
        # ファイルパス
        self.features_file = self.timestamp_manager.get_features_file()
        
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
        
        # 色の多様性（色相の分散）
        features['color_diversity'] = np.var(h)
        
        # 支配色の分析（HSV統計量を使用）
        features.update(self._extract_dominant_colors(image))
        
        # 明度分布（重複を避けるため、brightness_meanとbrightness_stdは使用しない）
        features.update(self._extract_brightness_distribution(v))
        
        return features
    
    def _extract_dominant_colors(self, image: np.ndarray) -> Dict[str, float]:
        """
        支配色の特徴量を抽出（HSV統計量を使用）
        
        解釈性を重視し、5つの支配色のRGB値を個別に使用せず、
        HSV色空間での統計量（平均色相、平均彩度、平均明度、標準偏差）を使用します。
        """
        features = {}
        
        # 画像を1次元に変換
        pixels = image.reshape(-1, 3)
        
        # K-meansで色をクラスタリング
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # 各クラスタの中心色（RGB）
            colors_rgb = kmeans.cluster_centers_
            
            # 各クラスタのサイズ（支配度）
            labels = kmeans.labels_
            cluster_sizes = np.bincount(labels)
            
            # RGB値をHSVに変換
            colors_hsv = []
            for rgb in colors_rgb:
                # RGB値を0-255の範囲に制限
                rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
                # 1画素のRGB配列を作成
                rgb_array = np.uint8([[rgb_uint8]])
                # HSVに変換
                hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
                colors_hsv.append(hsv[0][0])
            
            colors_hsv = np.array(colors_hsv)
            
            # HSV統計量を計算（解釈性の高い特徴量）
            # 色相（Hue）の統計量
            hues = colors_hsv[:, 0]
            features['dominant_color_hue_mean'] = float(np.mean(hues))
            features['dominant_color_hue_std'] = float(np.std(hues))
            
            # 彩度（Saturation）の統計量
            saturations = colors_hsv[:, 1]
            features['dominant_color_saturation_mean'] = float(np.mean(saturations))
            features['dominant_color_saturation_std'] = float(np.std(saturations))
            
            # 明度（Value）の統計量
            values = colors_hsv[:, 2]
            features['dominant_color_value_mean'] = float(np.mean(values))
            features['dominant_color_value_std'] = float(np.std(values))
            
            # 支配色の占有率の統計量（最大占有率と平均占有率）
            ratios = cluster_sizes / len(pixels)
            features['dominant_color_ratio_max'] = float(np.max(ratios))
            features['dominant_color_ratio_mean'] = float(np.mean(ratios))
            features['dominant_color_ratio_std'] = float(np.std(ratios))
                    
        except Exception as e:
            self.logger.warning(f"支配色抽出エラー: {e}")
            # エラー時はデフォルト値（0.0）を設定
            features['dominant_color_hue_mean'] = 0.0
            features['dominant_color_hue_std'] = 0.0
            features['dominant_color_saturation_mean'] = 0.0
            features['dominant_color_saturation_std'] = 0.0
            features['dominant_color_value_mean'] = 0.0
            features['dominant_color_value_std'] = 0.0
            features['dominant_color_ratio_max'] = 0.0
            features['dominant_color_ratio_mean'] = 0.0
            features['dominant_color_ratio_std'] = 0.0
        
        return features
    
    def _extract_brightness_distribution(self, v_channel: np.ndarray) -> Dict[str, float]:
        """
        明度分布の特徴量を抽出
        
        注意: brightness_meanとbrightness_stdはmean_valueとvalue_stdと重複するため、
        歪度、尖度、エントロピーのみを抽出します。
        """
        features = {}
        
        # ヒストグラムを計算
        hist, bins = np.histogram(v_channel, bins=10, range=(0, 256))
        
        # 分布の特徴量（重複を避けるため、meanとstdは使用しない）
        # mean_valueとvalue_stdが既に存在するため、brightness_meanとbrightness_stdは使用しない
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
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        テクスチャ特徴量を抽出（GLCM + LBP）
        
        Args:
            image: RGB画像配列
            
        Returns:
            テクスチャ特徴量の辞書
        """
        features = {}
        
        # グレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # GLCM特徴量
        try:
            # GLCMを計算（距離=1, 角度=0, 45, 90, 135度）
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(gray, distances=distances, angles=angles, 
                                levels=256, symmetric=True, normed=True)
            
            # 各特徴量を計算（全角度の平均）
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            features['texture_glcm_contrast'] = float(contrast)
            features['texture_glcm_dissimilarity'] = float(dissimilarity)
            features['texture_glcm_homogeneity'] = float(homogeneity)
            features['texture_glcm_energy'] = float(energy)
            features['texture_glcm_correlation'] = float(correlation)
        except Exception as e:
            self.logger.warning(f"GLCM特徴量抽出エラー: {e}")
            features['texture_glcm_contrast'] = 0.0
            features['texture_glcm_dissimilarity'] = 0.0
            features['texture_glcm_homogeneity'] = 0.0
            features['texture_glcm_energy'] = 0.0
            features['texture_glcm_correlation'] = 0.0
        
        # LBP特徴量
        try:
            # LBPを計算
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # LBPの基本統計量
            lbp_mean = np.mean(lbp)
            lbp_std = np.std(lbp)
            features['texture_lbp_mean'] = float(lbp_mean)
            features['texture_lbp_std'] = float(lbp_std)
            
            # LBPヒストグラムの統計量（解釈性を重視してビン特徴量は使用しない）
            # ヒストグラムを計算（統計量の計算に使用）
            hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, n_points + 2))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # 正規化
            
            # ヒストグラムの統計量を計算（解釈性の高い特徴量）
            hist_nonzero = hist[hist > 0]
            if len(hist_nonzero) > 0:
                # エントロピー: テクスチャパターンの多様性を表す
                entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
                features['texture_lbp_histogram_entropy'] = float(entropy)
                
                # 歪度: テクスチャパターン分布の非対称性を表す
                hist_centered = hist - np.mean(hist)
                hist_std = np.std(hist)
                if hist_std > 0:
                    skewness_val = np.mean((hist_centered / hist_std) ** 3)
                    features['texture_lbp_histogram_skewness'] = float(skewness_val)
                else:
                    features['texture_lbp_histogram_skewness'] = 0.0
                
                # 尖度: テクスチャパターン分布の尖り具合を表す
                if hist_std > 0:
                    kurtosis_val = np.mean((hist_centered / hist_std) ** 4) - 3
                    features['texture_lbp_histogram_kurtosis'] = float(kurtosis_val)
                else:
                    features['texture_lbp_histogram_kurtosis'] = 0.0
            else:
                features['texture_lbp_histogram_entropy'] = 0.0
                features['texture_lbp_histogram_skewness'] = 0.0
                features['texture_lbp_histogram_kurtosis'] = 0.0
                
        except Exception as e:
            self.logger.warning(f"LBP特徴量抽出エラー: {e}")
            # エラー時はデフォルト値（0.0）を設定
            features['texture_lbp_mean'] = 0.0
            features['texture_lbp_std'] = 0.0
            features['texture_lbp_histogram_entropy'] = 0.0
            features['texture_lbp_histogram_skewness'] = 0.0
            features['texture_lbp_histogram_kurtosis'] = 0.0
        
        return features
    
    def extract_edge_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        エッジ特徴量を抽出（Canny）
        
        Args:
            image: RGB画像配列
            
        Returns:
            エッジ特徴量の辞書
        """
        features = {}
        
        # グレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        try:
            # Cannyエッジ検出
            edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)
            
            # エッジ密度
            edge_density = np.sum(edges > 0) / edges.size
            features['edge_density'] = float(edge_density)
            
            # エッジコンポーネントの分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
            if num_labels > 1:
                # 背景（ラベル0）を除外
                component_sizes = stats[1:, cv2.CC_STAT_AREA]
                edge_mean_length = np.mean(component_sizes)
                edge_std_length = np.std(component_sizes)
                edge_count = num_labels - 1
            else:
                edge_mean_length = 0.0
                edge_std_length = 0.0
                edge_count = 0
            
            features['edge_mean_length'] = float(edge_mean_length)
            features['edge_std_length'] = float(edge_std_length)
            features['edge_count'] = float(edge_count)
            
            # エッジ方向性（Hough変換を使用）
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            if lines is not None and len(lines) > 0:
                angles = lines[:, 0, 1]  # 角度（ラジアン）
                edge_orientation_mean = np.mean(angles)
                edge_orientation_std = np.std(angles)
            else:
                edge_orientation_mean = 0.0
                edge_orientation_std = 0.0
            
            features['edge_orientation_mean'] = float(edge_orientation_mean)
            features['edge_orientation_std'] = float(edge_orientation_std)
            
            # エッジの滑らかさ（曲率の計算）
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                curvatures = []
                for contour in contours:
                    if len(contour) > 2:
                        # 曲率の簡易計算（隣接点間の角度変化）
                        # 輪郭を正規化して曲率を計算
                        contour_points = contour.reshape(-1, 2).astype(np.float32)
                        
                        # 曲率の計算（3点間の角度変化）
                        if len(contour_points) >= 3:
                            # 各点での曲率を計算
                            for i in range(1, len(contour_points) - 1):
                                p1 = contour_points[i - 1]
                                p2 = contour_points[i]
                                p3 = contour_points[i + 1]
                                
                                # ベクトルを計算
                                v1 = p2 - p1
                                v2 = p3 - p2
                                
                                # ベクトルの長さを正規化
                                norm1 = np.linalg.norm(v1)
                                norm2 = np.linalg.norm(v2)
                                
                                if norm1 > 0 and norm2 > 0:
                                    v1_norm = v1 / norm1
                                    v2_norm = v2 / norm2
                                    
                                    # 角度の変化を計算（曲率の近似）
                                    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                                    angle = np.arccos(dot_product)
                                    curvatures.append(angle)
                
                if len(curvatures) > 0:
                    edge_smoothness = 1.0 / (np.mean(curvatures) + 1e-7)
                    edge_curvature_mean = np.mean(curvatures)
                    edge_curvature_std = np.std(curvatures)
                else:
                    edge_smoothness = 0.0
                    edge_curvature_mean = 0.0
                    edge_curvature_std = 0.0
            else:
                edge_smoothness = 0.0
                edge_curvature_mean = 0.0
                edge_curvature_std = 0.0
            
            features['edge_smoothness'] = float(edge_smoothness)
            features['edge_curvature_mean'] = float(edge_curvature_mean)
            features['edge_curvature_std'] = float(edge_curvature_std)
            
        except Exception as e:
            self.logger.warning(f"エッジ特徴量抽出エラー: {e}")
            # エラー時はデフォルト値
            features['edge_density'] = 0.0
            features['edge_mean_length'] = 0.0
            features['edge_std_length'] = 0.0
            features['edge_count'] = 0.0
            features['edge_orientation_mean'] = 0.0
            features['edge_orientation_std'] = 0.0
            features['edge_smoothness'] = 0.0
            features['edge_curvature_mean'] = 0.0
            features['edge_curvature_std'] = 0.0
        
        return features
    
    def extract_all_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        すべての特徴量を抽出（色彩+テクスチャ+エッジ）
        
        Args:
            image: RGB画像配列
            
        Returns:
            統合された特徴量の辞書
        """
        features = {}
        
        # 色彩特徴量
        features.update(self.extract_color_features(image))
        
        # テクスチャ特徴量
        features.update(self.extract_texture_features(image))
        
        # エッジ特徴量
        features.update(self.extract_edge_features(image))
        
        return features
    
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
        
        # 数値特徴量のみを抽出
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['object_id']]
        
        # 結果を保存
        features_df.to_csv(self.features_file.with_suffix('.csv'), index=False)
        
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
    
    def extract_features_from_wikiart_vlm(self, generation_model: str = 'Stable-Diffusion', 
                                          max_samples: Optional[int] = None) -> None:
        """
        WikiArt_VLMデータセットから特徴量を抽出
        
        Args:
            generation_model: 使用する生成モデル ('Stable-Diffusion', 'FLUX', 'F-Lite')
            max_samples: 最大サンプル数（Noneの場合は全データ）
        """
        self.logger.info(f"WikiArt_VLMデータセットからの特徴量抽出を開始... (モデル: {generation_model})")
        
        # WikiArt_VLMデータローダーを初期化
        wikiart_loader = WikiArtVLMDataLoader(self.config)
        
        # 画像ペアを読み込み
        image_pairs_df = wikiart_loader.load_image_pairs(generation_model)
        
        # サンプル数を制限（テスト用）
        if max_samples is not None:
            image_pairs_df = image_pairs_df.head(max_samples)
            self.logger.info(f"サンプル数を制限: {len(image_pairs_df)}件")
        
        self.logger.info(f"対象画像ペア数: {len(image_pairs_df)}")
        
        features_list = []
        failed_count = 0
        
        for idx, row in image_pairs_df.iterrows():
            if idx % 500 == 0 and idx > 0:
                self.logger.info(f"進捗: {idx}/{len(image_pairs_df)} (成功: {len(features_list)}, 失敗: {failed_count})")
            
            image_path = Path(row['image_path'])
            image_id = row['image_id']
            label = row['label']  # 0: 本物, 1: フェイク
            source = row['source']
            
            if not image_path.exists():
                self.logger.warning(f"画像が見つかりません: {image_path}")
                failed_count += 1
                continue
            
            # 画像を前処理
            image = self.preprocess_image(image_path)
            if image is None:
                failed_count += 1
                continue
            
            # 特徴量を抽出（色彩+テクスチャ+エッジ）
            try:
                features = self.extract_all_features(image)
                features['image_id'] = image_id
                features['image_path'] = str(image_path)
                features['label'] = label  # 0: 本物, 1: フェイク
                features['source'] = source  # 'Original' or generation_model
                features['generation_model'] = generation_model if source != 'Original' else 'Original'
                
                features_list.append(features)
            except Exception as e:
                self.logger.warning(f"特徴量抽出エラー (ID: {image_id}): {e}")
                failed_count += 1
                continue
        
        # データフレームに変換
        features_df = pd.DataFrame(features_list)
        
        if len(features_df) == 0:
            raise ValueError("有効な特徴量が抽出できませんでした")
        
        # 数値特徴量のみを抽出（スケーリングは行わない - データリーク防止のため）
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['image_id', 'label']]
        
        # 保存ファイル名に生成モデル名を含める
        output_filename = f"wikiart_vlm_features_{generation_model}"
        features_file = self.features_dir / output_filename
        features_df.to_csv(features_file.with_suffix('.csv'), index=False)
        
        # データセット統計
        original_count = len(features_df[features_df['label'] == 0])
        fake_count = len(features_df[features_df['label'] == 1])
        
        self.logger.info(f"WikiArt_VLM特徴量抽出完了:")
        self.logger.info(f"  - 生成モデル: {generation_model}")
        self.logger.info(f"  - 成功: {len(features_df)}")
        self.logger.info(f"  - 失敗: {failed_count}")
        self.logger.info(f"  - 本物画像: {original_count}")
        self.logger.info(f"  - フェイク画像: {fake_count}")
        self.logger.info(f"  - 特徴量数: {len(feature_columns)}")
        self.logger.info(f"  - 保存先: {features_file.with_suffix('.csv')}")
    
    def merge_gestalt_scores(self, features_file: Path, 
                            gestalt_scores_file: Path) -> pd.DataFrame:
        """
        色彩特徴量とゲシュタルト原則スコアを統合
        
        Args:
            features_file: 色彩特徴量ファイルのパス
            gestalt_scores_file: ゲシュタルト原則スコアファイルのパス
            
        Returns:
            統合された特徴量DataFrame
        """
        self.logger.info("特徴量とゲシュタルト原則スコアを統合中...")
        
        # 特徴量を読み込み
        if not features_file.exists():
            raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_file}")
        
        features_df = pd.read_csv(features_file)
        self.logger.info(f"色彩特徴量: {len(features_df)}件, {len(features_df.columns)}列")
        
        # ゲシュタルト原則スコアを読み込み
        if not gestalt_scores_file.exists():
            self.logger.warning(f"ゲシュタルト原則スコアファイルが見つかりません: {gestalt_scores_file}")
            self.logger.warning("色彩特徴量のみを返します")
            return features_df
        
        gestalt_df = pd.read_csv(gestalt_scores_file)
        self.logger.info(f"ゲシュタルト原則スコア: {len(gestalt_df)}件, {len(gestalt_df.columns)}列")
        
        # マージキーを決定（image_pathまたはimage_name）
        merge_key = None
        if 'image_path' in features_df.columns and 'image_path' in gestalt_df.columns:
            merge_key = 'image_path'
        elif 'image_name' in features_df.columns and 'image_name' in gestalt_df.columns:
            merge_key = 'image_name'
        elif 'image_id' in features_df.columns and 'image_name' in gestalt_df.columns:
            # image_idからimage_nameを生成してマージ
            features_df['image_name_temp'] = features_df['image_id'].astype(str) + '.jpg'
            merge_key = 'image_name_temp'
            gestalt_df_merge = gestalt_df.copy()
            gestalt_df_merge['image_name_temp'] = gestalt_df_merge['image_name']
        
        if merge_key is None:
            # image_pathのパスを正規化してマッチングを試行
            features_df['image_path_normalized'] = features_df['image_path'].apply(
                lambda x: str(Path(x).name) if isinstance(x, str) else ''
            )
            gestalt_df['image_path_normalized'] = gestalt_df['image_path'].apply(
                lambda x: str(Path(x).name) if isinstance(x, str) else ''
            )
            merge_key = 'image_path_normalized'
        
        # マージ
        if merge_key == 'image_name_temp':
            merged_df = pd.merge(
                features_df,
                gestalt_df_merge,
                on=merge_key,
                how='left',
                suffixes=('', '_gestalt')
            )
            merged_df = merged_df.drop(columns=['image_name_temp'])
        else:
            merged_df = pd.merge(
                features_df,
                gestalt_df,
                on=merge_key,
                how='left',
                suffixes=('', '_gestalt')
            )
        
        # マージ結果の確認
        merged_count = merged_df[gestalt_df.columns[1]].notna().sum()  # 最初のゲシュタルトスコアカラムで確認
        self.logger.info(f"統合完了: {merged_count}/{len(features_df)}件がマッチしました")
        
        if merged_count == 0:
            self.logger.warning("ゲシュタルト原則スコアとのマッチングがありませんでした。色彩特徴量のみを返します")
            return features_df
        
        return merged_df
    
    def _find_existing_gestalt_scores(self, generation_model: str) -> Optional[Path]:
        """
        既存のゲシュタルトスコアファイルを検索
        
        Args:
            generation_model: 生成モデル名
            
        Returns:
            見つかったファイルのPath、見つからない場合はNone
        """
        latest_file = self.gestalt_dir / f'gestalt_scores_{generation_model}.csv'
        if latest_file.exists():
            self.logger.info(f"既存のゲシュタルトスコアファイルが見つかりました: {latest_file}")
            return latest_file

        # 互換性のため、過去のanalysis_*構造からも検索
        legacy_dirs = sorted(self.base_output_dir.glob("analysis_*"), reverse=True)
        for analysis_dir in legacy_dirs:
            candidate_file = analysis_dir / "features" / f'gestalt_scores_{generation_model}.csv'
            if candidate_file.exists():
                self.logger.info(f"既存のゲシュタルトスコアファイルが見つかりました: {candidate_file}")
                return candidate_file

        self.logger.warning(f"既存のゲシュタルトスコアファイルが見つかりませんでした: gestalt_scores_{generation_model}.csv")
        return None
    
    def extract_features_with_gestalt_scores(self, generation_model: str = 'Stable-Diffusion',
                                             max_samples: Optional[int] = None,
                                             include_gestalt: bool = True,
                                             gestalt_model: str = 'llava:7b') -> None:
        """
        WikiArt_VLMデータセットから特徴量を抽出し、ゲシュタルト原則スコアも統合
        
        Args:
            generation_model: 使用する生成モデル ('Stable-Diffusion', 'FLUX', 'F-Lite')
            max_samples: 最大サンプル数（Noneの場合は全データ）
            include_gestalt: ゲシュタルト原則スコアを含めるか
            gestalt_model: Ollamaモデル名（ゲシュタルトスコアリング用）
        """
        self.logger.info(f"特徴量抽出（ゲシュタルト原則スコア統合）を開始... (モデル: {generation_model})")
        
        # 1. 色彩特徴量を抽出
        self.extract_features_from_wikiart_vlm(
            generation_model=generation_model,
            max_samples=max_samples
        )
        
        # 2. ゲシュタルト原則スコアリング（オプション）
        if include_gestalt:
            try:
                # 既存のゲシュタルトスコアファイルを検索
                existing_gestalt_file = self._find_existing_gestalt_scores(generation_model)
                
                if existing_gestalt_file is not None:
                    self.logger.info(f"\n既存のゲシュタルトスコアファイルを使用します: {existing_gestalt_file}")
                    gestalt_scores_file = existing_gestalt_file
                else:
                    self.logger.info("\nゲシュタルト原則スコアリングを開始...")
                    # ✅ 同じTimestampManagerを使用（段階的実行対応）
                    scorer = GestaltScorer(self.config, model_name=gestalt_model, 
                                          timestamp_manager=self.timestamp_manager)
                    gestalt_scores_df, gestalt_scores_file_path = scorer.score_wikiart_vlm_images(
                        generation_model=generation_model,
                        max_samples=max_samples
                    )
                    gestalt_scores_file = Path(gestalt_scores_file_path)
                
                # 3. 特徴量とゲシュタルト原則スコアを統合
                features_file = self.features_dir / f"wikiart_vlm_features_{generation_model}.csv"
                
                merged_df = self.merge_gestalt_scores(features_file, gestalt_scores_file)
                
                # 統合された特徴量を保存
                merged_output_file = self.features_dir / f"wikiart_vlm_features_with_gestalt_{generation_model}.csv"
                
                # 数値特徴量の統計を収集
                numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
                feature_columns = [col for col in numeric_columns if col not in ['image_id', 'label']]
                merged_df.to_csv(merged_output_file, index=False)
                
                self.logger.info(f"\n統合特徴量を保存: {merged_output_file}")
                self.logger.info(f"  - 総特徴量数: {len(feature_columns)}")
                self.logger.info(f"  - 色彩特徴量: {len([c for c in feature_columns if not c.endswith('_score')])}")
                self.logger.info(f"  - ゲシュタルト原則スコア: {len([c for c in feature_columns if c.endswith('_score')])}")
                
            except Exception as e:
                self.logger.warning(f"ゲシュタルト原則スコアリングでエラー: {e}")
                self.logger.warning("色彩特徴量のみを返します")
