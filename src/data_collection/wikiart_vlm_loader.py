"""
WikiArt_VLM Dataset Loader
Load pairs of original images and AI-generated images to build datasets for classification tasks
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.model_selection import train_test_split
try:
    import openpyxl  # Excelファイル読み込み用
except ImportError:
    openpyxl = None

from ..utils.config_manager import ConfigManager
from ..utils.timestamp_manager import TimestampManager
from .artist_style_mapping import ArtistStyleMapping


class WikiArtVLMDataLoader:
    """WikiArt_VLM Dataset Loader"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.wikiart_config = config.get('wikiart_vlm', {})
        
        # WikiArt_VLM dataset path
        self.base_path = Path(self.wikiart_config.get('base_path', 'data/external/WikiArt_VLM-main'))
        self.original_dir = self.base_path / 'images' / 'Original'
        self.prompt_file = self.base_path / 'All_gpt4.1-mini_prompt.xlsx'
        
        # Generation model selection
        self.generation_models = self.wikiart_config.get('generation_models', ['Stable-Diffusion'])
        
        self.logger = logging.getLogger(__name__)
        
        # Data split ratios
        self.train_ratio = self.wikiart_config.get('train_ratio', 0.7)
        self.val_ratio = self.wikiart_config.get('val_ratio', 0.15)
        self.test_ratio = self.wikiart_config.get('test_ratio', 0.15)
        
        # Data cache
        self.image_pairs_cache = None
        self.prompts_cache = None
        
        # Filtering configuration
        self.filtering_config = config.get('wikiart_filters', {})
        self.filtering_enabled = self.filtering_config.get('enabled', False)
        
        # Artist-style mapping (for filtering)
        self.artist_style_mapper = None
        self.artist_style_mapping_cache = None
        
        # Check directory existence
        if not self.original_dir.exists():
            self.logger.warning(f"Original image directory not found: {self.original_dir}")
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """
        Get image files in directory
        
        Args:
            directory: Path to image directory
            
        Returns:
            List of image file paths
        """
        if not directory.exists():
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory.glob(f'*{ext}'))
        
        return sorted(image_files)
    
    def _extract_image_id(self, filepath: Path) -> str:
        """
        Extract image ID from filename
        
        Args:
            filepath: Path to image file
            
        Returns:
            Image ID (filename without extension)
        """
        return filepath.stem
    
    def _assign_representative_style(self) -> pd.DataFrame:
        """
        Assign representative style to each artist based on WikiArt dataset
        
        Returns:
            DataFrame with artist and representative style mapping
            (columns: image_id, artist, representative_style)
        """
        if self.artist_style_mapping_cache is not None:
            self.logger.debug("Returning artist-style mapping from cache")
            return self.artist_style_mapping_cache
        
        self.logger.info("Assigning representative styles to artists...")
        
        # Load prompts file
        prompts_df = self.load_prompts()
        
        if prompts_df.empty:
            self.logger.warning("Prompt file is empty. Cannot assign representative styles.")
            return pd.DataFrame()
        
        # Initialize artist-style mapper
        if self.artist_style_mapper is None:
            wikiart_data_path = self.filtering_config.get('wikiart_data_path', 'data/external/WikiArt/wikiart_data.csv')
            self.artist_style_mapper = ArtistStyleMapping(self.config, wikiart_data_path)
        
        # Create artist-style mapping
        artist_style_mapping = self.artist_style_mapper.create_artist_style_mapping()
        
        # Assign representative style to each image
        # Excelファイルの行インデックス（Unnamed: 0）が画像ID（0, 1, 2, ...）に対応
        image_id_col = 'Unnamed: 0' if 'Unnamed: 0' in prompts_df.columns else prompts_df.index.name if prompts_df.index.name else prompts_df.index
        
        # 画像IDとartist名の対応関係を作成
        mapping_data = []
        for idx, row in prompts_df.iterrows():
            image_id = row.get('Unnamed: 0', idx) if 'Unnamed: 0' in prompts_df.columns else idx
            artist = row.get('artist', '')
            
            if artist in artist_style_mapping:
                representative_style = artist_style_mapping[artist]['representative_style']
            else:
                # artistがマッピングに見つからない場合
                representative_style = None
                self.logger.warning(f"Artist '{artist}' not found in artist-style mapping")
            
            mapping_data.append({
                'image_id': str(image_id),
                'artist': artist,
                'representative_style': representative_style
            })
        
        mapping_df = pd.DataFrame(mapping_data)
        
        # 統計情報をログ出力
        style_counts = mapping_df['representative_style'].value_counts()
        self.logger.info(f"Assigned representative styles to {len(mapping_df)} images")
        self.logger.info(f"Style distribution:")
        for style, count in style_counts.head(10).items():
            self.logger.info(f"  {style}: {count} images")
        
        # キャッシュに保存
        self.artist_style_mapping_cache = mapping_df
        
        return mapping_df
    
    def _filter_by_style(self, image_ids: List[str], style_mapping_df: pd.DataFrame) -> List[str]:
        """
        Filter image IDs by style/artist conditions
        
        Args:
            image_ids: List of image IDs to filter
            style_mapping_df: DataFrame with artist-style mapping
            
        Returns:
            Filtered list of image IDs
        """
        if not self.filtering_enabled:
            return image_ids
        
        # フィルタリング条件を取得
        filter_styles = self.filtering_config.get('style', [])
        filter_artists = self.filtering_config.get('artist', [])
        include_unknown = self.filtering_config.get('include_unknown', False)
        
        # デフォルトでImpressionismに絞る（styleが空の場合）
        if not filter_styles and not filter_artists:
            # styleが空の場合は、デフォルトでImpressionismを適用
            filter_styles = ['Impressionism']
            self.logger.info("No filter conditions specified. Defaulting to Impressionism.")
        
        # フィルタリング条件を適用
        filtered_ids = []
        
        for image_id in image_ids:
            # 画像IDに対応するartist-style情報を取得
            # 画像IDは文字列として比較（型の違いに対応）
            image_id_str = str(image_id)
            matching_rows = style_mapping_df[style_mapping_df['image_id'].astype(str) == image_id_str]
            
            if len(matching_rows) == 0:
                # マッピングが見つからない場合、スキップまたは含める
                self.logger.debug(f"No mapping found for image_id: {image_id}")
                continue
            
            row = matching_rows.iloc[0]
            artist = row.get('artist', '')
            style = row.get('representative_style', None)
            
            # "Unknown Artist"の除外チェック
            if not include_unknown and (artist.lower() == 'unknown artist' or artist == ''):
                continue
            
            # styleフィルタリング
            style_match = False
            if filter_styles:
                if style and style in filter_styles:
                    style_match = True
            else:
                # styleフィルターが指定されていない場合、style条件は満たす
                style_match = True
            
            # artistフィルタリング
            artist_match = False
            if filter_artists:
                if artist in filter_artists:
                    artist_match = True
            else:
                # artistフィルターが指定されていない場合、artist条件は満たす
                artist_match = True
            
            # 両方の条件を満たす場合、画像IDを追加
            if style_match and artist_match:
                filtered_ids.append(image_id)
        
        return filtered_ids
    
    def load_image_pairs(self, generation_model: str = 'Stable-Diffusion') -> pd.DataFrame:
        """
        Load pairs of original and generated images
        
        Args:
            generation_model: Generation model to use ('Stable-Diffusion', 'FLUX', 'F-Lite')
            
        Returns:
            DataFrame of image pairs (columns: image_id, image_path, label, source)
        """
        if self.image_pairs_cache is not None and generation_model in self.image_pairs_cache:
            self.logger.info(f"Loading image pairs from cache: {generation_model}")
            return self.image_pairs_cache[generation_model]
        
        # Generated image directory
        generated_dir = self.base_path / 'images' / generation_model
        
        if not generated_dir.exists():
            self.logger.error(f"Generated image directory not found: {generated_dir}")
            raise FileNotFoundError(f"Generated image directory not found: {generated_dir}")
        
        # Get original and generated image files
        original_files = self._get_image_files(self.original_dir)
        generated_files = self._get_image_files(generated_dir)
        
        self.logger.info(f"Original images: {len(original_files)}")
        self.logger.info(f"{generation_model} images: {len(generated_files)}")
        
        # Build correspondence by image ID
        # image_idは文字列型として統一（ファイル名のstemは文字列）
        original_ids = {self._extract_image_id(f): f for f in original_files}
        generated_ids = {self._extract_image_id(f): f for f in generated_files}
        
        # Get common IDs (すべて文字列型として扱う)
        common_ids = set(str(img_id) for img_id in original_ids.keys()) & set(str(img_id) for img_id in generated_ids.keys())
        self.logger.info(f"Common image IDs: {len(common_ids)}")
        
        # Apply filtering if enabled
        if self.filtering_enabled:
            self.logger.info("Filtering enabled. Applying style/artist filters...")
            
            # Assign representative styles to artists
            style_mapping_df = self._assign_representative_style()
            
            if style_mapping_df.empty:
                self.logger.warning("Style mapping is empty. Filtering will be skipped.")
            else:
                # Convert common_ids to list for filtering (すでに文字列型)
                common_ids_list = list(common_ids)
                
                # Filter image IDs
                filtered_ids = self._filter_by_style(common_ids_list, style_mapping_df)
                
                # Convert back to set (文字列型のまま)
                common_ids = set(filtered_ids)
                
                self.logger.info(f"Filtered image IDs: {len(common_ids)} / {len(common_ids_list)} "
                               f"({len(common_ids)/len(common_ids_list)*100:.1f}%)")
        
        # Create DataFrame
        pairs = []
        for image_id in sorted(common_ids):
            # image_idは文字列型なので、そのまま使用
            # original_idsとgenerated_idsのキーも文字列型なので、そのまま使用可能
            image_id_key = image_id  # 文字列型のまま
            
            # Original images (class 0)
            pairs.append({
                'image_id': image_id,
                'image_path': str(original_ids[image_id_key]),
                'label': 0,  # Original
                'source': 'Original'
            })
            
            # Generated images (class 1)
            pairs.append({
                'image_id': f"{image_id}_generated",
                'image_path': str(generated_ids[image_id_key]),
                'label': 1,  # Fake
                'source': generation_model
            })
        
        df = pd.DataFrame(pairs)
        
        # Save to cache
        if self.image_pairs_cache is None:
            self.image_pairs_cache = {}
        self.image_pairs_cache[generation_model] = df
        
        self.logger.info(f"Image pairs: {len(df)} (Original: {len(df[df['label']==0])}, Fake: {len(df[df['label']==1])})")
        
        return df
    
    def load_prompts(self) -> pd.DataFrame:
        """
        Load prompt information from Excel file
        
        Returns:
            DataFrame of prompt information (columns: image_id, prompt, ...)
        """
        if self.prompts_cache is not None:
            self.logger.info("Loading prompt information from cache")
            return self.prompts_cache
        
        if not self.prompt_file.exists():
            self.logger.warning(f"Prompt file not found: {self.prompt_file}")
            return pd.DataFrame()
        
        try:
            if openpyxl is None:
                self.logger.warning("openpyxl is not installed. Cannot read Excel file.")
                return pd.DataFrame()
            
            # Load Excel file
            df = pd.read_excel(self.prompt_file, engine='openpyxl')
            self.logger.info(f"Loaded prompt information: {len(df)} entries")
            
            # Save to cache
            self.prompts_cache = df
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading prompt file: {e}")
            return pd.DataFrame()
    
    def split_dataset(self, df: pd.DataFrame, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/val/test
        
        Args:
            df: DataFrame of image pairs
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split into train and temp (val+test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=random_state,
            stratify=df['label']  # Preserve label distribution
        )
        
        # Split temp into val and test
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=random_state,
            stratify=temp_df['label']
        )
        
        self.logger.info(f"Dataset split completed:")
        self.logger.info(f"  Train: {len(train_df)} (Original: {len(train_df[train_df['label']==0])}, Fake: {len(train_df[train_df['label']==1])})")
        self.logger.info(f"  Val: {len(val_df)} (Original: {len(val_df[val_df['label']==0])}, Fake: {len(val_df[val_df['label']==1])})")
        self.logger.info(f"  Test: {len(test_df)} (Original: {len(test_df[test_df['label']==0])}, Fake: {len(test_df[test_df['label']==1])})")
        
        return train_df, val_df, test_df
    
    def get_dataset_stats(self, generation_model: str = 'Stable-Diffusion') -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Args:
            generation_model: Generation model to use
            
        Returns:
            Dictionary of statistics
        """
        df = self.load_image_pairs(generation_model)
        
        stats = {
            'total_images': len(df),
            'original_count': len(df[df['label'] == 0]),
            'generated_count': len(df[df['label'] == 1]),
            'generation_model': generation_model,
            'image_ids': df['image_id'].nunique() if 'image_id' in df.columns else 0
        }
        
        return stats
    
    def verify_dataset_integrity(self, generation_model: str = 'Stable-Diffusion') -> Dict[str, Any]:
        """
        Verify dataset integrity
        
        Args:
            generation_model: Generation model to use
            
        Returns:
            Dictionary of integrity check results
        """
        df = self.load_image_pairs(generation_model)
        
        checks = {
            'total_pairs': len(df),
            'has_original_images': len(df[df['label'] == 0]) > 0,
            'has_generated_images': len(df[df['label'] == 1]) > 0,
            'balanced_dataset': abs(len(df[df['label'] == 0]) - len(df[df['label'] == 1])) <= 10,
            'images_exist': []
        }
        
        # Check sample image existence
        sample_size = min(10, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        for idx, row in sample_df.iterrows():
            img_path = Path(row['image_path'])
            checks['images_exist'].append({
                'image_id': row.get('image_id', 'unknown'),
                'path': str(img_path),
                'exists': img_path.exists()
            })
        
        all_exist = all(check['exists'] for check in checks['images_exist'])
        checks['sample_images_exist'] = all_exist
        
        return checks
