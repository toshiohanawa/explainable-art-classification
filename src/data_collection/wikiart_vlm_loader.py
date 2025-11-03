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
        self.base_path = Path(self.wikiart_config.get('base_path', 'data/WikiArt_VLM-main/WikiArt_VLM-main'))
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
        original_ids = {self._extract_image_id(f): f for f in original_files}
        generated_ids = {self._extract_image_id(f): f for f in generated_files}
        
        # Get common IDs
        common_ids = set(original_ids.keys()) & set(generated_ids.keys())
        self.logger.info(f"Common image IDs: {len(common_ids)}")
        
        # Create DataFrame
        pairs = []
        for image_id in sorted(common_ids):
            # Original images (class 0)
            pairs.append({
                'image_id': image_id,
                'image_path': str(original_ids[image_id]),
                'label': 0,  # Original
                'source': 'Original'
            })
            
            # Generated images (class 1)
            pairs.append({
                'image_id': f"{image_id}_generated",
                'image_path': str(generated_ids[image_id]),
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

