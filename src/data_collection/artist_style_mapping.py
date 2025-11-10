"""
Artist-Styleマッピングモジュール
Hugging Faceデータセット（wikiart_data.csv）から各artistの代表スタイルを決定
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ..utils.config_manager import ConfigManager


class ArtistStyleMapping:
    """Artist-Styleマッピングクラス"""
    
    def __init__(self, config: Dict[str, Any], wikiart_data_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config: 設定辞書
            wikiart_data_path: Hugging Faceデータセット（wikiart_data.csv）のパス
                             Noneの場合はconfig.yamlから取得
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.wikiart_config = config.get('wikiart_vlm', {})
        
        # WikiArtデータセットのパス
        if wikiart_data_path:
            self.wikiart_data_path = Path(wikiart_data_path)
        else:
            wikiart_filters = config.get('wikiart_filters', {})
            default_path = Path('data/external/WikiArt/wikiart_data.csv')
            self.wikiart_data_path = Path(wikiart_filters.get('wikiart_data_path', default_path))
        
        self.logger = logging.getLogger(__name__)
        
        # キャッシュ
        self._mapping_cache = None
        self._wikiart_df_cache = None
    
    def _load_wikiart_dataset(self) -> pd.DataFrame:
        """
        Hugging Faceデータセット（wikiart_data.csv）を読み込む
        
        Returns:
            DataFrame of wikiart data (columns: index, artist, genre, style)
        """
        if self._wikiart_df_cache is not None:
            self.logger.debug("Loading WikiArt dataset from cache")
            return self._wikiart_df_cache
        
        if not self.wikiart_data_path.exists():
            self.logger.error(f"WikiArt data file not found: {self.wikiart_data_path}")
            raise FileNotFoundError(f"WikiArt data file not found: {self.wikiart_data_path}")
        
        try:
            df = pd.read_csv(self.wikiart_data_path)
            self.logger.info(f"Loaded WikiArt dataset: {len(df)} entries")
            
            # キャッシュに保存
            self._wikiart_df_cache = df
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading WikiArt dataset: {e}")
            raise
    
    def create_artist_style_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        各artistの代表スタイルを決定するマッピング辞書を作成
        
        Returns:
            Dictionary mapping artist name to representative style information
            {
                'artist_name': {
                    'representative_style': 'StyleName',
                    'style_count': count,
                    'total_works': total,
                    'all_styles': {style: count, ...}
                },
                ...
            }
        """
        if self._mapping_cache is not None:
            self.logger.debug("Returning artist-style mapping from cache")
            return self._mapping_cache
        
        self.logger.info("Creating artist-style mapping from WikiArt dataset...")
        
        # WikiArtデータセットを読み込み
        wikiart_df = self._load_wikiart_dataset()
        
        # 各artistのスタイル分布を計算
        artist_style_mapping = {}
        
        for artist in wikiart_df['artist'].unique():
            artist_df = wikiart_df[wikiart_df['artist'] == artist]
            style_counts = artist_df['style'].value_counts()
            
            # 最も作品数の多いスタイルを代表スタイルとして採用
            # 複数スタイルが同数の場合、最初に見つかったスタイルを採用
            representative_style = style_counts.index[0]
            count = int(style_counts.iloc[0])
            
            artist_style_mapping[artist] = {
                'representative_style': representative_style,
                'style_count': count,
                'total_works': len(artist_df),
                'all_styles': style_counts.to_dict()
            }
        
        self.logger.info(f"Created artist-style mapping for {len(artist_style_mapping)} artists")
        
        # キャッシュに保存
        self._mapping_cache = artist_style_mapping
        
        # 統計情報をログ出力
        single_style_count = sum(1 for info in artist_style_mapping.values() if len(info['all_styles']) == 1)
        multi_style_count = len(artist_style_mapping) - single_style_count
        
        self.logger.info(f"Artists with single style: {single_style_count}")
        self.logger.info(f"Artists with multiple styles: {multi_style_count}")
        
        return artist_style_mapping
    
    def get_representative_style(self, artist: str) -> Optional[str]:
        """
        指定されたartistの代表スタイルを取得
        
        Args:
            artist: Artist name
            
        Returns:
            Representative style name, or None if artist not found
        """
        if self._mapping_cache is None:
            self.create_artist_style_mapping()
        
        if artist in self._mapping_cache:
            return self._mapping_cache[artist]['representative_style']
        
        return None
    
    def get_style_distribution(self) -> Dict[str, int]:
        """
        代表スタイルの分布を取得
        
        Returns:
            Dictionary mapping style name to artist count
        """
        if self._mapping_cache is None:
            self.create_artist_style_mapping()
        
        style_distribution = {}
        for artist, info in self._mapping_cache.items():
            style = info['representative_style']
            style_distribution[style] = style_distribution.get(style, 0) + 1
        
        return style_distribution


def create_artist_style_mapping_from_config(config: Dict[str, Any], 
                                            wikiart_data_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    設定からartist-styleマッピング辞書を作成（便利関数）
    
    Args:
        config: 設定辞書
        wikiart_data_path: WikiArtデータセットのパス（オプション）
        
    Returns:
        Artist-styleマッピング辞書
    """
    mapper = ArtistStyleMapping(config, wikiart_data_path)
    return mapper.create_artist_style_mapping()
