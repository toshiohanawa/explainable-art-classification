"""
タイムスタンプ管理ユーティリティ
分析結果を時系列で管理するためのタイムスタンプ付きディレクトリを作成
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class TimestampManager:
    """タイムスタンプ管理クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.data_config = config['data']
        self.use_timestamp = self.data_config.get('use_timestamp', False)
        
        # ベース出力ディレクトリ
        self.base_output_dir = Path(self.data_config['output_dir'])
        
        # タイムスタンプ付きディレクトリ（分単位）
        if self.use_timestamp:
            self.timestamp = datetime.now().strftime("%y%m%d%H%M")
            self.output_dir = self.base_output_dir / f"analysis_{self.timestamp}"
        else:
            self.output_dir = self.base_output_dir
    
    def get_output_dir(self) -> Path:
        """出力ディレクトリを取得"""
        return self.output_dir
    
    def get_images_dir(self) -> Path:
        """画像ディレクトリを取得（生データは固定ディレクトリ）"""
        return self.base_output_dir / self.data_config['images_dir']
    
    def get_raw_data_dir(self) -> Path:
        """生データディレクトリを取得（生データは固定ディレクトリ）"""
        return self.base_output_dir / 'raw_data'
    
    def get_features_dir(self) -> Path:
        """特徴量ディレクトリを取得"""
        return self.output_dir / 'features'
    
    def get_models_dir(self) -> Path:
        """モデルディレクトリを取得"""
        return self.output_dir / 'models'
    
    def get_results_dir(self) -> Path:
        """結果ディレクトリを取得"""
        return self.output_dir / 'results'
    
    def get_visualizations_dir(self) -> Path:
        """可視化ディレクトリを取得"""
        return self.output_dir / 'visualizations'
    
    def get_shap_explanations_dir(self) -> Path:
        """SHAP説明ディレクトリを取得"""
        return self.output_dir / 'shap_explanations'
    
    def create_directories(self) -> None:
        """必要なディレクトリを作成"""
        # 分析結果用のディレクトリのみ作成
        analysis_directories = [
            self.output_dir,
            self.get_features_dir(),
            self.get_models_dir(),
            self.get_results_dir(),
            self.get_visualizations_dir(),
            self.get_shap_explanations_dir()
        ]
        
        for directory in analysis_directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 生データ用のディレクトリも作成（固定ディレクトリ）
        raw_directories = [
            self.get_images_dir(),
            self.get_raw_data_dir()
        ]
        
        for directory in raw_directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_metadata_file(self) -> Path:
        """メタデータファイルパスを取得"""
        return self.get_raw_data_dir() / self.data_config['metadata_file']
    
    def get_features_file(self) -> Path:
        """特徴量ファイルパスを取得"""
        return self.get_features_dir() / self.data_config['features_file']
    
    def get_model_file(self) -> Path:
        """モデルファイルパスを取得"""
        return self.get_models_dir() / 'random_forest_model.pkl'
    
    def get_scaler_file(self) -> Path:
        """スケーラーファイルパスを取得"""
        return self.get_models_dir() / 'scaler.pkl'
    
    def get_results_file(self) -> Path:
        """結果ファイルパスを取得"""
        return self.get_results_dir() / 'training_results.txt'
    
    def get_timestamp(self) -> str:
        """タイムスタンプを取得"""
        return self.timestamp if self.use_timestamp else "default"
