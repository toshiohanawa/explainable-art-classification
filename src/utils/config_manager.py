"""
設定管理クラス
YAML設定ファイルの読み込みと管理を行う
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """設定ファイルの管理クラス"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = Path(config_path)
        self._config = None
        self.load_config()
    
    def load_config(self) -> None:
        """設定ファイルを読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"設定ファイルの形式が正しくありません: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """設定を取得"""
        if self._config is None:
            self.load_config()
        return self._config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """特定のセクションの設定を取得"""
        config = self.get_config()
        if section not in config:
            raise KeyError(f"設定セクション '{section}' が見つかりません")
        return config[section]
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """設定を更新"""
        if self._config is None:
            self.load_config()
        
        for key, value in updates.items():
            if key in self._config:
                self._config[key].update(value)
            else:
                self._config[key] = value
    
    def save_config(self) -> None:
        """設定をファイルに保存"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
