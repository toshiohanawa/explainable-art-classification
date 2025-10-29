# -*- coding: utf-8 -*-
"""
最適化テスト用の簡単なスクリプト
"""

import sys
import traceback
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent))

try:
    from src.utils.config_manager import ConfigManager
    print("ConfigManager import: OK")
    
    config_manager = ConfigManager()
    print("ConfigManager instantiation: OK")
    
    config = config_manager.get_config()
    print(f"Config loaded: {type(config)}")
    print(f"Config keys: {list(config.keys()) if config else 'None'}")
    
    if config:
        print(f"API config: {config.get('api', 'Not found')}")
        print(f"Data config: {config.get('data', 'Not found')}")
        print(f"Hybrid config: {config.get('hybrid_collection', 'Not found')}")
    
except Exception as e:
    print(f"Error: {e}")
    print(f"Traceback: {traceback.format_exc()}")
