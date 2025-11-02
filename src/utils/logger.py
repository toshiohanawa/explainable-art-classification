"""
共通ログ設定ユーティリティ
プロジェクト全体で使用するログ設定を提供
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    ログ設定を初期化
    
    Args:
        log_file: ログファイルのパス（Noneの場合はファイル出力なし）
        log_level: ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_format: ログフォーマット（Noneの場合はデフォルト使用）
        config: 設定辞書（loggingセクションを指定可能）
    
    Returns:
        設定済みのLoggerインスタンス
    """
    # 設定から値を取得
    if config and 'logging' in config:
        logging_config = config['logging']
        log_level = log_level or logging_config.get('level', 'INFO')
        log_format = log_format or logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if log_file is None and 'file' in logging_config:
            log_file = logging_config['file']
    else:
        # デフォルトフォーマット
        log_format = log_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # ハンドラーのリスト
    handlers = []
    
    # コンソール出力用のハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # ファイル出力用のハンドラー（指定がある場合）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # ログ設定
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True  # 既存の設定を上書き
    )
    
    return logging.getLogger(__name__)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Loggerインスタンスを取得
    
    Args:
        name: Logger名（Noneの場合は呼び出し元のモジュール名）
    
    Returns:
        Loggerインスタンス
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'root')
    
    return logging.getLogger(name)

