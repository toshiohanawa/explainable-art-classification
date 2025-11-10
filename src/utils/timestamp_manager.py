"""
成果物ディレクトリ管理ユーティリティ
ステージごとに最新版ディレクトリを更新しつつ、必要に応じてスナップショットを作成
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class TimestampManager:
    """成果物ディレクトリ/タイムスタンプ管理クラス"""

    STAGE_NAMES = [
        'features',
        'models',
        'results',
        'visualizations',
        'shap_explanations',
        'gestalt'
    ]

    def __init__(self, config: Dict[str, Any], timestamp: Optional[str] = None):
        """初期化"""
        self.config = config
        self.data_config = config['data']
        self.use_timestamp = self.data_config.get('use_timestamp', False)
        self.latest_label = self.data_config.get('latest_label', 'latest')
        self.snapshot_prefix = self.data_config.get('snapshot_prefix', 'snapshot')

        self.base_output_dir = Path(self.data_config['output_dir'])
        artifacts_subdir = self.data_config.get('artifacts_subdir', '').strip()
        experiments_subdir = self.data_config.get('experiments_subdir', '').strip()

        self.artifacts_root = self.base_output_dir / artifacts_subdir if artifacts_subdir else self.base_output_dir
        self.experiments_root = self.base_output_dir / experiments_subdir if experiments_subdir else self.base_output_dir

        raw_data_dir_config = self.data_config.get('raw_data_dir', self.base_output_dir / 'raw_data')
        self.raw_data_dir = Path(raw_data_dir_config)

        images_dir_config = Path(self.data_config.get('images_dir', 'raw_images'))
        self.images_dir = images_dir_config if images_dir_config.is_absolute() else self.base_output_dir / images_dir_config

        if self.use_timestamp:
            self.timestamp = timestamp or datetime.now().strftime("%y%m%d%H%M")
            self.output_dir = self.experiments_root / f"analysis_{self.timestamp}"
        else:
            self.timestamp = self.latest_label
            self.output_dir = self.artifacts_root

    def _stage_dir(self, stage_name: str) -> Path:
        if self.use_timestamp:
            return self.output_dir / stage_name
        return self.artifacts_root / stage_name / self.latest_label

    def get_output_dir(self) -> Path:
        """出力ディレクトリを取得"""
        return self.output_dir

    def get_images_dir(self) -> Path:
        return self.images_dir

    def get_raw_data_dir(self) -> Path:
        return self.raw_data_dir

    def get_features_dir(self) -> Path:
        return self._stage_dir('features')

    def get_models_dir(self) -> Path:
        return self._stage_dir('models')

    def get_results_dir(self) -> Path:
        return self._stage_dir('results')

    def get_visualizations_dir(self) -> Path:
        return self._stage_dir('visualizations')

    def get_shap_explanations_dir(self) -> Path:
        return self._stage_dir('shap_explanations')

    def get_gestalt_dir(self) -> Path:
        return self._stage_dir('gestalt')

    def create_directories(self) -> None:
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.experiments_root.mkdir(parents=True, exist_ok=True)

        stage_dirs = [self._stage_dir(stage) for stage in self.STAGE_NAMES]

        if self.use_timestamp:
            stage_dirs.append(self.output_dir)

        for directory in stage_dirs:
            directory.mkdir(parents=True, exist_ok=True)

        raw_directories = [self.get_images_dir(), self.get_raw_data_dir()]
        for directory in raw_directories:
            directory.mkdir(parents=True, exist_ok=True)

    def snapshot_stage(self, stage_name: str, timestamp: Optional[str] = None) -> Path:
        """最新版をスナップショットとして保存"""
        if stage_name not in self.STAGE_NAMES:
            raise ValueError(f"未知のステージです: {stage_name}")

        src = self._stage_dir(stage_name)
        if not src.exists():
            raise FileNotFoundError(f"スナップショット対象ディレクトリが見つかりません: {src}")

        snapshot_ts = timestamp or datetime.now().strftime("%y%m%d%H%M")
        dest = self.artifacts_root / stage_name / f"{self.snapshot_prefix}_{snapshot_ts}"
        shutil.copytree(src, dest, dirs_exist_ok=True)
        return dest

    def get_metadata_file(self) -> Path:
        return self.get_raw_data_dir() / self.data_config['metadata_file']

    def get_features_file(self) -> Path:
        return self.get_features_dir() / self.data_config['features_file']

    def get_model_file(self) -> Path:
        return self.get_models_dir() / 'random_forest_model.pkl'

    def get_scaler_file(self) -> Path:
        return self.get_models_dir() / 'scaler.pkl'

    def get_results_file(self) -> Path:
        return self.get_results_dir() / 'training_results.txt'

    def get_gestalt_scores_file(self) -> Path:
        return self.get_gestalt_dir() / 'gestalt_scores.parquet'

    def get_timestamp(self) -> str:
        return self.timestamp
