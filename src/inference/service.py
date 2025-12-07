"""
推論サービス: 画像アップロードから特徴量抽出、モデル推論、SHAP/ゲシュタルト指標計算までをまとめる。
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from PIL import Image

from ..feature_extraction.color_extractor import ColorFeatureExtractor
from ..feature_extraction.gestalt_scorer import GestaltScorer
from ..utils.config_manager import ConfigManager
from ..utils.timestamp_manager import TimestampManager

logger = logging.getLogger(__name__)


class InferenceService:
    """ランダムフォレストモデルの推論を提供するサービスクラス。"""

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_gestalt: Optional[bool] = None,
        gestalt_model: Optional[str] = None,
    ) -> None:
        default_config = Path(__file__).resolve().parents[2] / "config.yaml"
        config_path = config_path or str(default_config)
        self.project_root = Path(config_path).resolve().parent
        self.config = ConfigManager(config_path).get_config()
        data_cfg = self.config.get("data", {})
        if data_cfg.get("output_dir") and not Path(data_cfg["output_dir"]).is_absolute():
            data_cfg["output_dir"] = str(self.project_root / data_cfg["output_dir"])
        if data_cfg.get("raw_data_dir") and not Path(data_cfg["raw_data_dir"]).is_absolute():
            data_cfg["raw_data_dir"] = str(self.project_root / data_cfg["raw_data_dir"])
        if data_cfg.get("images_dir") and not Path(data_cfg["images_dir"]).is_absolute():
            data_cfg["images_dir"] = str(self.project_root / data_cfg["images_dir"])
        self.timestamp_manager = TimestampManager(self.config)
        self.model_path = self.timestamp_manager.get_model_file()
        self.scaler_path = self.timestamp_manager.get_scaler_file()
        self.enable_gestalt = (
            enable_gestalt
            if enable_gestalt is not None
            else os.getenv("ENABLE_GESTALT", "false").lower() == "true"
        )
        self.gestalt_model = gestalt_model or self.config.get("gestalt_scoring", {}).get(
            "model_name", "llava:7b"
        )

        self.model = None
        self.scaler = None
        self.feature_columns: List[str] = []
        self.model_version = self._resolve_model_version()

        self.feature_extractor = ColorFeatureExtractor(
            self.config, timestamp_manager=self.timestamp_manager
        )

        self._load_model()

    def _resolve_model_version(self) -> str:
        """モデルファイルの更新日時をバージョンとして返す。"""
        if not self.model_path.exists():
            return "unknown"
        ts = datetime.fromtimestamp(self.model_path.stat().st_mtime)
        return ts.strftime("%Y-%m-%d %H:%M:%S")

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)
        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.scaler = None
        if self.scaler_path.exists():
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        logger.info(
            "モデル読み込み完了 | features=%d | scaler=%s",
            len(self.feature_columns),
            "yes" if self.scaler is not None else "no",
        )

    def _top_feature_importances(self, features: Dict[str, float], top_n: int = 10) -> List[Dict[str, float]]:
        """ランダムフォレストの特徴量重要度ランキング（特徴量値付き）"""
        if not hasattr(self.model, "feature_importances_"):
            return []
        importances = self.model.feature_importances_
        pairs = []
        for name, imp in zip(self.feature_columns, importances):
            pairs.append({"feature": name, "importance": float(imp), "value": float(features.get(name, 0.0))})
        pairs.sort(key=lambda x: x["importance"], reverse=True)
        return pairs[:top_n]

    def _validate_extension(self, filename: str) -> None:
        ext = Path(filename).suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"未対応の拡張子です: {ext} (許可: {', '.join(sorted(self.ALLOWED_EXTENSIONS))})")

    def _extract_metadata(self, image_path: Path) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"file_name": image_path.name}
        meta["file_size_bytes"] = image_path.stat().st_size if image_path.exists() else 0
        try:
            with Image.open(image_path) as img:
                meta["width"], meta["height"] = img.size
                meta["format"] = img.format
        except Exception as e:  # noqa: BLE001
            logger.warning("EXIFメタデータ取得に失敗しました: %s", e)
        return meta

    def _compute_gestalt_scores(self, image_path: Path) -> Tuple[Dict[str, float], str]:
        """ゲシュタルトスコアを計算。失敗時は0で埋め、ステータスを返す。"""
        if not self.enable_gestalt:
            return {}, "skipped (ENABLE_GESTALT is false)"
        try:
            scorer = GestaltScorer(
                self.config,
                model_name=self.gestalt_model,
                timestamp_manager=self.timestamp_manager,
            )
            scores = scorer.score_single_image(image_path)
            flattened: Dict[str, float] = {}
            for key in scorer.GESTALT_PRINCIPLES.keys():
                val = scores.get(key, {})
                if isinstance(val, dict):
                    flattened[f"{key}_score"] = float(val.get("score", 0.0))
                else:
                    flattened[f"{key}_score"] = float(val)
            return flattened, "ok"
        except Exception as e:  # noqa: BLE001
            logger.warning("ゲシュタルト計算に失敗しました: %s", e)
            return {}, f"failed: {e}"

    def _build_feature_vector(self, feature_dict: Dict[str, Any]) -> np.ndarray:
        return np.array([float(feature_dict.get(col, 0.0)) for col in self.feature_columns], dtype=float)

    def _predict_vector(self, X: np.ndarray, features: Dict[str, float]) -> Dict[str, Any]:
        proba = self.model.predict_proba(X)[0]
        prob_authentic = float(proba[0])
        prob_fake = float(proba[1])
        label = "fake" if prob_fake >= prob_authentic else "authentic"
        top_importances = self._top_feature_importances(features)
        return {
            "label": label,
            "probabilities": {"authentic": prob_authentic, "fake": prob_fake},
            "confidence": prob_fake if label == "fake" else prob_authentic,
            "feature_importances": top_importances,
        }

    def predict_image_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        include_gestalt: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """画像バイト列から推論を実行。"""
        self._validate_extension(filename)
        gestalt_flag = self.enable_gestalt if include_gestalt is None else include_gestalt

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            image = self.feature_extractor.preprocess_image(tmp_path)
            if image is None:
                raise ValueError("画像の読み込みまたは前処理に失敗しました。")

            features = self.feature_extractor.extract_all_features(image)
            gestalt_scores, gestalt_status = self._compute_gestalt_scores(tmp_path) if gestalt_flag else ({}, "skipped")
            features.update(gestalt_scores)

            vector = self._build_feature_vector(features).reshape(1, -1)
            if self.scaler is not None:
                vector = self.scaler.transform(vector)

            prediction = self._predict_vector(vector, features)
            metadata = self._extract_metadata(tmp_path)

            return {
                "prediction": prediction,
                "features": {k: float(v) for k, v in features.items()},
                "feature_columns": self.feature_columns,
                "gestalt_status": gestalt_status,
                "model_version": self.model_version,
                "metadata": metadata,
            }
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
