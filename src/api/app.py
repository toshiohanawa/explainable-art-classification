"""
FastAPIエントリポイント。画像アップロード→推論結果（本物/偽物確率、SHAP上位特徴、ゲシュタルト指標、色特徴、EXIF）を返す。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ..inference.service import InferenceService

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Explainable Art Authenticity API",
    description="学習済みランダムフォレストモデルで本物/偽物を判定し、SHAP・ゲシュタルト・色特徴を返すAPI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))  # 8MBデフォルト
inference_service = InferenceService()


@app.get("/health", summary="疎通確認")
def health() -> Dict[str, Any]:
    # モデルがない場合でも落とさず状態を返す
    return {
        "status": "ok",
        "model_version": inference_service.model_version,
        "feature_count": len(inference_service.feature_columns),
        "gestalt_enabled": inference_service.enable_gestalt,
        "model_error": inference_service.model_load_error,
        "model_loaded": inference_service.model is not None,
    }


@app.post("/predict", summary="画像を本物/偽物で判定")
async def predict(
    file: UploadFile = File(...),
    include_gestalt: bool = False,
) -> Dict[str, Any]:
    content = await file.read()
    filename = (file.filename or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="画像が空です。")
    if not filename:
        raise HTTPException(status_code=400, detail="ファイル名が空です。")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="ファイルサイズが上限を超えています。")

    try:
        result = inference_service.predict_image_bytes(
            content,
            filename=filename,
            include_gestalt=include_gestalt,
        )
        result["file_name"] = filename
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("推論中にエラーが発生しました")
        raise HTTPException(status_code=500, detail=f"推論エラー: {e}")
