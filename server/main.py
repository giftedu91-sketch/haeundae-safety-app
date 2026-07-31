from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "best.pt"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))

app = FastAPI(title="Haeundae Marine Safety YOLO API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 후에는 GitHub Pages 주소만 허용하는 것을 권장합니다.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model: YOLO | None = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="server 폴더에 best.pt 파일이 없습니다. 다운로드한 모델을 server/best.pt로 넣어주세요.",
            )
        _model = YOLO(str(MODEL_PATH))
    return _model


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "message": "Haeundae Marine Safety YOLO API"}


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_exists": MODEL_PATH.exists(), "model_path": str(MODEL_PATH)}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="이미지는 10MB 이하여야 합니다.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.") from exc

    model = get_model()
    results = model.predict(source=image, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]
    names = result.names
    detections: list[dict[str, object]] = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(names[class_id]),
                    "confidence": round(confidence, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

    detections.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return {
        "filename": file.filename,
        "image_width": image.width,
        "image_height": image.height,
        "count": len(detections),
        "detections": detections,
    }
