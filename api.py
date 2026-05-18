"""
API REST para el modelo de deteccion de senas.

Endpoints:
    GET  /            -> info y lista de clases
    GET  /classes     -> lista de clases (con nombre "bonito")
    POST /predict     -> imagen multipart -> JSON con detecciones
    POST /predict_b64 -> JSON {"image_b64": "..."} -> JSON con detecciones

Inicio:
    pip install fastapi uvicorn python-multipart pillow
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

CORS:
    Permite cualquier origin por defecto (ajustar para produccion).
"""

from __future__ import annotations

import base64
import io
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import jwt
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordRequestForm,
)
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "runs" / "detect" / "senas_v1" / "weights" / "best.pt"

# ---------- JWT config ----------
JWT_SECRET = os.environ.get("JWT_SECRET", "cambia-esto-en-produccion-por-favor-32chars")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MIN = 60 * 8  # 8 horas

# Usuario unico (demo)
USERS = {"admin": "admin123"}

bearer_scheme = HTTPBearer(
    bearerFormat="JWT",
    description="Pega el JWT obtenido desde POST /login o POST /token",
)


def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MIN),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def require_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    creds_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in USERS:
            raise creds_exc
        return username
    except jwt.PyJWTError:
        raise creds_exc

# Mapeo "bonito" -> mostrar HOLA-5 como HOLA
DISPLAY_NAME = {"HOLA-5": "HOLA"}
WORD_CLASSES = {"HOLA-5", "MI", "NOMBRE", "Te amo"}


def pretty(label: str) -> str:
    return DISPLAY_NAME.get(label, label)


# ---------- Carga del modelo (una vez) ----------
if not WEIGHTS.exists():
    raise FileNotFoundError(
        f"No se encontro {WEIGHTS}. Entrena primero con `python train.py`."
    )

print(f"Cargando modelo: {WEIGHTS}")
model = YOLO(str(WEIGHTS))
CLASS_NAMES = model.names  # dict {id: name}
print(f"Modelo cargado. {len(CLASS_NAMES)} clases.")


# ---------- App ----------
app = FastAPI(
    title="Deteccion de Senas Ecuatorianas",
    version="1.0",
    description="API que detecta senas en una imagen y devuelve cajas + etiquetas.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------
class Detection(BaseModel):
    label: str          # nombre "bonito"
    raw_label: str      # nombre original del modelo
    confidence: float
    is_word: bool
    x1: float
    y1: float
    x2: float
    y2: float


class PredictResponse(BaseModel):
    detections: List[Detection]
    image_width: int
    image_height: int
    inference_ms: float


class Base64Request(BaseModel):
    image_b64: str
    conf: Optional[float] = 0.40
    imgsz: Optional[int] = 416


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_min: int = JWT_EXPIRE_MIN
    user: str


# ---------- Helpers ----------
def run_inference(img: Image.Image, conf: float, imgsz: int) -> PredictResponse:
    import time
    arr = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB -> BGR para YOLO
    t0 = time.time()
    results = model.predict(arr, imgsz=imgsz, conf=conf, verbose=False)
    inf_ms = (time.time() - t0) * 1000
    r = results[0]

    detections: list[Detection] = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy, confs, cls):
            raw = CLASS_NAMES[int(k)]
            detections.append(Detection(
                label=pretty(raw),
                raw_label=raw,
                confidence=float(c),
                is_word=raw in WORD_CLASSES,
                x1=float(box[0]), y1=float(box[1]),
                x2=float(box[2]), y2=float(box[3]),
            ))
    return PredictResponse(
        detections=detections,
        image_width=arr.shape[1],
        image_height=arr.shape[0],
        inference_ms=round(inf_ms, 2),
    )


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {
        "name": "Deteccion de Senas Ecuatorianas",
        "version": "1.0",
        "classes": [pretty(n) for n in CLASS_NAMES.values()],
        "num_classes": len(CLASS_NAMES),
        "endpoints": ["/classes", "/predict", "/predict_b64"],
    }


@app.get("/classes")
def classes():
    return {
        "classes": [
            {
                "id": int(k),
                "label": pretty(v),
                "raw_label": v,
                "is_word": v in WORD_CLASSES,
            }
            for k, v in CLASS_NAMES.items()
        ]
    }


@app.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    """Login con JSON (lo usa el frontend Next.js)."""
    expected = USERS.get(req.username)
    if expected is None or expected != req.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contrasena invalidos",
        )
    return TokenResponse(access_token=create_token(req.username), user=req.username)


@app.post("/token", response_model=TokenResponse)
def token(form: OAuth2PasswordRequestForm = Depends()):
    """Login OAuth2 estandar (form-urlencoded). Lo usa Swagger UI 'Authorize'."""
    expected = USERS.get(form.username)
    if expected is None or expected != form.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contrasena invalidos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(access_token=create_token(form.username), user=form.username)


@app.get("/me")
def me(user: str = Depends(require_user)):
    return {"user": user}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float = 0.40,
    imgsz: int = 416,
    _user: str = Depends(require_user),
):
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen invalida: {e}")
    return run_inference(img, conf, imgsz)


@app.post("/predict_b64", response_model=PredictResponse)
def predict_b64(req: Base64Request, _user: str = Depends(require_user)):
    b64 = req.image_b64
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    try:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 invalido: {e}")
    return run_inference(img, req.conf or 0.40, req.imgsz or 416)
