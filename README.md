# Detección de Lenguaje de Señas Ecuatoriano — Backend

API REST con FastAPI + modelo YOLOv8n entrenado para detectar 33 señas
(letras, números y palabras) en tiempo real.

## Métricas del modelo
- mAP@0.5: **0.95**
- mAP@0.5-0.95: **0.67**
- Inferencia: **~3 ms / imagen** (GPU) · **~15 ms** (CPU)
- Tamaño: **6 MB** (.pt) / **3 MB** (TFLite INT8)

## Estructura
```
api.py              # FastAPI con JWT
train.py            # Entrenamiento YOLOv8n
webcam_test.py      # Prueba local con cámara + construcción de palabras
export_mobile.py    # Exporta a TFLite/ONNX/CoreML
generate_pdf.py     # Genera guía de integración móvil
generar_referencia.py  # Mosaico de las 33 señas
data.yaml           # Configuración del dataset
runs/detect/senas_v1/weights/best.pt  # Pesos entrenados
```

## Instalación
```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart pyjwt
```

Para GPU (opcional):
```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

## Correr la API
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Docs interactivos en http://localhost:8000/docs

## Endpoints
| Método | Endpoint | Auth | Descripción |
|---|---|---|---|
| POST | `/login` | no | JSON `{username, password}` → JWT |
| POST | `/token` | no | OAuth2 form (Swagger) |
| GET | `/me` | sí | Verifica el token |
| GET | `/classes` | no | Lista de 33 clases |
| POST | `/predict` | sí | Imagen multipart → detecciones |
| POST | `/predict_b64` | sí | JSON `{image_b64}` → detecciones |

Credenciales demo: `admin` / `admin123`

## Frontend
El cliente Next.js está en repo separado: `lenguajeSe-as_front`

## Deploy
- **NO usar Vercel** para este backend (límite 250 MB y PyTorch pesa ~3 GB)
- Recomendado: **Render.com**, **Railway**, **Fly.io**, **Hugging Face Spaces**

## Re-entrenamiento
El dataset NO está incluido en el repo (1700+ imágenes). Descárgalo desde
[Roboflow](https://universe.roboflow.com/ecuadorian-language-sign/ecuadorian-sign-language/dataset/3)
y descomprime en la raíz, luego:
```bash
python train.py
```
