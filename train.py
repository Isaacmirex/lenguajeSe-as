"""
Entrenamiento de un modelo YOLOv8n (nano) para deteccion de senas ecuatorianas.
Modelo ligero (~6 MB) pensado para correr en moviles via TFLite/ONNX/CoreML.

Uso:
    python train.py

Salida:
    runs/detect/senas_v1/weights/best.pt   <- modelo entrenado
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = ROOT / "data.yaml"

def main():
    # YOLOv8n: la variante mas pequena y rapida (~3.2M params, ~6 MB).
    # Si quieres aun mas chico para movil de gama baja, prueba "yolov8n.pt" + imgsz=320.
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA),
        epochs=50,
        imgsz=416,          # 416 da buen balance precision/velocidad en movil
        batch=32,           # GPU 8GB -> batch mayor
        device=0,           # GPU 0 (RTX 2070 SUPER); usa "cpu" si no hay CUDA
        workers=2,
        patience=15,        # early stopping
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.1, scale=0.5, fliplr=0.5,
        project=str(ROOT / "runs" / "detect"),
        name="senas_v1",
        exist_ok=True,
        verbose=True,
    )

    # Validacion final sobre el set test
    metrics = model.val(data=str(DATA), split="test", imgsz=416)
    print("\n=== Metricas test ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
