"""
Exporta el modelo entrenado a formatos optimizados para movil.

- TFLite (Android, Flutter, TF Lite Task Library)
- ONNX   (multiplataforma, NNAPI, ONNX Runtime Mobile)
- CoreML (iOS nativo)  - solo se intenta en macOS / si esta disponible

Uso:
    python export_mobile.py
    python export_mobile.py --weights runs/detect/senas_v1/weights/best.pt --imgsz 416
"""

import argparse
import platform
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "senas_v1" / "weights" / "best.pt"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--int8", action="store_true",
                   help="Cuantizacion INT8 para TFLite (mas chico, ~2-4x mas rapido)")
    return p.parse_args()

def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {weights}")

    model = YOLO(str(weights))
    print(f"Exportando desde: {weights}")

    print("\n[1/3] Exportando a ONNX...")
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)
    print(f"  -> {onnx_path}")

    print("\n[2/3] Exportando a TFLite...")
    try:
        tflite_path = model.export(
            format="tflite",
            imgsz=args.imgsz,
            int8=args.int8,
            data=str(ROOT / "data.yaml") if args.int8 else None,
        )
        print(f"  -> {tflite_path}")
    except Exception as e:
        print(f"  [aviso] TFLite fallo: {e}")
        print("  Instala: pip install tensorflow")

    print("\n[3/3] Exportando a CoreML...")
    if platform.system() == "Darwin":
        try:
            coreml_path = model.export(format="coreml", imgsz=args.imgsz, nms=True)
            print(f"  -> {coreml_path}")
        except Exception as e:
            print(f"  [aviso] CoreML fallo: {e}")
    else:
        print("  [omitido] CoreML solo se exporta correctamente desde macOS.")

    print("\nListo. Usa los archivos resultantes en tu app movil.")

if __name__ == "__main__":
    main()
