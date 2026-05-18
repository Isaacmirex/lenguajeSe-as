"""
Genera carpeta examples/ con una imagen por clase + un mosaico de referencia.
Toma una muestra representativa del set de entrenamiento para cada clase.
"""
from pathlib import Path
import random
import cv2
import numpy as np
import yaml


def imread_unicode(path):
    """cv2.imread compatible con rutas Unicode en Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok

ROOT = Path(__file__).parent
DATA = ROOT / "data.yaml"
IMAGES = ROOT / "train" / "images"
LABELS = ROOT / "train" / "labels"
OUT_DIR = ROOT / "examples"
MOSAIC = ROOT / "senas_referencia.png"

with open(DATA, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
NAMES = cfg["names"]
NC = cfg["nc"]

random.seed(0)
OUT_DIR.mkdir(exist_ok=True)

# 1) Para cada clase, juntar todas las (img, bbox) y elegir la del bbox mas grande
per_class = {i: [] for i in range(NC)}
for lbl in LABELS.glob("*.txt"):
    img = IMAGES / (lbl.stem + ".jpg")
    if not img.exists():
        continue
    for line in lbl.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid, xc, yc, w, h = parts
        cid = int(cid)
        area = float(w) * float(h)
        per_class[cid].append((area, img, (float(xc), float(yc), float(w), float(h))))

# 2) Recortar y guardar la mejor por clase (mayor area)
tiles = []
TILE_SIZE = 224
PAD = 0.15  # 15% de padding alrededor del bbox

for cid in range(NC):
    name = NAMES[cid]
    candidates = per_class[cid]
    if not candidates:
        print(f"[!] sin ejemplos para clase {cid} ({name})")
        # Tile placeholder
        tile = np.full((TILE_SIZE, TILE_SIZE, 3), 60, dtype=np.uint8)
        cv2.putText(tile, "sin ejemplo", (15, TILE_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    else:
        # Top 5 por area, elegir aleatoria (variedad) - seed fija
        candidates.sort(reverse=True)
        _, img_path, (xc, yc, w, h) = random.choice(candidates[:5])
        img = imread_unicode(img_path)
        if img is None:
            print(f"[!] no se pudo leer {img_path}")
            tile = np.full((TILE_SIZE, TILE_SIZE, 3), 60, dtype=np.uint8)
            cv2.putText(tile, "error lectura", (10, TILE_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            label_h = 40
            framed = np.full((TILE_SIZE + label_h, TILE_SIZE, 3), 30, dtype=np.uint8)
            framed[:TILE_SIZE] = tile
            cv2.putText(framed, name, (10, TILE_SIZE + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
            tiles.append(framed)
            continue
        ih, iw = img.shape[:2]
        # Convertir YOLO normalizado -> pixeles, con padding
        cx, cy = xc * iw, yc * ih
        bw, bh = w * iw, h * ih
        side = max(bw, bh) * (1 + PAD * 2)
        x1 = int(max(0, cx - side / 2))
        y1 = int(max(0, cy - side / 2))
        x2 = int(min(iw, cx + side / 2))
        y2 = int(min(ih, cy + side / 2))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img
        tile = cv2.resize(crop, (TILE_SIZE, TILE_SIZE))

        # Guardar individual
        safe_name = name.replace(" ", "_").replace("/", "_")
        imwrite_unicode(OUT_DIR / f"{safe_name}.jpg", tile)

    # Bordes y etiqueta debajo
    label_h = 40
    framed = np.full((TILE_SIZE + label_h, TILE_SIZE, 3), 30, dtype=np.uint8)
    framed[:TILE_SIZE] = tile
    cv2.rectangle(framed, (0, 0), (TILE_SIZE - 1, TILE_SIZE + label_h - 1),
                  (80, 80, 80), 1)
    # Calcular tamano de texto centrado
    text = name
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    tx = (TILE_SIZE - tw) // 2
    cv2.putText(framed, text, (tx, TILE_SIZE + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
    tiles.append(framed)

# 3) Componer mosaico (7 columnas)
cols = 7
rows = (len(tiles) + cols - 1) // cols
tile_w = tiles[0].shape[1]
tile_h = tiles[0].shape[0]
gap = 8
mosaic_w = cols * tile_w + (cols + 1) * gap
mosaic_h = rows * tile_h + (rows + 1) * gap + 60  # 60 para titulo
mosaic = np.full((mosaic_h, mosaic_w, 3), 20, dtype=np.uint8)

# Titulo
title = f"Lengua de senas ecuatoriana - {NC} clases"
cv2.putText(mosaic, title, (gap, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

for i, tile in enumerate(tiles):
    r, c = divmod(i, cols)
    y = 60 + gap + r * (tile_h + gap)
    x = gap + c * (tile_w + gap)
    mosaic[y:y + tile_h, x:x + tile_w] = tile

imwrite_unicode(MOSAIC, mosaic)
print(f"\nIndividuales en: {OUT_DIR}")
print(f"Mosaico: {MOSAIC}")
print(f"Total clases procesadas: {NC}")
