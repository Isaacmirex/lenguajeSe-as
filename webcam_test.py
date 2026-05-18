"""
Prueba en vivo del modelo de senas con construccion de palabras en tiempo real.

Caracteristicas:
- Muestra cajas + clase con confianza.
- "PALABRA ACTUAL": construye palabra letra-por-letra a medida que detecta signos estables.
- "HISTORIAL": muestra las palabras ya completadas.
- Optimizado para velocidad en CPU: resolucion configurable, NMS afinado.

Logica de construccion:
- Cada frame toma la deteccion con mas confianza por encima de --conf.
- Una clase debe aparecer en N de los ultimos M frames para "comprometerse" (estable).
- Tras comprometerse, se ignoran repetidas de la misma clase hasta cambio.
- Si no se detecta nada por --space-after segundos -> espacio (terminar palabra).
- Las clases que son palabras completas (HOLA-5, MI, NOMBRE, Te amo) se anaden enteras.
- Tecla c: limpiar todo. Tecla espacio: forzar espacio. Tecla z: borrar ultimo caracter.

Uso:
    python webcam_test.py
    python webcam_test.py --imgsz 320 --conf 0.30   # mas rapido
    python webcam_test.py --source 0
"""

import argparse
import time
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "senas_v1" / "weights" / "best.pt"
CAPTURES = ROOT / "captures"

# Clases que son palabras enteras (no letras)
WORD_CLASSES = {"HOLA-5", "MI", "NOMBRE", "Te amo"}

# Mapeo para mostrar nombres "bonitos" (sin tocar las clases entrenadas)
DISPLAY_NAME = {"HOLA-5": "HOLA"}

# Tras este tiempo sin deteccion ni actividad, se limpia todo
AUTO_CLEAR_SECONDS = 10.0


def pretty(label):
    return DISPLAY_NAME.get(label, label)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--conf", type=float, default=0.30,
                   help="Umbral de confianza (modelo de 5 epochs -> usa 0.25-0.35)")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Tamano de inferencia. Mas chico = mas rapido (256/320/416)")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--stable-frames", type=int, default=6,
                   help="Frames necesarios para comprometer una sena")
    p.add_argument("--window-frames", type=int, default=10,
                   help="Tamano de ventana deslizante")
    p.add_argument("--space-after", type=float, default=1.5,
                   help="Segundos sin deteccion para terminar palabra")
    p.add_argument("--skip", type=int, default=0,
                   help="Procesar 1 de cada N+1 frames (0 = procesar todos)")
    return p.parse_args()


class WordBuilder:
    """Acumula detecciones estables en palabras y frases."""

    def __init__(self, stable_frames, window_frames, space_after):
        self.stable = stable_frames
        self.window = deque(maxlen=window_frames)
        self.space_after = space_after
        self.current_word = ""
        self.history = []          # palabras completadas
        self.last_committed = None
        self.last_detection_time = time.time()
        self.last_commit_time = 0.0   # para efecto flash al comprometer
        self.commit_stream = deque(maxlen=12)  # historial visual de letras recientes

    def push(self, label):
        """Llamado cada frame con la clase detectada (o None)."""
        self.window.append(label)
        now = time.time()

        if label is not None:
            self.last_detection_time = now
        else:
            # Sin deteccion: si paso el tiempo, cerrar palabra
            if (self.current_word
                    and now - self.last_detection_time > self.space_after):
                self.history.append(self.current_word)
                if len(self.history) > 8:
                    self.history = self.history[-8:]
                self.current_word = ""
                self.last_committed = None
            return

        # Encontrar clase estable en la ventana
        counts = Counter([x for x in self.window if x is not None])
        if not counts:
            return
        top_label, top_count = counts.most_common(1)[0]
        if top_count < self.stable:
            return
        if top_label == self.last_committed:
            return  # ya comprometida, esperar a que cambie

        # Comprometer
        if top_label in WORD_CLASSES:
            # Palabra entera: cerrar la actual si hay, agregar al historial
            if self.current_word:
                self.history.append(self.current_word)
                self.current_word = ""
            self.history.append(pretty(top_label))
            if len(self.history) > 8:
                self.history = self.history[-8:]
        else:
            self.current_word += top_label
        self.last_committed = top_label
        self.last_commit_time = time.time()
        self.commit_stream.append(pretty(top_label))

    def maybe_auto_clear(self):
        """Limpia todo si lleva AUTO_CLEAR_SECONDS sin actividad."""
        idle = time.time() - self.last_detection_time
        if idle > AUTO_CLEAR_SECONDS and (self.current_word or self.history or self.commit_stream):
            self.clear()
            return True
        return False

    def force_space(self):
        if self.current_word:
            self.history.append(self.current_word)
            self.current_word = ""
            self.last_committed = None
            if len(self.history) > 8:
                self.history = self.history[-8:]

    def backspace(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.last_committed = None
        elif self.history:
            self.history.pop()

    def clear(self):
        self.current_word = ""
        self.history = []
        self.last_committed = None
        self.window.clear()


def compose_canvas(frame, builder, fps, conf, top):
    """Compone un canvas: arriba la camara, abajo un panel grande con el texto."""
    cam_h, cam_w = frame.shape[:2]
    panel_h = 220
    canvas_w = cam_w
    canvas_h = cam_h + panel_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=frame.dtype)
    canvas[:cam_h] = frame
    canvas[cam_h:] = (25, 25, 30)

    # ----- Barra superior sobre la camara -----
    cv2.rectangle(canvas, (0, 0), (cam_w, 38), (0, 0, 0), -1)
    cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if top is not None:
        lbl, c = top
        color = (0, 255, 255) if c >= conf else (120, 120, 120)
        cv2.putText(canvas, f"DETECTANDO: {pretty(lbl)}  ({c:.2f})",
                    (160, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    else:
        cv2.putText(canvas, "DETECTANDO: -",
                    (160, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)

    # Countdown de auto-clear (arriba derecha)
    idle = time.time() - builder.last_detection_time
    remain = max(0.0, AUTO_CLEAR_SECONDS - idle)
    if (builder.current_word or builder.history) and idle > 1.0:
        cd_color = (0, 255, 255) if remain > 3 else (0, 0, 255)
        cv2.putText(canvas, f"Auto-limpiar en {remain:.1f}s",
                    (cam_w - 270, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cd_color, 2)

    # ----- Palabra actual (GRANDE) -----
    y_word = cam_h + 60
    cv2.putText(canvas, "PALABRA:", (10, y_word - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    word_text = builder.current_word if builder.current_word else "_"
    flash = max(0.0, 0.4 - (time.time() - builder.last_commit_time)) / 0.4
    color = (int(120 * flash), 255, int(120 * flash))
    cv2.putText(canvas, word_text, (10, y_word + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)

    # ----- Historial (palabras completas) -----
    y_hist = cam_h + panel_h - 70
    cv2.putText(canvas, "PALABRAS:", (10, y_hist),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    hist = "  ·  ".join(builder.history[-6:]) if builder.history else "(vacio)"
    cv2.putText(canvas, hist[-90:], (10, y_hist + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 200, 0), 2)

    cv2.putText(canvas, "q salir   c limpiar   espacio nueva palabra   z borrar   s capturar",
                (10, canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
    return canvas


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo en {weights}. Entrena primero con `python train.py`."
        )

    print(f"Cargando modelo: {weights}")
    model = YOLO(str(weights))
    names = model.names
    print(f"Clases: {list(names.values())}")

    src = int(args.source) if args.source.isdigit() else args.source
    print(f"Abriendo camara (src={src}, backend=DSHOW)...")
    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # DirectShow -> mas rapido en Windows
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la fuente: {src}. "
            "Si tienes varias camaras prueba --source 1 o --source 2."
        )
    # Solo setear FPS (los width/height pueden colgarse en algunas camaras Windows)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("Camara abierta. Iniciando bucle...")

    builder = WordBuilder(args.stable_frames, args.window_frames, args.space_after)
    CAPTURES.mkdir(exist_ok=True)

    fps_t0, frames = time.time(), 0
    fps_smooth = 0.0
    last_top = None
    skip_counter = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        do_infer = (skip_counter == 0)
        skip_counter = (skip_counter + 1) % (args.skip + 1)

        top_label = None
        top_conf = 0.0
        if do_infer:
            results = model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                verbose=False, half=False,
            )
            r = results[0]
            # Dibujar cajas
            frame = r.plot(line_width=2, font_size=0.5)

            # Top deteccion
            if r.boxes is not None and len(r.boxes) > 0:
                confs = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                idx = confs.argmax()
                top_label = names[int(cls[idx])]
                top_conf = float(confs[idx])
            last_top = (top_label, top_conf) if top_label else None

        # Alimentar al builder cada frame (aunque saltamos inferencia, mantenemos timing)
        builder.push(top_label)
        builder.maybe_auto_clear()

        # FPS suavizado
        frames += 1
        if frames % 5 == 0:
            elapsed = time.time() - fps_t0
            fps_smooth = frames / elapsed if elapsed > 0 else 0
            if elapsed > 2.0:  # reiniciar ventana
                fps_t0 = time.time()
                frames = 0

        canvas = compose_canvas(frame, builder, fps_smooth, args.conf, last_top)
        cv2.imshow("Deteccion de Senas - q salir", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            builder.clear()
        elif key == ord(" "):
            builder.force_space()
        elif key == ord("z"):
            builder.backspace()
        elif key == ord("s"):
            out = CAPTURES / f"capture_{int(time.time())}.jpg"
            cv2.imwrite(str(out), frame)
            print(f"Guardado: {out}")

    cap.release()
    cv2.destroyAllWindows()

    if builder.current_word:
        builder.history.append(builder.current_word)
    print("\n=== FRASE FINAL ===")
    print(" ".join(builder.history))


if __name__ == "__main__":
    main()
