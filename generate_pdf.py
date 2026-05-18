"""
Genera el PDF: 'Integracion del modelo de senas en una app movil'.
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, ListFlowable, ListItem, Preformatted,
)

ROOT = Path(__file__).parent
OUT = ROOT / "integracion_mobile.pdf"

# ---------- estilos ----------
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18,
                    textColor=colors.HexColor("#1F3A93"), spaceAfter=12)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14,
                    textColor=colors.HexColor("#22313F"), spaceBefore=10, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=12,
                    textColor=colors.HexColor("#34495E"), spaceBefore=6, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10.5,
                      leading=14, alignment=TA_JUSTIFY)
SMALL = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=9, leading=12)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontSize=8.5, leading=11,
                      textColor=colors.HexColor("#222"), backColor=colors.HexColor("#F4F4F4"),
                      borderColor=colors.HexColor("#DDDDDD"), borderWidth=0.5,
                      borderPadding=6, leftIndent=4, rightIndent=4)

def p(text, style=BODY):
    return Paragraph(text, style)

def code(text):
    return Preformatted(text, CODE)

def bullets(items):
    return ListFlowable(
        [ListItem(p(it), leftIndent=12) for it in items],
        bulletType="bullet", leftIndent=12,
    )

# ---------- contenido ----------
story = []

story.append(p("Deteccion de Senas Ecuatorianas — Guia de Integracion Movil", H1))
story.append(p(
    "Este documento describe como llevar el modelo entrenado (YOLOv8n) a una "
    "aplicacion movil. Cubre la exportacion del modelo, los formatos recomendados "
    "por plataforma, y ejemplos de integracion en Android (Kotlin), iOS (Swift), "
    "Flutter y React Native. Tambien incluye consejos de rendimiento para "
    "ejecucion en dispositivos de gama baja.", BODY))
story.append(Spacer(1, 0.4 * cm))

# Resumen del modelo
story.append(p("1. Resumen del modelo", H2))
tbl = Table([
    ["Arquitectura", "YOLOv8n (nano)"],
    ["Parametros", "~3.2 M"],
    ["Tamano .pt", "~6 MB"],
    ["Tamano TFLite FP16", "~6 MB"],
    ["Tamano TFLite INT8", "~3 MB"],
    ["Resolucion entrada", "416x416 (recomendado movil); 320x320 (gama baja)"],
    ["Clases (33)", "10, 2, 4, 7, 8, 9, A, B, C, D, E, F, G, H, HOLA-5, I, K, L, M, "
                   "MI, N, NOMBRE, O, P, Q, R, S, T, Te amo, U, W, X, Y"],
    ["Inferencia movil", "~30-60 FPS (Snapdragon 8xx / A14+); ~10-15 FPS (gama media)"],
], colWidths=[4.2 * cm, 11.5 * cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8EEF7")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7C3")),
    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
]))
story.append(tbl)
story.append(Spacer(1, 0.3 * cm))

# Por que YOLOv8n
story.append(p("2. Por que YOLOv8n para movil", H2))
story.append(bullets([
    "<b>Ligero:</b> ~6 MB y ~3.2 M parametros. Cabe sin problemas en cualquier APK/IPA.",
    "<b>Rapido:</b> inferencia en milisegundos, ideal para video en tiempo real.",
    "<b>Exportable:</b> ultralytics exporta nativamente a TFLite, ONNX, CoreML, NCNN.",
    "<b>Cuantizable:</b> INT8 reduce el tamano ~3x y acelera ~2-4x en CPUs ARM con perdida minima.",
    "<b>Detector + clasificador en uno:</b> da bounding box y clase de la sena al mismo tiempo.",
]))

# Exportar
story.append(p("3. Exportar el modelo", H2))
story.append(p("Una vez entrenado (best.pt en runs/detect/senas_v1/weights/), ejecuta:", BODY))
story.append(code(
    "# FP32 / FP16\n"
    "python export_mobile.py\n\n"
    "# INT8 cuantizado (mas chico y rapido en movil)\n"
    "python export_mobile.py --int8 --imgsz 320"
))
story.append(p("Esto produce, segun el sistema:", BODY))
story.append(bullets([
    "<b>best.onnx</b> — multiplataforma, Android (NNAPI/onnxruntime-mobile), iOS, Flutter.",
    "<b>best_saved_model/best_float32.tflite</b> — Android y Flutter.",
    "<b>best_int8.tflite</b> — Android gama baja, Edge devices.",
    "<b>best.mlpackage</b> — iOS (solo se genera desde macOS).",
]))

# Recomendacion por plataforma
story.append(p("4. Formato recomendado por plataforma", H2))
plat = Table([
    ["Plataforma", "Formato recomendado", "Runtime / SDK"],
    ["Android nativo (Kotlin/Java)", "TFLite (INT8)", "TFLite Task Vision / LiteRT"],
    ["iOS nativo (Swift)", "CoreML (mlpackage)", "Vision + CoreML"],
    ["Flutter (Android + iOS)", "TFLite", "tflite_flutter / ultralytics_yolo"],
    ["React Native", "TFLite o ONNX", "react-native-fast-tflite / onnxruntime-react-native"],
    ["Cross-platform (alternativa)", "ONNX", "onnxruntime-mobile (Android + iOS)"],
], colWidths=[4.5 * cm, 4.5 * cm, 6.7 * cm])
plat.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A93")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7C3")),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
]))
story.append(plat)

story.append(PageBreak())

# Android
story.append(p("5. Integracion en Android (Kotlin)", H2))
story.append(p("5.1. Dependencias en <i>app/build.gradle</i>:", H3))
story.append(code(
    "dependencies {\n"
    "    implementation(\"org.tensorflow:tensorflow-lite:2.14.0\")\n"
    "    implementation(\"org.tensorflow:tensorflow-lite-support:0.4.4\")\n"
    "    implementation(\"org.tensorflow:tensorflow-lite-gpu:2.14.0\")\n"
    "    implementation(\"androidx.camera:camera-camera2:1.3.1\")\n"
    "    implementation(\"androidx.camera:camera-lifecycle:1.3.1\")\n"
    "    implementation(\"androidx.camera:camera-view:1.3.1\")\n"
    "}\n\n"
    "android {\n"
    "    aaptOptions { noCompress \"tflite\" }\n"
    "}"
))
story.append(p("5.2. Colocar el modelo en <i>app/src/main/assets/senas.tflite</i> "
               "y un <i>labels.txt</i> con una clase por linea (mismo orden de data.yaml).", BODY))
story.append(p("5.3. Cargar e inferir con CameraX + TFLite:", H3))
story.append(code(
    "class YoloDetector(context: Context) {\n"
    "    private val interpreter: Interpreter\n"
    "    private val inputSize = 416\n"
    "    private val labels: List<String>\n\n"
    "    init {\n"
    "        val model = FileUtil.loadMappedFile(context, \"senas.tflite\")\n"
    "        val options = Interpreter.Options().apply { setNumThreads(4) }\n"
    "        interpreter = Interpreter(model, options)\n"
    "        labels = FileUtil.loadLabels(context, \"labels.txt\")\n"
    "    }\n\n"
    "    fun detect(bitmap: Bitmap, confTh: Float = 0.45f): List<Detection> {\n"
    "        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)\n"
    "        val input = TensorImage(DataType.FLOAT32).apply { load(resized) }\n"
    "        // Output YOLOv8: [1, 4 + nc, 8400]\n"
    "        val output = Array(1) { Array(4 + labels.size) { FloatArray(8400) } }\n"
    "        interpreter.run(input.buffer, output)\n"
    "        return postProcessYolov8(output[0], confTh, labels)\n"
    "    }\n"
    "}"
))
story.append(p("5.4. Post-procesado (resumen):", H3))
story.append(bullets([
    "Transponer la matriz de salida y, por cada una de las 8400 anclas, obtener la clase con "
    "mayor probabilidad.",
    "Filtrar por umbral de confianza (ej. 0.45).",
    "Aplicar Non-Max Suppression con IoU ~0.5 para eliminar cajas duplicadas.",
    "Reescalar las coordenadas (x, y, w, h) de espacio 416x416 al tamano de la imagen original.",
]))
story.append(p("Alternativa mas simple: usar la libreria <b>ultralytics-android</b> "
               "(MIT), que ya incluye preprocesado, postprocesado y NMS.", BODY))

story.append(PageBreak())

# iOS
story.append(p("6. Integracion en iOS (Swift)", H2))
story.append(p("6.1. Arrastra <i>best.mlpackage</i> dentro del proyecto Xcode. Xcode "
               "genera automaticamente una clase Swift con el nombre del modelo.", BODY))
story.append(p("6.2. Inferencia con Vision (ya incluye NMS):", H3))
story.append(code(
    "import Vision\n"
    "import CoreML\n\n"
    "class SignDetector {\n"
    "    private let request: VNCoreMLRequest\n\n"
    "    init() throws {\n"
    "        let config = MLModelConfiguration()\n"
    "        config.computeUnits = .all   // CPU + GPU + Neural Engine\n"
    "        let model = try VNCoreMLModel(for: best(configuration: config).model)\n"
    "        request = VNCoreMLRequest(model: model)\n"
    "        request.imageCropAndScaleOption = .scaleFill\n"
    "    }\n\n"
    "    func detect(pixelBuffer: CVPixelBuffer,\n"
    "                completion: @escaping ([VNRecognizedObjectObservation]) -> Void) {\n"
    "        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,\n"
    "                                             orientation: .right)\n"
    "        try? handler.perform([request])\n"
    "        let results = request.results as? [VNRecognizedObjectObservation] ?? []\n"
    "        completion(results)\n"
    "    }\n"
    "}"
))
story.append(p("6.3. Conectar con AVCaptureSession para video en tiempo real:", BODY))
story.append(bullets([
    "Configura <i>AVCaptureVideoDataOutput</i> con formato <b>kCVPixelFormatType_32BGRA</b>.",
    "En el callback <i>captureOutput(_:didOutput:from:)</i>, pasa el <i>CVPixelBuffer</i> "
    "directo al detector — evita conversiones a UIImage.",
    "Dibuja las cajas sobre una capa <i>CAShapeLayer</i> sincronizada con el preview.",
]))

# Flutter
story.append(p("7. Integracion en Flutter", H2))
story.append(p("La opcion mas directa es <b>ultralytics_yolo</b> (paquete oficial) que "
               "soporta camara, deteccion y dibujado en una sola dependencia:", BODY))
story.append(code(
    "# pubspec.yaml\n"
    "dependencies:\n"
    "  ultralytics_yolo: ^0.1.21\n"
    "  camera: ^0.10.5\n"
))
story.append(code(
    "// main.dart\n"
    "final controller = UltralyticsYoloCameraController();\n"
    "final predictor = ObjectDetector(\n"
    "  modelPath: 'assets/senas.tflite',\n"
    "  metadataPath: 'assets/metadata.yaml',\n"
    ");\n"
    "await predictor.loadModel(useGpu: true);\n\n"
    "UltralyticsYoloCameraPreview(\n"
    "  predictor: predictor,\n"
    "  controller: controller,\n"
    "  onCameraCreated: () => predictor.setConfidenceThreshold(0.45),\n"
    ");"
))
story.append(p("Si prefieres TFLite plano usa <b>tflite_flutter</b> + "
               "<b>tflite_flutter_helper</b> y replica el postprocesado YOLOv8 "
               "(la logica es la misma que en Android).", BODY))

story.append(PageBreak())

# React Native
story.append(p("8. Integracion en React Native", H2))
story.append(p("Opcion recomendada: <b>react-native-fast-tflite</b> (basado en JSI, ~zero copy).", BODY))
story.append(code(
    "npm i react-native-fast-tflite react-native-vision-camera"
))
story.append(code(
    "import { loadTensorflowModel } from 'react-native-fast-tflite';\n\n"
    "const model = await loadTensorflowModel(require('./assets/senas.tflite'));\n"
    "const output = model.runSync([inputTensor]);\n"
    "// output[0] -> Float32Array de [1 * (4+nc) * 8400]"
))
story.append(p("Combina con <b>react-native-vision-camera</b> y un frame processor "
               "para obtener el frame en formato adecuado para el modelo.", BODY))

# Performance
story.append(p("9. Consejos de rendimiento en movil", H2))
story.append(bullets([
    "<b>Reduce la resolucion</b> a 320x320 si necesitas mas FPS en gama baja.",
    "<b>Cuantiza a INT8</b> con datos representativos (data.yaml) — reduce el tamano ~3x.",
    "<b>Habilita aceleracion por hardware</b>:"
    " GPU delegate o NNAPI en Android, Neural Engine en iOS.",
    "<b>Procesa cada N frames</b> (ej. 1 de cada 2) y reutiliza el resultado anterior "
    "para los demas para mantener UI fluida.",
    "<b>Pre-asigna buffers</b> de entrada/salida; no los crees en cada frame.",
    "<b>Confianza minima alta</b> (0.5+) reduce falsos positivos y trabajo de NMS.",
    "<b>Imagen en RGB float32 / 255</b>: respeta el formato esperado por YOLOv8.",
]))

# Flujo final
story.append(p("10. Flujo recomendado de extremo a extremo", H2))
story.append(bullets([
    "Entrenar: <i>python train.py</i> (50 epochs, imgsz 416).",
    "Validar en webcam: <i>python webcam_test.py</i>.",
    "Exportar: <i>python export_mobile.py --int8</i>.",
    "Copiar <i>best_int8.tflite</i> a <i>assets/</i> de la app + <i>labels.txt</i>.",
    "Integrar con CameraX (Android) / AVCaptureSession (iOS) / camera (Flutter).",
    "Probar en dispositivo real y medir FPS; ajustar imgsz/cuantizacion si hace falta.",
]))

story.append(Spacer(1, 0.4 * cm))
story.append(p("<i>Generado automaticamente por generate_pdf.py</i>", SMALL))

# ---------- build ----------
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    title="Integracion del modelo de senas en movil",
    author="Proyecto deteccion_senas",
)
doc.build(story)
print(f"PDF generado: {OUT}")
