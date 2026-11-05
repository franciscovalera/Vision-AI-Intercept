"""Detección de la pelota en directo con la OAK, viendo la imagen y las cajas.

Dos modos:
  - Red dentro de la OAK (por defecto): MonoCamera izq -> ImageManip -> YoloDetectionNetwork.
    Se muestra el frame exacto que ha procesado la red (passthrough), así que las
    cajas siempre coinciden con la imagen; por USB solo viajan esos frames.
  - Red en el ordenador (--modelo best.pt): la cámara manda la imagen mono y
    Ultralytics detecta en el host (CPU o GPU). Sirve para comparar con el blob.

    python codigo/detectar_en_vivo.py                                         # blob 416x256 en la OAK
    python codigo/detectar_en_vivo.py --blob modelos/oak/pelota_v1_320x192.blob
    python codigo/detectar_en_vivo.py --modelo modelos/entrenamientos/pelota_v1/weights/best.pt

La exposición es automática con límite (como en la grabación); --limite-auto la cambia.

Teclas: q / Esc salir · f pantalla completa · + / - subir o bajar la confianza mínima · g guardar captura
"""

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from grabacion import LIMITE_AUTO_US, RESOLUCIONES_MONO, Configuracion, aplicar_autoexposicion
from medir_oak import leer_config

RAIZ = Path(__file__).resolve().parents[1]
BLOB_POR_DEFECTO = RAIZ / "modelos" / "oak" / "pelota_v1_416x256.blob"
VENTANA = "Deteccion en directo"
VERDE, AMARILLO, GRIS = (0, 255, 0), (0, 255, 255), (170, 170, 170)


def crear_camara(pipeline, fps, limite_auto_us):
    mono = pipeline.create(dai.node.MonoCamera)
    mono.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono.setResolution(RESOLUCIONES_MONO[400][0])
    mono.setFps(fps)
    aplicar_autoexposicion(mono.initialControl, Configuracion(fps=fps, limite_auto_us=limite_auto_us))
    return mono


def pipeline_oak(blob, fps, limite_auto_us, hilos):
    """Red en la OAK. La confianza mínima de la red es baja (0,1): el umbral se aplica en el host
    para poder cambiarlo en directo con + / -."""
    ancho, alto, meta = leer_config(blob)
    pipeline = dai.Pipeline()
    mono = crear_camara(pipeline, fps, limite_auto_us)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(ancho, alto)
    manip.initialConfig.setKeepAspectRatio(False)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setMaxOutputFrameSize(ancho * alto * 3)
    mono.out.link(manip.inputImage)

    red = pipeline.create(dai.node.YoloDetectionNetwork)
    red.setBlobPath(str(blob))
    red.setNumClasses(meta["classes"])
    red.setCoordinateSize(meta["coordinates"])
    red.setAnchors(meta["anchors"])
    red.setAnchorMasks(meta["anchor_masks"])
    red.setIouThreshold(meta["iou_threshold"])
    red.setConfidenceThreshold(0.1)
    red.setNumInferenceThreads(hilos)
    red.input.setBlocking(False)
    red.input.setQueueSize(1)
    manip.out.link(red.input)

    for nombre, salida in (("detecciones", red.out), ("imagen", red.passthrough)):
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName(nombre)
        salida.link(xout.input)
    return pipeline


def pipeline_host(fps, limite_auto_us):
    pipeline = dai.Pipeline()
    mono = crear_camara(pipeline, fps, limite_auto_us)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("imagen")
    # Si el host va más lento que la cámara, los frames se descartan en la OAK en vez de acumularse
    xout.input.setBlocking(False)
    xout.input.setQueueSize(1)
    mono.out.link(xout.input)
    return pipeline


class FuenteOak:
    """Frames y detecciones de la red de la OAK, emparejados por número de secuencia."""

    def __init__(self, dispositivo):
        self.cola_det = dispositivo.getOutputQueue("detecciones", maxSize=8, blocking=False)
        self.cola_img = dispositivo.getOutputQueue("imagen", maxSize=8, blocking=False)
        self.detecciones, self.imagenes = {}, {}  # por número de secuencia, a la espera de su pareja

    def siguiente(self):
        """(imagen BGR, [(conf, x1, y1, x2, y2) normalizados], timestamp de captura).

        Devuelve la pareja más reciente. Si la ventana va más lenta que la red, las colas descartan
        mensajes y a veces falta la imagen de una detección (o al revés): esas se saltan, no se esperan.
        """
        while True:
            for paquete in self.cola_det.tryGetAll():
                self.detecciones[paquete.getSequenceNum()] = paquete
            for frame in self.cola_img.tryGetAll():
                self.imagenes[frame.getSequenceNum()] = frame
            comunes = self.detecciones.keys() & self.imagenes.keys()
            if comunes:
                secuencia = max(comunes)
                detecciones, frame = self.detecciones[secuencia], self.imagenes[secuencia]
                self.detecciones = {k: v for k, v in self.detecciones.items() if k > secuencia}
                self.imagenes = {k: v for k, v in self.imagenes.items() if k > secuencia}
                break
            # Esperar a que llegue lo que falta (sin acumular indefinidamente las que no tienen pareja)
            if not self.detecciones:
                paquete = self.cola_det.get()
                self.detecciones[paquete.getSequenceNum()] = paquete
            else:
                frame = self.cola_img.get()
                self.imagenes[frame.getSequenceNum()] = frame
            for pendientes in (self.detecciones, self.imagenes):
                for k in sorted(pendientes)[:-16]:
                    del pendientes[k]
        cajas = [(d.confidence, d.xmin, d.ymin, d.xmax, d.ymax) for d in detecciones.detections]
        imagen = cv2.resize(frame.getCvFrame(), (640, 400))  # deshace el estirado del ImageManip
        return imagen, cajas, detecciones.getTimestamp()


class FuenteHost:
    """Frames mono de la cámara y detección con Ultralytics en el ordenador."""

    def __init__(self, dispositivo, ruta_modelo, imgsz, dispositivo_torch):
        from ultralytics import YOLO
        self.cola = dispositivo.getOutputQueue("imagen", maxSize=1, blocking=False)
        self.modelo = YOLO(ruta_modelo)
        self.imgsz = imgsz
        self.dispositivo_torch = dispositivo_torch

    def siguiente(self):
        frames = self.cola.tryGetAll() or [self.cola.get()]
        frame = frames[-1]  # siempre el más reciente: si el host es lento, se saltan frames
        imagen = cv2.cvtColor(frame.getCvFrame(), cv2.COLOR_GRAY2BGR)
        resultado = self.modelo.predict(imagen, imgsz=self.imgsz, conf=0.1, device=self.dispositivo_torch,
                                        verbose=False)[0]
        alto, ancho = imagen.shape[:2]
        cajas = [(float(c), x1 / ancho, y1 / alto, x2 / ancho, y2 / alto)
                 for (x1, y1, x2, y2), c in zip(resultado.boxes.xyxy.tolist(), resultado.boxes.conf.tolist())]
        return imagen, cajas, frame.getTimestamp()


def texto(imagen, cadena, posicion, color=(255, 255, 255)):
    cv2.putText(imagen, cadena, posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(imagen, cadena, posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blob", type=Path, default=BLOB_POR_DEFECTO, help="red para la OAK (con su .json)")
    parser.add_argument("--modelo", type=Path, default=None, help="detectar en el ordenador con este .pt")
    parser.add_argument("--imgsz", type=int, default=640, help="tamaño de inferencia con --modelo")
    parser.add_argument("--dispositivo", default=None, help="con --modelo: cpu, 0 (GPU)...")
    parser.add_argument("--conf", type=float, default=0.5, help="confianza mínima inicial (cambia con + / -)")
    parser.add_argument("--fps", type=float, default=120, help="FPS de la cámara mono")
    parser.add_argument("--limite-auto", type=int, default=LIMITE_AUTO_US,
                        help=f"exposición máxima en automático, µs (por defecto {LIMITE_AUTO_US})")
    parser.add_argument("--hilos", type=int, choices=[1, 2], default=2, help="hilos de inferencia en la OAK")
    args = parser.parse_args()

    if args.modelo:
        # Con la red en el host no tiene sentido pedir 120 FPS por USB: la red no los procesa
        pipeline, nombre = pipeline_host(min(args.fps, 60), args.limite_auto), f"{args.modelo.name} (ordenador)"
    else:
        pipeline, nombre = pipeline_oak(args.blob, args.fps, args.limite_auto, args.hilos), f"{args.blob.stem} (OAK)"

    conf = args.conf
    capturas = RAIZ / "resultados" / "oak" / "capturas"
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(VENTANA, 1280, 800)

    with dai.Device(pipeline) as dispositivo:
        usb = dispositivo.getUsbSpeed().name
        fuente = (FuenteHost(dispositivo, args.modelo, args.imgsz, args.dispositivo) if args.modelo
                  else FuenteOak(dispositivo))
        instantes, latencias = deque(maxlen=60), deque(maxlen=60)
        print(f"{nombre} · USB {usb} · q para salir")
        while True:
            imagen, cajas, marca = fuente.siguiente()
            latencias.append((dai.Clock.now() - marca).total_seconds() * 1000)
            instantes.append(time.monotonic())
            fps = (len(instantes) - 1) / (instantes[-1] - instantes[0]) if len(instantes) > 1 else 0.0

            alto, ancho = imagen.shape[:2]
            visibles = [c for c in cajas if c[0] >= conf]
            for c, x1, y1, x2, y2 in visibles:
                p1, p2 = (int(x1 * ancho), int(y1 * alto)), (int(x2 * ancho), int(y2 * alto))
                cv2.rectangle(imagen, p1, p2, VERDE, 2)
                texto(imagen, f"{c:.2f}", (p1[0], max(14, p1[1] - 5)), VERDE)
            texto(imagen, f"{nombre} · USB {usb}", (10, 20))
            texto(imagen, f"{fps:5.1f} FPS · latencia {np.mean(latencias):5.1f} ms · conf >= {conf:.2f}", (10, 40),
                  AMARILLO)
            texto(imagen, "PELOTA" if visibles else "sin pelota", (10, 60), VERDE if visibles else GRIS)
            cv2.imshow(VENTANA, imagen)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27) or cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                break
            if tecla == ord("f"):
                completa = cv2.getWindowProperty(VENTANA, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(VENTANA, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL if completa else cv2.WINDOW_FULLSCREEN)
            elif tecla in (ord("+"), ord("=")):
                conf = min(0.95, conf + 0.05)
            elif tecla == ord("-"):
                conf = max(0.1, conf - 0.05)
            elif tecla == ord("g"):
                capturas.mkdir(parents=True, exist_ok=True)
                ruta = capturas / f"{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(str(ruta), imagen)
                print(f"Captura: {ruta}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
