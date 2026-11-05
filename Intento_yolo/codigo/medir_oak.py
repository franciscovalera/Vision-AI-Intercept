"""Mide el detector dentro de la OAK: FPS, latencia y detecciones de cada .blob.

Cadena en la cámara (la misma que usará el sistema final):
    MonoCamera izq (640x400, GRAY8) -> ImageManip (GRAY8 -> BGR888p, ancho x alto)
    -> YoloDetectionNetwork (decodificación y NMS en la OAK) -> host

La red recibe los frames sin bloquear la cámara: si es más lenta que la cámara,
descarta frames, y los FPS medidos son los que la red da de verdad.

Modo en vivo (por defecto): mide cada blob durante --segundos.
    python codigo/medir_oak.py                                   # todos los blobs de modelos/oak
    python codigo/medir_oak.py --blobs modelos/oak/pelota_v1_416x256.blob --ver

Modo clip: manda desde el host los frames de un clip grabado (uno a uno, sin
perder ninguno) para comprobar que el blob dentro de la cámara detecta bien.
Guarda una hoja con ejemplos de detecciones en resultados/oak/.
    python codigo/medir_oak.py --clip grabaciones/20261023_133016/clip_002/parte_5

Resultados en resultados/oak/medidas.csv (una fila por blob y modo).
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from grabacion import RESOLUCIONES_MONO
from lectura_clips import ClipGrabado

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_BLOBS = RAIZ / "modelos" / "oak"
CARPETA_RESULTADOS = RAIZ / "resultados" / "oak"


def leer_config(blob):
    """Ancho, alto y metadatos YOLO del JSON que acompaña al blob."""
    config = json.loads(blob.with_suffix(".json").read_text())["nn_config"]
    ancho, alto = (int(v) for v in config["input_size"].split("x"))
    return ancho, alto, config["NN_specific_metadata"]


def crear_pipeline(blob, conf, fps, desde_host, ver=False, hilos=2):
    ancho, alto, meta = leer_config(blob)
    pipeline = dai.Pipeline()

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(ancho, alto)
    manip.initialConfig.setKeepAspectRatio(False)  # estira: 640x400 tiene casi la misma proporción
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setMaxOutputFrameSize(ancho * alto * 3)

    if desde_host:
        entrada = pipeline.create(dai.node.XLinkIn)
        entrada.setStreamName("entrada")
        entrada.setMaxDataSize(1280 * 800)
        entrada.out.link(manip.inputImage)
    else:
        mono = pipeline.create(dai.node.MonoCamera)
        mono.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono.setResolution(RESOLUCIONES_MONO[400][0])
        mono.setFps(fps)
        mono.out.link(manip.inputImage)
        if ver:  # la imagen solo se manda al host si se va a mostrar (ocupa USB)
            xout_imagen = pipeline.create(dai.node.XLinkOut)
            xout_imagen.setStreamName("imagen")
            mono.out.link(xout_imagen.input)

    red = pipeline.create(dai.node.YoloDetectionNetwork)
    red.setBlobPath(str(blob))
    red.setNumClasses(meta["classes"])
    red.setCoordinateSize(meta["coordinates"])
    red.setAnchors(meta["anchors"])
    red.setAnchorMasks(meta["anchor_masks"])
    red.setIouThreshold(meta["iou_threshold"])
    red.setConfidenceThreshold(conf)
    red.setNumInferenceThreads(hilos)  # 2: más FPS; 1: menos latencia
    red.input.setBlocking(desde_host)  # en vivo: si la red no da abasto, descarta frames
    red.input.setQueueSize(1 if not desde_host else 4)
    manip.out.link(red.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("detecciones")
    red.out.link(xout.input)
    return pipeline


def dibujar(imagen, detecciones):
    alto, ancho = imagen.shape[:2]
    for d in detecciones:
        x1, y1, x2, y2 = int(d.xmin * ancho), int(d.ymin * alto), int(d.xmax * ancho), int(d.ymax * alto)
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(imagen, f"{d.confidence:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return imagen


def medir_en_vivo(dispositivo, blob, segundos, ver):
    cola = dispositivo.getOutputQueue("detecciones", maxSize=8, blocking=False)
    cola_imagen = dispositivo.getOutputQueue("imagen", maxSize=4, blocking=False) if ver else None
    latencias, n, con_pelota = [], 0, 0
    ultimas, imagenes = [], {}
    inicio = None
    while True:
        paquete = cola.get()
        if inicio is None:  # el primer resultado incluye el arranque de la red: no cuenta
            inicio = time.monotonic()
            continue
        latencias.append((dai.Clock.now() - paquete.getTimestamp()).total_seconds() * 1000)
        n += 1
        con_pelota += bool(paquete.detections)
        if ver:
            for frame in cola_imagen.tryGetAll():
                imagenes[frame.getSequenceNum()] = frame
            frame = imagenes.pop(paquete.getSequenceNum(), None)
            if frame is not None:
                imagenes = {k: v for k, v in imagenes.items() if k > paquete.getSequenceNum()}
                imagen = dibujar(cv2.cvtColor(frame.getCvFrame(), cv2.COLOR_GRAY2BGR), paquete.detections)
                cv2.putText(imagen, f"{blob.stem}  {n / max(1e-3, time.monotonic() - inicio):.0f} FPS", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("OAK", imagen)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        if time.monotonic() - inicio >= segundos:
            break
    duracion = time.monotonic() - inicio
    latencias = np.array(latencias)
    return {
        "fps": n / duracion,
        "latencia_media_ms": float(latencias.mean()),
        "latencia_p95_ms": float(np.percentile(latencias, 95)),
        "frames_con_pelota": f"{con_pelota}/{n}",
    }


def medir_clip(dispositivo, blob, clip, carpeta_hoja):
    entrada = dispositivo.getInputQueue("entrada")
    cola = dispositivo.getOutputQueue("detecciones", maxSize=8, blocking=True)
    stream = clip.streams["izq"]
    con_pelota, ejemplos, tiempos = 0, [], []
    for i in range(len(stream)):
        imagen = stream.leer(i)
        frame = dai.ImgFrame()
        frame.setType(dai.ImgFrame.Type.GRAY8)
        frame.setWidth(imagen.shape[1])
        frame.setHeight(imagen.shape[0])
        frame.setData(imagen.flatten())
        frame.setSequenceNum(i)
        t = time.monotonic()
        entrada.send(frame)
        detecciones = cola.get().detections
        tiempos.append(time.monotonic() - t)
        if detecciones:
            con_pelota += 1
            if i % 25 == 0 and len(ejemplos) < 24:
                ejemplos.append(dibujar(cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR), detecciones))
    if ejemplos:
        while len(ejemplos) % 6:
            ejemplos.append(np.zeros_like(ejemplos[0]))
        filas = [np.hstack([cv2.resize(e, (320, 200)) for e in ejemplos[k:k + 6]]) for k in range(0, len(ejemplos), 6)]
        ruta = carpeta_hoja / f"clip_{blob.stem}.jpg"
        cv2.imwrite(str(ruta), np.vstack(filas))
        print(f"    ejemplos: {ruta}")
    return {
        "frames_con_pelota": f"{con_pelota}/{len(stream)}",
        "ida_y_vuelta_ms": float(np.median(tiempos) * 1000),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blobs", nargs="+", type=Path, default=None,
                        help="blobs a medir (por defecto, todos los de modelos/oak)")
    parser.add_argument("--segundos", type=float, default=15, help="duración de cada medida en vivo")
    parser.add_argument("--fps", type=float, default=120, help="FPS de la cámara mono")
    parser.add_argument("--conf", type=float, default=0.5, help="confianza mínima de las detecciones")
    parser.add_argument("--ver", action="store_true", help="mostrar la imagen con las detecciones (resta algo de FPS)")
    parser.add_argument("--hilos", type=int, choices=[1, 2], default=2,
                        help="hilos de inferencia de la red (2 = más FPS, 1 = menos latencia)")
    parser.add_argument("--clip", type=Path, default=None, help="comprobar con un clip grabado en lugar de en vivo")
    args = parser.parse_args()

    blobs = args.blobs or sorted(CARPETA_BLOBS.glob("*.blob"), key=lambda b: -leer_config(b)[0])
    clip = ClipGrabado(args.clip) if args.clip else None
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta_csv = CARPETA_RESULTADOS / "medidas.csv"
    nuevo = not ruta_csv.exists()

    with open(ruta_csv, "a", newline="") as f:
        escritor = csv.writer(f)
        if nuevo:
            escritor.writerow(["fecha", "blob", "modo", "usb", "fps_camara", "hilos", "fps", "latencia_media_ms",
                               "latencia_p95_ms", "frames_con_pelota", "ida_y_vuelta_ms"])
        for blob in blobs:
            pipeline = crear_pipeline(blob, args.conf, args.fps, clip is not None, args.ver, args.hilos)
            with dai.Device(pipeline) as dispositivo:
                usb = dispositivo.getUsbSpeed().name
                print(f"{blob.name}  (USB {usb})")
                if clip is None:
                    r = medir_en_vivo(dispositivo, blob, args.segundos, args.ver)
                    print(f"    {r['fps']:.1f} FPS · latencia media {r['latencia_media_ms']:.1f} ms "
                          f"(p95 {r['latencia_p95_ms']:.1f}) · frames con pelota {r['frames_con_pelota']}")
                else:
                    r = medir_clip(dispositivo, blob, clip, CARPETA_RESULTADOS)
                    print(f"    frames con pelota {r['frames_con_pelota']} · "
                          f"ida y vuelta {r['ida_y_vuelta_ms']:.1f} ms/frame")
            escritor.writerow([datetime.now().isoformat(timespec="seconds"), blob.name,
                               "clip" if clip else "vivo", usb, args.fps, args.hilos,
                               *(round(r[k], 2) if isinstance(r.get(k), float) else r.get(k, "")
                                 for k in ("fps", "latencia_media_ms", "latencia_p95_ms", "frames_con_pelota",
                                           "ida_y_vuelta_ms"))])
            f.flush()
    if args.ver:
        cv2.destroyAllWindows()
    print(f"Medidas añadidas a {ruta_csv}")


if __name__ == "__main__":
    main()
