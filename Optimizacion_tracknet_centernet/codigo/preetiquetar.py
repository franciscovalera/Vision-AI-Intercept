"""Paso 1 del etiquetado denso: pasa el YOLO pelota_v1 (enfoque anterior) por TODOS los frames.

Para entrenar PelotaNet hace falta la posición de la pelota en cada frame de cada clip (la red
ve pilas de frames consecutivos y da una salida por frame). Etiquetar ~80.000 frames por sesión
a mano es inviable, así que se parte de un detector que ya funciona razonablemente bien:

    1. preetiquetar.py   YOLO en todos los frames -> candidatos (varias cajas por frame, confianza baja)
    2. limpiar_etiquetas.py   trayectorias, huecos rellenados, falsos positivos fuera, frames dudosos
    3. revisar_etiquetas.py   revisión manual solo de los dudosos

Aquí se guardan candidatos, no etiquetas: con un umbral de confianza bajo (0,1) y hasta 5 cajas
por frame, para que el paso 2 pueda recuperar pelotas que YOLO ve con poca seguridad y elegir
la buena por coherencia temporal. YOLO es solo una herramienta de etiquetado: la red propia no
depende de él.

Salida, por clip y cámara (izq y der):
    datos/etiquetas/<sesión>/<clip>[/parte_K]/candidatos_<cámara>.csv
        frame, t, x, y, ancho, alto, conf     (centro y tamaño de la caja en píxeles de 640x400)
Es reanudable: salta los archivos que ya existen (se escriben al terminar cada uno).

    python codigo/preetiquetar.py grabaciones/sesion_1
    python codigo/preetiquetar.py grabaciones --dispositivo 0 --lote 64      # todo, en GPU
"""

import argparse
import csv
import time
from pathlib import Path

import cv2

from lectura_clips import CAMARAS_MONO, ClipGrabado, buscar_clips

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_ETIQUETAS = RAIZ / "datos" / "etiquetas"
MODELO_POR_DEFECTO = RAIZ.parent / "Intento_yolo" / "modelos" / "entrenamientos" / "pelota_v1" / "weights" / "best.pt"


def preetiquetar_stream(modelo, stream, ruta, args):
    filas = []
    for inicio in range(0, len(stream), args.lote):
        indices = range(inicio, min(inicio + args.lote, len(stream)))
        imagenes = [cv2.cvtColor(stream.leer(i), cv2.COLOR_GRAY2BGR) for i in indices]
        resultados = modelo.predict(imagenes, imgsz=args.imgsz, conf=args.conf, max_det=args.max_cajas,
                                    device=args.dispositivo, verbose=False)
        for i, resultado in zip(indices, resultados):
            for (x, y, ancho, alto), conf in zip(resultado.boxes.xywh.tolist(), resultado.boxes.conf.tolist()):
                filas.append([i, f"{stream.tiempos[i]:.6f}", f"{x:.2f}", f"{y:.2f}", f"{ancho:.2f}",
                              f"{alto:.2f}", f"{conf:.4f}"])
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    with open(temporal, "w", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["frame", "t", "x", "y", "ancho", "alto", "conf"])
        escritor.writerows(filas)
    temporal.rename(ruta)  # solo existe completo: así se puede reanudar
    return len({fila[0] for fila in filas})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ruta", type=Path, help="clip, secuencia, sesión o carpeta de grabaciones")
    parser.add_argument("--modelo", type=Path, default=MODELO_POR_DEFECTO)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.1, help="confianza mínima de los candidatos")
    parser.add_argument("--max-cajas", type=int, default=5, help="candidatos por frame como máximo")
    parser.add_argument("--lote", type=int, default=16, help="frames por inferencia")
    parser.add_argument("--dispositivo", default=None, help="cpu, 0 (GPU)...")
    args = parser.parse_args()

    from ultralytics import YOLO
    modelo = YOLO(str(args.modelo))

    for carpeta in buscar_clips(args.ruta):
        clip = ClipGrabado(carpeta)
        for camara in CAMARAS_MONO:
            ruta = CARPETA_ETIQUETAS / clip.nombre / f"candidatos_{camara}.csv"
            if camara not in clip.streams or ruta.exists():
                continue
            inicio = time.monotonic()
            n = preetiquetar_stream(modelo, clip.streams[camara], ruta, args)
            duracion = time.monotonic() - inicio
            total = len(clip.streams[camara])
            print(f"{clip.nombre} {camara}: {n}/{total} frames con candidatos "
                  f"({total / duracion:.0f} frames/s)", flush=True)
        clip.cerrar()


if __name__ == "__main__":
    main()
