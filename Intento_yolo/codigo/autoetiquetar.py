"""Autoetiquetado de la pelota con un detector grande preentrenado en COCO (clase "sports ball").

Paso lento del pipeline de datos: pasa el modelo "profesor" por uno de cada N
frames de los clips y guarda TODAS sus detecciones (con confianza baja incluida)
en un CSV. Después crear_dataset.py decide con qué umbrales construir el dataset,
sin tener que repetir la inferencia. Se puede interrumpir y relanzar: continúa
donde lo dejó.

    python codigo/autoetiquetar.py grabaciones --cada 8
    python codigo/autoetiquetar.py grabaciones/pruebas_primer_dia --camaras izq der --cada 4

Salida: datos/autoetiquetado/detecciones_coco.csv con una fila por detección
(o una fila sin caja si el frame se procesó y no hubo detecciones).
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from lectura_clips import ClipGrabado, buscar_clips

RAIZ = Path(__file__).resolve().parents[1]
MODELO_PROFESOR = RAIZ / "modelos" / "preentrenados" / "yolo11l.pt"
SALIDA = RAIZ / "datos" / "autoetiquetado" / "detecciones_coco.csv"
CLASE_PELOTA_COCO = 32  # "sports ball"
COLUMNAS = ["clip", "camara", "frame", "secuencia", "t_s", "x1", "y1", "x2", "y2", "confianza"]


def frames_procesados(ruta_csv):
    if not ruta_csv.exists():
        return set()
    with open(ruta_csv, newline="") as f:
        return {(fila["clip"], fila["camara"], int(fila["frame"])) for fila in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rutas", nargs="+", help="clips, secuencias o sesiones (p. ej. grabaciones)")
    parser.add_argument("--cada", type=int, default=8, help="procesar uno de cada N frames (por defecto 8)")
    parser.add_argument("--camaras", nargs="+", choices=["izq", "der"], default=["izq"])
    parser.add_argument("--modelo", type=Path, default=MODELO_PROFESOR)
    parser.add_argument("--imgsz", type=int, default=1280, help="tamaño de inferencia (más grande = pelotas pequeñas)")
    parser.add_argument("--conf", type=float, default=0.05, help="confianza mínima que se guarda")
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--dispositivo", default=None,
                        help="cpu, 0 (primera GPU NVIDIA), mps (Mac)... por defecto, GPU si hay")
    args = parser.parse_args()

    carpetas = [c for ruta in args.rutas for c in buscar_clips(ruta)]
    hechos = frames_procesados(args.salida)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not args.salida.exists()
    modelo = YOLO(str(args.modelo))

    pendientes = []
    for carpeta in carpetas:
        clip = ClipGrabado(carpeta)
        for camara in args.camaras:
            if camara in clip.streams:
                for i in range(0, len(clip.streams[camara]), args.cada):
                    if (clip.nombre, camara, i) not in hechos:
                        pendientes.append((carpeta, clip.nombre, camara, i))
        clip.cerrar()
    print(f"{len(carpetas)} clips, {len(pendientes)} frames pendientes ({len(hechos)} ya procesados)", flush=True)

    inicio = time.monotonic()
    clip_abierto = None
    with open(args.salida, "a", newline="") as f:
        escritor = csv.writer(f)
        if nuevo:
            escritor.writerow(COLUMNAS)
        for k, (carpeta, nombre, camara, i) in enumerate(pendientes):
            if clip_abierto is None or clip_abierto.carpeta != carpeta:
                if clip_abierto is not None:
                    clip_abierto.cerrar()
                clip_abierto = ClipGrabado(carpeta)
            stream = clip_abierto.streams[camara]
            imagen = cv2.cvtColor(stream.leer(i), cv2.COLOR_GRAY2BGR)
            resultado = modelo.predict(imagen, imgsz=args.imgsz, classes=[CLASE_PELOTA_COCO], conf=args.conf,
                                       device=args.dispositivo, verbose=False)[0]
            base = [nombre, camara, i, int(stream.secuencias[i]), f"{stream.tiempos[i] - clip_abierto.t0:.4f}"]
            cajas = resultado.boxes
            if len(cajas) == 0:
                escritor.writerow(base + ["", "", "", "", ""])
            for (x1, y1, x2, y2), confianza in zip(cajas.xyxy.tolist(), cajas.conf.tolist()):
                escritor.writerow(base + [f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{confianza:.4f}"])
            if (k + 1) % 50 == 0:
                f.flush()
                transcurrido = time.monotonic() - inicio
                restante = transcurrido / (k + 1) * (len(pendientes) - k - 1)
                print(f"{k + 1}/{len(pendientes)}  {transcurrido / (k + 1):.2f} s/frame  "
                      f"quedan {restante / 60:.0f} min", flush=True)
    if clip_abierto is not None:
        clip_abierto.cerrar()
    print("Terminado.", flush=True)


if __name__ == "__main__":
    main()
