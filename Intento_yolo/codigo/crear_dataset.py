"""Construye el dataset YOLO de la pelota a partir del autoetiquetado (detecciones_coco.csv).

- Positivos: en cada frame, la mejor detección del profesor si supera --conf y
  tiene forma de pelota (casi cuadrada, tamaño mínimo).
- Negativos (desactivados por defecto): frames sin ninguna detección, una
  fracción --negativos del total. Muchos contienen la pelota en la mano o
  lejos, que el profesor no ve, así que solo deben usarse tras revisarlos con
  las hojas de revisión. Sin ellos el modelo aprende igualmente el fondo del
  resto de cada imagen positiva.
- Validación: clips completos (--val), nunca frames sueltos, para que no haya
  frames casi idénticos a uno y otro lado.
- rechazados.txt (junto al CSV): frames descartados tras revisarlos, uno por
  línea como "clip|camara|frame". Se excluyen siempre.

Las imágenes se guardan con los bytes JPEG originales de la cámara (sin recomprimir).

    python codigo/crear_dataset.py --nombre pelota_v1
    python codigo/crear_dataset.py --nombre pelota_v1 --revision    # además, hojas para revisar a ojo
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from lectura_clips import ClipGrabado

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_AUTO = RAIZ / "datos" / "autoetiquetado"
VAL_POR_DEFECTO = ["pruebas_primer_dia/clip_001/parte_5", "20261023_133016/clip_002/parte_5"]


def leer_detecciones(ruta_csv):
    """{(clip, camara, frame): [(conf, x1, y1, x2, y2), ...]} incluidos los frames sin detecciones."""
    frames = defaultdict(list)
    with open(ruta_csv, newline="") as f:
        for fila in csv.DictReader(f):
            clave = (fila["clip"], fila["camara"], int(fila["frame"]))
            frames[clave]
            if fila["confianza"]:
                frames[clave].append((float(fila["confianza"]), *(float(fila[k]) for k in ("x1", "y1", "x2", "y2"))))
    return frames


def leer_rechazados(ruta):
    if not ruta.exists():
        return set()
    rechazados = set()
    for linea in ruta.read_text().splitlines():
        linea = linea.split("#")[0].strip()
        if linea:
            clip, camara, frame = linea.split("|")
            rechazados.add((clip, camara, int(frame)))
    return rechazados


def forma_de_pelota(x1, y1, x2, y2, lado_minimo):
    ancho, alto = x2 - x1, y2 - y1
    return min(ancho, alto) >= lado_minimo and 0.6 <= ancho / alto <= 1.6


def nombre_archivo(clave):
    clip, camara, frame = clave
    return f"{clip.replace('/', '__')}__{camara}__{frame:05d}"


def hoja_revision(muestras, clips, ruta, titulo, recorte):
    """Mosaico numerado: recortes alrededor de cada caja (positivos) o frames reducidos (negativos)."""
    miniaturas = []
    for n, (clave, caja) in enumerate(muestras):
        clip, camara, frame = clave
        imagen = cv2.cvtColor(clips[clip].streams[camara].leer(frame), cv2.COLOR_GRAY2BGR)
        if caja is not None and recorte:
            _, x1, y1, x2, y2 = caja
            cx, cy, lado = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 2.5 + 20
            a, b = int(max(0, cx - lado / 2)), int(max(0, cy - lado / 2))
            cv2.rectangle(imagen, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            miniatura = cv2.resize(imagen[b:int(cy + lado / 2), a:int(cx + lado / 2)], (128, 128))
        else:
            miniatura = cv2.resize(imagen, (256, 160))
        cv2.putText(miniatura, str(n), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if caja is not None:
            cv2.putText(miniatura, f"{caja[0]:.2f}", (3, miniatura.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1)
        miniaturas.append(miniatura)
    columnas = 10 if recorte else 6
    while len(miniaturas) % columnas:
        miniaturas.append(np.zeros_like(miniaturas[0]))
    filas = [np.hstack(miniaturas[k:k + columnas]) for k in range(0, len(miniaturas), columnas)]
    hoja = np.vstack(filas)
    cv2.imwrite(str(ruta), hoja)
    print(f"  {titulo}: {ruta}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nombre", default="pelota_v1", help="carpeta del dataset dentro de datos/")
    parser.add_argument("--detecciones", type=Path, default=CARPETA_AUTO / "detecciones_coco.csv")
    parser.add_argument("--conf", type=float, default=0.5, help="confianza mínima para un positivo")
    parser.add_argument("--conf-negativo", type=float, default=0.05,
                        help="un frame es negativo si ninguna detección llega a esta confianza")
    parser.add_argument("--negativos", type=float, default=0.0,
                        help="fracción de negativos sobre el total (revisarlos antes con --revision)")
    parser.add_argument("--lado-minimo", type=float, default=4, help="lado mínimo de la caja en píxeles")
    parser.add_argument("--val", nargs="+", default=VAL_POR_DEFECTO, help="clips que van a validación")
    parser.add_argument("--revision", action="store_true", help="generar hojas de revisión de positivos y negativos")
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()

    detecciones = leer_detecciones(args.detecciones)
    rechazados = leer_rechazados(args.detecciones.parent / "rechazados.txt")

    positivos, candidatos_negativos = {}, []
    for clave, cajas in detecciones.items():
        if clave in rechazados:
            continue
        validas = [c for c in cajas if c[0] >= args.conf and forma_de_pelota(*c[1:], args.lado_minimo)]
        if validas:
            positivos[clave] = max(validas)
        elif not cajas or max(c[0] for c in cajas) < args.conf_negativo:
            candidatos_negativos.append(clave)
    random.Random(args.semilla).shuffle(candidatos_negativos)
    n_negativos = int(len(positivos) * args.negativos / (1 - args.negativos))
    negativos = sorted(candidatos_negativos[:n_negativos])

    destino = RAIZ / "datos" / args.nombre
    if destino.exists():
        shutil.rmtree(destino)
    for parte in ("train", "val"):
        (destino / "images" / parte).mkdir(parents=True)
        (destino / "labels" / parte).mkdir(parents=True)

    clips = {}
    def abrir(clip):
        if clip not in clips:
            sesion, resto = clip.split("/", 1)
            clips[clip] = ClipGrabado(RAIZ / "grabaciones" / sesion / resto)
        return clips[clip]

    cuenta = defaultdict(int)
    for clave in sorted(positivos) + negativos:
        clip, camara, frame = clave
        stream = abrir(clip).streams[camara]
        parte = "val" if clip in args.val else "train"
        base = nombre_archivo(clave)
        stream._f.seek(int(stream.offsets[frame]))
        (destino / "images" / parte / f"{base}.jpg").write_bytes(stream._f.read(int(stream.tamanos[frame])))
        etiqueta = ""
        if clave in positivos:
            _, x1, y1, x2, y2 = positivos[clave]
            alto, ancho = stream.leer(frame).shape
            etiqueta = (f"0 {(x1 + x2) / 2 / ancho:.6f} {(y1 + y2) / 2 / alto:.6f} "
                        f"{(x2 - x1) / ancho:.6f} {(y2 - y1) / alto:.6f}\n")
        (destino / "labels" / parte / f"{base}.txt").write_text(etiqueta)
        cuenta[(parte, "positivo" if clave in positivos else "negativo")] += 1

    (destino / "data.yaml").write_text(
        f"# Generado por codigo/crear_dataset.py a partir de {args.detecciones.name}\n"
        f"path: {destino}\ntrain: images/train\nval: images/val\nnames:\n  0: pelota\n")

    print(f"Dataset en {destino}")
    for parte in ("train", "val"):
        print(f"  {parte}: {cuenta[(parte, 'positivo')]} positivos, {cuenta[(parte, 'negativo')]} negativos")
    print(f"  rechazados a mano: {len(rechazados)} | candidatos a negativo: {len(candidatos_negativos)}")

    if args.revision:
        carpeta = destino / "revision"
        carpeta.mkdir()
        lista_pos = sorted(positivos.items())
        with open(carpeta / "indice.csv", "w", newline="") as f:
            escritor = csv.writer(f)
            escritor.writerow(["hoja", "numero", "clip", "camara", "frame"])
            for k in range(0, len(lista_pos), 200):
                bloque = lista_pos[k:k + 200]
                hoja_revision(bloque, clips, carpeta / f"positivos_{k // 200:02d}.jpg", "positivos", True)
                escritor.writerows([f"positivos_{k // 200:02d}", n, *c] for n, (c, _) in enumerate(bloque))
            for k in range(0, len(negativos), 60):
                bloque = [(c, None) for c in negativos[k:k + 60]]
                hoja_revision(bloque, clips, carpeta / f"negativos_{k // 60:02d}.jpg", "negativos", False)
                escritor.writerows([f"negativos_{k // 60:02d}", n, *c] for n, (c, _) in enumerate(bloque))

    for clip in clips.values():
        clip.cerrar()


if __name__ == "__main__":
    main()
