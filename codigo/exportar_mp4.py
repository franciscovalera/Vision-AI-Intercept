"""Exporta clips grabados a MP4 (H.264) con la duración real.

Los frames que la cámara capturó pero no se guardaron se rellenan repitiendo el
anterior, de modo que el vídeo dura exactamente lo mismo que la grabación y se
puede ver en cualquier reproductor. Con --velocidad se exporta en cámara lenta.

    python codigo/exportar_mp4.py grabaciones/pruebas_primer_dia/clip_001
    python codigo/exportar_mp4.py grabaciones/pruebas_primer_dia --camara estereo --velocidad 0.25

Por defecto cada vídeo se guarda dentro de la carpeta de su clip
(izq.mp4, estereo_x0.25.mp4...).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from lectura_clips import ClipGrabado, buscar_clips

ROJO, BLANCO = (60, 60, 230), (255, 255, 255)


def abrir_escritor(ruta, fps, tamano):
    for codec in ("avc1", "mp4v"):  # H.264 si está disponible; si no, MPEG-4
        escritor = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*codec), fps, tamano)
        if escritor.isOpened():
            return escritor
    raise RuntimeError(f"no se pudo crear {ruta}")


def texto(imagen, cadena, posicion, color=BLANCO):
    cv2.putText(imagen, cadena, posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(imagen, cadena, posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def componer(clip, camara, i):
    """Imagen BGR del frame i de la cámara (o el par estéreo, referido a la izquierda)."""
    if camara == "estereo":
        izq = clip.streams["izq"].leer_bgr(i)
        j = clip.pareja("der", i)
        der = clip.streams["der"].leer_bgr(j) if j is not None else np.zeros_like(izq)
        return np.hstack([izq, der])
    return clip.streams[camara].leer_bgr(i)


def exportar(clip, camara, velocidad, con_texto, ruta_salida):
    stream = clip.streams["izq" if camara == "estereo" else camara]
    fps = stream.fps_nominal
    periodo = 1 / fps
    escritor = None
    escritos = rellenos = 0
    try:
        for i in range(len(stream)):
            imagen = componer(clip, camara, i)
            if escritor is None:
                alto, ancho = imagen.shape[:2]
                escritor = abrir_escritor(ruta_salida, fps * velocidad, (ancho, alto))
            # Huecos hasta el siguiente frame guardado: se repite este frame
            repeticiones = 1
            if i + 1 < len(stream):
                repeticiones += int(stream.secuencias[i + 1] - stream.secuencias[i] - 1)
            t = stream.tiempos[i] - clip.t0
            for r in range(repeticiones):
                salida = imagen.copy() if con_texto else imagen
                if con_texto:
                    texto(salida, f"{clip.nombre}  t = {t + r * periodo:6.3f} s", (10, 22))
                    if r > 0:
                        texto(salida, "frame perdido (repetido)", (10, 44), ROJO)
                escritor.write(salida)
            escritos += repeticiones
            rellenos += repeticiones - 1
    finally:
        if escritor is not None:
            escritor.release()
    return escritos, rellenos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ruta", help="carpeta de un clip, de una secuencia (clip_NNN) o de una sesión")
    parser.add_argument("--camara", choices=["izq", "der", "estereo", "color"], default="izq",
                        help="cámara a exportar; estereo pone izquierda y derecha lado a lado; "
                             "color se exporta reducido a 400 px de alto (por defecto izq)")
    parser.add_argument("--velocidad", type=float, default=1.0,
                        help="1 = tiempo real; 0.25 = cámara lenta x4 (por defecto 1)")
    parser.add_argument("--sin-texto", action="store_true", help="no escribir el tiempo ni los avisos sobre el vídeo")
    parser.add_argument("--salida", type=Path, default=None,
                        help="carpeta donde guardar los MP4 (por defecto, la carpeta de cada clip)")
    args = parser.parse_args()
    if args.velocidad <= 0:
        parser.error("--velocidad debe ser mayor que 0")

    carpetas = buscar_clips(args.ruta)
    if not carpetas:
        sys.exit(f"No hay clips (carpetas con tiempos.csv) en {args.ruta}")
    if args.salida:
        args.salida.mkdir(parents=True, exist_ok=True)

    sufijo = "" if args.velocidad == 1 else f"_x{args.velocidad:g}"
    for carpeta in carpetas:
        clip = ClipGrabado(carpeta)
        necesarias = ["izq", "der"] if args.camara == "estereo" else [args.camara]
        if any(c not in clip.streams for c in necesarias):
            print(f"{clip.nombre}: no tiene la cámara {args.camara}, se omite")
            clip.cerrar()
            continue

        nombre = f"{args.camara}{sufijo}.mp4"
        if args.salida:
            ruta = args.salida / f"{clip.nombre.replace('/', '_')}_{nombre}"
        else:
            ruta = carpeta / nombre
        escritos, rellenos = exportar(clip, args.camara, args.velocidad, not args.sin_texto, ruta)
        fps = clip.streams["izq" if args.camara == "estereo" else args.camara].fps_nominal
        duracion = escritos / (fps * args.velocidad)
        print(f"{clip.nombre}: {ruta}  ({duracion:.2f} s de vídeo, {rellenos} de {escritos} frames rellenados)")
        clip.cerrar()


if __name__ == "__main__":
    main()
