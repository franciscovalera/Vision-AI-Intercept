"""Vídeo de las detecciones de PelotaNet sobre un clip grabado, con izq y der lado a lado.

Usa los predicciones.csv que escribe evaluar.py (uno por red), las etiquetas y, si existen,
los candidatos de YOLO; no vuelve a pasar la red. Dibuja en cada frame:
    etiqueta      círculo verde
    red 1, 2...   círculos azul, naranja, magenta... (en el orden de --predicciones)
    YOLO          caja fina gris (la de mayor confianza con conf >= 0,5)
y en la esquina, por red, la puntuación del mapa de calor y el resultado con el criterio de
evaluar.py (error < 4 px y < diámetro/2, umbral --umbral):
    ACIERTO  hay pelota y la predicción está dentro del margen
    FN       hay pelota y no se predice (o se predice fuera del margen: entonces "FP+FN")
    FP       se predice y no hay pelota
    VN       ni pelota ni predicción
    dudoso   frame dudoso (no cuenta en las métricas)
Los frames sin predicción en el CSV (los de pilas con frames perdidos, que la red no procesa)
salen sin círculo de la red y con "-".

El vídeo va a la velocidad real con tiempos.csv, como exportar_mp4.py: el frame izquierdo
marca el ritmo, los perdidos se rellenan repitiendo el anterior y el derecho se empareja por
tiempo. Se codifica en H.264 con ffmpeg (2 hilos).

    python codigo/video_predicciones.py sesion_2/clip_001/parte_3 \\
        --predicciones resultados/entrenamiento/a16_f3/predicciones.csv resultados/entrenamiento/a24_f3/predicciones.csv \\
        --lento 4

Salida por defecto en resultados/entrenamiento/videos/<clip>[_lentoN].mp4.
"""

import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np

from lectura_clips import CAMARAS_MONO, ClipGrabado

RAIZ = Path(__file__).resolve().parents[1]
VERDE, GRIS, BLANCO = (0, 220, 0), (170, 170, 170), (255, 255, 255)
COLORES_RED = [(255, 120, 0), (0, 140, 255), (255, 0, 255), (0, 255, 255)]  # BGR: azul, naranja...
COLOR_ESTADO = {"ACIERTO": (0, 220, 0), "FN": (0, 0, 255), "FP": (0, 0, 255), "FP+FN": (0, 0, 255),
                "VN": (200, 200, 200), "dudoso": (0, 200, 255), "-": (150, 150, 150)}


def leer_csv(ruta):
    with open(ruta, newline="") as f:
        return list(csv.DictReader(f))


def numero(v):
    return float(v) if v not in ("", "nan") else np.nan


def predicciones_red(ruta, clip):
    """(cámara, frame) → (puntuación, x, y, diámetro) de un predicciones.csv, solo del clip."""
    return {(fila["camara"], int(fila["frame"])): tuple(numero(fila[k]) for k in
                                                        ("puntuacion", "x_pred", "y_pred", "diametro_pred"))
            for fila in leer_csv(ruta) if fila["clip"] == clip}


def etiquetas(clip):
    """cámara → lista por frame de (x, y, diámetro, visibilidad, dudoso)."""
    resultado = {}
    for camara in CAMARAS_MONO:
        ruta = RAIZ / "datos" / "etiquetas" / clip / f"etiquetas_{camara}.csv"
        if ruta.exists():
            resultado[camara] = [(numero(f["x"]), numero(f["y"]), numero(f["diametro"]), int(f["visibilidad"]),
                                  f["dudoso"] == "1") for f in leer_csv(ruta)]
    return resultado


def yolo(clip, conf_min=0.5):
    """(cámara, frame) → caja (x, y, ancho, alto, conf) de mayor confianza."""
    mejores = {}
    for camara in CAMARAS_MONO:
        ruta = RAIZ / "datos" / "etiquetas" / clip / f"candidatos_{camara}.csv"
        if not ruta.exists():
            continue
        for f in leer_csv(ruta):
            conf, clave = float(f["conf"]), (camara, int(f["frame"]))
            if conf >= conf_min and conf > mejores.get(clave, (0,) * 5)[4]:
                mejores[clave] = (float(f["x"]), float(f["y"]), float(f["ancho"]), float(f["alto"]), conf)
    return mejores


def estado(pred, et, umbral):
    if pred is None:
        return "-"
    x, y, d, visibilidad, dudoso = et
    if dudoso:
        return "dudoso"
    hay = pred[0] >= umbral
    if visibilidad == 0:
        return "FP" if hay else "VN"
    if not hay:
        return "FN"
    return "ACIERTO" if np.hypot(pred[1] - x, pred[2] - y) < min(4.0, d / 2) else "FP+FN"


def texto(imagen, cadena, posicion, color=BLANCO, escala=0.45):
    cv2.putText(imagen, cadena, posicion, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(imagen, cadena, posicion, cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)


def dibujar(imagen, camara, i, et, redes, cajas_yolo, umbral):
    """Dibuja sobre la imagen BGR del frame i de la cámara."""
    e = et[camara][i]
    caja = cajas_yolo.get((camara, i))
    if caja is not None:
        x, y, w, h, _ = caja
        cv2.rectangle(imagen, (int(round(x - w / 2)), int(round(y - h / 2))),
                      (int(round(x + w / 2)), int(round(y + h / 2))), GRIS, 1)
    if e[3] > 0:
        cv2.circle(imagen, (int(round(e[0])), int(round(e[1]))), max(2, int(round(e[2] / 2))), VERDE, 1, cv2.LINE_AA)
    for k, (nombre, pred) in enumerate(redes):
        p = pred.get((camara, i))
        if p is not None and p[0] >= umbral:
            cv2.circle(imagen, (int(round(p[1])), int(round(p[2]))), max(2, int(round(p[3] / 2)) + 2 + k),
                       COLORES_RED[k], 1, cv2.LINE_AA)
        s = estado(p, e, umbral)
        punt = "-" if p is None else f"{p[0]:.2f}"
        texto(imagen, f"{nombre} {punt}", (8, 40 + 18 * k), COLORES_RED[k])
        texto(imagen, s, (150, 40 + 18 * k), COLOR_ESTADO[s])
    leyenda = f"{camara}  frame {i}" + ("  (dudoso)" if e[4] else "")
    texto(imagen, leyenda, (8, 20))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="clip relativo a grabaciones/ (p. ej. sesion_2/clip_001/parte_3)")
    parser.add_argument("--predicciones", type=Path, nargs="+", required=True,
                        help="predicciones.csv de evaluar.py, una por red (el nombre es su carpeta)")
    parser.add_argument("--umbral", type=float, default=0.5)
    parser.add_argument("--lento", type=float, default=1.0, help="factor de cámara lenta (4 = 4 veces más lento)")
    parser.add_argument("--desde", type=float, default=0.0, help="segundo del clip en que empieza el vídeo")
    parser.add_argument("--hasta", type=float, default=None, help="segundo del clip en que acaba")
    parser.add_argument("--sin-yolo", action="store_true")
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args()

    clip = ClipGrabado(RAIZ / "grabaciones" / args.clip)
    et = etiquetas(args.clip)
    redes = [(ruta.parent.name, predicciones_red(ruta, args.clip)) for ruta in args.predicciones]
    cajas_yolo = {} if args.sin_yolo else yolo(args.clip)
    izq = clip.streams["izq"]
    sufijo = "" if args.lento == 1 else f"_lento{args.lento:g}"
    salida = args.salida or RAIZ / "resultados" / "entrenamiento" / "videos" / f"{args.clip.replace('/', '_')}{sufijo}.mp4"
    salida.parent.mkdir(parents=True, exist_ok=True)

    alto, ancho = izq.leer(0).shape
    fps = izq.fps_nominal / args.lento
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * ancho}x{alto}",
         "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-threads", "2", str(salida)], stdin=subprocess.PIPE)
    escritos = 0
    try:
        for i in range(len(izq)):
            t = izq.tiempos[i] - clip.t0
            if t < args.desde or (args.hasta is not None and t > args.hasta):
                continue
            imagen_izq = cv2.cvtColor(izq.leer(i), cv2.COLOR_GRAY2BGR)
            dibujar(imagen_izq, "izq", i, et, redes, cajas_yolo, args.umbral)
            j = clip.pareja("der", i)
            if j is not None:
                imagen_der = cv2.cvtColor(clip.streams["der"].leer(j), cv2.COLOR_GRAY2BGR)
                dibujar(imagen_der, "der", j, et, redes, cajas_yolo, args.umbral)
            else:
                imagen_der = np.zeros_like(imagen_izq)
            imagen = np.hstack([imagen_izq, imagen_der])
            texto(imagen, f"t = {t:6.3f} s" + (f"  (x1/{args.lento:g})" if args.lento != 1 else ""),
                  (ancho - 80, alto - 10))
            texto(imagen, "verde: etiqueta   gris: YOLO   " +
                  "   ".join(f"{['azul', 'naranja', 'magenta', 'amarillo'][k]}: {n}" for k, (n, _) in enumerate(redes)),
                  (8, alto - 10), BLANCO, 0.4)
            # Frames perdidos: se repite el último para conservar la duración real
            repeticiones = 1 + (int(izq.secuencias[i + 1] - izq.secuencias[i] - 1) if i + 1 < len(izq) else 0)
            for _ in range(repeticiones):
                ffmpeg.stdin.write(imagen.tobytes())
            escritos += repeticiones
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        clip.cerrar()
    print(f"{salida}: {escritos} frames, {escritos / fps:.1f} s de vídeo")


if __name__ == "__main__":
    main()
