"""Evalúa el detector de pelota entrenado en clips completos (todos los frames), no solo en el dataset.

La validación de entrenar_detector.py solo contiene frames en los que el
profesor (yolo11l de COCO) estaba seguro, así que mide los casos fáciles. Aquí
se pasan el alumno y el profesor por TODOS los frames de los clips y se compara:

- Referencia: la mejor caja del profesor con conf >= --conf-ref y forma de
  pelota. Sensibilidad del alumno (IoU >= 0,5) por tamaño de la pelota
  (lejos / media / cerca / muy cerca).
- Frames donde solo detecta el alumno: pueden ser aciertos que el profesor no
  ve (pelota en la mano, borrosa, tapada) o falsos positivos. Se guardan en
  hojas de miniaturas para revisarlos a ojo, igual que los fallos.
- Gráfica temporal de confianza y tamaño de la caja de ambos modelos.

    python codigo/evaluar_detector.py --modelo modelos/entrenamientos/pelota_v1/weights/best.pt --dispositivo 0

Salida en resultados/detector/<nombre>/: detecciones.csv (todas las cajas de
ambos modelos), metricas.txt, linea_temporal_*.png y hojas de ejemplos.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from crear_dataset import VAL_POR_DEFECTO, forma_de_pelota
from lectura_clips import ClipGrabado

RAIZ = Path(__file__).resolve().parents[1]
PROFESOR = RAIZ / "modelos" / "preentrenados" / "yolo11l.pt"
CLASE_PELOTA_COCO = 32
# Lado de la caja (px, imagen de 640x400) que separa lejos / media / cerca / muy cerca
TAMANOS = [(0, 15, "lejos (<15 px)"), (15, 30, "media (15-30 px)"), (30, 50, "cerca (30-50 px)"),
           (50, 1e9, "muy cerca (>50 px)")]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detectar(modelo, imagenes, **kwargs):
    """Lista (por imagen) de cajas [x1, y1, x2, y2, conf]."""
    resultados = modelo.predict(imagenes, verbose=False, **kwargs)
    return [[[*caja, conf] for caja, conf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())] for r in resultados]


def hoja(ejemplos, ruta, titulo, columnas=8):
    """Recortes de 160x160 alrededor de la pelota (alumno en verde, profesor en rojo)."""
    if not ejemplos:
        return
    miniaturas = []
    for imagen, alumno, profesor, texto in ejemplos:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        for cajas, color in ((profesor, (0, 0, 255)), (alumno, (0, 255, 0))):
            for x1, y1, x2, y2, _ in cajas:
                cv2.rectangle(imagen, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        caja = (alumno or profesor)[0]
        cx, cy = (caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2
        lado = int(max(80, 2.5 * max(caja[2] - caja[0], caja[3] - caja[1])))
        borde = cv2.copyMakeBorder(imagen, lado, lado, lado, lado, cv2.BORDER_CONSTANT)
        x, y = int(cx) + lado - lado // 2, int(cy) + lado - lado // 2
        miniatura = cv2.resize(borde[y:y + lado, x:x + lado], (160, 160))
        cv2.putText(miniatura, texto, (3, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        miniaturas.append(miniatura)
    while len(miniaturas) % columnas:
        miniaturas.append(np.zeros_like(miniaturas[0]))
    cv2.imwrite(str(ruta), np.vstack([np.hstack(miniaturas[k:k + columnas])
                                      for k in range(0, len(miniaturas), columnas)]))
    print(f"  {titulo}: {ruta}")


def linea_temporal(filas, ruta, titulo, conf_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_conf, ax_lado) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for fuente, color in (("profesor", "tab:red"), ("alumno", "tab:green")):
        t = [f["t_s"] for f in filas if f[fuente]]
        ax_conf.plot(t, [f[fuente][4] for f in filas if f[fuente]], ".", ms=2, color=color, label=fuente)
        ax_lado.plot(t, [max(f[fuente][2] - f[fuente][0], f[fuente][3] - f[fuente][1]) for f in filas if f[fuente]],
                     ".", ms=2, color=color, label=fuente)
    ax_conf.axhline(conf_ref, color="gray", lw=0.8, ls="--")
    ax_conf.set_ylabel("confianza (mejor caja)")
    ax_conf.set_ylim(0, 1)
    ax_conf.legend(loc="lower right")
    ax_lado.set_ylabel("lado de la caja (px)")
    ax_lado.set_xlabel("tiempo (s)")
    ax_conf.set_title(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    print(f"  línea temporal: {ruta}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", type=Path, required=True, help="pesos del alumno (best.pt)")
    parser.add_argument("--clips", nargs="+", default=VAL_POR_DEFECTO, help="clips a evaluar (sesion/clip[/parte])")
    parser.add_argument("--camara", default="izq", choices=["izq", "der"])
    parser.add_argument("--conf", type=float, default=0.25, help="confianza mínima del alumno")
    parser.add_argument("--conf-ref", type=float, default=0.5, help="confianza mínima del profesor como referencia")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--nombre", default=None, help="carpeta en resultados/detector/ (por defecto, la del modelo)")
    parser.add_argument("--dispositivo", default=None)
    parser.add_argument("--ejemplos", type=int, default=48, help="miniaturas por hoja")
    args = parser.parse_args()

    nombre = args.nombre or args.modelo.resolve().parents[1].name
    destino = RAIZ / "resultados" / "detector" / nombre
    destino.mkdir(parents=True, exist_ok=True)
    alumno = YOLO(str(args.modelo))
    profesor = YOLO(str(PROFESOR))

    filas = []  # una por frame: clip, frame, t_s, alumno (mejor caja o None), profesor (ídem)
    imagenes_guardadas = {}
    with open(destino / "detecciones.csv", "w", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["clip", "camara", "frame", "t_s", "modelo", "x1", "y1", "x2", "y2", "confianza"])
        for nombre_clip in args.clips:
            sesion, resto = nombre_clip.split("/", 1)
            clip = ClipGrabado(RAIZ / "grabaciones" / sesion / resto)
            stream = clip.streams[args.camara]
            print(f"{nombre_clip}: {len(stream)} frames", flush=True)
            for k in range(0, len(stream), 32):
                indices = range(k, min(k + 32, len(stream)))
                grises = [stream.leer(i) for i in indices]
                bgr = [cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) for g in grises]
                cajas_a = detectar(alumno, bgr, imgsz=args.imgsz, conf=args.conf, device=args.dispositivo)
                cajas_p = detectar(profesor, bgr, imgsz=1280, conf=0.05, classes=[CLASE_PELOTA_COCO],
                                   device=args.dispositivo)
                for i, gris, ca, cp in zip(indices, grises, cajas_a, cajas_p):
                    t = stream.tiempos[i] - clip.t0
                    for modelo, cajas in (("alumno", ca), ("profesor", cp)):
                        for caja in cajas:
                            escritor.writerow([nombre_clip, args.camara, i, f"{t:.4f}", modelo,
                                               *(f"{v:.1f}" for v in caja[:4]), f"{caja[4]:.4f}"])
                    filas.append({"clip": nombre_clip, "frame": i, "t_s": t, "imagen": gris,
                                  "alumno": max(ca, key=lambda c: c[4]) if ca else None,
                                  "profesor_todas": cp,
                                  "profesor": max(cp, key=lambda c: c[4]) if cp else None})
            clip.cerrar()

    # Métricas
    por_tamano = defaultdict(lambda: [0, 0])  # tamaño -> [referencias, acertadas]
    fallos, solo_alumno, aciertos_dificiles, ious = [], [], [], []
    n_ref = n_ref_ok = n_alumno = n_ninguno = 0
    for f in filas:
        validas = [c for c in f["profesor_todas"] if c[4] >= args.conf_ref and forma_de_pelota(*c[:4], 4)]
        ref = max(validas, key=lambda c: c[4]) if validas else None
        a = f["alumno"]
        n_alumno += a is not None
        texto = f"{f['clip'].split('/')[-1]} f{f['frame']}"
        if ref is not None:
            lado = max(ref[2] - ref[0], ref[3] - ref[1])
            bin_ = next(nombre for lo, hi, nombre in TAMANOS if lo <= lado < hi)
            por_tamano[bin_][0] += 1
            n_ref += 1
            if a is not None and iou(a, ref) >= 0.5:
                por_tamano[bin_][1] += 1
                n_ref_ok += 1
                ious.append(iou(a, ref))
            else:
                fallos.append((f["imagen"], [a] if a else [], [ref], texto + (f" a{a[4]:.2f}" if a else " -")))
        elif a is not None:
            p = f["profesor"]
            solo_alumno.append((f["imagen"], [a], [p] if p else [],
                                texto + f" a{a[4]:.2f} p{p[4]:.2f}" if p else texto + f" a{a[4]:.2f} p-"))
        else:
            n_ninguno += 1

    lineas = [
        f"Modelo: {args.modelo}",
        f"Clips: {', '.join(args.clips)} (cámara {args.camara}, todos los frames)",
        f"Alumno conf >= {args.conf}; referencia = profesor conf >= {args.conf_ref}",
        "",
        f"Frames: {len(filas)}",
        f"  con referencia del profesor: {n_ref}",
        f"  alumno detecta (algo): {n_alumno}",
        f"  solo el alumno detecta (sin referencia): {len(solo_alumno)}",
        f"  ninguno de los dos: {n_ninguno}",
        "",
        f"Sensibilidad respecto al profesor (IoU >= 0,5): {n_ref_ok}/{n_ref} = {n_ref_ok / max(n_ref, 1):.3f}",
        f"IoU medio en aciertos: {np.mean(ious) if ious else 0:.3f}",
    ]
    for _, _, nombre_bin in TAMANOS:
        total, ok = por_tamano[nombre_bin]
        if total:
            lineas.append(f"  {nombre_bin:20s} {ok:5d}/{total:<5d} = {ok / total:.3f}")
    texto_metricas = "\n".join(lineas)
    print("\n" + texto_metricas + "\n")
    (destino / "metricas.txt").write_text(texto_metricas + "\n")

    rng = np.random.default_rng(0)
    def muestra(lista):
        if len(lista) <= args.ejemplos:
            return lista
        return [lista[i] for i in sorted(rng.choice(len(lista), args.ejemplos, replace=False))]
    hoja(muestra(fallos), destino / "fallos.jpg", "fallos (profesor sí, alumno no)")
    hoja(muestra(solo_alumno), destino / "solo_alumno.jpg", "solo el alumno detecta")
    for nombre_clip in args.clips:
        linea_temporal([f for f in filas if f["clip"] == nombre_clip],
                       destino / f"linea_temporal_{nombre_clip.replace('/', '__')}.png", nombre_clip, args.conf_ref)


if __name__ == "__main__":
    main()
