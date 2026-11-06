"""Paso 2 del etiquetado denso: de los candidatos de YOLO a una etiqueta por frame.

Usa que a 120 FPS la pelota se mueve poco y de forma suave entre frames:

  1. Trayectorias: se recorren los frames en orden y cada candidato se une a la trayectoria
     cuya posición predicha (última posición + velocidad x tiempo transcurrido) tiene más cerca,
     si cae dentro de una puerta que crece con el diámetro y la velocidad y tiene un tamaño
     parecido. Los que no encajan empiezan una trayectoria nueva. Una trayectoria muere si pasa
     más de --hueco-max segundos sin candidatos (la pelota ha salido o está tapada mucho rato).
  2. Se quedan las trayectorias creíbles (al menos --min-frames frames y alguno con confianza
     >= 0,5); el resto son falsos positivos sueltos. Se descartan también los objetos fijos:
     trayectorias que no se mueven (desviación de la posición < max(1,5 px, 10 % del diámetro))
     durante más de --fijo-max segundos con confianza media < 0,5 (una rueda, un borde oscuro, algo
     redondo cortado por el borde de la imagen). YOLO los ve con poca seguridad en casi todos los
     frames y, por su longitud, ganarían a la pelota de verdad en confianza acumulada.
     Si dos trayectorias se solapan en el tiempo (solo hay una pelota), en cada frame gana la de
     mayor confianza acumulada.
  3. Huecos dentro de una trayectoria (frames sin detección): se rellenan ajustando una parábola
     en el tiempo a x(t) e y(t) con los puntos de alrededor (es el movimiento de una pelota en el
     aire visto en la imagen, bien aproximado en tramos cortos); el diámetro, lineal.
  4. Frames dudosos, que se revisan a mano con revisar_etiquetas.py:
       conf baja (< 0,5), hueco relleno de 3 o más frames, punto que se sale de la parábola de
       sus vecinos (fallo o bote), salto de diámetro, principio y fin de cada trayectoria (la
       pelota puede verse antes de que YOLO la detecte), y detecciones seguras descartadas.

Visibilidad: 0 = sin pelota, 1 = visible entera, 2 = parcial (caja pegada al borde de la imagen
o muy poco cuadrada: cortada, tapada o en la mano). Con visibilidad 2 el diámetro no es fiable.

Salida: datos/etiquetas/<clip>/etiquetas_<cámara>.csv, una fila por frame:
    frame, t, x, y, diametro, visibilidad, origen, conf, dudoso, motivo, pista, revisado
    origen: yolo | interpolada | manual | ninguna
Las filas ya revisadas a mano (revisado = 1) se conservan al volver a ejecutar.

    python codigo/limpiar_etiquetas.py datos/etiquetas                 # todo lo preetiquetado
    python codigo/limpiar_etiquetas.py datos/etiquetas/sesion_1 --solo izq
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from lectura_clips import ClipGrabado

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_GRABACIONES = RAIZ / "grabaciones"
COLUMNAS = ["frame", "t", "x", "y", "diametro", "visibilidad", "origen", "conf", "dudoso", "motivo", "pista",
            "revisado"]
CONF_SEGURA = 0.5


def leer_candidatos(ruta):
    with open(ruta, newline="") as f:
        filas = list(csv.DictReader(f))
    return {campo: np.array([float(fila[campo]) for fila in filas]) for campo in
            ("frame", "t", "x", "y", "ancho", "alto", "conf")}


def seguir(c, tiempos, args):
    """Une los candidatos en trayectorias. Devuelve una lista de listas de índices de candidato."""
    diametros = (c["ancho"] + c["alto"]) / 2
    por_frame = {}
    for k, frame in enumerate(c["frame"].astype(int)):
        por_frame.setdefault(frame, []).append(k)

    pistas, activas = [], []  # activa: dict con miembros, posición, velocidad (px/s), t y diámetro
    for frame in sorted(por_frame):
        t = tiempos[frame]
        activas = [p for p in activas if t - p["t"] <= args.hueco_max]
        parejas = []
        for k in por_frame[frame]:
            for p in activas:
                dt = t - p["t"]
                prevista = p["pos"] + p["vel"] * dt
                puerta = 2 * p["d"] + 10 + 0.3 * np.linalg.norm(p["vel"]) * dt
                distancia = np.linalg.norm(np.array([c["x"][k], c["y"][k]]) - prevista)
                if distancia < puerta and 0.6 < diametros[k] / p["d"] < 1.6:
                    parejas.append((distancia / puerta, k, id(p), p))
        usados_k, usadas_p = set(), set()
        for _, k, clave, p in sorted(parejas, key=lambda e: e[0]):  # asignación voraz por coste
            if k in usados_k or clave in usadas_p:
                continue
            usados_k.add(k)
            usadas_p.add(clave)
            pos = np.array([c["x"][k], c["y"][k]])
            dt = t - p["t"]
            p["vel"] = 0.5 * p["vel"] + 0.5 * (pos - p["pos"]) / dt
            p["pos"], p["t"], p["d"] = pos, t, 0.7 * p["d"] + 0.3 * diametros[k]
            p["miembros"].append(k)
        for k in por_frame[frame]:
            if k not in usados_k:
                nueva = {"miembros": [k], "pos": np.array([c["x"][k], c["y"][k]]), "vel": np.zeros(2),
                         "t": t, "d": diametros[k]}
                pistas.append(nueva)
                activas.append(nueva)
    return [p["miembros"] for p in pistas]


def ajuste_parabola(ts, valores, t):
    grado = 2 if len(ts) >= 4 else 1
    t0 = ts.mean()  # centrar el tiempo: polyfit con segundos absolutos (cientos) está mal condicionado
    return np.polyval(np.polyfit(ts - t0, valores, grado), t - t0)


def es_objeto_fijo(c, p, tiempos, args):
    """Trayectoria larga, inmóvil y de confianza baja: un objeto del fondo, no la pelota."""
    frames = c["frame"][p].astype(int)
    if tiempos[frames.max()] - tiempos[frames.min()] < args.fijo_max or c["conf"][p].mean() >= CONF_SEGURA:
        return False
    diametro = np.median((c["ancho"][p] + c["alto"][p]) / 2)
    return max(c["x"][p].std(), c["y"][p].std()) < max(1.5, 0.1 * diametro)


def etiquetar(c, pistas, tiempos, ancho_img, alto_img, args):
    n = len(tiempos)
    et = {"x": np.full(n, np.nan), "y": np.full(n, np.nan), "diametro": np.full(n, np.nan),
          "visibilidad": np.zeros(n, int), "origen": np.array(["ninguna"] * n, dtype=object),
          "conf": np.zeros(n), "pista": np.full(n, -1), "motivo": [[] for _ in range(n)]}

    fijas = [p for p in pistas if es_objeto_fijo(c, p, tiempos, args)]
    de_fijas = {k for p in fijas for k in p}
    buenas = [p for p in pistas if len(p) >= args.min_frames and c["conf"][p].max() >= CONF_SEGURA
              and p[0] not in de_fijas]
    buenas.sort(key=lambda p: -c["conf"][p].sum())  # la de más confianza acumulada gana los frames
    elegidos = set()
    for numero, p in enumerate(buenas):
        frames = c["frame"][p].astype(int)
        libres = [k for k, f in zip(p, frames) if et["pista"][f] < 0]
        if not libres:
            continue
        for k in libres:
            f = int(c["frame"][k])
            et["x"][f], et["y"][f] = c["x"][k], c["y"][k]
            et["diametro"][f] = (c["ancho"][k] + c["alto"][k]) / 2
            et["conf"][f], et["origen"][f], et["pista"][f] = c["conf"][k], "yolo", numero
            et["visibilidad"][f] = visibilidad(c, k, ancho_img, alto_img)
            elegidos.add(k)
            if c["conf"][k] < CONF_SEGURA:
                et["motivo"][f].append("conf baja")
        frames = sorted(int(c["frame"][k]) for k in libres)
        et["motivo"][frames[0]].append("inicio")
        et["motivo"][frames[-1]].append("fin")
        rellenar_huecos(et, frames, tiempos, numero, args)

    revisar_coherencia(et, args)
    for k in np.flatnonzero(c["conf"] >= CONF_SEGURA):  # detecciones seguras que no se han usado
        f = int(c["frame"][k])
        if k in de_fijas:
            continue
        if k not in elegidos and et["pista"][f] < 0:
            et["motivo"][f].append("detección descartada")
        elif k not in elegidos:
            et["motivo"][f].append("otra pelota posible")
    et["fijas"] = len(fijas)
    return et


def visibilidad(c, k, ancho_img, alto_img):
    x1, x2 = c["x"][k] - c["ancho"][k] / 2, c["x"][k] + c["ancho"][k] / 2
    y1, y2 = c["y"][k] - c["alto"][k] / 2, c["y"][k] + c["alto"][k] / 2
    en_borde = x1 < 2 or y1 < 2 or x2 > ancho_img - 2 or y2 > alto_img - 2
    proporcion = max(c["ancho"][k], c["alto"][k]) / max(1e-6, min(c["ancho"][k], c["alto"][k]))
    return 2 if en_borde or proporcion > 1.3 else 1


def rellenar_huecos(et, frames, tiempos, numero, args):
    """Huecos entre frames detectados de una misma pista: parábola en el tiempo con los vecinos."""
    frames = np.array(frames)
    for a, b in zip(frames[:-1], frames[1:]):
        if b - a <= 1:
            continue
        vecinos = np.concatenate([frames[frames <= a][-args.vecinos:], frames[frames >= b][:args.vecinos]])
        ts = tiempos[vecinos]
        for f in range(a + 1, b):
            t = tiempos[f]
            et["x"][f] = ajuste_parabola(ts, et["x"][vecinos], t)
            et["y"][f] = ajuste_parabola(ts, et["y"][vecinos], t)
            et["diametro"][f] = np.interp(t, [tiempos[a], tiempos[b]], [et["diametro"][a], et["diametro"][b]])
            et["visibilidad"][f] = 1 if et["visibilidad"][a] == et["visibilidad"][b] == 1 else 2
            et["origen"][f], et["pista"][f] = "interpolada", numero
            if b - a - 1 >= 3:
                et["motivo"][f].append(f"hueco de {b - a - 1}")


def revisar_coherencia(et, args):
    """Marca los puntos de YOLO que se salen de la parábola de sus vecinos o cambian de tamaño de golpe."""
    for f in np.flatnonzero(et["origen"] == "yolo"):
        mismos = np.flatnonzero(et["pista"] == et["pista"][f])
        vecinos = mismos[(np.abs(mismos - f) <= args.vecinos) & (mismos != f)]
        if len(vecinos) < 4:
            continue
        ts = vecinos.astype(float)  # en frames: solo importa la forma, y así no depende de la escala
        error = np.hypot(ajuste_parabola(ts, et["x"][vecinos], f) - et["x"][f],
                         ajuste_parabola(ts, et["y"][vecinos], f) - et["y"][f])
        if error > max(3.0, 0.25 * et["diametro"][f]):
            et["motivo"][f].append(f"fuera de trayectoria ({error:.0f} px)")
        mediana = np.median(et["diametro"][vecinos])
        if abs(et["diametro"][f] - mediana) > 0.25 * mediana:
            et["motivo"][f].append("salto de diámetro")


def escribir(ruta, et, tiempos, revisadas):
    with open(ruta, "w", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(COLUMNAS)
        for i, t in enumerate(tiempos):
            if i in revisadas:
                escritor.writerow([revisadas[i][c] for c in COLUMNAS])
                continue
            hay = not np.isnan(et["x"][i])
            escritor.writerow([
                i, f"{t:.6f}",
                f"{et['x'][i]:.2f}" if hay else "", f"{et['y'][i]:.2f}" if hay else "",
                f"{et['diametro'][i]:.2f}" if hay else "", et["visibilidad"][i], et["origen"][i],
                f"{et['conf'][i]:.3f}" if et["origen"][i] == "yolo" else "",
                int(bool(et["motivo"][i])), "; ".join(et["motivo"][i]),
                et["pista"][i] if hay else "", 0,
            ])


def leer_revisadas(ruta):
    if not ruta.exists():
        return {}
    with open(ruta, newline="") as f:
        return {int(fila["frame"]): fila for fila in csv.DictReader(f) if fila["revisado"] == "1"}


def limpiar(ruta_candidatos, args):
    camara = ruta_candidatos.stem.removeprefix("candidatos_")
    relativa = ruta_candidatos.parent.relative_to(args.etiquetas)
    clip = ClipGrabado(CARPETA_GRABACIONES / relativa)
    stream = clip.streams[camara]
    alto_img, ancho_img = stream.leer(0).shape[:2]
    tiempos = stream.tiempos.copy()
    clip.cerrar()

    c = leer_candidatos(ruta_candidatos)
    pistas = seguir(c, tiempos, args) if len(c["frame"]) else []
    et = etiquetar(c, pistas, tiempos, ancho_img, alto_img, args)
    ruta = ruta_candidatos.with_name(f"etiquetas_{camara}.csv")
    revisadas = leer_revisadas(ruta)
    escribir(ruta, et, tiempos, revisadas)

    con_pelota = int((et["visibilidad"] > 0).sum())
    print(f"{relativa} {camara}: {len(tiempos)} frames, {con_pelota} con pelota "
          f"({int((et['origen'] == 'yolo').sum())} YOLO + {int((et['origen'] == 'interpolada').sum())} "
          f"interpoladas), {len(set(et['pista'][et['pista'] >= 0]))} trayectorias, "
          f"{et['fijas']} objetos fijos descartados, "
          f"{sum(bool(m) for m in et['motivo'])} dudosos, {len(revisadas)} revisados conservados")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ruta", type=Path, nargs="?", default=RAIZ / "datos" / "etiquetas")
    parser.add_argument("--solo", choices=["izq", "der"], default=None)
    parser.add_argument("--hueco-max", type=float, default=0.15, help="s sin detección antes de cortar la pista")
    parser.add_argument("--min-frames", type=int, default=4, help="frames mínimos de una trayectoria creíble")
    parser.add_argument("--vecinos", type=int, default=4, help="puntos a cada lado para ajustar la parábola")
    parser.add_argument("--fijo-max", type=float, default=1.0,
                        help="s que puede estar inmóvil una trayectoria de confianza baja antes de tomarla por objeto fijo")
    args = parser.parse_args()
    args.etiquetas = RAIZ / "datos" / "etiquetas"

    patron = f"candidatos_{args.solo}.csv" if args.solo else "candidatos_*.csv"
    for ruta in sorted(Path(args.ruta).resolve().rglob(patron)):
        limpiar(ruta, args)


if __name__ == "__main__":
    main()
