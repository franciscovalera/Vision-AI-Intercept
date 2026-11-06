"""Evaluación de PelotaNet frame a frame contra las etiquetas, y comparación con YOLO.

1. DECODIFICACIÓN (lo mismo que se hará con las salidas de la OAK)
   Por cada frame de la pila: máximo del mapa de calor (sigmoide de los logits). Si supera el
   umbral (0,5 por defecto) hay pelota en la celda (j, i) del máximo, con centro
       x = 4 * (j + dx),  y = 4 * (i + dy)
   y diámetro el del canal de diámetro en esa celda. Solo una pelota por frame.

2. ACIERTOS
   Solo cuentan los frames no dudosos. Una predicción es correcta si el error del centro es menor
   que 4 px y que medio diámetro (para las pelotas pequeñas el margen se reduce).
       VP: hay pelota, se predice y está dentro del margen
       FP: se predice y no hay pelota, o hay pero fuera del margen
       FN: hay pelota y no se predice, o se predice fuera del margen (cuenta como FP y como FN)
   precisión = VP / (VP + FP), sensibilidad = VP / (VP + FN). Error medio del centro y del
   diámetro sobre los VP (el del diámetro solo con visibilidad 1). Sensibilidad por tamaño de la
   pelota (< 12, 12-20, > 20 px) y por posición del frame en la pila (f0 = el más antiguo).

3. YOLO COMO REFERENCIA
   Con los mismos frames, la detección de mayor confianza de candidatos_<cámara>.csv con
   conf >= 0,5. Ojo, la comparación favorece a YOLO: las etiquetas salen de sus propias
   detecciones (limpiadas con la coherencia temporal), así que donde YOLO ve la pelota su error
   es casi cero por construcción, y donde no la ve el frame suele quedar sin pelota o dudoso.
   Sus fallos solo se cuentan cuando la limpieza los ha corregido (falsos positivos fuera de la
   trayectoria, huecos interpolados con seguridad).

4. IoU (solape entre la pelota predicha y la etiquetada)
   La red no da cajas sino centro y diámetro, así que la IoU se calcula de dos formas:
       cajas    cuadrados de lado = diámetro centrados en el centro (comparable con la de YOLO)
       círculos área de la intersección de los dos círculos / área de la unión (la correcta
                geométricamente para una pelota)
   Por detector: media y mediana de la IoU en los VP; % de pelotas etiquetadas con IoU >= 0,5 y
   >= 0,75 (sin predicción cuenta como IoU 0); AP50 y AP50-95 estilo COCO (una predicción por
   frame ordenada por puntuación, interpolación de 101 puntos, umbrales de IoU 0,50:0,05:0,95).
   Todo también por tamaño de la pelota. En el desglose por tamaño, como en COCO, cada
   predicción cuenta en el grupo de la pelota etiquetada de su frame; si el frame no tiene
   pelota, en el del diámetro predicho.
   La IoU de YOLO es ~1 por construcción (las etiquetas son sus cajas); solo baja en los frames
   en los que la limpieza movió o interpoló la etiqueta.
   Con --desde-predicciones se recalcula a partir de predicciones.csv de una evaluación anterior,
   sin volver a pasar la red:
       python codigo/evaluar.py --desde-predicciones resultados/entrenamiento/a16_f3/predicciones.csv \
           resultados/entrenamiento/a24_f3/predicciones.csv --nombre comparacion

Las pilas no se solapan (paso = F), como en la OAK, donde cada frame se procesa una vez.

    python codigo/evaluar.py --pesos modelos/entrenamientos/a16_f3/mejor.pt
    python codigo/evaluar.py --pesos ... --sesiones sesion_2 --umbral 0.5

Salida en resultados/entrenamiento/<nombre>/: metricas.json, tabla.md (con YOLO),
predicciones.csv (un frame por fila, para dibujar o buscar fallos) y hojas de recortes al azar
de fallos (fallos_fn.jpg, fallos_fp.jpg) y aciertos (aciertos.jpg) del primer modelo: verde =
etiqueta, rojo = predicción con su puntuación.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datos import REDUCCION, PilasPelota, buscar_fuentes, preparar_cache
from modelo import PelotaNet

RAIZ = Path(__file__).resolve().parents[1]
TAMANOS = [("< 12 px", 0, 12), ("12-20 px", 12, 20), ("> 20 px", 20, 1e9)]


def decodificar(calor, desplazamiento, diametro):
    """Salidas de la red (calor en logits) → puntuación, x, y, diámetro por frame: tensores B x F."""
    B, F, h, w = calor.shape
    prob = torch.sigmoid(calor.float()).flatten(2)
    puntuacion, celda = prob.max(2)
    i, j = celda // w, celda % w
    desplazamiento = desplazamiento.float().view(B, F, 2, h * w)
    dx = desplazamiento[:, :, 0].gather(2, celda[..., None])[..., 0]
    dy = desplazamiento[:, :, 1].gather(2, celda[..., None])[..., 0]
    d = diametro.float().flatten(2).gather(2, celda[..., None])[..., 0]
    return puntuacion, REDUCCION * (j + dx), REDUCCION * (i + dy), d


@torch.no_grad()
def predecir(modelo, ds, dispositivo, lote=64, trabajadores=8):
    """Pasa la red por todas las pilas de ds. Devuelve arrays por frame (N*F) de predicción y etiqueta."""
    modelo.eval()
    cargador = DataLoader(ds, lote, shuffle=False, num_workers=trabajadores, pin_memory=True)
    salida = {k: [] for k in ("puntuacion", "x", "y", "d")}
    for x, _ in cargador:
        with torch.autocast("cuda", torch.bfloat16, enabled=dispositivo.type == "cuda"):
            x = x.to(dispositivo, non_blocking=True).float().contiguous(memory_format=torch.channels_last)
            calor, desplazamiento, diametro = modelo(x)
        for k, v in zip(salida, decodificar(calor, desplazamiento, diametro)):
            salida[k].append(v.cpu().numpy().ravel())
    pred = {k: np.concatenate(v) for k, v in salida.items()}
    return pred, etiquetas_por_frame(ds)


def etiquetas_por_frame(ds):
    """Etiquetas (y origen: fuente, frame) de cada frame de cada pila de ds, en el orden de predecir()."""
    filas = {k: [] for k in ("x", "y", "d", "visibilidad", "dudoso", "fuente", "frame", "posicion")}
    for k, i in ds.indices:
        fuente = ds.fuentes[k]
        sl = slice(i, i + ds.F)
        filas["x"].append(fuente.x[sl]); filas["y"].append(fuente.y[sl]); filas["d"].append(fuente.diametro[sl])
        filas["visibilidad"].append(fuente.visibilidad[sl]); filas["dudoso"].append(fuente.dudoso[sl])
        filas["fuente"].append(np.full(ds.F, k)); filas["frame"].append(np.arange(i, i + ds.F))
        filas["posicion"].append(np.arange(ds.F))
    return {k: np.concatenate(v) for k, v in filas.items()}


def metricas(hay_pred, px, py, pd, et):
    """Métricas de la sección 2. hay_pred, px, py, pd: arrays por frame; et: etiquetas_por_frame()."""
    validos = ~et["dudoso"]
    pelota = (et["visibilidad"] > 0) & validos
    error = np.hypot(px - et["x"], py - et["y"])
    margen = np.minimum(4.0, et["d"] / 2)
    vp = pelota & hay_pred & (error < margen)
    fp = validos & hay_pred & ~vp
    fn = pelota & ~vp
    n_vp, n_fp, n_fn = int(vp.sum()), int(fp.sum()), int(fn.sum())
    vp_diam = vp & (et["visibilidad"] == 1)
    resultado = {
        "frames": int(validos.sum()), "con_pelota": int(pelota.sum()),
        "vp": n_vp, "fp": n_fp, "fn": n_fn,
        "precision": n_vp / max(1, n_vp + n_fp), "sensibilidad": n_vp / max(1, n_vp + n_fn),
        "error_px": float(error[vp].mean()) if n_vp else None,
        "error_diametro_px": float(np.abs(pd - et["d"])[vp_diam].mean()) if vp_diam.any() else None,
        "fp_sin_pelota": int((fp & (et["visibilidad"] == 0)).sum()),
        "sensibilidad_por_tamano": {}, "sensibilidad_por_posicion": {},
    }
    resultado["f1"] = 2 * resultado["precision"] * resultado["sensibilidad"] / max(
        1e-9, resultado["precision"] + resultado["sensibilidad"])
    for nombre, lo, hi in TAMANOS:
        m = pelota & (et["d"] >= lo) & (et["d"] < hi)
        resultado["sensibilidad_por_tamano"][nombre] = [float(vp[m].mean()) if m.any() else None, int(m.sum())]
    for f in np.unique(et["posicion"]):
        m = pelota & (et["posicion"] == f)
        resultado["sensibilidad_por_posicion"][f"f{f}"] = float(vp[m].mean()) if m.any() else None
    return resultado


def prediccion_yolo(ds, et, conf_min=0.5, con_conf=False):
    """Detección de mayor confianza de YOLO por frame (conf >= conf_min), en el orden de et.

    Con con_conf devuelve además la confianza (0 donde no hay detección), para la AP.
    """
    mejores = {}  # (fuente, frame) → (conf, x, y, d)
    for k, fuente in enumerate(ds.fuentes):
        ruta = RAIZ / "datos" / "etiquetas" / fuente.clip / f"candidatos_{fuente.camara}.csv"
        with open(ruta, newline="") as f:
            for fila in csv.DictReader(f):
                conf = float(fila["conf"])
                clave = (k, int(fila["frame"]))
                if conf >= conf_min and conf > mejores.get(clave, (0,))[0]:
                    d = (float(fila["ancho"]) + float(fila["alto"])) / 2
                    mejores[clave] = (conf, float(fila["x"]), float(fila["y"]), d)
    n = len(et["frame"])
    hay, conf, px, py, pd = np.zeros(n, bool), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for m, clave in enumerate(zip(et["fuente"], et["frame"])):
        if clave in mejores:
            hay[m] = True
            conf[m], px[m], py[m], pd[m] = mejores[clave]
    return (hay, conf, px, py, pd) if con_conf else (hay, px, py, pd)


def iou_cajas(x1, y1, d1, x2, y2, d2):
    """IoU de cuadrados de lado d centrados en (x, y). Arrays; NaN donde falta alguno."""
    ancho = np.clip(np.minimum(x1 + d1 / 2, x2 + d2 / 2) - np.maximum(x1 - d1 / 2, x2 - d2 / 2), 0, None)
    alto = np.clip(np.minimum(y1 + d1 / 2, y2 + d2 / 2) - np.maximum(y1 - d1 / 2, y2 - d2 / 2), 0, None)
    interseccion = ancho * alto
    return interseccion / (d1 ** 2 + d2 ** 2 - interseccion)


def iou_circulos(x1, y1, d1, x2, y2, d2):
    """IoU de círculos de diámetro d centrados en (x, y) (área de la lente de intersección)."""
    r1, r2 = np.abs(d1) / 2, np.abs(d2) / 2
    dist = np.hypot(x1 - x2, y1 - y2)
    with np.errstate(divide="ignore", invalid="ignore"):
        a1 = np.arccos(np.clip((dist ** 2 + r1 ** 2 - r2 ** 2) / (2 * dist * r1), -1, 1))
        a2 = np.arccos(np.clip((dist ** 2 + r2 ** 2 - r1 ** 2) / (2 * dist * r2), -1, 1))
        lente = r1 ** 2 * (a1 - np.sin(2 * a1) / 2) + r2 ** 2 * (a2 - np.sin(2 * a2) / 2)
    interseccion = np.where(dist >= r1 + r2, 0.0,
                            np.where(dist <= np.abs(r1 - r2), np.pi * np.minimum(r1, r2) ** 2, lente))
    return interseccion / (np.pi * r1 ** 2 + np.pi * r2 ** 2 - interseccion)


def ap_coco(puntuacion, iou, es_pelota, n_pelotas, umbral_iou):
    """AP con una predicción por frame. puntuacion, iou: por predicción (iou NaN si el frame no tiene
    pelota); es_pelota: el frame tiene pelota etiquetada. Interpolación de 101 puntos como COCO."""
    if n_pelotas == 0:
        return None
    orden = np.argsort(-puntuacion, kind="stable")
    vp = (es_pelota & (np.nan_to_num(iou) >= umbral_iou))[orden]
    vp_acum, fp_acum = np.cumsum(vp), np.cumsum(~vp)
    sensibilidad = vp_acum / n_pelotas
    precision = vp_acum / np.maximum(1, vp_acum + fp_acum)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # envolvente
    puntos = np.linspace(0, 1, 101)
    k = np.searchsorted(sensibilidad, puntos, side="left")
    return float(np.mean(np.where(k < len(precision), precision[np.minimum(k, len(precision) - 1)], 0)))


def metricas_iou(hay_pred, puntuacion, px, py, pd, et):
    """Métricas de IoU de la sección 4. hay_pred: predicción por encima del umbral; puntuacion: para la AP."""
    validos = ~et["dudoso"]
    pelota = (et["visibilidad"] > 0) & validos
    error = np.hypot(px - et["x"], py - et["y"])
    vp = pelota & hay_pred & (error < np.minimum(4.0, et["d"] / 2))
    candidatas = validos & (puntuacion > 0)  # predicciones que entran en la AP
    resultado = {}
    for tipo, funcion in (("cajas", iou_cajas), ("circulos", iou_circulos)):
        iou = funcion(px, py, pd, et["x"], et["y"], et["d"])
        iou_pelota = np.where(pelota & hay_pred, np.nan_to_num(iou), 0.0)
        grupos = {"todas": (pelota, np.ones_like(pelota))}
        for nombre, lo, hi in TAMANOS:
            de_et = pelota & (et["d"] >= lo) & (et["d"] < hi)
            # predicciones del grupo: las de frames con pelota de ese tamaño, o sin pelota y diámetro predicho del grupo
            de_pred = de_et | (validos & ~pelota & (pd >= lo) & (pd < hi))
            grupos[nombre] = (de_et, de_pred)
        resultado[tipo] = {}
        for nombre, (de_et, de_pred) in grupos.items():
            n = int(de_et.sum())
            m_vp, sel = vp & de_et, candidatas & de_pred
            aps = [ap_coco(puntuacion[sel], iou[sel], pelota[sel], n, u) for u in np.arange(0.5, 0.96, 0.05)]
            resultado[tipo][nombre] = {
                "pelotas": n, "vp": int(m_vp.sum()),
                "iou_media_vp": float(np.mean(iou[m_vp])) if m_vp.any() else None,
                "iou_mediana_vp": float(np.median(iou[m_vp])) if m_vp.any() else None,
                "iou_50": float(np.mean(iou_pelota[de_et] >= 0.5)) if n else None,
                "iou_75": float(np.mean(iou_pelota[de_et] >= 0.75)) if n else None,
                "ap50": aps[0], "ap50_95": float(np.mean(aps)) if aps[0] is not None else None,
            }
    return resultado


def tabla_iou(filas):
    """filas: lista de (nombre, metricas_iou)."""
    def p(v):
        return "-" if v is None else f"{100 * v:.1f}"

    def n(v):
        return "-" if v is None else f"{v:.3f}"

    lineas = []
    for tipo, titulo in (("cajas", "IoU de cajas cuadradas (lado = diámetro)"),
                         ("circulos", "IoU de círculos")):
        lineas += [f"### {titulo}", "",
                   "| Detector | Tamaño | Pelotas | VP | IoU media VP | IoU mediana VP | IoU ≥ 0,5 % | IoU ≥ 0,75 % | AP50 % | AP50-95 % |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
        for nombre, m in filas:
            for grupo, g in m[tipo].items():
                lineas.append(f"| {nombre} | {grupo} | {g['pelotas']} | {g['vp']} | {n(g['iou_media_vp'])} | "
                              f"{n(g['iou_mediana_vp'])} | {p(g['iou_50'])} | {p(g['iou_75'])} | {p(g['ap50'])} | "
                              f"{p(g['ap50_95'])} |")
        lineas.append("")
    return "\n".join(lineas)


def leer_predicciones(ruta):
    """predicciones.csv → (claves (clip, cámara, frame), etiquetas como etiquetas_por_frame(), predicción)."""
    with open(ruta, newline="") as f:
        filas = list(csv.DictReader(f))
    col = lambda k, tipo=float: np.array([tipo(fila[k]) for fila in filas])
    et = {"x": col("x"), "y": col("y"), "d": col("diametro"), "visibilidad": col("visibilidad", int),
          "dudoso": col("dudoso", int).astype(bool), "posicion": col("posicion", int)}
    pred = {"puntuacion": col("puntuacion"), "x": col("x_pred"), "y": col("y_pred"), "d": col("diametro_pred")}
    claves = [(fila["clip"], fila["camara"], int(fila["frame"])) for fila in filas]
    return claves, et, pred


def yolo_por_claves(claves, conf_min):
    """Como prediccion_yolo, pero para frames dados por (clip, cámara, frame)."""
    mejores = {}
    for clip, camara in sorted({(c, k) for c, k, _ in claves}):
        with open(RAIZ / "datos" / "etiquetas" / clip / f"candidatos_{camara}.csv", newline="") as f:
            for fila in csv.DictReader(f):
                conf, clave = float(fila["conf"]), (clip, camara, int(fila["frame"]))
                if conf >= conf_min and conf > mejores.get(clave, (0,))[0]:
                    mejores[clave] = (conf, float(fila["x"]), float(fila["y"]),
                                      (float(fila["ancho"]) + float(fila["alto"])) / 2)
    n = len(claves)
    conf, px, py, pd = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for m, clave in enumerate(claves):
        if clave in mejores:
            conf[m], px[m], py[m], pd[m] = mejores[clave]
    return conf, px, py, pd


def iou_desde_predicciones(rutas, umbral, nombre):
    filas, todo, base = [], {"umbral": umbral}, None
    for ruta in rutas:
        claves, et, pred = leer_predicciones(ruta)
        if base is None:
            base = (claves, et)
        elif claves != base[0]:
            raise ValueError(f"{ruta} no tiene los mismos frames que {rutas[0]}")
        m = metricas_iou(pred["puntuacion"] >= umbral, pred["puntuacion"], pred["x"], pred["y"], pred["d"], et)
        filas.append((f"PelotaNet {ruta.parent.name}", m))
        todo[ruta.parent.name] = m
    claves, et = base
    conf, px, py, pd = yolo_por_claves(claves, 0.0)  # todas las candidatas (conf >= 0,1) para la AP
    yolo = metricas_iou(conf >= 0.5, conf, px, py, pd, et)
    filas.append(("YOLO pelota_v1 (conf ≥ 0,5)*", yolo))
    todo["yolo"] = yolo
    guardar_iou(filas, todo, umbral, RAIZ / "resultados" / "entrenamiento" / nombre)


def guardar_iou(filas, todo, umbral, salida):
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "iou.json").write_text(json.dumps(todo, indent=1, ensure_ascii=False))
    texto = (f"IoU en sesion_2 (mismos frames no dudosos, pilas sin solapar; umbral {umbral} para IoU media, "
             f"mediana y % de pelotas; AP con todas las predicciones ordenadas por puntuación).\n\n"
             + tabla_iou(filas) + "\n" +
             "VP = acierto con el criterio de tabla.md (error < 4 px y < diámetro/2). "
             "IoU ≥ x % = pelotas etiquetadas con predicción (umbral) e IoU ≥ x, sobre todas las etiquetadas.\n\n"
             "\\* La IoU de YOLO es ~1 por construcción: las etiquetas son sus propias cajas (diámetro = media de "
             "ancho y alto). Solo baja en los frames en los que limpiar_etiquetas.py movió o interpoló la "
             "etiqueta. No es comparable con PelotaNet; se pone como referencia del techo.\n")
    (salida / "iou.md").write_text(texto)
    print(texto)


def curva(pred, et, umbrales=np.round(np.arange(0.1, 0.95, 0.05), 2)):
    """Precisión y sensibilidad para varios umbrales del mapa de calor."""
    return [{"umbral": float(u), **{k: v for k, v in metricas(pred["puntuacion"] >= u, pred["x"], pred["y"],
                                                                 pred["d"], et).items()
                                     if k in ("precision", "sensibilidad", "f1")}} for u in umbrales]


def tabla_markdown(filas):
    """filas: lista de (nombre, métricas)."""
    def p(v):
        return "-" if v is None else f"{100 * v:.1f}"

    def n(v):
        return "-" if v is None else f"{v:.2f}"

    lineas = ["| Detector | Precisión % | Sensibilidad % | F1 % | Error centro px | Error diámetro px | "
              + " | ".join(f"Sens. {t[0]} %" for t in TAMANOS) + " | FP sin pelota |",
              "|---" * (7 + len(TAMANOS)) + "|"]
    for nombre, m in filas:
        lineas.append(f"| {nombre} | {p(m['precision'])} | {p(m['sensibilidad'])} | {p(m['f1'])} | "
                      f"{n(m['error_px'])} | {n(m['error_diametro_px'])} | "
                      + " | ".join(p(m["sensibilidad_por_tamano"][t[0]][0]) for t in TAMANOS)
                      + f" | {m['fp_sin_pelota']} |")
    tam = filas[0][1]["sensibilidad_por_tamano"]
    lineas.append("")
    lineas.append(f"Frames evaluados (no dudosos): {filas[0][1]['frames']}, con pelota: {filas[0][1]['con_pelota']} "
                  f"(" + ", ".join(f"{t}: {v[1]}" for t, v in tam.items()) + ").")
    return "\n".join(lineas)


def hoja(ds, et, pred, filas, ruta, n=48, lado=96, semilla=0):
    """Recortes de lado x lado px alrededor de la etiqueta (o de la predicción) de n frames al azar."""
    if len(filas) == 0:
        return
    filas = np.random.default_rng(semilla).choice(filas, min(n, len(filas)), replace=False)
    recortes = []
    for m in filas:
        fuente = ds.fuentes[et["fuente"][m]]
        imagen = cv2.cvtColor(np.array(fuente.frames[et["frame"][m]]), cv2.COLOR_GRAY2BGR)
        hay_et = et["visibilidad"][m] > 0
        cx, cy = (et["x"][m], et["y"][m]) if hay_et else (pred["x"][m], pred["y"][m])
        x0 = int(np.clip(cx - lado / 2, 0, imagen.shape[1] - lado))
        y0 = int(np.clip(cy - lado / 2, 0, imagen.shape[0] - lado))
        r = cv2.resize(imagen[y0:y0 + lado, x0:x0 + lado], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        if hay_et:
            cv2.circle(r, (int(2 * (et["x"][m] - x0)), int(2 * (et["y"][m] - y0))),
                       max(2, int(et["d"][m])), (0, 255, 0), 1)
        cv2.drawMarker(r, (int(2 * (pred["x"][m] - x0)), int(2 * (pred["y"][m] - y0))), (0, 0, 255),
                       cv2.MARKER_CROSS, 10, 1)
        cv2.putText(r, f"{pred['puntuacion'][m]:.2f}", (3, 14), 0, 0.45, (0, 0, 255), 1)
        cv2.putText(r, f"{fuente.clip.split('/')[0][-9:]} {fuente.camara} {et['frame'][m]}",
                    (3, 2 * lado - 5), 0, 0.35, (0, 255, 255), 1)
        recortes.append(r)
    while len(recortes) % 8:
        recortes.append(np.zeros_like(recortes[0]))
    cv2.imwrite(str(ruta), np.vstack([np.hstack(recortes[i:i + 8]) for i in range(0, len(recortes), 8)]))


def hojas(ds, et, pred, umbral, salida):
    validos = ~et["dudoso"]
    pelota = (et["visibilidad"] > 0) & validos
    hay = pred["puntuacion"] >= umbral
    error = np.hypot(pred["x"] - et["x"], pred["y"] - et["y"])
    vp = pelota & hay & (error < np.minimum(4.0, et["d"] / 2))
    hoja(ds, et, pred, np.flatnonzero(pelota & ~vp), salida / "fallos_fn.jpg")
    hoja(ds, et, pred, np.flatnonzero(validos & hay & ~vp), salida / "fallos_fp.jpg")
    hoja(ds, et, pred, np.flatnonzero(vp), salida / "aciertos.jpg")


def cargar_modelo(ruta, frames, anchura, dispositivo):
    modelo = PelotaNet(frames, frames, anchura)
    modelo.load_state_dict(torch.load(ruta, map_location="cpu"))
    return modelo.to(dispositivo).to(memory_format=torch.channels_last)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pesos", type=Path, nargs="+",
                        help="uno o varios state_dict (mejor.pt); con varios se comparan en la misma tabla")
    parser.add_argument("--anchura", type=int, nargs="+", default=[16], help="una por pesos (o una para todos)")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--sesiones", nargs="+", default=["sesion_2"])
    parser.add_argument("--umbral", type=float, default=0.5)
    parser.add_argument("--nombre", default=None, help="carpeta de salida (por defecto, la del primer entrenamiento)")
    parser.add_argument("--desde-predicciones", type=Path, nargs="+", default=None,
                        help="solo IoU, a partir de predicciones.csv ya calculados (sin red ni GPU)")
    parser.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.desde_predicciones:
        iou_desde_predicciones(args.desde_predicciones, args.umbral, args.nombre or "comparacion")
        return
    if not args.pesos:
        parser.error("hace falta --pesos o --desde-predicciones")

    dispositivo = torch.device(args.dispositivo)
    fuentes = buscar_fuentes(args.sesiones)
    preparar_cache(fuentes)
    ds = PilasPelota(fuentes, args.frames, aumentar=False, paso=args.frames)
    anchuras = args.anchura * len(args.pesos) if len(args.anchura) == 1 else args.anchura

    filas, todo = [], {"sesiones": args.sesiones, "umbral": args.umbral, "pilas": len(ds)}
    et, filas_iou = None, []
    for ruta, anchura in zip(args.pesos, anchuras):
        nombre = ruta.parent.name
        modelo = cargar_modelo(ruta, args.frames, anchura, dispositivo)
        pred, et = predecir(modelo, ds, dispositivo)
        m = metricas(pred["puntuacion"] >= args.umbral, pred["x"], pred["y"], pred["d"], et)
        m["curva"] = curva(pred, et)
        filas.append((f"PelotaNet {nombre}", m))
        filas_iou.append((f"PelotaNet {nombre}", metricas_iou(pred["puntuacion"] >= args.umbral, pred["puntuacion"],
                                                             pred["x"], pred["y"], pred["d"], et)))
        todo[nombre] = m
        print(f"{nombre}: precisión {100 * m['precision']:.1f} %, sensibilidad {100 * m['sensibilidad']:.1f} %, "
              f"error {m['error_px']:.2f} px", flush=True)
        if len(filas) == 1:
            primera = pred
    yolo = metricas(*prediccion_yolo(ds, et), et)
    filas.append(("YOLO pelota_v1 (conf ≥ 0,5)", yolo))
    todo["yolo"] = yolo
    hay, conf, px, py, pd = prediccion_yolo(ds, et, conf_min=0.0, con_conf=True)
    filas_iou.append(("YOLO pelota_v1 (conf ≥ 0,5)*", metricas_iou(conf >= 0.5, conf, px, py, pd, et)))

    salida = RAIZ / "resultados" / "entrenamiento" / (args.nombre or args.pesos[0].parent.name)
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "metricas.json").write_text(json.dumps(todo, indent=1, ensure_ascii=False))
    tabla = tabla_markdown(filas)
    (salida / "tabla.md").write_text(
        f"Evaluación en {', '.join(args.sesiones)} (pilas sin solapar, umbral {args.umbral}).\n\n{tabla}\n\n"
        "Acierto: error del centro < 4 px y < diámetro/2. Las etiquetas salen de YOLO (limpiadas con "
        "coherencia temporal), así que la comparación favorece a YOLO.\n")
    with open(salida / "predicciones.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "camara", "frame", "posicion", "x", "y", "diametro", "visibilidad", "dudoso",
                    "puntuacion", "x_pred", "y_pred", "diametro_pred"])
        for m in range(len(et["frame"])):
            fuente = ds.fuentes[et["fuente"][m]]
            w.writerow([fuente.clip, fuente.camara, et["frame"][m], et["posicion"][m],
                        f"{et['x'][m]:.2f}", f"{et['y'][m]:.2f}", f"{et['d'][m]:.2f}", et["visibilidad"][m],
                        int(et["dudoso"][m]), f"{primera['puntuacion'][m]:.3f}", f"{primera['x'][m]:.2f}",
                        f"{primera['y'][m]:.2f}", f"{primera['d'][m]:.2f}"])
    hojas(ds, et, primera, args.umbral, salida)
    guardar_iou(filas_iou, {"umbral": args.umbral, **{n: m for n, m in filas_iou}}, args.umbral, salida)
    print(tabla)
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    main()
