"""Exporta un detector YOLO (v8 / 11 de Ultralytics) a .blob para la OAK-D (RVC2, depthai v2).

Sustituye la cabeza Detect por una equivalente a la de las herramientas de
Luxonis (tools.luxonis.com) para que el decodificador de YoloDetectionNetwork
la entienda: tres salidas output{1,2,3}_yolov6r2 de forma [1, 4+1+nc, H, W] con
las distancias izquierda/arriba/derecha/abajo en celdas (DFL ya resuelto), la
confianza (máximo de las clases) y las clases con sigmoide. En depthai v2 se
configura con anchors vacíos (decodificación sin anchors).

Para cada tamaño:
  1. ONNX (opset 12, simplificado).
  2. Verificación: el mismo tensor de entrada por el .pt original y por el
     ONNX (onnxruntime); se comparan las salidas brutas y las cajas tras NMS.
     También se mide, frente al .pt a 640, cuántas pelotas se siguen
     detectando al reducir la entrada (anticipo de la pérdida de precisión).
  3. .blob con blobconverter (servidor de Luxonis, OpenVINO 2022.1, FP16) y
     un JSON de configuración con el formato de los ejemplos de depthai v2.

La entrada del blob es U8 BGR planar (0-255; la normalización /255 va dentro
del blob). Con las cámaras mono hay que convertir GRAY8 -> BGR888p y
redimensionar a ancho x alto con ImageManip (estirando: 640x400 ya tiene casi
la misma proporción que los tamaños propuestos).

    python codigo/exportar_oak.py --modelo modelos/entrenamientos/pelota_v1/weights/best.pt
    python codigo/exportar_oak.py --modelo ... --tamanos 416x256 --sin-blob   # solo ONNX + verificación

Salida en modelos/oak/: <nombre>_<ancho>x<alto>.{onnx,blob,json}.
"""

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

RAIZ = Path(__file__).resolve().parents[1]
NOMBRES_SALIDA = ["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"]
TAMANOS_POR_DEFECTO = ["640x384", "512x320", "416x256", "320x192"]


class CabezaOak(nn.Module):
    """Cabeza Detect de YOLOv8/11 con las salidas que espera el decodificador YOLO de la OAK."""

    def __init__(self, detect):
        super().__init__()
        self.nl, self.nc, self.reg_max = detect.nl, detect.nc, detect.reg_max
        self.cv2, self.cv3 = detect.cv2, detect.cv3
        self.f, self.i, self.stride = detect.f, detect.i, detect.stride
        # DFL como convolución 1x1 con pesos 0..reg_max-1 (esperanza de la distribución)
        self.proyeccion = nn.Conv2d(self.reg_max, 1, 1, bias=False).requires_grad_(False)
        self.proyeccion.weight.data[:] = torch.arange(self.reg_max, dtype=torch.float).view(1, self.reg_max, 1, 1)

    def forward(self, x):
        salidas = []
        for i in range(self.nl):
            caja = self.cv2[i](x[i])
            n, _, alto, ancho = caja.shape
            caja = caja.view(n, 4, self.reg_max, alto * ancho).permute(0, 2, 1, 3)
            caja = self.proyeccion(F.softmax(caja, dim=1))[:, 0].reshape(n, 4, alto, ancho)
            clases = self.cv3[i](x[i]).sigmoid()
            confianza, _ = clases.max(1, keepdim=True)
            salidas.append(torch.cat([caja, confianza, clases], 1))
        return salidas


def cargar(ruta_pt):
    """(modelo original fusionado, modelo con CabezaOak, nombres de clase)."""
    original = YOLO(str(ruta_pt)).model.float().eval()
    original.fuse()
    if not isinstance(original.model[-1], Detect) or getattr(original.model[-1], "end2end", False):
        raise ValueError("Solo se admiten modelos de detección YOLOv8/11 (cabeza Detect sin end2end)")
    oak = copy.deepcopy(original)
    oak.model[-1] = CabezaOak(oak.model[-1])
    return original, oak.eval(), original.names


def exportar_onnx(modelo, ancho, alto, ruta):
    import onnx
    import onnxsim

    entrada = torch.zeros(1, 3, alto, ancho)
    torch.onnx.export(modelo, entrada, str(ruta), opset_version=12, input_names=["images"],
                      output_names=NOMBRES_SALIDA, do_constant_folding=True, dynamo=False)
    modelo_onnx, ok = onnxsim.simplify(onnx.load(str(ruta)))
    if not ok:
        raise RuntimeError("onnxsim no pudo validar el modelo simplificado")
    onnx.save(modelo_onnx, str(ruta))


def preparar(gris, ancho, alto):
    """Imagen mono -> tensor [1, 3, alto, ancho] en 0-1, estirada como lo haría ImageManip."""
    imagen = cv2.resize(gris, (ancho, alto), interpolation=cv2.INTER_LINEAR)
    return np.repeat(imagen[None, None].astype(np.float32) / 255, 3, axis=1)


def nms(cajas, puntuaciones, conf, iou):
    """Cajas xyxy [N, 4] y puntuaciones [N] -> array [M, 5] (x1, y1, x2, y2, conf)."""
    cajas, puntuaciones = torch.as_tensor(cajas, dtype=torch.float32), torch.as_tensor(puntuaciones, dtype=torch.float32)
    validas = puntuaciones >= conf
    cajas, puntuaciones = cajas[validas], puntuaciones[validas]
    quedan = torchvision.ops.nms(cajas, puntuaciones, iou)
    return torch.cat([cajas[quedan], puntuaciones[quedan, None]], 1).numpy()


def aplanar_oak(salidas, ancho, alto):
    """Salidas de la CabezaOak -> (cajas xyxy en píxeles de la entrada [N, 4], confianza [N], clases [N, nc]).

    Mismo orden de anclas que Ultralytics (nivel a nivel, filas y columnas), como decodifica la OAK.
    """
    cajas, confianzas, clases = [], [], []
    for salida in salidas:
        salida = np.asarray(salida)[0]
        _, h, w = salida.shape
        paso = alto / h
        gy, gx = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
        izq, arr, der, aba = salida[:4]
        cajas.append(np.stack([(gx - izq) * paso, (gy - arr) * paso, (gx + der) * paso, (gy + aba) * paso],
                              -1).reshape(-1, 4))
        confianzas.append(salida[4].reshape(-1))
        clases.append(salida[5:].reshape(salida.shape[0] - 5, -1).T)
    return np.concatenate(cajas), np.concatenate(confianzas), np.concatenate(clases)


def aplanar_original(salida):
    """Salida de Ultralytics [1, 4+nc, N] (xywh en píxeles) -> (cajas xyxy [N, 4], clases [N, nc])."""
    salida = salida[0].numpy()
    cx, cy, w, h = salida[:4]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1), salida[4:].T


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def verificar(ruta_onnx, original, imagenes, ancho, alto, conf, umbral_iou, referencia):
    """Compara el ONNX con el .pt sobre el mismo tensor; devuelve un dict con el resumen."""
    import onnxruntime as ort

    sesion = ort.InferenceSession(str(ruta_onnx), providers=["CPUExecutionProvider"])
    dif_cajas, dif_puntuacion, iguales, total_pt, recuperadas, total_ref = 0.0, 0.0, 0, 0, 0, 0
    for ruta, gris in imagenes:
        entrada = preparar(gris, ancho, alto)
        with torch.no_grad():
            cajas_pt, clases_pt = aplanar_original(original(torch.from_numpy(entrada))[0])
        cajas_onnx, conf_onnx, clases_onnx = aplanar_oak(sesion.run(None, {"images": entrada}), ancho, alto)
        # Salidas brutas, solo en las anclas con alguna puntuación apreciable
        relevantes = clases_pt.max(1) > 0.05
        if relevantes.any():
            dif_cajas = max(dif_cajas, float(np.abs(cajas_pt[relevantes] - cajas_onnx[relevantes]).max()))
            dif_puntuacion = max(dif_puntuacion, float(np.abs(clases_pt[relevantes] - clases_onnx[relevantes]).max()))
        # Cajas tras NMS (en la OAK se filtra por la confianza y se puntúa con la clase; con una clase coinciden)
        det_pt = nms(cajas_pt, clases_pt.max(1), conf, umbral_iou)
        det_onnx = nms(cajas_onnx, np.where(conf_onnx >= conf, clases_onnx.max(1), 0), conf, umbral_iou)
        total_pt += len(det_pt)
        iguales += sum(any(iou(a, b) > 0.95 and abs(a[4] - b[4]) < 0.02 for b in det_onnx) for a in det_pt)
        iguales -= max(0, len(det_onnx) - len(det_pt))  # cajas de más en el ONNX cuentan como fallo
        # Frente al .pt a su tamaño de entrenamiento (cajas en coordenadas de la imagen original)
        alto_img, ancho_img = gris.shape
        escala = np.array([ancho_img / ancho, alto_img / alto] * 2)
        for caja_ref in referencia.get(ruta, []):
            total_ref += 1
            recuperadas += any(iou(caja_ref, caja[:4] * escala) >= 0.5 for caja in det_onnx)
    return {"max_dif_cajas_px": round(dif_cajas, 4), "max_dif_puntuacion": round(dif_puntuacion, 5),
            "detecciones_pt": total_pt, "detecciones_iguales": iguales,
            "pelotas_ref": total_ref, "pelotas_recuperadas": recuperadas}


def compilar_blob(ruta_onnx, ruta_blob, shaves, version):
    import blobconverter

    blobconverter.set_defaults(silent=True)
    with tempfile.TemporaryDirectory() as carpeta:
        blob = blobconverter.from_onnx(
            model=str(ruta_onnx), data_type="FP16", shaves=shaves, version=version, output_dir=carpeta,
            use_cache=False,
            # Entrada U8 0-255; la normalización de YOLO (/255, sin media) va dentro del blob.
            # --reverse_input_channels: el modelo se entrenó en RGB y la OAK entrega BGR
            # (con imágenes mono replicadas en 3 canales da igual).
            optimizer_params=["--mean_values=[0,0,0]", "--scale_values=[255,255,255]", "--reverse_input_channels"],
        )
        shutil.copy(blob, ruta_blob)


def configuracion(nombres, ancho, alto, conf, umbral_iou, blob):
    """JSON con el formato de los ejemplos de depthai v2 (tools.luxonis.com)."""
    return {
        "model": {"blob": blob.name},
        "nn_config": {
            "output_format": "detection",
            "NN_family": "YOLO",
            "input_size": f"{ancho}x{alto}",
            "NN_specific_metadata": {
                "classes": len(nombres),
                "coordinates": 4,
                "anchors": [],
                "anchor_masks": {},
                "iou_threshold": umbral_iou,
                "confidence_threshold": conf,
            },
        },
        "mappings": {"labels": [nombres[i] for i in sorted(nombres)]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", type=Path, required=True, help="pesos .pt (best.pt de un entrenamiento)")
    parser.add_argument("--tamanos", nargs="+", default=TAMANOS_POR_DEFECTO, help="ANCHOxALTO, múltiplos de 32")
    parser.add_argument("--nombre", default=None, help="prefijo de los archivos (por defecto, el del entrenamiento)")
    parser.add_argument("--salida", type=Path, default=RAIZ / "modelos" / "oak")
    parser.add_argument("--imagenes", type=Path, default=RAIZ / "datos" / "pelota_v1" / "images" / "val",
                        help="imágenes para verificar el ONNX")
    parser.add_argument("--conf", type=float, default=0.5, help="umbral de confianza del JSON y de la verificación")
    parser.add_argument("--iou", type=float, default=0.5, help="umbral de IoU del NMS")
    parser.add_argument("--shaves", type=int, default=6)
    parser.add_argument("--version-openvino", default="2022.1")
    parser.add_argument("--sin-blob", action="store_true", help="solo ONNX y verificación (sin conexión)")
    args = parser.parse_args()

    nombre = args.nombre or args.modelo.resolve().parents[1].name
    args.salida.mkdir(parents=True, exist_ok=True)
    original, oak, nombres = cargar(args.modelo)

    rutas = sorted(args.imagenes.glob("*.jpg"))
    imagenes = [(r, cv2.imread(str(r), cv2.IMREAD_GRAYSCALE)) for r in rutas]
    print(f"{len(imagenes)} imágenes de verificación en {args.imagenes}")
    referencia = {}
    for (ruta, _), resultado in zip(imagenes, YOLO(str(args.modelo)).predict([str(r) for r in rutas], imgsz=640,
                                                                             conf=args.conf, verbose=False)):
        referencia[ruta] = resultado.boxes.xyxy.tolist()

    resumen = []
    for tamano in args.tamanos:
        ancho, alto = (int(v) for v in tamano.lower().split("x"))
        if ancho % 32 or alto % 32:
            raise ValueError(f"{tamano}: ancho y alto deben ser múltiplos de 32")
        base = args.salida / f"{nombre}_{ancho}x{alto}"
        print(f"\n== {ancho}x{alto} ==", flush=True)
        exportar_onnx(oak, ancho, alto, base.with_suffix(".onnx"))
        v = verificar(base.with_suffix(".onnx"), original, imagenes, ancho, alto, args.conf, args.iou, referencia)
        print(f"  ONNX vs .pt (mismo tensor): máx. diferencia cajas {v['max_dif_cajas_px']} px, "
              f"puntuaciones {v['max_dif_puntuacion']}; cajas iguales tras NMS {v['detecciones_iguales']}/"
              f"{v['detecciones_pt']}")
        print(f"  pelotas del .pt a 640 que se mantienen a {ancho}x{alto}: "
              f"{v['pelotas_recuperadas']}/{v['pelotas_ref']}")
        equivalente = v["detecciones_iguales"] == v["detecciones_pt"] and v["max_dif_cajas_px"] < 0.5
        if not equivalente:
            print("  ¡El ONNX no equivale al .pt! No se compila el blob.")
        elif not args.sin_blob:
            compilar_blob(base.with_suffix(".onnx"), base.with_suffix(".blob"), args.shaves, args.version_openvino)
            base.with_suffix(".json").write_text(json.dumps(
                configuracion(nombres, ancho, alto, args.conf, args.iou, base.with_suffix(".blob")), indent=2) + "\n")
            print(f"  {base.with_suffix('.blob')} ({base.with_suffix('.blob').stat().st_size / 1e6:.1f} MB)")
        resumen.append({"tamano": f"{ancho}x{alto}", "equivalente": equivalente, **v})

    (args.salida / f"{nombre}_verificacion.json").write_text(json.dumps(
        {"modelo": str(args.modelo), "imagenes": str(args.imagenes), "conf": args.conf, "resultados": resumen},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
