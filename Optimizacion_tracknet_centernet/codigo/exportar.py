"""Exporta variantes de PelotaNet a ONNX y a .blob para la OAK (Myriad X / RVC2, depthai v2).

Pasos para cada variante:
  1. Construir la red (pesos aleatorios o los de un entrenamiento con --pesos) y fusionarla
     (BatchNorm y división por 255 dentro de las convoluciones, ver modelo.fusionar).
     Sin --pesos, antes se calibran las BatchNorm con unas pasadas sobre imágenes aleatorias:
     así las activaciones quedan normalizadas como en una red entrenada y las salidas dependen
     de verdad de los frames (sin esto son casi constantes y la comprobación no prueba nada).
  2. Envolverla para el despliegue (clase Desplegable):
       - una entrada por frame (f0 = el más antiguo ... f{F-1} = el más reciente), de forma
         1 x 1 x alto x ancho: así cada entrada recibe directamente un ImgFrame GRAY8 de la
         cámara y no hay que copiar bytes para apilarlos fuera de la red; la red los concatena;
       - sigmoide en el mapa de calor, para que la salida ya sea una probabilidad.
  3. ONNX (opset 12) y comprobación con onnxruntime: misma entrada por PyTorch (sin fusionar)
     y por el ONNX; las salidas deben coincidir salvo redondeo.
  4. .blob con blobconverter (compilador de Luxonis en la nube, OpenVINO 2022.1, FP16) con
     entrada U8: la cámara manda los bytes 0-255 tal cual. Sin mean/scale en el optimizador
     (blobconverter los añade por defecto), porque la normalización ya está en los pesos.
  5. JSON con la descripción de la variante (lo lee medir_oak.py).

Nombre de una variante: a<anchura>_f<frames>_<ancho>x<alto>[_cc]  (cc = cabeza compartida).
Con F frames de entrada la red da F salidas (todas las posiciones de la pila: MIMO).

    python codigo/exportar.py                               # las variantes de la primera medida
    python codigo/exportar.py --variantes a16_f3_640x400 a8_f1_320x192_cc --shaves 6
    python codigo/exportar.py --variantes a16_f3_640x400 --pesos modelos/entrenamientos/x/mejor.pt
    python codigo/exportar.py --sin-blob                    # solo ONNX + comprobación

Salida en modelos/variantes/pelotanet_<variante>.{onnx,blob,json}.
"""

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from modelo import PelotaNet, contar_macs, contar_parametros, fusionar

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_VARIANTES = RAIZ / "modelos" / "variantes"
NOMBRES_SALIDA = ["calor", "desplazamiento", "diametro"]
# Primera medida: anchura x frames x resolución, más la cabeza compartida en la variante central
VARIANTES_POR_DEFECTO = [f"a{a}_f{f}_{r}" for a in (8, 16, 24) for f in (1, 3) for r in ("640x400", "320x192")]
VARIANTES_POR_DEFECTO += ["a16_f3_640x400_cc", "a16_f3_320x192_cc"]


def leer_variante(nombre):
    """'a16_f3_640x400_cc' -> dict con anchura, frames, ancho, alto y cabeza_compartida."""
    m = re.fullmatch(r"a(\d+)_f(\d+)_(\d+)x(\d+)(_cc)?", nombre)
    if m is None:
        raise ValueError(f"variante no válida: {nombre} (formato a16_f3_640x400 o a16_f3_640x400_cc)")
    anchura, frames, ancho, alto = (int(g) for g in m.groups()[:4])
    if ancho % 16 or alto % 16:
        raise ValueError(f"{nombre}: ancho y alto deben ser múltiplos de 16 (la red baja hasta 1/16)")
    return {"anchura": anchura, "frames": frames, "ancho": ancho, "alto": alto,
            "cabeza_compartida": m.group(5) is not None}


@torch.no_grad()
def calibrar_batchnorm(modelo, frames, ancho, alto, pasadas=20):
    """Estadísticas de BatchNorm (media móvil) a partir de imágenes aleatorias 0-255."""
    modelo.train()
    for capa in modelo.modules():
        if isinstance(capa, nn.BatchNorm2d):
            capa.momentum = None  # media acumulada exacta, no exponencial
    for _ in range(pasadas):
        modelo(torch.rand(2, frames, alto, ancho) * 255)


class Desplegable(nn.Module):
    """Red fusionada con una entrada por frame y sigmoide en el calor."""

    def __init__(self, modelo_fusionado):
        super().__init__()
        self.modelo = modelo_fusionado

    def forward(self, *frames):
        x = torch.cat(frames, 1) if len(frames) > 1 else frames[0]
        calor, desplazamiento, diametro = self.modelo(x)
        return torch.sigmoid(calor), desplazamiento, diametro


def exportar_onnx(desplegable, frames, ancho, alto, ruta):
    entradas = tuple(torch.zeros(1, 1, alto, ancho) for _ in range(frames))
    torch.onnx.export(desplegable, entradas, str(ruta), opset_version=12, dynamo=False,
                      input_names=[f"f{i}" for i in range(frames)], output_names=NOMBRES_SALIDA)


@torch.no_grad()
def comprobar_onnx(modelo, ruta_onnx, frames, ancho, alto):
    """Máxima diferencia entre el modelo sin fusionar (PyTorch) y el ONNX, con frames aleatorios 0-255."""
    import onnxruntime

    x = torch.randint(0, 256, (1, frames, alto, ancho)).float()
    calor, desplazamiento, diametro = modelo.eval()(x)
    referencia = [torch.sigmoid(calor), desplazamiento, diametro]
    sesion = onnxruntime.InferenceSession(str(ruta_onnx), providers=["CPUExecutionProvider"])
    salidas = sesion.run(NOMBRES_SALIDA, {f"f{i}": x[:, i:i + 1].numpy() for i in range(frames)})
    return {nombre: float(np.abs(r.numpy() - s).max()) for nombre, r, s in zip(NOMBRES_SALIDA, referencia, salidas)}


def compilar_blob(ruta_onnx, shaves, ruta_blob):
    import blobconverter

    blobconverter.set_defaults(silent=True)
    with tempfile.TemporaryDirectory() as temporal:
        generado = blobconverter.from_onnx(model=str(ruta_onnx), data_type="FP16", shaves=shaves,
                                           optimizer_params=[], compile_params=["-ip U8"],
                                           output_dir=temporal, use_cache=False)
        shutil.move(generado, ruta_blob)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variantes", nargs="+", default=VARIANTES_POR_DEFECTO)
    parser.add_argument("--pesos", type=Path, default=None,
                        help="state_dict de un entrenamiento (sin él, pesos aleatorios: vale para medir velocidad)")
    parser.add_argument("--shaves", type=int, default=6,
                        help="núcleos SHAVE por hilo de inferencia (con 2 hilos quedan 13 libres: 6)")
    parser.add_argument("--sin-blob", action="store_true", help="solo ONNX y comprobación")
    parser.add_argument("--salida", type=Path, default=CARPETA_VARIANTES)
    args = parser.parse_args()
    variantes = [(nombre, leer_variante(nombre)) for nombre in args.variantes]  # falla antes de empezar
    args.salida.mkdir(parents=True, exist_ok=True)

    for nombre, v in variantes:
        torch.manual_seed(0)
        modelo = PelotaNet(v["frames"], v["frames"], v["anchura"], v["cabeza_compartida"])
        if args.pesos is not None:
            modelo.load_state_dict(torch.load(args.pesos, map_location="cpu"))
        else:
            calibrar_batchnorm(modelo, v["frames"], v["ancho"], v["alto"])
        modelo.eval()
        base = args.salida / f"pelotanet_{nombre}"

        exportar_onnx(Desplegable(fusionar(modelo)), v["frames"], v["ancho"], v["alto"], base.with_suffix(".onnx"))
        diferencias = comprobar_onnx(modelo, base.with_suffix(".onnx"), v["frames"], v["ancho"], v["alto"])
        descripcion = {
            **v,
            "entradas": [f"f{i}" for i in range(v["frames"])],
            "salidas": {"calor": [v["frames"], v["alto"] // 4, v["ancho"] // 4],
                        "desplazamiento": [2 * v["frames"], v["alto"] // 4, v["ancho"] // 4],
                        "diametro": [v["frames"], v["alto"] // 4, v["ancho"] // 4]},
            "parametros": contar_parametros(modelo),
            "gmacs": round(contar_macs(modelo, v["alto"], v["ancho"]) / 1e9, 4),
            "pesos": str(args.pesos) if args.pesos else None,
            "shaves": None if args.sin_blob else args.shaves,
            "dif_max_onnx": diferencias,
        }
        texto = (f"{nombre:>20}: {descripcion['parametros']:>9,} parámetros, {descripcion['gmacs']:.3f} GMACs, "
                 f"dif. ONNX {max(diferencias.values()):.1e}")
        if not args.sin_blob:
            compilar_blob(base.with_suffix(".onnx"), args.shaves, base.with_suffix(".blob"))
            texto += f", blob {base.with_suffix('.blob').stat().st_size / 1e6:.2f} MB"
        base.with_suffix(".json").write_text(json.dumps(descripcion, indent=2, ensure_ascii=False))
        print(texto, flush=True)


if __name__ == "__main__":
    main()
