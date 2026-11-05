"""Mide variantes de PelotaNet dentro de la OAK: inferencias/s, posiciones/s y latencia.

Cadena en la cámara (la misma que usará el sistema final):

    MonoCamera izq (640x400 GRAY8, 120 FPS)
      [-> ImageManip (si la red usa otra resolución: reduce en GRAY8)]
      -> Script "agrupador" (solo si F > 1)
      -> NeuralNetwork con F entradas f0..f{F-1}  -> host (calor, desplazamiento, diámetro)

Cómo llegan F frames a la red: la red tiene una entrada por frame (ver exportar.Desplegable) y un
nodo Script, que corre en el procesador de la propia cámara, junta F frames CONSECUTIVOS (por
número de secuencia) y manda cada uno a su entrada. Solo reenvía referencias a los ImgFrame: no
copia ni apila bytes. Si se pierde un frame (la red va saturada), el grupo se descarta y se
empieza otro, para que la red nunca vea una pila con huecos. Los grupos no se solapan: cada
inferencia da la posición de F frames nuevos (MIMO), así que posiciones/s = F x inferencias/s.
Con F = 1 la cámara va directa a la red.

La red no frena a la cámara: si no da abasto, los frames sobrantes se descartan, y lo medido es
lo que la red da de verdad.

Latencia = reloj del host al recibir el resultado - marca de tiempo del resultado (la del frame
de la pila del que la hereda depthai; ver la columna latencia y la explicación en la memoria).
Incluye la transferencia por USB de las tres salidas y su lectura en numpy.

Modos:
    python codigo/medir_oak.py                                   # todas las variantes de modelos/variantes
    python codigo/medir_oak.py --variantes a16_f3_640x400 a8_f1_320x192 --segundos 15
    python codigo/medir_oak.py --comprobar --variantes a16_f3_640x400
        manda frames de un clip grabado al blob dentro de la OAK y compara sus salidas con las del
        ONNX en el ordenador (comprueba entradas, orden de los frames y error de FP16)

Resultados en resultados/oak/medidas.csv (una fila por variante y medida).
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from grabacion import CARPETA_GRABACIONES, RESOLUCIONES_MONO
from lectura_clips import ClipGrabado, buscar_clips

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_VARIANTES = RAIZ / "modelos" / "variantes"
CARPETA_RESULTADOS = RAIZ / "resultados" / "oak"
SALIDAS = ["calor", "desplazamiento", "diametro"]
ANCHO_CAMARA, ALTO_CAMARA = 640, 400

# Corre en la OAK (intérprete de Python de depthai). Agrupa F frames consecutivos.
SCRIPT_AGRUPADOR = """
F = {frames}
grupo = []
anterior = -1
while True:
    frame = node.io['img'].get()
    secuencia = frame.getSequenceNum()
    if secuencia != anterior + 1:
        grupo = []  # se ha perdido algún frame: la pila tendría un hueco
    anterior = secuencia
    grupo.append(frame)
    if len(grupo) == F:
        for i in range(F):
            node.io['f' + str(i)].send(grupo[i])
        grupo = []
"""


def cargar_variante(nombre):
    blob = CARPETA_VARIANTES / f"pelotanet_{nombre}.blob"
    if not blob.exists():
        raise FileNotFoundError(f"{blob} no existe: ejecuta antes exportar.py --variantes {nombre}")
    return blob, json.loads(blob.with_suffix(".json").read_text())


def crear_red(pipeline, blob, v, hilos):
    red = pipeline.create(dai.node.NeuralNetwork)
    red.setBlobPath(str(blob))
    red.setNumInferenceThreads(hilos)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("red")
    red.out.link(xout.input)
    return red


def pipeline_en_vivo(blob, v, fps, hilos):
    pipeline = dai.Pipeline()
    mono = pipeline.create(dai.node.MonoCamera)
    mono.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono.setResolution(RESOLUCIONES_MONO[ALTO_CAMARA][0])
    mono.setFps(fps)
    # El agrupador retiene F frames y la red otros F mientras infiere: sin buffers de sobra la
    # cámara (3 por defecto) se queda sin memoria donde escribir y se para
    buffers = 3 * v["frames"] + 4
    mono.setNumFramesPool(buffers)
    fuente = mono.out

    if (v["ancho"], v["alto"]) != (ANCHO_CAMARA, ALTO_CAMARA):
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setResize(v["ancho"], v["alto"])
        manip.initialConfig.setKeepAspectRatio(False)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
        manip.setMaxOutputFrameSize(v["ancho"] * v["alto"])
        manip.setNumFramesPool(buffers)
        manip.inputImage.setBlocking(False)
        manip.inputImage.setQueueSize(2)
        fuente.link(manip.inputImage)
        fuente = manip.out

    red = crear_red(pipeline, blob, v, hilos)
    if v["frames"] == 1:
        fuente.link(red.inputs["f0"])
        red.inputs["f0"].setBlocking(False)
        red.inputs["f0"].setQueueSize(1)
    else:
        script = pipeline.create(dai.node.Script)
        script.setScript(SCRIPT_AGRUPADOR.format(frames=v["frames"]))
        fuente.link(script.inputs["img"])
        script.inputs["img"].setBlocking(False)  # si el agrupador espera a la red, se pierden frames
        script.inputs["img"].setQueueSize(2)     # (y el grupo empieza de nuevo), nunca se frena la cámara
        for i in range(v["frames"]):
            script.outputs[f"f{i}"].link(red.inputs[f"f{i}"])
            red.inputs[f"f{i}"].setBlocking(True)  # la pila entera llega a la red o espera, no se rompe
            red.inputs[f"f{i}"].setQueueSize(1)
    return pipeline


def pipeline_desde_host(blob, v):
    pipeline = dai.Pipeline()
    red = crear_red(pipeline, blob, v, hilos=1)
    for i in range(v["frames"]):
        entrada = pipeline.create(dai.node.XLinkIn)
        entrada.setStreamName(f"f{i}")
        entrada.setMaxDataSize(v["ancho"] * v["alto"])
        entrada.out.link(red.inputs[f"f{i}"])
        red.inputs[f"f{i}"].setBlocking(True)
        red.inputs[f"f{i}"].setQueueSize(2)
    return pipeline


def leer_salidas(paquete, v):
    """NNData -> dict de arrays float32 con la forma de cada salida (canales, alto/4, ancho/4).

    Se leen los bytes FP16 directamente del buffer (getData + desplazamiento de cada capa):
    getLayerFp16 los convierte a una lista de Python y es unas 10 veces más lento.
    """
    datos = paquete.getData()
    desplazamientos = {capa.name: capa.offset for capa in paquete.getAllLayers()}
    salidas = {}
    for nombre in SALIDAS:
        forma = v["salidas"][nombre]
        inicio = desplazamientos[nombre]
        salidas[nombre] = np.frombuffer(datos[inicio:inicio + 2 * int(np.prod(forma))].tobytes(),
                                        dtype=np.float16).astype(np.float32).reshape(forma)
    return salidas


def medir_en_vivo(dispositivo, v, segundos):
    cola = dispositivo.getOutputQueue("red", maxSize=4, blocking=False)
    latencias, lectura_ms, n, inicio = [], [], 0, None
    while True:
        paquete = cola.get()
        if inicio is None:  # el primer resultado incluye el arranque de la red: no cuenta
            inicio = time.monotonic()
            continue
        latencias.append((dai.Clock.now() - paquete.getTimestamp()).total_seconds() * 1000)
        t = time.perf_counter()
        salidas = leer_salidas(paquete, v)
        salidas["calor"].reshape(v["frames"], -1).argmax(1)  # el pico de cada frame: lo mínimo del decodificado
        lectura_ms.append((time.perf_counter() - t) * 1000)
        n += 1
        if time.monotonic() - inicio >= segundos:
            break
    duracion = time.monotonic() - inicio
    latencias = np.array(latencias)
    return {
        "inferencias_s": round(n / duracion, 1),
        "posiciones_s": round(n * v["frames"] / duracion, 1),
        "latencia_media_ms": round(float(latencias.mean()), 1),
        "latencia_p95_ms": round(float(np.percentile(latencias, 95)), 1),
        "lectura_host_ms": round(float(np.mean(lectura_ms)), 2),
    }


def comprobar(dispositivo, blob, v, clip, grupos):
    """Mismos frames por el blob en la OAK y por el ONNX en el host; devuelve las diferencias.

    Control: la diferencia con el ONNX si se le dan los frames en orden inverso. Si es mucho mayor
    que la diferencia OAK-ONNX, el blob recibe los frames en el orden correcto (f0 = el más antiguo).
    """
    import onnxruntime

    sesion = onnxruntime.InferenceSession(str(blob.with_suffix(".onnx")), providers=["CPUExecutionProvider"])
    colas = [dispositivo.getInputQueue(f"f{i}") for i in range(v["frames"])]
    salida = dispositivo.getOutputQueue("red", maxSize=4, blocking=True)
    stream = clip.principal
    paso = max(1, (len(stream) - v["frames"]) // grupos)
    diferencias = {nombre: [] for nombre in SALIDAS}
    escala = {nombre: [] for nombre in SALIDAS}
    invertido = []
    for g in range(grupos):
        frames = [stream.leer(g * paso + i) for i in range(v["frames"])]
        frames = [cv2.resize(f, (v["ancho"], v["alto"]), interpolation=cv2.INTER_AREA)
                  if f.shape != (v["alto"], v["ancho"]) else f for f in frames]
        for i, (cola, f) in enumerate(zip(colas, frames)):
            imagen = dai.ImgFrame()
            imagen.setType(dai.ImgFrame.Type.GRAY8)
            imagen.setWidth(v["ancho"])
            imagen.setHeight(v["alto"])
            imagen.setData(f.reshape(-1))
            imagen.setSequenceNum(g)
            cola.send(imagen)
        oak = leer_salidas(salida.get(), v)
        onnx = ejecutar_onnx(sesion, frames)
        for nombre in SALIDAS:
            diferencias[nombre].append(float(np.abs(oak[nombre] - onnx[nombre]).max()))
            escala[nombre].append(float(np.abs(onnx[nombre]).max()))
        if v["frames"] > 1:
            invertido.append(float(np.abs(oak["calor"] - ejecutar_onnx(sesion, frames[::-1])["calor"]).max()))
    resultado = {f"dif_max_{n}": round(max(diferencias[n]), 5) for n in SALIDAS}
    resultado.update({f"rango_{n}": round(max(escala[n]), 3) for n in SALIDAS})
    if invertido:
        resultado["dif_calor_orden_invertido"] = round(min(invertido), 5)
    return resultado


def ejecutar_onnx(sesion, frames):
    salidas = sesion.run(SALIDAS, {f"f{i}": f[None, None].astype(np.float32) for i, f in enumerate(frames)})
    return {nombre: s[0] for nombre, s in zip(SALIDAS, salidas)}


def guardar(fila):
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    archivo = CARPETA_RESULTADOS / "medidas.csv"
    columnas = ["fecha", "modo", "variante", "anchura", "frames", "entrada", "cabeza_compartida", "parametros",
                "gmacs", "shaves", "hilos", "usb", "inferencias_s", "posiciones_s", "latencia_media_ms",
                "latencia_p95_ms", "lectura_host_ms", "segundos", "detalle"]
    nuevo = not archivo.exists()
    with open(archivo, "a", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        if nuevo:
            escritor.writeheader()
        escritor.writerow(fila)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variantes", nargs="+", default=None,
                        help="por defecto, todas las que tengan .blob en modelos/variantes")
    parser.add_argument("--segundos", type=float, default=10, help="duración de cada medida en vivo")
    parser.add_argument("--hilos", type=int, choices=[1, 2], default=2, help="hilos de inferencia en la OAK")
    parser.add_argument("--fps", type=float, default=120, help="FPS de la cámara mono")
    parser.add_argument("--usb2", action="store_true",
                        help="forzar USB 2 (si en USB 3 falla con X_LINK_DEVICE_NOT_FOUND: cable o conector)")
    parser.add_argument("--comprobar", action="store_true", help="comparar el blob con el ONNX usando un clip")
    parser.add_argument("--clip", type=Path, default=None, help="clip para --comprobar (por defecto, el primero)")
    parser.add_argument("--grupos", type=int, default=20, help="pilas de frames que se comparan con --comprobar")
    args = parser.parse_args()

    nombres = args.variantes or sorted(p.stem.removeprefix("pelotanet_") for p in CARPETA_VARIANTES.glob("*.blob"))
    if not nombres:
        raise SystemExit(f"No hay blobs en {CARPETA_VARIANTES}: ejecuta antes exportar.py")
    clip = ClipGrabado(args.clip or buscar_clips(CARPETA_GRABACIONES)[0]) if args.comprobar else None

    for nombre in nombres:
        blob, v = cargar_variante(nombre)
        if args.comprobar:
            pipeline = pipeline_desde_host(blob, v)
        else:
            pipeline = pipeline_en_vivo(blob, v, args.fps, args.hilos)
        velocidad = dai.UsbSpeed.HIGH if args.usb2 else dai.UsbSpeed.SUPER_PLUS
        with dai.Device(pipeline, maxUsbSpeed=velocidad) as dispositivo:
            usb = dispositivo.getUsbSpeed().name
            fila = {"fecha": datetime.now().isoformat(timespec="seconds"), "variante": nombre,
                    "anchura": v["anchura"], "frames": v["frames"], "entrada": f"{v['ancho']}x{v['alto']}",
                    "cabeza_compartida": v["cabeza_compartida"], "parametros": v["parametros"],
                    "gmacs": v["gmacs"], "shaves": v["shaves"], "usb": usb}
            if args.comprobar:
                resultado = comprobar(dispositivo, blob, v, clip, args.grupos)
                fila.update(modo="comprobar", hilos=1, detalle=json.dumps({"clip": clip.nombre, **resultado}))
                print(f"{nombre:>20}: {resultado}", flush=True)
            else:
                resultado = medir_en_vivo(dispositivo, v, args.segundos)
                fila.update(modo="en vivo", hilos=args.hilos, segundos=args.segundos, **resultado)
                print(f"{nombre:>20}: {v['gmacs']:.2f} GMACs | {resultado['inferencias_s']:5.1f} inf/s | "
                      f"{resultado['posiciones_s']:5.1f} pos/s | latencia {resultado['latencia_media_ms']:5.1f} ms "
                      f"(p95 {resultado['latencia_p95_ms']:.0f}) | lectura host {resultado['lectura_host_ms']:.1f} ms",
                      flush=True)
        guardar(fila)


if __name__ == "__main__":
    main()
