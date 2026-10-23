"""Grabación de vídeos con la OAK-D S2 para el dataset y las pruebas offline.

Graba las dos cámaras mono (global shutter, par estéreo) comprimidas en MJPEG
dentro de la propia OAK, y opcionalmente la cámara a color en H.265. Por cada
frame se guardan la secuencia y las marcas de tiempo, necesarias para el
filtro de Kalman y para emparejar izquierda/derecha al calcular profundidad.

Estructura de salida:
    grabaciones/AAAAMMDD_HHMMSS/
        metadatos.json      configuración de la sesión y estadísticas por clip
        calibracion.json    calibración de fábrica de la cámara
        clip_001/
            izq.mjpeg       cámara mono izquierda (CAM_B)
            der.mjpeg       cámara mono derecha (CAM_C)
            color.h265      cámara a color (CAM_A), solo con --color
            tiempos.csv     una fila por frame y cámara

Controles de la ventana de vista previa:
    r       empezar / parar un clip
    q, Esc  salir

Ejemplos:
    python codigo/grabar.py
    python codigo/grabar.py --fps 120 --exposicion 1000 --iso 800
    python codigo/grabar.py --resolucion 800 --fps 60 --color
    python codigo/grabar.py --sin-preview --duracion 10
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

RAIZ = Path(__file__).resolve().parents[1]

RESOLUCIONES_MONO = {
    400: (dai.MonoCameraProperties.SensorResolution.THE_400_P, 120),
    720: (dai.MonoCameraProperties.SensorResolution.THE_720_P, 60),
    800: (dai.MonoCameraProperties.SensorResolution.THE_800_P, 60),
}
FPS_COLOR = 30


def parsear_argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolucion", type=int, choices=RESOLUCIONES_MONO, default=400,
                   help="altura en píxeles de las cámaras mono (por defecto 400)")
    p.add_argument("--fps", type=float, default=None,
                   help="FPS de las cámaras mono (por defecto, el máximo de la resolución)")
    p.add_argument("--exposicion", type=int, default=None,
                   help="exposición manual en µs; sin ella se usa autoexposición")
    p.add_argument("--iso", type=int, default=800, help="sensibilidad con exposición manual (100-1600)")
    p.add_argument("--calidad", type=int, default=95, help="calidad MJPEG (1-100)")
    p.add_argument("--color", action="store_true", help="grabar también la cámara a color (1080p H.265)")
    p.add_argument("--salida", type=Path, default=RAIZ / "grabaciones", help="carpeta de grabaciones")
    p.add_argument("--sin-preview", action="store_true",
                   help="sin ventana: graba un único clip desde el arranque hasta --duracion o Ctrl+C")
    p.add_argument("--duracion", type=float, default=None, help="duración máxima de cada clip en segundos")
    args = p.parse_args()

    resolucion, fps_max = RESOLUCIONES_MONO[args.resolucion]
    args.resolucion_dai = resolucion
    if args.fps is None:
        args.fps = fps_max
    elif args.fps > fps_max:
        p.error(f"a {args.resolucion}p el máximo es {fps_max} FPS")
    if args.exposicion is not None and args.exposicion > 1e6 / args.fps:
        p.error(f"a {args.fps:g} FPS la exposición no puede superar {int(1e6 / args.fps)} µs")
    return args


def crear_pipeline(args):
    pipeline = dai.Pipeline()

    for nombre, socket in (("izq", dai.CameraBoardSocket.CAM_B), ("der", dai.CameraBoardSocket.CAM_C)):
        mono = pipeline.create(dai.node.MonoCamera)
        mono.setBoardSocket(socket)
        mono.setResolution(args.resolucion_dai)
        mono.setFps(args.fps)
        if args.exposicion is not None:
            mono.initialControl.setManualExposure(args.exposicion, args.iso)

        encoder = pipeline.create(dai.node.VideoEncoder)
        encoder.setDefaultProfilePreset(args.fps, dai.VideoEncoderProperties.Profile.MJPEG)
        encoder.setQuality(args.calidad)
        mono.out.link(encoder.input)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName(nombre)
        encoder.out.link(xout.input)

    if args.color:
        color = pipeline.create(dai.node.ColorCamera)
        color.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color.setFps(FPS_COLOR)

        encoder = pipeline.create(dai.node.VideoEncoder)
        encoder.setDefaultProfilePreset(FPS_COLOR, dai.VideoEncoderProperties.Profile.H265_MAIN)
        color.video.link(encoder.input)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("color")
        encoder.out.link(xout.input)

    return pipeline


class Clip:
    """Escribe los streams de un clip y lleva la cuenta de frames perdidos."""

    EXTENSIONES = {"izq": "mjpeg", "der": "mjpeg", "color": "h265"}

    def __init__(self, carpeta, streams):
        carpeta.mkdir(parents=True)
        self.carpeta = carpeta
        self.archivos = {s: open(carpeta / f"{s}.{self.EXTENSIONES[s]}", "wb") for s in streams}
        self.csv_archivo = open(carpeta / "tiempos.csv", "w", newline="")
        self.csv = csv.writer(self.csv_archivo)
        self.csv.writerow(["camara", "secuencia", "t_dispositivo_s", "t_host_s", "exposicion_us", "iso", "bytes"])
        self.frames = {s: 0 for s in streams}
        self.perdidos = {s: 0 for s in streams}
        self.ultima_secuencia = {}
        self.inicio = time.monotonic()

    def escribir(self, stream, frame):
        datos = frame.getData()
        secuencia = frame.getSequenceNum()
        anterior = self.ultima_secuencia.get(stream)
        if anterior is not None and secuencia > anterior + 1:
            self.perdidos[stream] += secuencia - anterior - 1
        self.ultima_secuencia[stream] = secuencia

        self.archivos[stream].write(datos)
        self.csv.writerow([
            stream,
            secuencia,
            f"{frame.getTimestampDevice().total_seconds():.6f}",
            f"{frame.getTimestamp().total_seconds():.6f}",
            int(frame.getExposureTime().total_seconds() * 1e6),
            frame.getSensitivity(),
            len(datos),
        ])
        self.frames[stream] += 1

    def duracion(self):
        return time.monotonic() - self.inicio

    def cerrar(self):
        for archivo in self.archivos.values():
            archivo.close()
        self.csv_archivo.close()
        return {
            "carpeta": self.carpeta.name,
            "duracion_s": round(self.duracion(), 2),
            "frames": self.frames,
            "perdidos": self.perdidos,
        }


def dibujar_preview(jpeg, grabando, clip, fps_medido):
    imagen = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    texto = f"{fps_medido:5.1f} FPS"
    if grabando:
        cv2.circle(imagen, (20, 20), 8, (0, 0, 255), -1)
        texto += f" | REC {clip.carpeta.name} {clip.duracion():5.1f} s | perdidos {sum(clip.perdidos.values())}"
    else:
        texto += " | r: grabar  q: salir"
    cv2.putText(imagen, texto, (40, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return imagen


def main():
    args = parsear_argumentos()
    streams = ["izq", "der"] + (["color"] if args.color else [])

    with dai.Device(crear_pipeline(args)) as dispositivo:
        velocidad_usb = dispositivo.getUsbSpeed()
        print(f"Cámara: {dispositivo.getProductName()} | USB: {velocidad_usb.name}")
        if velocidad_usb not in (dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS):
            print("AVISO: la cámara no está en USB 3; pueden perderse frames a FPS altos.")

        sesion = args.salida / datetime.now().strftime("%Y%m%d_%H%M%S")
        sesion.mkdir(parents=True)
        dispositivo.readCalibration().eepromToJsonFile(str(sesion / "calibracion.json"))
        metadatos = {
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "camara": dispositivo.getProductName(),
            "mxid": dispositivo.getMxId(),
            "usb": velocidad_usb.name,
            "mono": {
                "resolucion": args.resolucion,
                "fps": args.fps,
                "exposicion_us": args.exposicion or "auto",
                "iso": args.iso if args.exposicion is not None else "auto",
                "calidad_mjpeg": args.calidad,
            },
            "color": {"resolucion": 1080, "fps": FPS_COLOR, "codec": "H265"} if args.color else None,
            "clips": [],
        }

        def guardar_metadatos():
            (sesion / "metadatos.json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False))

        guardar_metadatos()
        colas = {s: dispositivo.getOutputQueue(s, maxSize=120, blocking=False) for s in streams}

        clip = None

        def empezar_clip():
            nuevo = Clip(sesion / f"clip_{len(metadatos['clips']) + 1:03d}", streams)
            print(f"Grabando {nuevo.carpeta.name}...")
            return nuevo

        def terminar_clip():
            resumen = clip.cerrar()
            metadatos["clips"].append(resumen)
            guardar_metadatos()
            print(f"  {resumen['carpeta']}: {resumen['duracion_s']} s, frames {resumen['frames']}, "
                  f"perdidos {resumen['perdidos']}")

        if args.sin_preview:
            clip = empezar_clip()

        ultimo_jpeg = None
        ultima_preview = 0.0
        contador_fps, inicio_fps, fps_medido = 0, time.monotonic(), 0.0
        print(f"Sesión: {sesion}")

        try:
            while True:
                hay_datos = False
                for stream, cola in colas.items():
                    for frame in cola.tryGetAll():
                        hay_datos = True
                        if clip is not None:
                            clip.escribir(stream, frame)
                        if stream == "izq":
                            ultimo_jpeg = frame.getData()
                            contador_fps += 1

                ahora = time.monotonic()
                if ahora - inicio_fps >= 1.0:
                    fps_medido = contador_fps / (ahora - inicio_fps)
                    contador_fps, inicio_fps = 0, ahora

                if clip is not None and args.duracion is not None and clip.duracion() >= args.duracion:
                    terminar_clip()
                    clip = None
                    if args.sin_preview:
                        break

                if args.sin_preview:
                    if not hay_datos:
                        time.sleep(0.001)
                    continue

                # La vista previa se refresca a ~30 FPS para no frenar la escritura
                if ultimo_jpeg is not None and ahora - ultima_preview >= 1 / 30:
                    cv2.imshow("Grabacion OAK (izquierda)", dibujar_preview(ultimo_jpeg, clip is not None, clip, fps_medido))
                    ultima_preview = ahora
                    tecla = cv2.waitKey(1) & 0xFF
                    if tecla in (ord("q"), 27):
                        break
                    if tecla == ord("r"):
                        if clip is None:
                            clip = empezar_clip()
                        else:
                            terminar_clip()
                            clip = None
                elif not hay_datos:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            if clip is not None:
                terminar_clip()
            cv2.destroyAllWindows()

    if not metadatos["clips"]:
        # Sesión sin clips: no dejar carpetas vacías
        for archivo in sesion.iterdir():
            archivo.unlink()
        sesion.rmdir()
        print("No se grabó ningún clip.")


if __name__ == "__main__":
    main()
