"""Núcleo de grabación con la OAK-D S2: pipeline, sesiones, clips e hilo de captura.

Lo usan grabar.py (línea de comandos) e interfaz_grabacion.py (interfaz gráfica).

Se graban las dos cámaras mono (global shutter, par estéreo) comprimidas en
MJPEG dentro de la propia OAK, y opcionalmente la cámara a color en H.265. Por
cada frame se guardan la secuencia y las marcas de tiempo, necesarias para el
filtro de Kalman y para emparejar izquierda/derecha al calcular profundidad.

Estructura de salida:
    grabaciones/AAAAMMDD_HHMMSS/
        metadatos.json      cámara y resumen de cada clip (configuración, etiqueta, pérdidas)
        calibracion.json    calibración de fábrica de la cámara
        clip_001/
            izq.mjpeg       cámara mono izquierda (CAM_B)
            der.mjpeg       cámara mono derecha (CAM_C)
            color.h265      cámara a color (CAM_A), solo si se activa
            tiempos.csv     una fila por frame y cámara
        clip_002/           una secuencia: varias partes con la misma estructura
            parte_1/
            parte_2/ ...
"""

import csv
import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import depthai as dai

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_GRABACIONES = RAIZ / "grabaciones"

# altura en píxeles -> (resolución del sensor, FPS máximos)
RESOLUCIONES_MONO = {
    400: (dai.MonoCameraProperties.SensorResolution.THE_400_P, 120),
    720: (dai.MonoCameraProperties.SensorResolution.THE_720_P, 60),
    800: (dai.MonoCameraProperties.SensorResolution.THE_800_P, 60),
}
FPS_COLOR = 30
TAMANO_PREVIEW_COLOR = (320, 180)
EXPOSICION_MIN_US = 20
ISO_MIN, ISO_MAX = 100, 1600

ETIQUETAS = ["prueba", "fondo", "tiro raso", "tiro alto", "rebote", "pelota estática", "negativo"]


def exposicion_maxima_us(fps):
    return int(1e6 / fps)


@dataclass
class Configuracion:
    resolucion: int = 400
    fps: float = 120
    exposicion_us: int | None = None  # None = autoexposición
    iso: int = 800
    calidad: int = 95
    color: bool = False

    def validar(self):
        if self.resolucion not in RESOLUCIONES_MONO:
            raise ValueError(f"resolución no válida: {self.resolucion}")
        fps_max = RESOLUCIONES_MONO[self.resolucion][1]
        if not 1 <= self.fps <= fps_max:
            raise ValueError(f"a {self.resolucion}p los FPS deben estar entre 1 y {fps_max}")
        if self.exposicion_us is not None:
            if not EXPOSICION_MIN_US <= self.exposicion_us <= exposicion_maxima_us(self.fps):
                raise ValueError(f"a {self.fps:g} FPS la exposición debe estar entre "
                                 f"{EXPOSICION_MIN_US} y {exposicion_maxima_us(self.fps)} µs")
            if not ISO_MIN <= self.iso <= ISO_MAX:
                raise ValueError(f"el ISO debe estar entre {ISO_MIN} y {ISO_MAX}")
        if not 1 <= self.calidad <= 100:
            raise ValueError("la calidad MJPEG debe estar entre 1 y 100")


def crear_pipeline(config):
    pipeline = dai.Pipeline()

    # Entrada para cambiar la exposición en vivo en las dos cámaras mono
    control = pipeline.create(dai.node.XLinkIn)
    control.setStreamName("control")

    for nombre, socket in (("izq", dai.CameraBoardSocket.CAM_B), ("der", dai.CameraBoardSocket.CAM_C)):
        mono = pipeline.create(dai.node.MonoCamera)
        mono.setBoardSocket(socket)
        mono.setResolution(RESOLUCIONES_MONO[config.resolucion][0])
        mono.setFps(config.fps)
        if config.exposicion_us is not None:
            mono.initialControl.setManualExposure(config.exposicion_us, config.iso)
        control.out.link(mono.inputControl)

        encoder = pipeline.create(dai.node.VideoEncoder)
        encoder.setDefaultProfilePreset(config.fps, dai.VideoEncoderProperties.Profile.MJPEG)
        encoder.setQuality(config.calidad)
        mono.out.link(encoder.input)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName(nombre)
        encoder.out.link(xout.input)

    if config.color:
        color = pipeline.create(dai.node.ColorCamera)
        color.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color.setFps(FPS_COLOR)

        encoder = pipeline.create(dai.node.VideoEncoder)
        encoder.setDefaultProfilePreset(FPS_COLOR, dai.VideoEncoderProperties.Profile.H265_MAIN)
        # Keyframe cada medio segundo: cada clip empieza en uno (ver Clip.escribir)
        encoder.setKeyframeFrequency(FPS_COLOR // 2)
        color.video.link(encoder.input)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("color")
        encoder.out.link(xout.input)

        # Vista previa pequeña sin comprimir para la interfaz
        color.setPreviewSize(*TAMANO_PREVIEW_COLOR)
        color.setInterleaved(True)
        color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        xout_preview = pipeline.create(dai.node.XLinkOut)
        xout_preview.setStreamName("color_preview")
        color.preview.link(xout_preview.input)

    return pipeline


class Sesion:
    """Carpeta con los clips de una ejecución. Solo se crea en disco al grabar el primer clip.

    Sin nombre se usa la fecha y hora; si la carpeta ya existe, se continúa esa
    sesión y los clips nuevos se numeran a partir del último. Al abrir una sesión
    existente se quitan de metadatos.json los clips cuyas carpetas se han borrado.
    """

    def __init__(self, carpeta_raiz=CARPETA_GRABACIONES, nombre=None):
        self.carpeta = Path(carpeta_raiz) / (nombre or datetime.now().strftime("%Y%m%d_%H%M%S"))
        archivo = self.carpeta / "metadatos.json"
        if archivo.exists():
            self.metadatos = json.loads(archivo.read_text())
            if self.limpiar():
                self.guardar()
        else:
            self.metadatos = {"fecha": datetime.now().isoformat(timespec="seconds"), "camara": None, "clips": []}

    def limpiar(self):
        """Quita los clips sin carpeta (o sin tiempos.csv) en disco. Devuelve los nombres quitados."""
        quitados = [c["carpeta"] for c in self.clips if not (self.carpeta / c["carpeta"] / "tiempos.csv").exists()]
        self.metadatos["clips"] = [c for c in self.clips if c["carpeta"] not in quitados]
        return quitados

    @property
    def clips(self):
        return self.metadatos["clips"]

    def preparar(self, info_camara, calibracion):
        self.carpeta.mkdir(parents=True, exist_ok=True)
        if not (self.carpeta / "calibracion.json").exists():
            calibracion.eepromToJsonFile(str(self.carpeta / "calibracion.json"))
        self.metadatos["camara"] = info_camara
        self.guardar()

    def siguiente_clip(self):
        """Nombre del siguiente clip_NNN, sin reutilizar ninguno que esté en disco o en metadatos.json."""
        nombres = [p.name for p in self.carpeta.glob("clip_[0-9][0-9][0-9]")]
        nombres += [c["carpeta"].split("/")[0] for c in self.clips]
        numeros = [int(n[5:]) for n in nombres if n[5:].isdigit()]
        return f"clip_{max(numeros, default=0) + 1:03d}"

    def registrar(self, resumen):
        self.clips.append(resumen)
        self.guardar()

    def guardar(self):
        (self.carpeta / "metadatos.json").write_text(json.dumps(self.metadatos, indent=2, ensure_ascii=False))


class Clip:
    """Escribe los streams de un clip y lleva la cuenta de frames perdidos."""

    EXTENSIONES = {"izq": "mjpeg", "der": "mjpeg", "color": "h265"}

    def __init__(self, carpeta, nombre, streams, config, etiqueta, notas, extra=None):
        carpeta.mkdir(parents=True)
        self.carpeta = carpeta
        self.nombre = nombre  # ruta relativa a la sesión, p. ej. "clip_002/parte_1"
        self.config = config
        self.etiqueta = etiqueta
        self.notas = notas
        self.extra = extra or {}
        self.archivos = {s: open(carpeta / f"{s}.{self.EXTENSIONES[s]}", "wb") for s in streams}
        self.csv_archivo = open(carpeta / "tiempos.csv", "w", newline="")
        self.csv = csv.writer(self.csv_archivo)
        self.csv.writerow(["camara", "secuencia", "t_dispositivo_s", "t_host_s", "exposicion_us", "iso", "bytes"])
        self.frames = {s: 0 for s in streams}
        self.perdidos = {s: 0 for s in streams}
        self.bytes = 0
        self.ultima_secuencia = {}
        self.inicio = time.monotonic()

    def escribir(self, stream, frame, datos):
        # El H.265 solo se puede decodificar desde un keyframe: hasta el primero no se guarda
        # nada, así cada fila de tiempos.csv corresponde a un frame decodificable
        if stream == "color" and stream not in self.ultima_secuencia:
            if frame.getFrameType() != dai.EncodedFrame.FrameType.I:
                return
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
        self.bytes += len(datos)

    def duracion(self):
        return time.monotonic() - self.inicio

    def cerrar(self):
        for archivo in self.archivos.values():
            archivo.close()
        self.csv_archivo.close()
        return {
            "carpeta": self.nombre,
            "etiqueta": self.etiqueta,
            "notas": self.notas,
            "duracion_s": round(self.duracion(), 2),
            "frames": self.frames,
            "perdidos": self.perdidos,
            "tamano_mb": round(self.bytes / 1e6, 1),
            "configuracion": asdict(self.config),
            **self.extra,
        }


class Grabadora:
    """Abre la OAK en un hilo propio, lee los streams continuamente y escribe el clip activo.

    Uso:
        grabadora = Grabadora(config, sesion)
        grabadora.iniciar()
        grabadora.lista.wait()          # cámara abierta (o error en grabadora.error)
        grabadora.empezar_clip("tiro raso")
        ...
        resumen = grabadora.terminar_clip()
        grabadora.detener()
    """

    def __init__(self, config, sesion):
        config.validar()
        self.config = config
        self.sesion = sesion
        self.streams = ["izq", "der"] + (["color"] if config.color else [])

        self.lista = threading.Event()
        self.error = None
        self.info_camara = None
        self.usb_super = False

        # Lo que lee la interfaz: último JPEG de cada cámara mono y estadísticas del último segundo
        self.ultimo_jpeg = {"izq": None, "der": None}
        self.ultimo_color = None  # imagen BGR pequeña de la cámara a color (si está activa)
        self.fps = {s: 0.0 for s in self.streams}
        self.perdidos_por_segundo = {s: 0 for s in self.streams}

        self._clip = None
        self._calibracion = None
        self._comandos = []
        self._lock = threading.Lock()
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)

    # --- control desde fuera del hilo ---

    def iniciar(self):
        self._hilo.start()

    def detener(self):
        if self.grabando:
            self.terminar_clip()
        self._parar.set()
        if self._hilo.is_alive():
            self._hilo.join()

    @property
    def activa(self):
        return self._hilo.is_alive() and self.error is None

    @property
    def grabando(self):
        return self._clip is not None

    @property
    def clip(self):
        return self._clip

    def empezar_clip(self, etiqueta="prueba", notas="", nombre=None, extra=None):
        """Empieza a escribir un clip en sesion/nombre (por defecto, el siguiente clip_NNN).

        extra: campos adicionales que se añaden al resumen del clip en metadatos.json.
        """
        if not self.lista.is_set() or self.error is not None:
            raise RuntimeError("la cámara no está lista")
        with self._lock:
            if self._clip is None:
                self.sesion.preparar(self.info_camara, self._calibracion)
                nombre = nombre or self.sesion.siguiente_clip()
                self._clip = Clip(self.sesion.carpeta / nombre, nombre, self.streams, self.config,
                                  etiqueta, notas, extra)
            return self._clip

    def terminar_clip(self):
        with self._lock:
            clip, self._clip = self._clip, None
        if clip is None:
            return None
        resumen = clip.cerrar()
        self.sesion.registrar(resumen)
        return resumen

    def ajustar_exposicion(self, exposicion_us, iso):
        """Exposición manual en vivo; se aplica también a los siguientes reinicios."""
        self.config.exposicion_us, self.config.iso = exposicion_us, iso
        control = dai.CameraControl()
        control.setManualExposure(exposicion_us, iso)
        with self._lock:
            self._comandos.append(control)

    def exposicion_automatica(self):
        self.config.exposicion_us = None
        control = dai.CameraControl()
        control.setAutoExposureEnable()
        with self._lock:
            self._comandos.append(control)

    # --- hilo de captura ---

    def _bucle(self):
        try:
            with dai.Device(crear_pipeline(self.config)) as dispositivo:
                self._calibracion = dispositivo.readCalibration()
                velocidad_usb = dispositivo.getUsbSpeed()
                self.usb_super = velocidad_usb in (dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS)
                self.info_camara = {
                    "producto": dispositivo.getProductName(),
                    "mxid": dispositivo.getMxId(),
                    "usb": velocidad_usb.name,
                }
                colas = {s: dispositivo.getOutputQueue(s, maxSize=120, blocking=False) for s in self.streams}
                cola_preview = (dispositivo.getOutputQueue("color_preview", maxSize=2, blocking=False)
                                if self.config.color else None)
                cola_control = dispositivo.getInputQueue("control")
                self.lista.set()

                ultima_secuencia = {}
                frames = {s: 0 for s in self.streams}
                perdidos = {s: 0 for s in self.streams}
                inicio_ventana = time.monotonic()

                while not self._parar.is_set():
                    with self._lock:
                        comandos, self._comandos = self._comandos, []
                    for comando in comandos:
                        cola_control.send(comando)

                    hay_datos = False
                    for stream, cola in colas.items():
                        for frame in cola.tryGetAll():
                            hay_datos = True
                            datos = frame.getData()
                            secuencia = frame.getSequenceNum()
                            anterior = ultima_secuencia.get(stream)
                            if anterior is not None and secuencia > anterior + 1:
                                perdidos[stream] += secuencia - anterior - 1
                            ultima_secuencia[stream] = secuencia
                            frames[stream] += 1

                            with self._lock:
                                if self._clip is not None:
                                    self._clip.escribir(stream, frame, datos)
                            if stream in self.ultimo_jpeg:
                                self.ultimo_jpeg[stream] = datos

                    if cola_preview is not None:
                        preview = cola_preview.tryGet()
                        if preview is not None:
                            self.ultimo_color = preview.getCvFrame()

                    ahora = time.monotonic()
                    if ahora - inicio_ventana >= 1.0:
                        for s in self.streams:
                            self.fps[s] = frames[s] / (ahora - inicio_ventana)
                            self.perdidos_por_segundo[s] = perdidos[s]
                            frames[s] = perdidos[s] = 0
                        inicio_ventana = ahora

                    if not hay_datos:
                        time.sleep(0.001)
        except Exception as e:
            self.error = str(e)
        finally:
            self.lista.set()


class Secuencia:
    """Graba varias partes de duración fija separadas por pausas, dentro de una misma carpeta clip_NNN.

    Salida: sesion/clip_NNN/parte_1, parte_2... Cada parte queda en metadatos.json
    como un clip más, con un campo "secuencia" que la identifica.

    No usa hilos: hay que llamar a actualizar() periódicamente (cada pocos ms).
    """

    def __init__(self, grabadora, partes, duracion_s, pausa_s, etiqueta="prueba", notas=""):
        if partes < 1 or duracion_s <= 0 or pausa_s < 0:
            raise ValueError("la secuencia necesita al menos 1 parte, duración > 0 y pausa >= 0")
        self.grabadora = grabadora
        self.partes = partes
        self.duracion_s = duracion_s
        self.pausa_s = pausa_s
        self.etiqueta = etiqueta
        self.notas = notas
        self.nombre = grabadora.sesion.siguiente_clip()
        self.parte = 0
        self.estado = None  # "grabando", "pausa", "terminada" o "cancelada"
        self._inicio_pausa = None
        self._empezar_parte()

    @property
    def activa(self):
        return self.estado in ("grabando", "pausa")

    def restante_s(self):
        """Segundos que faltan para acabar la parte actual o la pausa."""
        if self.estado == "grabando":
            return max(0.0, self.duracion_s - self.grabadora.clip.duracion())
        if self.estado == "pausa":
            return max(0.0, self.pausa_s - (time.monotonic() - self._inicio_pausa))
        return 0.0

    def actualizar(self):
        """Avanza la secuencia. Devuelve el resumen de la parte que acaba de terminar, si la hay."""
        if self.estado == "grabando" and self.grabadora.clip.duracion() >= self.duracion_s:
            resumen = self.grabadora.terminar_clip()
            if self.parte == self.partes:
                self.estado = "terminada"
            else:
                self.estado = "pausa"
                self._inicio_pausa = time.monotonic()
            return resumen
        if self.estado == "pausa" and self.restante_s() <= 0:
            self._empezar_parte()
        return None

    def cancelar(self):
        """Detiene la secuencia; la parte en curso se guarda tal como esté."""
        resumen = self.grabadora.terminar_clip() if self.estado == "grabando" else None
        if self.activa:
            self.estado = "cancelada"
        return resumen

    def _empezar_parte(self):
        self.parte += 1
        extra = {"secuencia": {
            "clip": self.nombre,
            "parte": self.parte,
            "partes": self.partes,
            "duracion_s": self.duracion_s,
            "pausa_s": self.pausa_s,
        }}
        self.grabadora.empezar_clip(self.etiqueta, self.notas, f"{self.nombre}/parte_{self.parte}", extra)
        self.estado = "grabando"
