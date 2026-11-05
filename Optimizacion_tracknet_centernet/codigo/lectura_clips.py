"""Lectura de clips grabados con grabacion.py.

Da acceso aleatorio a los frames de izq.mjpeg / der.mjpeg (usando los tamaños
guardados en tiempos.csv) y de color.h265, con sus tiempos reales de cámara.
Lo usan ver_clip.py, exportar_mp4.py y el procesado offline.

    clip = ClipGrabado("grabaciones/pruebas_primer_dia/clip_001/parte_1")
    izq = clip.streams["izq"]
    imagen = izq.leer(100)            # frame 100 en escala de grises
    t = izq.tiempos[100] - clip.t0    # segundos desde el inicio del clip
    j = clip.pareja("der", 100)       # frame derecho simultáneo (o None)
    k = clip.pareja("color", 100)     # frame a color más cercano en el tiempo (o None)
"""

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

CAMARAS_MONO = ("izq", "der")
ALTO_COLOR = 400  # el color se reduce a la altura de las mono al cargarlo


def _orden_natural(ruta):
    """Ordena parte_2 antes que parte_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(ruta))]


def buscar_clips(ruta):
    """Carpetas de clip (las que tienen tiempos.csv) en ruta: un clip, una secuencia o una sesión entera."""
    ruta = Path(ruta)
    if (ruta / "tiempos.csv").exists():
        return [ruta]
    return sorted((p.parent for p in ruta.rglob("tiempos.csv")), key=_orden_natural)


class _Stream:
    """Datos comunes de una cámara: secuencia, tiempo, exposición y tamaño de cada frame."""

    def __init__(self, archivo, filas):
        self.archivo = Path(archivo)
        self.secuencias = np.array([int(f["secuencia"]) for f in filas])
        self.tiempos = np.array([float(f["t_dispositivo_s"]) for f in filas])
        self.exposiciones_us = np.array([int(f["exposicion_us"]) for f in filas])
        self.isos = np.array([int(f["iso"]) for f in filas])
        self.tamanos = np.array([int(f["bytes"]) for f in filas])
        self.completo = self.archivo.stat().st_size == self.tamanos.sum()
        dt = np.diff(self.tiempos) / np.maximum(np.diff(self.secuencias), 1)
        self.fps_nominal = float(1 / np.median(dt)) if len(dt) else 0.0

    def __len__(self):
        return len(self.secuencias)

    def perdidos_antes(self, i):
        """Frames que la cámara capturó pero no llegaron a guardarse justo antes del frame i."""
        return int(self.secuencias[i] - self.secuencias[i - 1] - 1) if i > 0 else 0

    def indice_en(self, t):
        """Último frame con tiempo <= t (0 si t es anterior al primero)."""
        return max(0, int(np.searchsorted(self.tiempos, t, side="right")) - 1)

    def cerrar(self):
        pass


class StreamGrabado(_Stream):
    """Cámara mono (MJPEG): cada frame se lee directamente de su posición en el archivo."""

    def __init__(self, archivo, filas):
        super().__init__(archivo, filas)
        self.offsets = np.concatenate([[0], np.cumsum(self.tamanos)[:-1]])
        self._f = open(self.archivo, "rb")

    def leer(self, i, color=False):
        self._f.seek(int(self.offsets[i]))
        datos = np.frombuffer(self._f.read(int(self.tamanos[i])), np.uint8)
        return cv2.imdecode(datos, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)

    def leer_bgr(self, i):
        return self.leer(i, color=True)

    def cerrar(self):
        self._f.close()


class StreamColor(_Stream):
    """Cámara a color (H.265). Al leer el primer frame se decodifica entero una vez y se guarda en
    memoria como JPEG reducido a ALTO_COLOR píxeles de alto (~20 MB por cada 20 s)."""

    def __init__(self, archivo, filas):
        super().__init__(archivo, filas)
        self._jpegs = None
        self.aviso = None

    def _cargar(self):
        captura = cv2.VideoCapture(str(self.archivo))
        jpegs = []
        while True:
            ok, imagen = captura.read()
            if not ok:
                break
            escala = ALTO_COLOR / imagen.shape[0]
            imagen = cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
            jpegs.append(cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, 90])[1])
        captura.release()
        if len(jpegs) != len(self):
            self.aviso = f"color: {len(jpegs)} frames decodificados y {len(self)} en tiempos.csv"
            n = min(len(jpegs), len(self))
            jpegs = jpegs[:n]
            for nombre in ("secuencias", "tiempos", "exposiciones_us", "isos", "tamanos"):
                setattr(self, nombre, getattr(self, nombre)[:n])
        self._jpegs = jpegs

    def __len__(self):
        return len(self._jpegs) if self._jpegs is not None else len(self.secuencias)

    def leer_bgr(self, i):
        if self._jpegs is None:
            self._cargar()
        return cv2.imdecode(self._jpegs[i], cv2.IMREAD_COLOR)

    def leer(self, i, color=True):
        imagen = self.leer_bgr(i)
        return imagen if color else cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


class ClipGrabado:
    def __init__(self, carpeta):
        self.carpeta = Path(carpeta)
        with open(self.carpeta / "tiempos.csv", newline="") as f:
            filas = list(csv.DictReader(f))

        self.streams = {}
        for camara, extension, clase in (("izq", "mjpeg", StreamGrabado), ("der", "mjpeg", StreamGrabado),
                                         ("color", "h265", StreamColor)):
            archivo = self.carpeta / f"{camara}.{extension}"
            filas_camara = [f for f in filas if f["camara"] == camara]
            if archivo.exists() and filas_camara:
                self.streams[camara] = clase(archivo, filas_camara)
        if not any(c in self.streams for c in CAMARAS_MONO):
            raise ValueError(f"{self.carpeta} no tiene vídeo de las cámaras mono")

        self.t0 = self.principal.tiempos[0]
        self.fps_nominal = self.principal.fps_nominal
        self.metadatos = self._buscar_metadatos()

    @property
    def principal(self):
        return self.streams.get("izq") or self.streams["der"]

    @property
    def tiene_color(self):
        return "color" in self.streams

    @property
    def nombre(self):
        """p. ej. "pruebas_primer_dia/clip_001/parte_1"."""
        partes = self.carpeta.parts
        for i in range(len(partes) - 1, -1, -1):
            if (Path(*partes[:i + 1]) / "metadatos.json").exists():
                return "/".join(partes[i:])
        return self.carpeta.name

    def duracion(self):
        return float(self.principal.tiempos[-1] - self.t0)

    def pareja(self, camara, i, referencia="izq"):
        """Índice del frame de `camara` más cercano en el tiempo al frame i de `referencia`.

        None si ese frame no se guardó (más lejos que medio periodo de `camara`).
        """
        if camara not in self.streams or referencia not in self.streams:
            return None
        objetivo = self.streams[camara]
        t = self.streams[referencia].tiempos[i]
        j = int(np.searchsorted(objetivo.tiempos, t))
        candidatos = [k for k in (j - 1, j) if 0 <= k < len(objetivo)]
        if not candidatos:
            return None
        mejor = min(candidatos, key=lambda k: abs(objetivo.tiempos[k] - t))
        tolerancia = 0.5 / objetivo.fps_nominal if objetivo.fps_nominal else 0.004
        return mejor if abs(objetivo.tiempos[mejor] - t) <= tolerancia else None

    def _buscar_metadatos(self):
        """Entrada de este clip en el metadatos.json de su sesión, si existe."""
        for sesion in self.carpeta.parents:
            archivo = sesion / "metadatos.json"
            if archivo.exists():
                relativa = self.carpeta.relative_to(sesion).as_posix()
                for clip in json.loads(archivo.read_text()).get("clips", []):
                    if clip["carpeta"] == relativa:
                        return clip
                return None
        return None

    def cerrar(self):
        for stream in self.streams.values():
            stream.cerrar()
