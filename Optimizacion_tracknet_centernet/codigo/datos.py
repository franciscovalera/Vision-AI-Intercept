"""Dataset de PelotaNet: pilas de F frames consecutivos con sus F etiquetas (entrada y salida MIMO).

1. MUESTRAS
   Cada cámara de cada clip (izq y der) es una fuente independiente. Una muestra es una pila de
   F frames consecutivos según el número de secuencia de la cámara: si entre ellos falta alguno
   (frame perdido al grabar) la pila se descarta, porque la red aprende del movimiento entre
   frames a intervalos fijos de 1/120 s. Las etiquetas salen de limpiar_etiquetas.py
   (datos/etiquetas/<clip>/etiquetas_<cámara>.csv), una por frame.

2. OBJETIVOS (a 1/4 de resolución, como las salidas de la red)
   Para cada frame f de la pila con pelota (visibilidad 1 o 2) y centro (x, y) en píxeles:
       celda (j, i) = (floor(x / 4), floor(y / 4))
       calor[f]          gaussiana de pico 1 en la celda, sigma según el diámetro (estilo CenterNet)
       desplazamiento    (x / 4 - j, y / 4 - i) en [0, 1): dónde cae el centro dentro de la celda
       diámetro          en píxeles de la imagen de entrada
   Decodificar es lo inverso: x = 4 * (j + dx), y = 4 * (i + dy).
   Coordenadas continuas: el píxel k ocupa [k, k + 1), su centro está en k + 0,5 (como las cajas
   de YOLO de las que salen las etiquetas).

   Máscaras de la pérdida:
       peso_calor[f]  0 si el frame es dudoso (no se sabe si la etiqueta es buena: ni positivo ni
                      negativo); 1 en el resto, también en los frames sin pelota (negativos).
       peso_diametro  0 con visibilidad 2 (pelota cortada, tapada o en la mano: el diámetro de la
                      caja no es el de la pelota).

3. AUMENTOS (los mismos en los F frames, para no inventar movimiento)
   Volteo horizontal, escala y traslación pequeñas (se ajustan las etiquetas; la pelota que se
   sale de la imagen pasa a no visible), brillo, contraste, gamma y ruido gaussiano. El ruido
   es lo más caro de generar en CPU (~7 ms por pila), así que aquí solo se sortea su desviación
   (objetivo "ruido") y lo suma entrenar.py ya en la GPU.

4. CACHÉ
   Decodificar MJPEG cuesta ~2 ms por frame y cada muestra lee F frames, así que en el primer uso
   se decodifica cada cámara de cada clip una sola vez a datos/cache/<clip>/<cámara>.u8 (memmap
   uint8, N x 400 x 640). Ocupa unos 256 KB por frame; el sistema operativo guarda en memoria lo
   que quepa.

    from datos import buscar_fuentes, PilasPelota
    fuentes = buscar_fuentes(["sesion_1", "pruebas_primer_dia"])
    ds = PilasPelota(fuentes, frames=3, aumentar=True)
    x, objetivos = ds[0]      # x: F x 400 x 640 uint8; objetivos: dict de tensores
"""

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lectura_clips import CAMARAS_MONO, ClipGrabado, buscar_clips

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_GRABACIONES = RAIZ / "grabaciones"
CARPETA_ETIQUETAS = RAIZ / "datos" / "etiquetas"
CARPETA_CACHE = RAIZ / "datos" / "cache"
REDUCCION = 4  # las salidas de la red están a 1/4 de la entrada


class Fuente:
    """Una cámara de un clip: frames (memmap), números de secuencia y etiquetas por frame."""

    def __init__(self, clip, camara):
        self.clip, self.camara = clip, camara
        self.cache = CARPETA_CACHE / clip / f"{camara}.u8"
        with open(CARPETA_ETIQUETAS / clip / f"etiquetas_{camara}.csv", newline="") as f:
            filas = list(csv.DictReader(f))
        numero = lambda v: float(v) if v else np.nan
        self.x = np.array([numero(fila["x"]) for fila in filas], np.float32)
        self.y = np.array([numero(fila["y"]) for fila in filas], np.float32)
        self.diametro = np.array([numero(fila["diametro"]) for fila in filas], np.float32)
        self.visibilidad = np.array([int(fila["visibilidad"]) for fila in filas], np.int64)
        self.dudoso = np.array([int(fila["dudoso"]) for fila in filas], bool)
        clip_grabado = ClipGrabado(CARPETA_GRABACIONES / clip)
        stream = clip_grabado.streams[camara]
        self.secuencias = stream.secuencias.copy()
        self.alto, self.ancho = stream.leer(0).shape
        clip_grabado.cerrar()
        if len(filas) != len(self.secuencias):
            raise ValueError(f"{clip} {camara}: {len(filas)} etiquetas para {len(self.secuencias)} frames")
        self._frames = None

    def __len__(self):
        return len(self.secuencias)

    @property
    def frames(self):
        # Se abre en cada proceso la primera vez que se usa (los memmap no se heredan bien entre procesos)
        if self._frames is None:
            self._frames = np.memmap(self.cache, np.uint8, "r", shape=(len(self), self.alto, self.ancho))
        return self._frames

    def __getstate__(self):
        estado = self.__dict__.copy()
        estado["_frames"] = None
        return estado

    def inicios_validos(self, frames):
        """Índices i tales que los frames i .. i+F-1 son consecutivos (sin perdidos entre medias)."""
        if len(self) < frames:
            return np.zeros(0, np.int64)
        salto = self.secuencias[frames - 1:] - self.secuencias[:len(self) - frames + 1]
        return np.flatnonzero(salto == frames - 1)


def _decodificar(clip, camara):
    """Escribe la caché de una cámara de un clip (en un proceso aparte)."""
    destino = CARPETA_CACHE / clip / f"{camara}.u8"
    clip_grabado = ClipGrabado(CARPETA_GRABACIONES / clip)
    stream = clip_grabado.streams[camara]
    alto, ancho = stream.leer(0).shape
    temporal = destino.with_suffix(".tmp")
    destino.parent.mkdir(parents=True, exist_ok=True)
    frames = np.memmap(temporal, np.uint8, "w+", shape=(len(stream), alto, ancho))
    for i in range(len(stream)):
        frames[i] = stream.leer(i)
    frames.flush()
    del frames
    clip_grabado.cerrar()
    temporal.rename(destino)  # solo existe completa
    return clip, camara


def preparar_cache(fuentes, procesos=12):
    pendientes = [(f.clip, f.camara) for f in fuentes if not f.cache.exists()]
    if not pendientes:
        return
    print(f"Decodificando {len(pendientes)} vídeos a {CARPETA_CACHE} ...", flush=True)
    with ProcessPoolExecutor(procesos) as ejecutor:
        for clip, camara in ejecutor.map(_decodificar, *zip(*pendientes)):
            print(f"  {clip} {camara}", flush=True)


def buscar_fuentes(sesiones):
    """Fuentes (clip, cámara) con etiquetas de las sesiones dadas (nombres de carpeta en grabaciones/)."""
    fuentes = []
    for sesion in sesiones:
        for carpeta in buscar_clips(CARPETA_GRABACIONES / sesion):
            clip = carpeta.relative_to(CARPETA_GRABACIONES).as_posix()
            for camara in CAMARAS_MONO:
                if (CARPETA_ETIQUETAS / clip / f"etiquetas_{camara}.csv").exists():
                    fuentes.append(Fuente(clip, camara))
    return fuentes


def gaussiana(calor, cx, cy, sigma):
    """Pinta (máximo) una gaussiana de pico 1 centrada en la celda entera (cx, cy)."""
    alto, ancho = calor.shape
    radio = int(np.ceil(3 * sigma))
    y0, y1 = max(0, cy - radio), min(alto, cy + radio + 1)
    x0, x1 = max(0, cx - radio), min(ancho, cx + radio + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    np.maximum(calor[y0:y1, x0:x1], g, out=calor[y0:y1, x0:x1])


def sigma_calor(diametro):
    """Sigma de la gaussiana en celdas de 1/4: un sexto del diámetro en celdas, como mínimo 0,8."""
    return max(0.8, diametro / REDUCCION / 6)


def objetivos(etiquetas, alto, ancho):
    """Mapas objetivo de una pila. etiquetas: lista de F tuplas (x, y, diametro, visibilidad, dudoso)."""
    F = len(etiquetas)
    h, w = alto // REDUCCION, ancho // REDUCCION
    calor = np.zeros((F, h, w), np.float32)
    desplazamiento = np.zeros((2 * F, h, w), np.float32)
    diametro = np.zeros((F, h, w), np.float32)
    positivo = np.zeros((F, h, w), np.float32)  # 1 en la celda del centro
    peso_calor = np.ones(F, np.float32)
    peso_diametro = np.zeros(F, np.float32)
    for f, (x, y, d, visibilidad, dudoso) in enumerate(etiquetas):
        if dudoso:
            peso_calor[f] = 0
            continue
        if visibilidad == 0:
            continue
        gx, gy = x / REDUCCION, y / REDUCCION
        j, i = int(np.floor(gx)), int(np.floor(gy))
        if not (0 <= j < w and 0 <= i < h):
            continue
        gaussiana(calor[f], j, i, sigma_calor(d))
        desplazamiento[2 * f, i, j], desplazamiento[2 * f + 1, i, j] = gx - j, gy - i
        diametro[f, i, j] = d
        positivo[f, i, j] = 1
        peso_diametro[f] = 1.0 if visibilidad == 1 else 0.0
    return {"calor": calor, "desplazamiento": desplazamiento, "diametro": diametro, "positivo": positivo,
            "peso_calor": peso_calor, "peso_diametro": peso_diametro}


class PilasPelota(Dataset):
    """Pilas de `frames` frames consecutivos de varias fuentes, con aumentos opcionales.

    paso: distancia entre inicios de pila (1 = todas las pilas solapadas; F = sin solapar, como
    en la OAK, donde cada frame se procesa una vez).
    """

    def __init__(self, fuentes, frames=3, aumentar=False, paso=1, semilla=0):
        self.fuentes, self.F, self.aumentar = fuentes, frames, aumentar
        indices = []
        for k, fuente in enumerate(fuentes):
            inicios = fuente.inicios_validos(frames)
            if paso > 1:
                inicios = sin_solapar(inicios, paso)
            indices.append(np.stack([np.full(len(inicios), k), inicios], 1))
        self.indices = np.concatenate(indices) if indices else np.zeros((0, 2), np.int64)
        self.rng = np.random.default_rng(semilla)

    def __len__(self):
        return len(self.indices)

    def con_pelota(self):
        """Por pila: True si algún frame tiene pelota visible y no dudosa (para sobremuestrear)."""
        resultado = np.zeros(len(self), bool)
        for n, (k, i) in enumerate(self.indices):
            fuente = self.fuentes[k]
            sl = slice(i, i + self.F)
            resultado[n] = np.any((fuente.visibilidad[sl] > 0) & ~fuente.dudoso[sl])
        return resultado

    def etiquetas(self, n):
        k, i = self.indices[n]
        fuente = self.fuentes[k]
        return [(float(fuente.x[i + f]), float(fuente.y[i + f]), float(fuente.diametro[i + f]),
                 int(fuente.visibilidad[i + f]), bool(fuente.dudoso[i + f])) for f in range(self.F)]

    def __getitem__(self, n):
        k, i = self.indices[n]
        fuente = self.fuentes[k]
        pila = np.array(fuente.frames[i:i + self.F])  # copia: F x alto x ancho uint8
        etiquetas = self.etiquetas(n)
        ruido = 0.0
        if self.aumentar:
            pila, etiquetas, ruido = aumentar(pila, etiquetas, np.random.default_rng())
        mapas = objetivos(etiquetas, fuente.alto, fuente.ancho)
        mapas["ruido"] = np.float32(ruido)
        # uint8: 4 veces menos datos entre procesos y hacia la GPU; se pasa a float allí
        return torch.from_numpy(pila), {nombre: torch.as_tensor(valor) for nombre, valor in mapas.items()}


def sin_solapar(inicios, paso):
    """De los inicios válidos, los que dan pilas sin solapar: se avanza `paso` desde cada inicio elegido."""
    elegidos, siguiente = [], -1
    for i in inicios:
        if i >= siguiente:
            elegidos.append(i)
            siguiente = i + paso
    return np.array(elegidos, np.int64)


def aumentar(pila, etiquetas, rng):
    """Mismos aumentos en todos los frames de la pila.

    Devuelve la pila (uint8), las etiquetas y la desviación del ruido gaussiano que hay que sumar
    (0 = sin ruido).
    """
    F, alto, ancho = pila.shape
    etiquetas = [list(e) for e in etiquetas]

    # Geometría: volteo, escala y traslación alrededor del centro de la imagen. En índices de
    # píxel (centro del píxel k en k), así que u = índice + 0,5 en coordenadas continuas.
    volteo = rng.random() < 0.5
    escala = rng.uniform(0.85, 1.15)
    tx, ty = rng.uniform(-0.06, 0.06) * ancho, rng.uniform(-0.06, 0.06) * alto
    cx, cy = (ancho - 1) / 2, (alto - 1) / 2
    sx = -escala if volteo else escala
    matriz = np.array([[sx, 0, cx - sx * cx + tx], [0, escala, cy - escala * cy + ty]], np.float32)
    pila = np.stack([cv2.warpAffine(frame, matriz, (ancho, alto), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE) for frame in pila])
    for e in etiquetas:
        if e[3] == 0:
            continue
        u = matriz @ np.array([e[0] - 0.5, e[1] - 0.5, 1.0]) + 0.5
        e[0], e[1], e[2] = float(u[0]), float(u[1]), e[2] * escala
        if not (0 <= e[0] < ancho and 0 <= e[1] < alto):
            e[3] = 0  # el centro se ha salido de la imagen

    # Fotometría: contraste, brillo y gamma con una tabla de 256 valores (mucho más barato que
    # operar píxel a píxel en coma flotante), y después ruido
    contraste, brillo, gamma = rng.uniform(0.7, 1.3), rng.uniform(-30, 30), rng.uniform(0.7, 1.4)
    tabla = np.clip((np.arange(256) - 128) * contraste + 128 + brillo, 0, 255)
    tabla = np.clip(255 * (tabla / 255) ** gamma + 0.5, 0, 255).astype(np.uint8)
    pila = np.stack([cv2.LUT(frame, tabla) for frame in pila])
    ruido = rng.uniform(1, 6) if rng.random() < 0.5 else 0.0
    return pila, [tuple(e) for e in etiquetas], ruido
