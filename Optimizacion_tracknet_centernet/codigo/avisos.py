"""Avisos sonoros de las secuencias de grabación, para quien tira sin mirar la pantalla.

    - pitidos cortos en los últimos 3 s de cada cuenta atrás (espera inicial y pausas),
    - pitido largo y agudo al empezar a grabar cada parte,
    - dos pitidos graves al pasar a pausa y tres al terminar la secuencia.

QApplication.beep() y el carácter BEL no suenan en la mayoría de escritorios Linux, así
que los tonos se generan una vez como WAV y se reproducen con paplay/pw-play/aplay en un
proceso aparte (no bloquea la captura). Si no hay ninguno, se intenta el BEL del terminal.
"""

import atexit
import shutil
import subprocess
import tempfile
import wave
from math import ceil
from pathlib import Path

import numpy as np

MUESTREO = 44100
REPRODUCTORES = ("paplay", "pw-play", "aplay")

# nombre -> lista de (frecuencia Hz, duración s); frecuencia 0 = silencio
TONOS = {
    "tic": [(880, 0.08)],
    "grabar": [(1320, 0.45)],
    "pausa": [(520, 0.15), (0, 0.08), (520, 0.15)],
    "fin": [(520, 0.15), (0, 0.08), (520, 0.15), (0, 0.08), (390, 0.35)],
}


def _escribir_wav(ruta, notas):
    trozos = []
    for frecuencia, duracion in notas:
        t = np.arange(int(MUESTREO * duracion)) / MUESTREO
        onda = np.sin(2 * np.pi * frecuencia * t) if frecuencia else np.zeros_like(t)
        rampa = np.minimum(1, np.minimum(t, duracion - t) / 0.01)  # 10 ms de subida/bajada: sin chasquidos
        trozos.append(0.6 * onda * rampa)
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(MUESTREO)
        w.writeframes((np.concatenate(trozos) * 32767).astype(np.int16).tobytes())


class Pitidos:
    def __init__(self):
        self.reproductor = next((r for r in REPRODUCTORES if shutil.which(r)), None)
        self.carpeta = Path(tempfile.mkdtemp(prefix="pitidos_"))
        atexit.register(shutil.rmtree, self.carpeta, ignore_errors=True)
        for nombre, notas in TONOS.items():
            _escribir_wav(self.carpeta / f"{nombre}.wav", notas)

    def sonar(self, nombre):
        if self.reproductor is None:
            print("\a", end="", flush=True)
            return
        subprocess.Popen([self.reproductor, str(self.carpeta / f"{nombre}.wav")],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class AvisosSecuencia:
    """Llamar a actualizar(secuencia) en cada vuelta del bucle; suena según los cambios de estado."""

    SEGUNDOS_TIC = 3

    def __init__(self, pitidos=None):
        self.pitidos = pitidos or Pitidos()
        self._estado = None
        self._segundo = None

    def actualizar(self, secuencia):
        estado = secuencia.estado
        if estado != self._estado:
            if estado == "grabando":
                self.pitidos.sonar("grabar")
            elif estado == "pausa":
                self.pitidos.sonar("pausa")
            elif estado == "terminada":
                self.pitidos.sonar("fin")
            self._estado = estado
            self._segundo = None
        if estado in ("espera", "pausa"):
            segundo = ceil(secuencia.restante_s())
            if segundo != self._segundo and 1 <= segundo <= self.SEGUNDOS_TIC:
                self.pitidos.sonar("tic")
            self._segundo = segundo
