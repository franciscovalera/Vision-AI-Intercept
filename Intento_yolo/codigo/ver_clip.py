"""Reproductor de clips grabados, a velocidad real según los tiempos de la cámara.

Los .mjpeg no guardan FPS ni tiempos, así que los reproductores normales los
muestran acelerados. Este usa tiempos.csv: respeta los huecos de los frames
perdidos, muestra izquierda y derecha emparejadas por tiempo y permite cámara
lenta y avance frame a frame.

    python codigo/ver_clip.py grabaciones/pruebas_primer_dia                  # toda la sesión
    python codigo/ver_clip.py grabaciones/pruebas_primer_dia/clip_001         # una secuencia
    python codigo/ver_clip.py grabaciones/pruebas_primer_dia/clip_001/parte_3 --velocidad 0.25
    python codigo/ver_clip.py grabaciones/pruebas_primer_dia --pantalla-completa
    python codigo/ver_clip.py grabaciones/20261023_133016/clip_002/parte_5 \
        --modelo modelos/entrenamientos/pelota_v1/weights/best.pt      # dibuja las detecciones

Con --modelo se pasa el detector (.pt de Ultralytics, en el host) por cada
frame mostrado de izquierda y derecha y se dibujan las cajas con su confianza.
Sin GPU la reproducción a ×1 se salta frames; usar cámara lenta o frame a frame.

Teclas:
    Espacio        pausa / seguir
    ← →  (a d)     frame anterior / siguiente (pausa la reproducción)
    1 2 3 4        velocidad ×1, ×0.5, ×0.25, ×0.1
    r              volver al inicio del clip
    n p            clip siguiente / anterior
    c              mostrar / ocultar la cámara a color (si el clip la tiene)
    f  doble clic  pantalla completa / ventana
    Esc            salir de pantalla completa (en ventana, salir)
    q              salir
"""

import argparse
import sys
import time

import cv2
import numpy as np

from lectura_clips import ClipGrabado, buscar_clips

VENTANA = "Clip"
ANCHO_MAXIMO = 1600
VELOCIDADES = {ord("1"): 1.0, ord("2"): 0.5, ord("3"): 0.25, ord("4"): 0.1}
# Códigos de flecha según el backend de ventanas de OpenCV (GTK/X11 y Qt)
IZQUIERDA = {65361, 2424832, 16777234, ord("a")}
DERECHA = {65363, 2555904, 16777236, ord("d")}
ROJO, BLANCO, GRIS, VERDE = (60, 60, 230), (255, 255, 255), (170, 170, 170), (80, 220, 80)


def en_pantalla_completa():
    return cv2.getWindowProperty(VENTANA, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN


def poner_pantalla_completa(activar):
    cv2.setWindowProperty(VENTANA, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if activar else cv2.WINDOW_NORMAL)


def cargar_color(clip, mostrar):
    """La primera lectura del color decodifica todo el H.265; se avisa porque tarda unos segundos."""
    if mostrar and clip.tiene_color:
        print(f"{clip.nombre}: cargando color...", flush=True)
        clip.streams["color"].leer_bgr(0)
        if clip.streams["color"].aviso:
            print(f"AVISO: {clip.streams['color'].aviso}")


def texto(imagen, cadena, posicion, color=BLANCO, escala=0.5):
    cv2.putText(imagen, cadena, posicion, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(imagen, cadena, posicion, cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)


class Detector:
    """Detector YOLO en el host; recuerda el último resultado de cada cámara (en pausa no se repite)."""

    def __init__(self, ruta, conf, imgsz, dispositivo):
        from ultralytics import YOLO

        self.modelo = YOLO(str(ruta))
        self.conf, self.imgsz, self.dispositivo = conf, imgsz, dispositivo
        self._cache = {}
        self.ms = 0.0

    def dibujar(self, imagen, clave):
        """Dibuja sobre imagen las cajas del frame clave = (clip, cámara, índice)."""
        camara = clave[:2]
        if self._cache.get(camara, (None,))[0] != clave:
            inicio = time.perf_counter()
            resultado = self.modelo.predict(imagen, conf=self.conf, imgsz=self.imgsz, device=self.dispositivo,
                                            verbose=False)[0]
            self.ms = (time.perf_counter() - inicio) * 1000
            self._cache[camara] = (clave, list(zip(resultado.boxes.xyxy.tolist(), resultado.boxes.conf.tolist())))
        for (x1, y1, x2, y2), confianza in self._cache[camara][1]:
            cv2.rectangle(imagen, (round(x1), round(y1)), (round(x2), round(y2)), VERDE, 1)
            texto(imagen, f"{confianza:.2f}", (round(x1), max(12, round(y1) - 4)), VERDE, 0.4)
        return len(self._cache[camara][1])


class Reproductor:
    def __init__(self, clip, posicion_clip, velocidad, solo_izq, mostrar_color, detector=None):
        self.clip = clip
        self.posicion_clip = posicion_clip  # (índice, total) dentro de la lista de clips
        self.velocidad = velocidad
        self.solo_izq = solo_izq
        self.mostrar_color = mostrar_color
        self.detector = detector
        self.stream = clip.principal
        self.i = 0
        self.t = 0.0  # segundos de vídeo desde el inicio del clip
        self.pausado = False
        self._moviendo_barra = False
        self._alternar_pantalla = False  # pedido por doble clic; se aplica en el bucle

    def _raton(self, evento, x, y, flags, parametro):
        if evento == cv2.EVENT_LBUTTONDBLCLK:
            self._alternar_pantalla = True

    def ir_a(self, i):
        self.i = int(np.clip(i, 0, len(self.stream) - 1))
        self.t = float(self.stream.tiempos[self.i] - self.clip.t0)

    def _barra_movida(self, posicion):
        if not self._moviendo_barra:
            self.ir_a(posicion)

    def componer(self):
        clip, i = self.clip, self.i
        izq = self.stream.leer_bgr(i)
        alto = izq.shape[0]
        detecciones = []
        if self.detector is not None:
            detecciones.append(self.detector.dibujar(izq, (clip.nombre, "principal", i)))
        paneles = [izq]
        if not self.solo_izq and "der" in clip.streams and self.stream is clip.streams.get("izq"):
            j = clip.pareja("der", i)
            if j is not None:
                der = clip.streams["der"].leer_bgr(j)
                if self.detector is not None:
                    detecciones.append(self.detector.dibujar(der, (clip.nombre, "der", j)))
            else:
                der = np.zeros_like(izq)
                texto(der, "frame derecho no guardado", (20, alto // 2), ROJO, 0.7)
            paneles.append(der)
        if self.mostrar_color and clip.tiene_color:
            k = clip.pareja("color", i, referencia="izq" if "izq" in clip.streams else "der")
            if k is not None:
                color = clip.streams["color"].leer_bgr(k)
                color = cv2.resize(color, (round(color.shape[1] * alto / color.shape[0]), alto))
            else:
                color = np.zeros((alto, round(alto * 16 / 9), 3), np.uint8)
                texto(color, "frame de color no guardado", (20, alto // 2), ROJO, 0.7)
            paneles.append(color)
        imagen = np.hstack(paneles)

        escala = min(1.0, ANCHO_MAXIMO / imagen.shape[1])
        if escala < 1:
            imagen = cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

        indice, total = self.posicion_clip
        estado = "PAUSA" if self.pausado else f"x{self.velocidad:g}"
        texto(imagen, f"[{indice + 1}/{total}] {clip.nombre}", (10, 22))
        texto(imagen, f"t = {self.t:6.3f} / {clip.duracion():.3f} s   frame {i + 1}/{len(self.stream)}   "
                      f"{clip.fps_nominal:.0f} FPS nominales   {estado}", (10, 44))
        texto(imagen, f"exposicion {self.stream.exposiciones_us[i]} us  ISO {self.stream.isos[i]}", (10, 66), GRIS)
        fila = 88
        if self.detector is not None:
            texto(imagen, f"detecciones {' / '.join(map(str, detecciones))}  (conf >= {self.detector.conf:g}, "
                          f"{self.detector.ms:.0f} ms)", (10, fila), VERDE)
            fila += 22
        perdidos = self.stream.perdidos_antes(i)
        if perdidos:
            texto(imagen, f"+{perdidos} frame(s) perdido(s) antes de este", (10, fila), ROJO)
        return imagen

    def ejecutar(self):
        """Reproduce el clip. Devuelve 'siguiente', 'anterior' o 'salir'."""
        cv2.createTrackbar("frame", VENTANA, 0, max(1, len(self.stream) - 1), self._barra_movida)
        cv2.setMouseCallback(VENTANA, self._raton)
        ultimo_mostrado = None
        reloj = time.monotonic()

        while True:
            ahora = time.monotonic()
            if not self.pausado:
                self.t += (ahora - reloj) * self.velocidad
                self.i = self.stream.indice_en(self.clip.t0 + self.t)
                if self.i == len(self.stream) - 1:
                    self.pausado = True  # se queda en el último frame
            reloj = ahora

            if self.i != ultimo_mostrado or self.pausado:
                imagen = self.componer()
                if ultimo_mostrado is None and not en_pantalla_completa():
                    cv2.resizeWindow(VENTANA, imagen.shape[1], imagen.shape[0])
                cv2.imshow(VENTANA, imagen)
                self._moviendo_barra = True
                cv2.setTrackbarPos("frame", VENTANA, self.i)
                self._moviendo_barra = False
                ultimo_mostrado = self.i

            tecla = cv2.waitKeyEx(30 if self.pausado else 1)
            if self._alternar_pantalla:
                self._alternar_pantalla = False
                poner_pantalla_completa(not en_pantalla_completa())
            if tecla == -1:
                if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                    return "salir"
                continue
            tecla_baja = tecla & 0xFF
            if tecla_baja == 27 and en_pantalla_completa():
                poner_pantalla_completa(False)
                continue
            if tecla_baja in (ord("q"), 27):
                return "salir"
            if tecla_baja == ord("f"):
                poner_pantalla_completa(not en_pantalla_completa())
                continue
            if tecla_baja == ord("c") and self.clip.tiene_color:
                self.mostrar_color = not self.mostrar_color
                cargar_color(self.clip, self.mostrar_color)
                ultimo_mostrado = None  # la imagen cambia de ancho: reajustar la ventana
                continue
            if tecla_baja == ord("n"):
                return "siguiente"
            if tecla_baja == ord("p"):
                return "anterior"
            if tecla_baja == ord(" "):
                if self.pausado and self.i == len(self.stream) - 1:
                    self.ir_a(0)
                self.pausado = not self.pausado
            elif tecla_baja == ord("r"):
                self.ir_a(0)
            elif tecla_baja in VELOCIDADES:
                self.velocidad = VELOCIDADES[tecla_baja]
            elif tecla in IZQUIERDA or tecla_baja == ord("a"):
                self.pausado = True
                self.ir_a(self.i - 1)
            elif tecla in DERECHA or tecla_baja == ord("d"):
                self.pausado = True
                self.ir_a(self.i + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ruta", help="carpeta de un clip, de una secuencia (clip_NNN) o de una sesión")
    parser.add_argument("--velocidad", type=float, default=1.0, help="velocidad inicial (1 = tiempo real)")
    parser.add_argument("--solo-izq", action="store_true", help="mostrar solo la cámara izquierda")
    parser.add_argument("--sin-color", action="store_true",
                        help="no mostrar la cámara a color al empezar (se puede activar con c)")
    parser.add_argument("--pantalla-completa", action="store_true", help="empezar en pantalla completa")
    parser.add_argument("--modelo", help="detector .pt (Ultralytics) para dibujar las detecciones")
    parser.add_argument("--conf", type=float, default=0.25, help="confianza mínima de las detecciones")
    parser.add_argument("--imgsz", type=int, default=640, help="tamaño de inferencia del detector")
    parser.add_argument("--dispositivo", default=None, help="cpu, 0 (GPU)... por defecto, GPU si hay")
    args = parser.parse_args()

    carpetas = buscar_clips(args.ruta)
    if not carpetas:
        sys.exit(f"No hay clips (carpetas con tiempos.csv) en {args.ruta}")
    detector = Detector(args.modelo, args.conf, args.imgsz, args.dispositivo) if args.modelo else None

    # Ventana redimensionable que conserva la proporción de la imagen (necesario para pantalla completa)
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    if args.pantalla_completa:
        poner_pantalla_completa(True)
    k, velocidad, mostrar_color = 0, args.velocidad, not args.sin_color
    while True:
        clip = ClipGrabado(carpetas[k])
        if not clip.principal.completo:
            print(f"AVISO: {clip.nombre}: el tamaño del vídeo no cuadra con tiempos.csv (¿grabación interrumpida?)")
        cargar_color(clip, mostrar_color)
        reproductor = Reproductor(clip, (k, len(carpetas)), velocidad, args.solo_izq, mostrar_color, detector)
        accion = reproductor.ejecutar()
        velocidad, mostrar_color = reproductor.velocidad, reproductor.mostrar_color
        clip.cerrar()
        if accion == "salir":
            break
        k = (k + (1 if accion == "siguiente" else -1)) % len(carpetas)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
