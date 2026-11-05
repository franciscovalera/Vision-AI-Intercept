"""Paso 3 del etiquetado denso: revisión manual de las etiquetas (sobre todo de los frames dudosos).

Abre una ventana con el frame ampliado, la etiqueta dibujada (círculo con el diámetro), la
trayectoria de alrededor y un zoom de la zona de la pelota. Se salta de dudoso en dudoso y se
corrige con el ratón. Recorre todos los clips con etiquetas que haya bajo la ruta indicada.

Colores: verde = YOLO, amarillo = interpolada, azul claro = manual. Texto rojo = dudoso y motivo.

Teclas y ratón:
    n / p           siguiente / anterior frame dudoso sin revisar (pasa al siguiente clip al acabar)
    d / a           frame siguiente / anterior        D / A   12 frames (0,1 s) adelante / atrás
    espacio         reproducir / parar (a 30 FPS)
    clic izquierdo  poner el centro de la pelota aquí (visibilidad 1 si no había pelota)
    rueda, + / -    diámetro +-0,5 px                 ] / [    diámetro +-2 px
    0 / 1 / 2       sin pelota / visible entera / parcial (tapada, cortada, en la mano)
    Intro o c       dar el frame por bueno tal como está y saltar al siguiente dudoso
    i               interpolar desde la etiqueta anterior hasta la siguiente (rellenar un hueco)
    u               deshacer el último cambio
    g               guardar                          q / Esc   guardar y salir

Todo frame tocado queda con revisado = 1 y dudoso = 0; limpiar_etiquetas.py lo respeta si se
vuelve a ejecutar.

    python codigo/revisar_etiquetas.py datos/etiquetas/sesion_1 --camara izq
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from lectura_clips import ClipGrabado
from limpiar_etiquetas import CARPETA_GRABACIONES, COLUMNAS, RAIZ

VENTANA = "Revisar etiquetas"
COLORES = {"yolo": (80, 220, 80), "interpolada": (0, 220, 255), "manual": (255, 200, 80)}
ZOOM, LADO_ZOOM = 4, 40  # el recuadro de zoom muestra 40x40 px ampliados x4


class Revisor:
    def __init__(self, archivos, camara, escala):
        self.archivos = archivos
        self.camara = camara
        self.escala = escala
        self.indice_archivo = -1
        self.deshacer = []
        self.reproduciendo = False
        self.salir = False
        self.abrir_siguiente(1)

    # --- archivos ---

    def abrir_siguiente(self, sentido):
        nuevo = self.indice_archivo + sentido
        if not 0 <= nuevo < len(self.archivos):
            return False
        if self.indice_archivo >= 0:
            self.guardar()
            self.clip.cerrar()
        self.indice_archivo = nuevo
        self.ruta = self.archivos[nuevo]
        relativa = self.ruta.parent.relative_to(RAIZ / "datos" / "etiquetas")
        self.nombre = str(relativa)
        self.clip = ClipGrabado(CARPETA_GRABACIONES / relativa)
        self.stream = self.clip.streams[self.camara]
        with open(self.ruta, newline="") as f:
            self.filas = list(csv.DictReader(f))
        self.deshacer = []
        self.cambios = False
        pendientes = self.pendientes()
        self.frame = pendientes[0] if pendientes else 0
        if sentido < 0 and pendientes:
            self.frame = pendientes[-1]
        return True

    def guardar(self):
        if not self.cambios:
            return
        temporal = self.ruta.with_suffix(".tmp")
        with open(temporal, "w", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=COLUMNAS)
            escritor.writeheader()
            escritor.writerows(self.filas)
        temporal.rename(self.ruta)
        self.cambios = False
        print(f"guardado {self.ruta.relative_to(RAIZ)}")

    def pendientes(self):
        return [i for i, fila in enumerate(self.filas) if fila["dudoso"] == "1" and fila["revisado"] != "1"]

    # --- edición ---

    def editar(self, i, **campos):
        self.deshacer.append((i, dict(self.filas[i])))
        self.filas[i].update({k: str(v) for k, v in campos.items()}, dudoso="0", revisado="1")
        self.cambios = True

    def diametro_cercano(self, i):
        """Diámetro de la etiqueta más cercana en el tiempo (para un punto nuevo)."""
        con = [j for j, fila in enumerate(self.filas) if fila["diametro"]]
        if not con:
            return 20.0
        return float(self.filas[min(con, key=lambda j: abs(j - i))]["diametro"])

    def poner_centro(self, x, y):
        fila = self.filas[self.frame]
        diametro = float(fila["diametro"]) if fila["diametro"] else self.diametro_cercano(self.frame)
        visibilidad = fila["visibilidad"] if fila["visibilidad"] in ("1", "2") else "1"
        self.editar(self.frame, x=f"{x:.2f}", y=f"{y:.2f}", diametro=f"{diametro:.2f}", visibilidad=visibilidad,
                    origen="manual", conf="")

    def cambiar_diametro(self, delta):
        fila = self.filas[self.frame]
        if fila["diametro"]:
            self.editar(self.frame, diametro=f"{max(2.0, float(fila['diametro']) + delta):.2f}", origen="manual")

    def poner_visibilidad(self, valor):
        if valor == 0:
            self.editar(self.frame, x="", y="", diametro="", visibilidad=0, origen="ninguna", conf="", pista="")
        elif self.filas[self.frame]["x"]:
            self.editar(self.frame, visibilidad=valor)

    def interpolar(self):
        """Rellena linealmente entre la etiqueta anterior y la siguiente al frame actual."""
        con = [j for j, fila in enumerate(self.filas) if fila["x"]]
        antes = [j for j in con if j < self.frame]
        despues = [j for j in con if j > self.frame]
        if not antes or not despues:
            return
        a, b = antes[-1], despues[0]
        t = np.array([float(self.filas[j]["t"]) for j in (a, b)])
        for j in range(a + 1, b):
            tj = float(self.filas[j]["t"])
            valores = {c: np.interp(tj, t, [float(self.filas[a][c]), float(self.filas[b][c])])
                       for c in ("x", "y", "diametro")}
            visible = self.filas[a]["visibilidad"] == self.filas[b]["visibilidad"] == "1"
            self.editar(j, **{c: f"{v:.2f}" for c, v in valores.items()}, visibilidad=1 if visible else 2,
                        origen="interpolada", conf="")

    def deshacer_ultimo(self):
        if self.deshacer:
            i, fila = self.deshacer.pop()
            self.filas[i] = fila
            self.frame = i
            self.cambios = True

    # --- navegación ---

    def ir(self, frame):
        self.frame = int(np.clip(frame, 0, len(self.filas) - 1))

    def saltar_dudoso(self, sentido):
        pendientes = self.pendientes()
        candidatos = [i for i in pendientes if (i > self.frame if sentido > 0 else i < self.frame)]
        if candidatos:
            self.frame = candidatos[0] if sentido > 0 else candidatos[-1]
        elif not self.abrir_siguiente(sentido):
            print("No quedan dudosos en esa dirección.")

    # --- dibujo ---

    def dibujar(self):
        gris = self.stream.leer(self.frame)
        imagen = cv2.resize(cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR), None, fx=self.escala, fy=self.escala,
                            interpolation=cv2.INTER_NEAREST)
        e = self.escala
        for j in range(max(0, self.frame - 30), min(len(self.filas), self.frame + 31)):  # trayectoria
            fila = self.filas[j]
            if fila["x"] and j != self.frame:
                centro = (int(float(fila["x"]) * e), int(float(fila["y"]) * e))
                cv2.circle(imagen, centro, 2, (0, 140, 255) if j < self.frame else (255, 120, 0), -1)
        fila = self.filas[self.frame]
        if fila["x"]:
            x, y, d = float(fila["x"]), float(fila["y"]), float(fila["diametro"])
            color = COLORES.get(fila["origen"], (255, 255, 255))
            cv2.circle(imagen, (int(x * e), int(y * e)), max(1, int(d / 2 * e)), color, 1, cv2.LINE_AA)
            cv2.drawMarker(imagen, (int(x * e), int(y * e)), color, cv2.MARKER_CROSS, 8, 1)
            self.dibujar_zoom(imagen, gris, x, y, d, color)

        dudoso = fila["dudoso"] == "1" and fila["revisado"] != "1"
        lineas = [
            f"{self.nombre} [{self.camara}]  frame {self.frame}/{len(self.filas) - 1}  "
            f"pendientes {len(self.pendientes())}  archivo {self.indice_archivo + 1}/{len(self.archivos)}",
            f"origen {fila['origen']}  visibilidad {fila['visibilidad']}  conf {fila['conf'] or '-'}  "
            f"diam {fila['diametro'] or '-'}  {'REVISADO' if fila['revisado'] == '1' else ''}",
        ]
        if dudoso:
            lineas.append(f"DUDOSO: {fila['motivo']}")
        for k, texto in enumerate(lineas):
            color = (60, 60, 255) if k == 2 else (255, 255, 255)
            cv2.putText(imagen, texto, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(imagen, texto, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.imshow(VENTANA, imagen)

    def dibujar_zoom(self, imagen, gris, x, y, d, color):
        """Recuadro arriba a la derecha con la zona de la pelota ampliada."""
        lado = max(LADO_ZOOM, int(d * 1.6))
        x0, y0 = int(round(x)) - lado // 2, int(round(y)) - lado // 2
        recorte = cv2.copyMakeBorder(gris, lado, lado, lado, lado, cv2.BORDER_CONSTANT)[
            y0 + lado:y0 + 2 * lado, x0 + lado:x0 + 2 * lado]
        tamano = LADO_ZOOM * ZOOM
        factor = tamano / lado
        zoom = cv2.cvtColor(cv2.resize(recorte, (tamano, tamano), interpolation=cv2.INTER_NEAREST),
                            cv2.COLOR_GRAY2BGR)
        centro = (int((x - x0) * factor), int((y - y0) * factor))
        cv2.circle(zoom, centro, int(d / 2 * factor), color, 1, cv2.LINE_AA)
        cv2.drawMarker(zoom, centro, color, cv2.MARKER_CROSS, 10, 1)
        alto, ancho = imagen.shape[:2]
        if tamano + 10 < min(alto, ancho):
            imagen[10:10 + tamano, ancho - tamano - 10:ancho - 10] = zoom
            cv2.rectangle(imagen, (ancho - tamano - 11, 9), (ancho - 9, 11 + tamano), (255, 255, 255), 1)

    # --- eventos ---

    def raton(self, evento, x, y, banderas, _):
        if evento == cv2.EVENT_LBUTTONDOWN:
            self.poner_centro(x / self.escala, y / self.escala)
        elif evento == cv2.EVENT_MOUSEWHEEL:
            self.cambiar_diametro(0.5 if cv2.getMouseWheelDelta(banderas) > 0 else -0.5)
        else:
            return
        self.dibujar()

    def tecla(self, k):
        acciones = {
            ord("n"): lambda: self.saltar_dudoso(1), ord("p"): lambda: self.saltar_dudoso(-1),
            ord("d"): lambda: self.ir(self.frame + 1), ord("a"): lambda: self.ir(self.frame - 1),
            ord("D"): lambda: self.ir(self.frame + 12), ord("A"): lambda: self.ir(self.frame - 12),
            ord("+"): lambda: self.cambiar_diametro(0.5), ord("-"): lambda: self.cambiar_diametro(-0.5),
            ord("]"): lambda: self.cambiar_diametro(2), ord("["): lambda: self.cambiar_diametro(-2),
            ord("0"): lambda: self.poner_visibilidad(0), ord("1"): lambda: self.poner_visibilidad(1),
            ord("2"): lambda: self.poner_visibilidad(2), ord("i"): self.interpolar,
            ord("u"): self.deshacer_ultimo, ord("g"): self.guardar,
        }
        if k in (13, 10, ord("c")):
            self.editar(self.frame)
            self.saltar_dudoso(1)
        elif k == ord(" "):
            self.reproduciendo = not self.reproduciendo
        elif k in (ord("q"), 27):
            self.guardar()
            self.salir = True
        elif k in acciones:
            acciones[k]()

    def bucle(self):
        cv2.namedWindow(VENTANA, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(VENTANA, self.raton)
        self.dibujar()
        while not self.salir:
            k = cv2.waitKey(33 if self.reproduciendo else 0) & 0xFF
            if k == 255:  # sin tecla: solo pasa reproduciendo
                if self.frame >= len(self.filas) - 1:
                    self.reproduciendo = False
                self.ir(self.frame + 1)
            else:
                self.tecla(k)
            if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                self.guardar()
                break
            if not self.salir:
                self.dibujar()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ruta", type=Path, nargs="?", default=RAIZ / "datos" / "etiquetas")
    parser.add_argument("--camara", choices=["izq", "der"], default="izq")
    parser.add_argument("--escala", type=float, default=2.0, help="ampliación de la imagen en pantalla")
    parser.add_argument("--todos", action="store_true", help="abrir también los clips sin dudosos pendientes")
    args = parser.parse_args()

    archivos = sorted(Path(args.ruta).resolve().rglob(f"etiquetas_{args.camara}.csv"))
    if not args.todos:
        archivos = [a for a in archivos if any(f["dudoso"] == "1" and f["revisado"] != "1"
                                               for f in csv.DictReader(open(a, newline="")))]
    if not archivos:
        raise SystemExit("No hay etiquetas pendientes de revisar (usa --todos para abrir igualmente).")
    Revisor(archivos, args.camara, args.escala).bucle()


if __name__ == "__main__":
    main()
