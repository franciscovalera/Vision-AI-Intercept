"""Interfaz gráfica para grabar con la OAK-D S2.

Muestra en vivo las dos cámaras mono, permite ajustar la exposición mientras se
ve el resultado y graba clips etiquetados (tiro raso, fondo, negativo...) dentro
de una misma sesión. El formato de salida está descrito en grabacion.py.

También graba secuencias automáticas: N partes de D segundos separadas por
pausas de P segundos, todas dentro de una misma carpeta clip_NNN.

    python codigo/interfaz_grabacion.py [--sesion NOMBRE] [--salida CARPETA]

Con --sesion se continúa (o se crea) grabaciones/NOMBRE en lugar de una
carpeta con la fecha y hora.

Atajos: Espacio o R para empezar/parar un clip.
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPushButton, QSizePolicy, QSlider, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from grabacion import (CARPETA_GRABACIONES, ETIQUETAS, EXPOSICION_MIN_US, ISO_MAX, ISO_MIN, RESOLUCIONES_MONO,
                       TAMANO_PREVIEW_COLOR, Configuracion, Grabadora, Secuencia, Sesion, exposicion_maxima_us)

UMBRAL_SATURACION = 250
COLOR_ROJO = "#d32f2f"
COLOR_VERDE = "#2e7d32"
COLOR_NARANJA = "#ef6c00"


def jpeg_a_pixmap(jpeg, resaltar_saturacion):
    gris = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    rgb = np.dstack([gris] * 3)
    if resaltar_saturacion:
        rgb[gris >= UMBRAL_SATURACION] = (255, 0, 0)
    alto, ancho = gris.shape
    imagen = QImage(rgb.data, ancho, alto, 3 * ancho, QImage.Format_RGB888)
    return QPixmap.fromImage(imagen.copy())


class VistaCamara(QLabel):
    """Imagen de una cámara escalada al tamaño disponible, con borde rojo al grabar."""

    def __init__(self, titulo):
        super().__init__(f"{titulo}\n(esperando cámara)")
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(320, 200)
        self._pixmap = None
        self.marcar_grabando(False)

    def mostrar(self, pixmap):
        self._pixmap = pixmap
        self._redibujar()

    def resizeEvent(self, evento):
        self._redibujar()
        super().resizeEvent(evento)

    def _redibujar(self):
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def marcar_grabando(self, grabando):
        borde = f"4px solid {COLOR_ROJO}" if grabando else "4px solid #444"
        self.setStyleSheet(f"background: #111; color: #aaa; border: {borde};")


class VentanaGrabacion(QMainWindow):
    def __init__(self, carpeta_salida, nombre_sesion=None):
        super().__init__()
        self.setWindowTitle("Grabación OAK-D · Vision-AI-Intercept")
        self.sesion = Sesion(carpeta_salida, nombre_sesion)
        self.grabadora = None
        self.secuencia = None
        self._modo = None  # último estado aplicado por _actualizar_modo_grabacion
        self._ultimo_mostrado = {"izq": None, "der": None}

        self._construir_interfaz()
        self._actualizar_limites_exposicion()

        QShortcut(QKeySequence(Qt.Key_Space), self, self.alternar_grabacion)
        QShortcut(QKeySequence(Qt.Key_R), self, self.alternar_grabacion)

        self.temporizador = QTimer(self)
        self.temporizador.timeout.connect(self.actualizar)
        self.temporizador.start(33)

        self.arrancar_camara()

    # --- construcción de la interfaz ---

    def _construir_interfaz(self):
        self.vista_izq = VistaCamara("Izquierda (CAM_B)")
        self.vista_der = VistaCamara("Derecha (CAM_C)")
        vistas = QHBoxLayout()
        vistas.addWidget(self.vista_izq)
        vistas.addWidget(self.vista_der)

        self.tabla_clips = QTableWidget(0, 7)
        self.tabla_clips.setHorizontalHeaderLabels(
            ["Clip", "Etiqueta", "Duración", "Frames izq", "Perdidos izq", "MB", "Notas"])
        self.tabla_clips.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tabla_clips.horizontalHeader().setStretchLastSection(True)
        self.tabla_clips.verticalHeader().setVisible(False)
        self.tabla_clips.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabla_clips.setMaximumHeight(170)

        izquierda = QVBoxLayout()
        izquierda.addLayout(vistas, stretch=1)
        izquierda.addWidget(QLabel("Clips de esta sesión"))
        izquierda.addWidget(self.tabla_clips)

        panel = QVBoxLayout()
        panel.addWidget(self._grupo_color())
        panel.addWidget(self._grupo_clip())
        panel.addWidget(self._grupo_secuencia())
        panel.addWidget(self._grupo_estado())
        panel.addWidget(self._grupo_exposicion())
        panel.addWidget(self._grupo_camara())
        panel.addStretch()
        contenedor_panel = QWidget()
        contenedor_panel.setLayout(panel)
        contenedor_panel.setFixedWidth(340)

        principal = QHBoxLayout()
        principal.addLayout(izquierda, stretch=1)
        principal.addWidget(contenedor_panel)
        central = QWidget()
        central.setLayout(principal)
        self.setCentralWidget(central)
        self.resize(1500, 720)

    def _grupo_color(self):
        grupo = QGroupBox("Color (CAM_A)")
        self.vista_color = QLabel("Desactivada: marca «Grabar también color»\ny pulsa «Aplicar y reiniciar»")
        self.vista_color.setAlignment(Qt.AlignCenter)
        self.vista_color.setFixedSize(*TAMANO_PREVIEW_COLOR)
        self.vista_color.setStyleSheet("background: #111; color: #aaa;")
        self._ultimo_color = None
        disposicion = QVBoxLayout(grupo)
        disposicion.addWidget(self.vista_color, alignment=Qt.AlignCenter)
        return grupo

    def _grupo_clip(self):
        grupo = QGroupBox("Clip")
        self.combo_etiqueta = QComboBox()
        self.combo_etiqueta.addItems(ETIQUETAS)
        self.texto_notas = QLineEdit()
        self.texto_notas.setPlaceholderText("p. ej. palo derecho, pelota blanca, 3 m")

        self.boton_grabar = QPushButton()
        self.boton_grabar.setFocusPolicy(Qt.NoFocus)  # que Espacio no lo pulse dos veces
        self.boton_grabar.setMinimumHeight(48)
        self.boton_grabar.clicked.connect(self.alternar_grabacion)
        self._estilo_boton_grabar(False)

        formulario = QFormLayout()
        formulario.addRow("Etiqueta:", self.combo_etiqueta)
        formulario.addRow("Notas:", self.texto_notas)
        disposicion = QVBoxLayout(grupo)
        disposicion.addLayout(formulario)
        disposicion.addWidget(self.boton_grabar)
        return grupo

    def _grupo_secuencia(self):
        grupo = QGroupBox("Secuencia automática")
        self.spin_partes = QSpinBox()
        self.spin_partes.setRange(1, 100)
        self.spin_partes.setValue(5)
        self.spin_duracion = QSpinBox()
        self.spin_duracion.setRange(1, 600)
        self.spin_duracion.setValue(20)
        self.spin_duracion.setSuffix(" s")
        self.spin_pausa = QSpinBox()
        self.spin_pausa.setRange(0, 600)
        self.spin_pausa.setValue(10)
        self.spin_pausa.setSuffix(" s")

        fila = QHBoxLayout()
        for texto, spin in (("Partes", self.spin_partes), ("Duración", self.spin_duracion),
                            ("Pausa", self.spin_pausa)):
            fila.addWidget(QLabel(texto))
            fila.addWidget(spin)

        self.boton_secuencia = QPushButton("Iniciar secuencia")
        self.boton_secuencia.setFocusPolicy(Qt.NoFocus)
        self.boton_secuencia.clicked.connect(self.alternar_secuencia)
        self.etiqueta_secuencia = QLabel("Etiqueta y notas: las del grupo Clip")
        self.etiqueta_secuencia.setWordWrap(True)

        disposicion = QVBoxLayout(grupo)
        disposicion.addLayout(fila)
        disposicion.addWidget(self.boton_secuencia)
        disposicion.addWidget(self.etiqueta_secuencia)
        self.controles_secuencia = (self.spin_partes, self.spin_duracion, self.spin_pausa)
        return grupo

    def _grupo_estado(self):
        grupo = QGroupBox("Estado")
        self.etiqueta_camara = QLabel("Conectando…")
        self.etiqueta_camara.setWordWrap(True)
        self.etiqueta_fps = QLabel("–")
        self.etiqueta_perdidos = QLabel("–")
        self.etiqueta_clip = QLabel("–")
        self.etiqueta_clip.setWordWrap(True)
        self.etiqueta_disco = QLabel("–")
        self.etiqueta_sesion = QLabel(self.sesion.carpeta.name)

        formulario = QFormLayout(grupo)
        formulario.addRow("Cámara:", self.etiqueta_camara)
        formulario.addRow("FPS:", self.etiqueta_fps)
        formulario.addRow("Pérdidas/s:", self.etiqueta_perdidos)
        formulario.addRow("Clip actual:", self.etiqueta_clip)
        formulario.addRow("Disco libre:", self.etiqueta_disco)
        formulario.addRow("Sesión:", self.etiqueta_sesion)
        return grupo

    def _grupo_exposicion(self):
        grupo = QGroupBox("Exposición (se aplica en vivo)")
        self.check_auto = QCheckBox("Automática")
        self.check_auto.setChecked(True)
        self.check_auto.toggled.connect(self._cambio_exposicion)

        self.slider_exposicion = QSlider(Qt.Horizontal)
        self.spin_exposicion = QSpinBox()
        self.spin_exposicion.setSuffix(" µs")
        for control in (self.slider_exposicion, self.spin_exposicion):
            control.setRange(EXPOSICION_MIN_US, exposicion_maxima_us(120))
        self.spin_exposicion.setValue(1000)
        self._enlazar(self.slider_exposicion, self.spin_exposicion)

        self.slider_iso = QSlider(Qt.Horizontal)
        self.spin_iso = QSpinBox()
        for control in (self.slider_iso, self.spin_iso):
            control.setRange(ISO_MIN, ISO_MAX)
            control.setSingleStep(100)
        self.spin_iso.setValue(800)
        self._enlazar(self.slider_iso, self.spin_iso)

        self.check_saturacion = QCheckBox("Resaltar píxeles saturados en rojo")
        self.check_saturacion.setChecked(True)

        rejilla = QGridLayout(grupo)
        rejilla.addWidget(self.check_auto, 0, 0, 1, 2)
        rejilla.addWidget(QLabel("Tiempo:"), 1, 0)
        rejilla.addWidget(self.slider_exposicion, 1, 1)
        rejilla.addWidget(self.spin_exposicion, 1, 2)
        rejilla.addWidget(QLabel("ISO:"), 2, 0)
        rejilla.addWidget(self.slider_iso, 2, 1)
        rejilla.addWidget(self.spin_iso, 2, 2)
        rejilla.addWidget(self.check_saturacion, 3, 0, 1, 3)
        self._actualizar_habilitado_exposicion()
        return grupo

    def _grupo_camara(self):
        grupo = QGroupBox("Cámara (requiere reiniciar)")
        self.combo_resolucion = QComboBox()
        for alto, (_, fps_max) in RESOLUCIONES_MONO.items():
            ancho = 640 if alto == 400 else 1280
            self.combo_resolucion.addItem(f"{ancho}×{alto} (hasta {fps_max} FPS)", alto)
        self.combo_resolucion.currentIndexChanged.connect(self._cambio_resolucion)

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 120)
        self.spin_fps.setValue(120)
        self.spin_fps.valueChanged.connect(self._actualizar_limites_exposicion)

        self.spin_calidad = QSpinBox()
        self.spin_calidad.setRange(1, 100)
        self.spin_calidad.setValue(95)

        self.check_color = QCheckBox("Grabar también color (1080p)")
        self.check_color.setToolTip("Cámara a color en H.265 a 30 FPS, sin vista previa")
        self.boton_aplicar = QPushButton("Aplicar y reiniciar cámara")
        self.boton_aplicar.clicked.connect(self.arrancar_camara)

        formulario = QFormLayout(grupo)
        formulario.addRow("Resolución mono:", self.combo_resolucion)
        formulario.addRow("FPS:", self.spin_fps)
        formulario.addRow("Calidad MJPEG:", self.spin_calidad)
        formulario.addRow(self.check_color)
        formulario.addRow(self.boton_aplicar)
        self.grupo_camara = grupo
        return grupo

    def _enlazar(self, slider, spin):
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        spin.valueChanged.connect(self._cambio_exposicion)

    def _estilo_boton_grabar(self, grabando):
        if grabando:
            self.boton_grabar.setText("■  Parar clip  (Espacio)")
            color = "#555"
        else:
            self.boton_grabar.setText("●  Grabar clip  (Espacio)")
            color = COLOR_ROJO
        self.boton_grabar.setStyleSheet(
            f"QPushButton {{ background: {color}; color: white; font-size: 16px; font-weight: bold; }}"
            "QPushButton:disabled { background: #999; }")

    # --- cámara ---

    def _configuracion_actual(self):
        return Configuracion(
            resolucion=self.combo_resolucion.currentData(),
            fps=self.spin_fps.value(),
            exposicion_us=None if self.check_auto.isChecked() else self.spin_exposicion.value(),
            iso=self.spin_iso.value(),
            calidad=self.spin_calidad.value(),
            color=self.check_color.isChecked(),
        )

    def arrancar_camara(self):
        if self.grabadora is not None:
            if self.grabadora.grabando:
                return
            self.grabadora.detener()
        try:
            self.grabadora = Grabadora(self._configuracion_actual(), self.sesion)
        except ValueError as e:
            QMessageBox.warning(self, "Configuración no válida", str(e))
            self.grabadora = None
            return
        self._ultimo_mostrado = {"izq": None, "der": None}
        self._ultimo_color = None
        self.vista_color.clear()
        self.vista_color.setText("Conectando…" if self.grabadora.config.color else
                                 "Desactivada: marca «Grabar también color»\ny pulsa «Aplicar y reiniciar»")
        self.etiqueta_camara.setText("Conectando…")
        self.grabadora.iniciar()

    def _camara_lista(self):
        g = self.grabadora
        return g is not None and g.lista.is_set() and not g.error

    def alternar_grabacion(self):
        if not self._camara_lista() or self._secuencia_activa():
            return
        g = self.grabadora
        if g.grabando:
            self._registrar_clip(g.terminar_clip())
        else:
            g.empezar_clip(self.combo_etiqueta.currentText(), self.texto_notas.text().strip())
        self._actualizar_modo_grabacion()

    def _secuencia_activa(self):
        return self.secuencia is not None and self.secuencia.activa

    def alternar_secuencia(self):
        if self._secuencia_activa():
            self._registrar_clip(self.secuencia.cancelar())
            self.etiqueta_secuencia.setText(f"{self.secuencia.nombre}: cancelada en la parte "
                                            f"{self.secuencia.parte}/{self.secuencia.partes}")
        elif self._camara_lista() and not self.grabadora.grabando:
            self.secuencia = Secuencia(self.grabadora, self.spin_partes.value(), self.spin_duracion.value(),
                                       self.spin_pausa.value(), self.combo_etiqueta.currentText(),
                                       self.texto_notas.text().strip())
            QApplication.beep()
        self._actualizar_modo_grabacion()

    def _avanzar_secuencia(self):
        s = self.secuencia
        if not self._secuencia_activa():
            return
        estado_anterior = s.estado
        self._registrar_clip(s.actualizar())
        if s.estado != estado_anterior:
            QApplication.beep()  # aviso sonoro al empezar y acabar cada parte
            self._actualizar_modo_grabacion()

        if s.estado == "grabando":
            texto, color = f"● {s.nombre} · parte {s.parte}/{s.partes} · quedan {s.restante_s():.0f} s", COLOR_ROJO
        elif s.estado == "pausa":
            texto, color = f"Pausa · parte {s.parte + 1}/{s.partes} empieza en {s.restante_s():.0f} s", COLOR_NARANJA
        else:
            texto, color = f"{s.nombre}: secuencia terminada ({s.partes} partes)", COLOR_VERDE
        self.etiqueta_secuencia.setText(texto)
        self.etiqueta_secuencia.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: bold;")

    def _actualizar_modo_grabacion(self):
        """Habilita/deshabilita controles según el estado; solo actúa si el estado ha cambiado."""
        lista = self._camara_lista()
        grabando = self.grabadora is not None and self.grabadora.grabando
        en_secuencia = self._secuencia_activa()
        if (lista, grabando, en_secuencia) == self._modo:
            return
        self._modo = (lista, grabando, en_secuencia)

        self._estilo_boton_grabar(grabando)
        self.boton_grabar.setEnabled(lista and not en_secuencia)
        self.boton_secuencia.setText("Cancelar secuencia" if en_secuencia else "Iniciar secuencia")
        self.boton_secuencia.setEnabled(en_secuencia or (lista and not grabando))
        self.vista_izq.marcar_grabando(grabando)
        self.vista_der.marcar_grabando(grabando)
        ocupado = grabando or en_secuencia
        for widget in (self.grupo_camara, self.combo_etiqueta, self.texto_notas, *self.controles_secuencia):
            widget.setEnabled(not ocupado)

    def _registrar_clip(self, resumen):
        if resumen is None:
            return
        fila = self.tabla_clips.rowCount()
        self.tabla_clips.insertRow(fila)
        valores = [
            resumen["carpeta"],
            resumen["etiqueta"],
            f"{resumen['duracion_s']:.1f} s",
            str(resumen["frames"]["izq"]),
            str(resumen["perdidos"]["izq"]),
            f"{resumen['tamano_mb']:.1f}",
            resumen["notas"],
        ]
        for columna, valor in enumerate(valores):
            self.tabla_clips.setItem(fila, columna, QTableWidgetItem(valor))
        self.tabla_clips.scrollToBottom()

    # --- exposición ---

    def _cambio_resolucion(self):
        fps_max = RESOLUCIONES_MONO[self.combo_resolucion.currentData()][1]
        self.spin_fps.setMaximum(fps_max)
        self.spin_fps.setValue(fps_max)

    def _actualizar_limites_exposicion(self):
        maximo = exposicion_maxima_us(self.spin_fps.value())
        for control in (self.slider_exposicion, self.spin_exposicion):
            control.setRange(EXPOSICION_MIN_US, maximo)

    def _actualizar_habilitado_exposicion(self):
        manual = not self.check_auto.isChecked()
        for widget in (self.slider_exposicion, self.spin_exposicion, self.slider_iso, self.spin_iso):
            widget.setEnabled(manual)

    def _cambio_exposicion(self):
        self._actualizar_habilitado_exposicion()
        if self.grabadora is None or not self.grabadora.activa:
            return
        if self.check_auto.isChecked():
            self.grabadora.exposicion_automatica()
        else:
            self.grabadora.ajustar_exposicion(self.spin_exposicion.value(), self.spin_iso.value())

    # --- refresco periódico ---

    def actualizar(self):
        g = self.grabadora
        if g is None:
            return
        if g.error:
            self._mostrar_estado(f"Error: {g.error}. Conecta la cámara y pulsa «Aplicar y reiniciar».", COLOR_ROJO)
            if self._secuencia_activa():
                self._registrar_clip(self.secuencia.cancelar())
            elif g.grabando:
                self._registrar_clip(g.terminar_clip())
            self._actualizar_modo_grabacion()
            return
        if not g.lista.is_set():
            self._mostrar_estado("Conectando…", None)
            self._actualizar_modo_grabacion()
            return

        self._avanzar_secuencia()
        self._actualizar_modo_grabacion()
        info = g.info_camara
        if g.usb_super:
            self._mostrar_estado(f"{info['producto']} · USB {info['usb']}", COLOR_VERDE)
        else:
            self._mostrar_estado(f"{info['producto']} · USB {info['usb']} (USB 2: se pierden frames)", COLOR_ROJO)

        resaltar = self.check_saturacion.isChecked()
        for stream, vista in (("izq", self.vista_izq), ("der", self.vista_der)):
            jpeg = g.ultimo_jpeg[stream]
            if jpeg is not None and jpeg is not self._ultimo_mostrado[stream]:
                vista.mostrar(jpeg_a_pixmap(jpeg, resaltar))
                self._ultimo_mostrado[stream] = jpeg

        color = g.ultimo_color
        if color is not None and color is not self._ultimo_color:
            rgb = np.ascontiguousarray(color[:, :, ::-1])
            alto, ancho = rgb.shape[:2]
            imagen = QImage(rgb.data, ancho, alto, 3 * ancho, QImage.Format_RGB888)
            self.vista_color.setPixmap(QPixmap.fromImage(imagen.copy()))
            self._ultimo_color = color

        objetivo = g.config.fps
        self.etiqueta_fps.setText(" · ".join(f"{s} {g.fps[s]:.0f}" for s in g.streams) + f"  (objetivo {objetivo:g})")
        perdidos = g.perdidos_por_segundo
        self.etiqueta_perdidos.setText(" · ".join(f"{s} {perdidos[s]}" for s in g.streams))
        self.etiqueta_perdidos.setStyleSheet(f"color: {COLOR_ROJO if any(perdidos.values()) else COLOR_VERDE};")

        clip = g.clip
        if clip is not None:
            self.etiqueta_clip.setText(
                f"● {clip.nombre} · {clip.duracion():.1f} s · {clip.bytes / 1e6:.0f} MB · "
                f"perdidos izq {clip.perdidos['izq']}")
            self.etiqueta_clip.setStyleSheet(f"color: {COLOR_ROJO}; font-weight: bold;")
        else:
            self.etiqueta_clip.setText("sin grabar")
            self.etiqueta_clip.setStyleSheet("")

        self._actualizar_disco()

    def _mostrar_estado(self, texto, color):
        self.etiqueta_camara.setText(texto)
        self.etiqueta_camara.setStyleSheet(f"color: {color};" if color else "")

    def _actualizar_disco(self):
        carpeta = self.sesion.carpeta
        while not carpeta.exists():
            carpeta = carpeta.parent
        libre_gb = shutil.disk_usage(carpeta).free / 1e9
        self.etiqueta_disco.setText(f"{libre_gb:.1f} GB")
        self.etiqueta_disco.setStyleSheet(f"color: {COLOR_ROJO};" if libre_gb < 10 else "")

    def closeEvent(self, evento):
        self.temporizador.stop()
        if self._secuencia_activa():
            self._registrar_clip(self.secuencia.cancelar())
        if self.grabadora is not None:
            if self.grabadora.grabando:
                self._registrar_clip(self.grabadora.terminar_clip())
            self.grabadora.detener()
        if self.sesion.clips:
            print(f"Sesión guardada en {self.sesion.carpeta} ({len(self.sesion.clips)} clips)")
        super().closeEvent(evento)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--salida", type=Path, default=CARPETA_GRABACIONES, help="carpeta de grabaciones")
    parser.add_argument("--sesion", default=None,
                        help="nombre de la sesión; si ya existe se continúa (por defecto, fecha y hora)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    ventana = VentanaGrabacion(args.salida, args.sesion)
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
