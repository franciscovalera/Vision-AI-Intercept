"""Grabación sin interfaz: graba un clip hasta --duracion o Ctrl+C, o una secuencia con --partes.

Para grabar viendo la imagen y ajustando la exposición en vivo, usar
interfaz_grabacion.py. El formato de salida está descrito en grabacion.py.

Ejemplos:
    python codigo/grabar.py --duracion 10
    python codigo/grabar.py --fps 110 --exposicion 1000 --iso 800 --etiqueta "tiro raso"
    python codigo/grabar.py --resolucion 800 --fps 60 --color
    python codigo/grabar.py --sesion sesion_1 --partes 5 --duracion 20 --pausa 10 --espera 5

Sin --sesion se crea la siguiente grabaciones/sesion_N. Las secuencias empiezan tras una
cuenta atrás (--espera) y avisan con pitidos al empezar cada parte y al pasar a pausa.
    python codigo/grabar.py --limite-auto 1500 --zona-auto 0 0.5 1 0.5 --franja noche
    python codigo/grabar.py --etiqueta "tiro raso" --tiro-ancho centro --tiro-velocidad fuerte
"""

import argparse
import sys
import time
from pathlib import Path

from avisos import AvisosSecuencia
from grabacion import (CARPETA_GRABACIONES, ETIQUETAS, FPS_POR_DEFECTO, FRANJAS, ISO_MAX, ISO_MIN, LIMITE_AUTO_US,
                       RESOLUCIONES_MONO, ZONAS_TIRO, Configuracion, Grabadora, Secuencia, Sesion, franja_actual)


def parsear_argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolucion", type=int, choices=RESOLUCIONES_MONO, default=400,
                   help="altura en píxeles de las cámaras mono (por defecto 400)")
    p.add_argument("--fps", type=float, default=None,
                   help="FPS de las cámaras mono (por defecto 110 a 400p: el máximo sin perder frames con las dos)")
    p.add_argument("--exposicion", type=int, default=None,
                   help="exposición manual en µs; sin ella se usa autoexposición")
    p.add_argument("--iso", type=int, default=800, help=f"sensibilidad con exposición manual ({ISO_MIN}-{ISO_MAX})")
    p.add_argument("--limite-auto", type=int, default=LIMITE_AUTO_US,
                   help=f"exposición máxima en automático, en µs; 0 = sin límite (por defecto {LIMITE_AUTO_US})")
    p.add_argument("--zona-auto", type=float, nargs=4, default=None, metavar=("X", "Y", "ANCHO", "ALTO"),
                   help="zona donde se mide la luz en automático, en fracciones de la imagen "
                        "(p. ej. 0 0.5 1 0.5 = mitad inferior); por defecto, la imagen entera")
    p.add_argument("--calidad", type=int, default=95, help="calidad MJPEG (1-100)")
    p.add_argument("--color", action="store_true", help="grabar también la cámara a color (1080p H.265)")
    p.add_argument("--etiqueta", default="prueba", help=f"tipo de clip, p. ej. {', '.join(ETIQUETAS)}")
    p.add_argument("--notas", default="", help="texto libre que se guarda en metadatos.json")
    p.add_argument("--franja", choices=FRANJAS, default=None, help="franja horaria (por defecto, según la hora)")
    for campo, opciones in ZONAS_TIRO.items():
        p.add_argument(f"--tiro-{campo}", choices=[o for o in opciones if o], default="",
                       help=f"{campo} del tiro (opcional)")
    p.add_argument("--salida", type=Path, default=CARPETA_GRABACIONES, help="carpeta de grabaciones")
    p.add_argument("--sesion", default=None,
                   help="nombre de la sesión; si ya existe se continúa (por defecto, la siguiente sesion_N)")
    p.add_argument("--duracion", type=float, default=None, help="duración del clip (o de cada parte) en segundos")
    p.add_argument("--partes", type=int, default=None,
                   help="grabar una secuencia de N partes dentro de un mismo clip_NNN (requiere --duracion)")
    p.add_argument("--pausa", type=float, default=10, help="segundos entre partes de la secuencia (por defecto 10)")
    p.add_argument("--espera", type=float, default=5,
                   help="cuenta atrás antes de la primera parte de la secuencia (por defecto 5)")
    args = p.parse_args()
    if args.partes is not None and (args.partes < 1 or args.duracion is None or args.pausa < 0 or args.espera < 0):
        p.error("--partes necesita un número >= 1, --duracion, una --pausa >= 0 y una --espera >= 0")

    config = Configuracion(
        resolucion=args.resolucion,
        fps=args.fps or FPS_POR_DEFECTO[args.resolucion],
        exposicion_us=args.exposicion,
        iso=args.iso,
        limite_auto_us=args.limite_auto or None,
        zona_auto=tuple(args.zona_auto) if args.zona_auto else None,
        calidad=args.calidad,
        color=args.color,
    )
    try:
        config.validar()
    except ValueError as e:
        p.error(str(e))
    args.extra = {"franja": args.franja or franja_actual()}
    tiro = {campo: getattr(args, f"tiro_{campo}") for campo in ZONAS_TIRO if getattr(args, f"tiro_{campo}")}
    if tiro:
        args.extra["tiro"] = tiro
    return args, config


def imprimir_resumen(resumen):
    if resumen is not None:
        print(f"  {resumen['carpeta']}: {resumen['duracion_s']} s, {resumen['tamano_mb']} MB, "
              f"frames {resumen['frames']}, perdidos {resumen['perdidos']}")


def grabar_clip(grabadora, args):
    clip = grabadora.empezar_clip(args.etiqueta, args.notas, extra=args.extra)
    print(f"Grabando {grabadora.sesion.carpeta / clip.nombre} (Ctrl+C para parar)...")
    try:
        while grabadora.activa and (args.duracion is None or clip.duracion() < args.duracion):
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    imprimir_resumen(grabadora.terminar_clip())


def grabar_secuencia(grabadora, args):
    secuencia = Secuencia(grabadora, args.partes, args.duracion, args.pausa, args.etiqueta, args.notas, args.extra,
                          espera_s=args.espera)
    avisos = AvisosSecuencia()
    print(f"Secuencia en {grabadora.sesion.carpeta / secuencia.nombre}: {args.partes} partes de "
          f"{args.duracion:g} s con pausas de {args.pausa:g} s (Ctrl+C para cancelar)")
    print(f"Empieza en {args.espera:g} s...")
    parte_anunciada = 0
    try:
        while grabadora.activa and secuencia.activa:
            resumen = secuencia.actualizar()
            avisos.actualizar(secuencia)
            if resumen is not None:
                imprimir_resumen(resumen)
                if secuencia.estado == "pausa":
                    print(f"Pausa de {args.pausa:g} s...")
            if secuencia.estado == "grabando" and secuencia.parte != parte_anunciada:
                print(f"Grabando parte {secuencia.parte}/{args.partes}...")
                parte_anunciada = secuencia.parte
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Secuencia cancelada.")
    imprimir_resumen(secuencia.cancelar())


def main():
    args, config = parsear_argumentos()
    sesion = Sesion(args.salida, args.sesion)
    grabadora = Grabadora(config, sesion)
    grabadora.iniciar()
    grabadora.lista.wait()
    if grabadora.error:
        sys.exit(f"Error al abrir la cámara: {grabadora.error}")

    info = grabadora.info_camara
    print(f"Cámara: {info['producto']} | USB: {info['usb']}")
    if not grabadora.usb_super:
        print("AVISO: la cámara no está en USB 3; pueden perderse frames a FPS altos.")

    if args.partes is None:
        grabar_clip(grabadora, args)
    else:
        grabar_secuencia(grabadora, args)

    grabadora.detener()
    if grabadora.error:
        print(f"Error durante la grabación: {grabadora.error}")


if __name__ == "__main__":
    main()
