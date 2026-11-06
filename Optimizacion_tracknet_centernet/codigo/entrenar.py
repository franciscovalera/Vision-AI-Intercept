"""Entrenamiento de PelotaNet con las pilas de datos.py.

1. PÉRDIDA (la de CenterNet, por frame de la pila)
   Mapa de calor: pérdida focal sobre los logits, con la gaussiana objetivo Y
       en el centro (Y = 1):  -(1 - p)^2 log(p)
       en el resto:           -(1 - Y)^4 p^2 log(1 - p)
   El término (1 - Y)^4 reduce el castigo cerca del centro (allí un poco de calor no es tan
   grave) y p^2 / (1 - p)^2 hacen que pesen poco los miles de celdas de fondo fáciles. Se
   normaliza por el número de centros del lote. Los frames dudosos no cuentan.
   Desplazamiento y diámetro: L1 solo en la celda del centro. El diámetro (en px, del orden de
   10-40) lleva peso 0,1 para quedar en la escala del desplazamiento (en celdas, 0-1), y no
   cuenta con visibilidad 2.
       pérdida = focal + 1 * L1(desplazamiento) + 0,1 * L1(diámetro)

2. OPTIMIZACIÓN
   AdamW con decaimiento de pesos 1e-4, tasa con calentamiento lineal (1 época) y bajada en
   coseno, precisión mixta bf16 (la pérdida se calcula en float32). Cada época son
   --muestras-epoca pilas sacadas al azar con reemplazo, de forma que ~70 % tengan pelota.

3. VALIDACIÓN
   Toda sesion_2 (otro día, con otra luz y ropa), en pilas sin solapar. Al final de cada época:
   pérdida y precisión/sensibilidad con la decodificación de evaluar.py. mejor.pt es el de mayor
   F1 con umbral 0,5.

Salida en modelos/entrenamientos/<nombre>/: mejor.pt y ultimo.pt (state_dict de PelotaNet sin
fusionar, el que carga exportar.py --pesos), historial.csv y argumentos.json. Además estado.pt
(red, optimizador, planificador y época) en cada época: con --reanudar el entrenamiento sigue
donde se cortó, con los mismos argumentos.

    python codigo/entrenar.py --nombre a16_f3 --epocas 30
    python codigo/entrenar.py --nombre prueba --epocas 2 --muestras-epoca 4000   # prueba rápida
    python codigo/entrenar.py --nombre a16_f3 --epocas 30 --reanudar              # tras un corte
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, WeightedRandomSampler

from datos import PilasPelota, buscar_fuentes, preparar_cache
from evaluar import metricas, predecir
from modelo import PelotaNet, contar_parametros

RAIZ = Path(__file__).resolve().parents[1]
SESIONES_ENTRENAMIENTO = ["sesion_1", "20261023_133016", "pruebas_primer_dia"]
SESIONES_VALIDACION = ["sesion_2"]


def perdida_focal(logits, objetivo, positivo, peso_frame):
    """Pérdida focal de CenterNet. logits, objetivo, positivo: B x F x h x w; peso_frame: B x F."""
    logits = logits.float()
    peso = peso_frame[:, :, None, None]
    log_p, log_1p = Fn.logsigmoid(logits), Fn.logsigmoid(-logits)
    p = torch.sigmoid(logits)
    pos = -((1 - p) ** 2) * log_p * positivo
    neg = -((1 - objetivo) ** 4) * p ** 2 * log_1p * (1 - positivo)
    n = (positivo * peso).sum().clamp(min=1)
    return ((pos + neg) * peso).sum() / n


def perdida_total(salidas, obj):
    calor, desplazamiento, diametro = salidas
    B, F = obj["peso_calor"].shape
    focal = perdida_focal(calor, obj["calor"], obj["positivo"], obj["peso_calor"])
    pos = obj["positivo"] * obj["peso_calor"][:, :, None, None]  # B x F x h x w
    n = pos.sum().clamp(min=1)
    h, w = calor.shape[-2:]
    l1_desp = (desplazamiento.float().view(B, F, 2, h, w) - obj["desplazamiento"].view(B, F, 2, h, w)).abs()
    l1_desp = (l1_desp * pos[:, :, None]).sum() / n
    pos_d = pos * obj["peso_diametro"][:, :, None, None]
    l1_diam = ((diametro.float() - obj["diametro"]).abs() * pos_d).sum() / pos_d.sum().clamp(min=1)
    total = focal + l1_desp + 0.1 * l1_diam
    return total, {"focal": focal.item(), "desplazamiento": l1_desp.item(), "diametro": l1_diam.item()}


def a_entrada(x, dispositivo, ruido=None):
    """Pila uint8 → float 0-255 en la GPU (channels_last), con el ruido gaussiano del aumento."""
    x = x.to(dispositivo, non_blocking=True).float()
    if ruido is not None:
        x = (x + torch.randn_like(x) * ruido[:, None, None, None]).clamp_(0, 255)
    return x.contiguous(memory_format=torch.channels_last)


def a_dispositivo(obj, dispositivo):
    return {k: v.to(dispositivo, non_blocking=True) for k, v in obj.items()}


@torch.no_grad()
def perdida_validacion(modelo, cargador, dispositivo, max_lotes):
    modelo.eval()
    totales = []
    for n, (x, obj) in enumerate(cargador):
        if n >= max_lotes:
            break
        with torch.autocast("cuda", torch.bfloat16):
            salidas = modelo(a_entrada(x, dispositivo))
        totales.append(perdida_total(salidas, a_dispositivo(obj, dispositivo))[0].item())
    return float(np.mean(totales))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nombre", required=True)
    parser.add_argument("--anchura", type=int, default=16)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--muestras-epoca", type=int, default=40000)
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--con-pelota", type=float, default=0.7, help="fracción de pilas con pelota por época")
    parser.add_argument("--trabajadores", type=int, default=12)
    parser.add_argument("--limite-horas", type=float, default=None,
                        help="no empieza una época nueva si no da tiempo a acabarla antes de este límite")
    parser.add_argument("--fin", default=None,
                        help="hora HH:MM límite (la próxima vez que llegue): no empieza una época que no acabe antes")
    parser.add_argument("--reanudar", action="store_true", help="seguir desde estado.pt de --nombre")
    parser.add_argument("--dispositivo", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dispositivo = torch.device(args.dispositivo)
    salida = RAIZ / "modelos" / "entrenamientos" / args.nombre
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "argumentos.json").write_text(json.dumps(vars(args), indent=1))

    entrenamiento = buscar_fuentes(SESIONES_ENTRENAMIENTO)
    validacion = buscar_fuentes(SESIONES_VALIDACION)
    preparar_cache(entrenamiento + validacion)
    ds = PilasPelota(entrenamiento, args.frames, aumentar=True)
    ds_val = PilasPelota(validacion, args.frames, aumentar=False, paso=args.frames)

    # Sobremuestreo: pesos para que una fracción `con_pelota` de las pilas sacadas tengan pelota
    con = ds.con_pelota()
    pesos = np.where(con, args.con_pelota / con.sum(), (1 - args.con_pelota) / max(1, (~con).sum()))
    muestreador = WeightedRandomSampler(torch.from_numpy(pesos), args.muestras_epoca, replacement=True)
    cargador = DataLoader(ds, args.lote, sampler=muestreador, num_workers=args.trabajadores,
                          pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    cargador_val = DataLoader(ds_val, 64, shuffle=False, num_workers=8, pin_memory=True)
    print(f"Entrenamiento: {len(ds)} pilas ({100 * con.mean():.0f} % con pelota) de {len(entrenamiento)} vídeos; "
          f"validación: {len(ds_val)} pilas de {len(validacion)} vídeos", flush=True)

    modelo = PelotaNet(args.frames, args.frames, args.anchura).to(dispositivo).to(memory_format=torch.channels_last)
    print(f"PelotaNet a{args.anchura} f{args.frames}: {contar_parametros(modelo) / 1e3:.1f} k parámetros", flush=True)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    pasos_epoca = len(cargador)
    total = args.epocas * pasos_epoca
    calentamiento = pasos_epoca

    def tasa(paso):
        if paso < calentamiento:
            return (paso + 1) / calentamiento
        return 0.5 * (1 + math.cos(math.pi * (paso - calentamiento) / max(1, total - calentamiento))) * 0.99 + 0.01

    planificador = torch.optim.lr_scheduler.LambdaLR(optimizador, tasa)

    historial = salida / "historial.csv"
    columnas = ["epoca", "segundos", "lr", "perdida", "focal", "desplazamiento", "diametro", "perdida_val",
                "precision_val", "sensibilidad_val", "f1_val", "error_px_val"]
    mejor_f1, inicio, duracion_epoca, primera = -1.0, time.time(), None, 1
    if args.reanudar and (salida / "estado.pt").exists():
        estado = torch.load(salida / "estado.pt", map_location=dispositivo)
        modelo.load_state_dict(estado["modelo"])
        optimizador.load_state_dict(estado["optimizador"])
        planificador.load_state_dict(estado["planificador"])
        mejor_f1, primera = estado["mejor_f1"], estado["epoca"] + 1
        inicio -= estado["segundos"]
        print(f"Reanudando en la época {primera} (mejor F1 {100 * mejor_f1:.1f} %)", flush=True)
    else:
        with open(historial, "w", newline="") as f:
            csv.writer(f).writerow(columnas)

    fin = None
    if args.fin:
        ahora = datetime.now()
        fin = ahora.replace(hour=int(args.fin[:2]), minute=int(args.fin[3:5]), second=0, microsecond=0)
        if fin <= ahora:
            fin += timedelta(days=1)
        fin = fin.timestamp()
    for epoca in range(primera, args.epocas + 1):
        # Duración prevista de la época: la última medida o, al empezar o reanudar, 3 minutos
        prevista = duracion_epoca or 180
        if (args.limite_horas and time.time() - inicio + prevista > args.limite_horas * 3600) or \
                (fin and time.time() + prevista > fin):
            print(f"Parada por límite de tiempo antes de la época {epoca}", flush=True)
            break
        t0 = time.time()
        modelo.train()
        sumas, n = {"perdida": 0.0, "focal": 0.0, "desplazamiento": 0.0, "diametro": 0.0}, 0
        for paso, (x, obj) in enumerate(cargador):
            obj = a_dispositivo(obj, dispositivo)
            x = a_entrada(x, dispositivo, obj["ruido"])
            with torch.autocast("cuda", torch.bfloat16):
                salidas = modelo(x)
            perdida, partes = perdida_total(salidas, obj)
            optimizador.zero_grad(set_to_none=True)
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 10.0)
            optimizador.step()
            planificador.step()
            sumas["perdida"] += perdida.item()
            for k, v in partes.items():
                sumas[k] += v
            n += 1
            if paso % 200 == 0:
                print(f"  época {epoca} paso {paso}/{pasos_epoca} pérdida {perdida.item():.3f} "
                      f"({', '.join(f'{k} {v:.3f}' for k, v in partes.items())})", flush=True)
        medias = {k: v / n for k, v in sumas.items()}

        perdida_val = perdida_validacion(modelo, cargador_val, dispositivo, max_lotes=40)
        pred, et = predecir(modelo, ds_val, dispositivo)
        m = metricas(pred["puntuacion"] >= 0.5, pred["x"], pred["y"], pred["d"], et)
        duracion_epoca = time.time() - t0
        fila = [epoca, round(time.time() - inicio), f"{planificador.get_last_lr()[0]:.2e}",
                *(f"{medias[k]:.4f}" for k in ("perdida", "focal", "desplazamiento", "diametro")),
                f"{perdida_val:.4f}", f"{m['precision']:.4f}", f"{m['sensibilidad']:.4f}", f"{m['f1']:.4f}",
                f"{m['error_px']:.3f}" if m["error_px"] is not None else ""]
        with open(historial, "a", newline="") as f:
            csv.writer(f).writerow(fila)
        estado = {k: v.detach().cpu() for k, v in modelo.state_dict().items()}
        torch.save(estado, salida / "ultimo.pt")
        if m["f1"] > mejor_f1:
            mejor_f1 = m["f1"]
            torch.save(estado, salida / "mejor.pt")
        torch.save({"modelo": modelo.state_dict(), "optimizador": optimizador.state_dict(),
                    "planificador": planificador.state_dict(), "epoca": epoca, "mejor_f1": mejor_f1,
                    "segundos": time.time() - inicio}, salida / "estado.pt")
        print(f"Época {epoca}/{args.epocas} ({duracion_epoca:.0f} s): pérdida {medias['perdida']:.4f}, "
              f"val {perdida_val:.4f}, precisión {100 * m['precision']:.1f} %, sensibilidad "
              f"{100 * m['sensibilidad']:.1f} %, F1 {100 * m['f1']:.1f} % (mejor {100 * mejor_f1:.1f} %)", flush=True)
    print(f"Terminado en {(time.time() - inicio) / 60:.1f} min. Pesos en {salida}")


if __name__ == "__main__":
    main()
