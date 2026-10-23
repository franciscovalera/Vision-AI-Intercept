"""Entrena el detector de pelota (YOLO de una clase) sobre un dataset de crear_dataset.py.

Parte de los pesos de COCO de un modelo pequeño (yolo11n por defecto), que es
el tamaño que puede ejecutarse en tiempo real dentro de la OAK.

    python codigo/entrenar_detector.py --dataset pelota_v1 --epocas 40

Resultado en modelos/entrenamientos/<nombre>/ (weights/best.pt, gráficas y métricas).
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="pelota_v1", help="carpeta dentro de datos/ con data.yaml")
    parser.add_argument("--modelo", default=str(RAIZ / "modelos" / "preentrenados" / "yolo11n.pt"))
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--nombre", default=None, help="nombre del entrenamiento (por defecto, el del dataset)")
    parser.add_argument("--dispositivo", default="cpu", help="cpu, 0 (GPU)...")
    args = parser.parse_args()

    modelo = YOLO(args.modelo)
    modelo.train(
        data=str(RAIZ / "datos" / args.dataset / "data.yaml"),
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.dispositivo,
        workers=4,
        project=str(RAIZ / "modelos" / "entrenamientos"),
        name=args.nombre or args.dataset,
        exist_ok=True,
        patience=15,
        # Imágenes en escala de grises: sin aumentos de tono/saturación; sí de brillo
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        fliplr=0.5,
        scale=0.5,
        mosaic=1.0,
        close_mosaic=5,
        plots=True,
    )


if __name__ == "__main__":
    main()
