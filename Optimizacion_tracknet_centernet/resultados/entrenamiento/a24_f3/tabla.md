Evaluación en sesion_2 (pilas sin solapar, umbral 0.5).

| Detector | Precisión % | Sensibilidad % | F1 % | Error centro px | Error diámetro px | Sens. < 12 px % | Sens. 12-20 px % | Sens. > 20 px % | FP sin pelota |
|---|---|---|---|---|---|---|---|---|---|
| PelotaNet a24_f3 | 96.6 | 96.0 | 96.3 | 0.55 | 0.50 | 88.8 | 97.1 | 95.7 | 326 |
| YOLO pelota_v1 (conf ≥ 0,5) | 93.2 | 97.6 | 95.3 | 0.00 | 0.00 | 97.8 | 96.7 | 98.3 | 734 |

Frames evaluados (no dudosos): 18795, con pelota: 11949 (< 12 px: 535, 12-20 px: 5178, > 20 px: 6236).

Acierto: error del centro < 4 px y < diámetro/2. Las etiquetas salen de YOLO (limpiadas con coherencia temporal), así que la comparación favorece a YOLO.
