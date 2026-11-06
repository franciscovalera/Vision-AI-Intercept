IoU en sesion_2 (mismos frames no dudosos, pilas sin solapar; umbral 0.5 para IoU media, mediana y % de pelotas; AP con todas las predicciones ordenadas por puntuación).

### IoU de cajas cuadradas (lado = diámetro)

| Detector | Tamaño | Pelotas | VP | IoU media VP | IoU mediana VP | IoU ≥ 0,5 % | IoU ≥ 0,75 % | AP50 % | AP50-95 % |
|---|---|---|---|---|---|---|---|---|---|
| PelotaNet a16_f3 | todas | 11949 | 11233 | 0.921 | 0.935 | 94.8 | 92.8 | 98.2 | 82.6 |
| PelotaNet a16_f3 | < 12 px | 535 | 366 | 0.881 | 0.885 | 68.4 | 66.7 | 97.9 | 73.7 |
| PelotaNet a16_f3 | 12-20 px | 5178 | 5011 | 0.919 | 0.930 | 96.8 | 95.5 | 98.2 | 83.6 |
| PelotaNet a16_f3 | > 20 px | 6236 | 5856 | 0.924 | 0.940 | 95.4 | 92.8 | 98.3 | 83.0 |
| PelotaNet a24_f3 | todas | 11949 | 11466 | 0.927 | 0.941 | 96.5 | 94.7 | 98.4 | 84.3 |
| PelotaNet a24_f3 | < 12 px | 535 | 475 | 0.868 | 0.883 | 88.8 | 80.6 | 97.8 | 71.8 |
| PelotaNet a24_f3 | 12-20 px | 5178 | 5026 | 0.930 | 0.939 | 97.1 | 96.2 | 98.4 | 86.4 |
| PelotaNet a24_f3 | > 20 px | 6236 | 5965 | 0.929 | 0.945 | 96.7 | 94.7 | 97.7 | 84.0 |
| YOLO pelota_v1 (conf ≥ 0,5)* | todas | 11949 | 11659 | 1.000 | 1.000 | 97.6 | 97.6 | 96.4 | 96.4 |
| YOLO pelota_v1 (conf ≥ 0,5)* | < 12 px | 535 | 523 | 1.000 | 0.999 | 97.8 | 97.8 | 97.0 | 97.0 |
| YOLO pelota_v1 (conf ≥ 0,5)* | 12-20 px | 5178 | 5009 | 1.000 | 0.999 | 96.7 | 96.7 | 94.6 | 94.6 |
| YOLO pelota_v1 (conf ≥ 0,5)* | > 20 px | 6236 | 6127 | 1.000 | 1.000 | 98.3 | 98.3 | 97.9 | 97.9 |

### IoU de círculos

| Detector | Tamaño | Pelotas | VP | IoU media VP | IoU mediana VP | IoU ≥ 0,5 % | IoU ≥ 0,75 % | AP50 % | AP50-95 % |
|---|---|---|---|---|---|---|---|---|---|
| PelotaNet a16_f3 | todas | 11949 | 11233 | 0.921 | 0.934 | 94.8 | 92.8 | 98.2 | 82.6 |
| PelotaNet a16_f3 | < 12 px | 535 | 366 | 0.879 | 0.884 | 68.4 | 66.7 | 97.9 | 73.2 |
| PelotaNet a16_f3 | 12-20 px | 5178 | 5011 | 0.919 | 0.930 | 96.8 | 95.6 | 98.2 | 83.6 |
| PelotaNet a16_f3 | > 20 px | 6236 | 5856 | 0.924 | 0.941 | 95.4 | 92.8 | 98.3 | 82.9 |
| PelotaNet a24_f3 | todas | 11949 | 11466 | 0.927 | 0.941 | 96.5 | 94.8 | 98.4 | 84.2 |
| PelotaNet a24_f3 | < 12 px | 535 | 475 | 0.867 | 0.884 | 88.8 | 80.4 | 97.8 | 71.8 |
| PelotaNet a24_f3 | 12-20 px | 5178 | 5026 | 0.930 | 0.939 | 97.1 | 96.2 | 98.4 | 86.2 |
| PelotaNet a24_f3 | > 20 px | 6236 | 5965 | 0.929 | 0.945 | 96.7 | 94.8 | 97.7 | 84.1 |
| YOLO pelota_v1 (conf ≥ 0,5)* | todas | 11949 | 11659 | 1.000 | 1.000 | 97.6 | 97.6 | 96.4 | 96.4 |
| YOLO pelota_v1 (conf ≥ 0,5)* | < 12 px | 535 | 523 | 1.000 | 0.999 | 97.8 | 97.8 | 97.0 | 97.0 |
| YOLO pelota_v1 (conf ≥ 0,5)* | 12-20 px | 5178 | 5009 | 1.000 | 0.999 | 96.7 | 96.7 | 94.6 | 94.6 |
| YOLO pelota_v1 (conf ≥ 0,5)* | > 20 px | 6236 | 6127 | 1.000 | 1.000 | 98.3 | 98.3 | 97.9 | 97.9 |

VP = acierto con el criterio de tabla.md (error < 4 px y < diámetro/2). IoU ≥ x % = pelotas etiquetadas con predicción (umbral) e IoU ≥ x, sobre todas las etiquetadas.

\* La IoU de YOLO es ~1 por construcción: las etiquetas son sus propias cajas (diámetro = media de ancho y alto). Solo baja en los frames en los que limpiar_etiquetas.py movió o interpoló la etiqueta. No es comparable con PelotaNet; se pone como referencia del techo.
