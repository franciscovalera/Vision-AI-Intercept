"""PelotaNet: red propia para detectar la pelota (entrada multi-frame tipo TrackNet + salida tipo CenterNet).

1. EL PROBLEMA Y LAS DECISIONES
   Hay que encontrar UN objeto (la pelota), redondo, de tamaño acotado (de ~6 a ~100 px), en
   imágenes en escala de grises de una cámara FIJA que graba a 120 FPS, y hacerlo dentro de la
   VPU Myriad X de la OAK (RVC2). Un detector genérico como YOLO resuelve un problema mucho más
   difícil (80 clases, cajas de cualquier tamaño y proporción, NMS) y lo paga en cómputo. Aquí:
   - Entrada multi-frame (idea de TrackNet): se apilan F frames consecutivos como canales. Con la
     cámara fija, lo que cambia entre frames es casi solo lo que se mueve, y eso delata a la
     pelota aunque sea pequeña o salga borrosa. La primera convolución puede aprender cualquier
     combinación lineal de los frames (p. ej. su diferencia), así que no hace falta calcularla aparte.
   - Salida tipo CenterNet: en vez de cajas, un mapa de calor cuyo máximo es el centro de la
     pelota, un desplazamiento subpíxel (porque el mapa está a 1/4 de resolución) y el DIÁMETRO
     en píxeles. El diámetro aparente es una de las dos medidas de profundidad que se fusionarán
     en el filtro de Kalman (la otra es el estéreo), así que se entrena directamente para él.
   - Varias salidas por inferencia (idea de TrackNetV2): con frames_salida = F la red da la
     posición de la pelota en los F frames de entrada a la vez; a 40 inferencias/s salen 120
     posiciones/s.

2. ARQUITECTURA (codificador-decodificador pequeño, con atajos como una U-Net/FPN mínima)
   Con anchura c y entrada F x 400 x 640 (la imagen mono completa: 400 y 640 son múltiplos de 16):

       entrada F x 400 x 640  (0-255)
       e2   2 conv (la 1.ª con paso 2)   ->  c  x 200 x 320   (1/2)
       e4   2 conv (la 1.ª con paso 2)   -> 2c  x 100 x 160   (1/4)  --------------------+
       e8   3 conv (la 1.ª con paso 2)   -> 4c  x  50 x  80   (1/8)  ----------+         |
       e16  3 conv (la 1.ª con paso 2)   -> 4c  x  25 x  40   (1/16)           |         |
       d8   subir x2, sumar e8, conv     -> 4c  x  50 x  80   <----------------+         |
       d4   conv 1x1 a 2c, subir x2, sumar e4, conv -> 2c x 100 x 160  <---------------+
       cabezas (conv 3x3 + conv 1x1) sobre d4:
           calor        frames_salida      canales  (logits; sigmoide = probabilidad de centro)
           desplazamiento 2*frames_salida  canales  (dx, dy dentro de la celda de 4x4 px)
           diámetro     frames_salida      canales  (en píxeles de la imagen de entrada)
       Cada cabeza tiene su propia conv 3x3; con cabeza_compartida=True las tres comparten una sola
       (las cabezas son ~28 % del coste con anchura 16; compartirla ahorra unos 2/3 de eso).

   - Bajar hasta 1/16 da contexto (campo receptivo de ~200 px) para pelotas cercanas grandes y
     para distinguir la pelota de zapatillas o manchas; los atajos devuelven el detalle fino de
     1/4 y 1/8 que necesitan las pelotas lejanas de 6-10 px.
   - Los atajos se SUMAN (no se concatenan): con el mismo número de canales cuesta lo mismo que
     una suma, mientras que concatenar dobla los canales de la convolución siguiente.
   - Antes de subir de 1/8 a 1/4 se reducen los canales con una conv 1x1: así se sube un mapa
     más delgado (más barato) y encaja con los 2c canales de e4.

3. PENSADA PARA EL MYRIAD X
   Solo usa operaciones que el compilador de OpenVINO para RVC2 ejecuta bien: convolución 3x3/1x1,
   BatchNorm (desaparece al fusionarla en la convolución), ReLU, suma y Upsample por vecino más
   cercano. Sin convoluciones depthwise (en esta VPU no siempre compensan; se medirán aparte),
   sin atención ni activaciones exóticas. Los canales son múltiplos de 8.
   La normalización 0-255 -> 0-1 va dentro de la red y, al fusionar, se integra en los pesos de la
   primera convolución: el blob recibe los bytes de la cámara tal cual, sin coste extra.

4. USO
       modelo = PelotaNet(frames_entrada=3, frames_salida=3, anchura=16, cabeza_compartida=False)
       calor, desplazamiento, diametro = modelo(x)        # x: (lotes, F, alto, ancho), valores 0-255
       desplegable = fusionar(modelo)                     # para exportar a ONNX / blob
       python codigo/modelo.py                            # tabla de variantes y comprobaciones
"""

import copy

import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    """Convolución sin sesgo + BatchNorm + ReLU. El relleno k // 2 conserva el tamaño (con paso 1)."""

    def __init__(self, canales_entrada, canales_salida, nucleo=3, paso=1):
        super().__init__(
            nn.Conv2d(canales_entrada, canales_salida, nucleo, paso, nucleo // 2, bias=False),
            nn.BatchNorm2d(canales_salida),
            nn.ReLU(inplace=True),
        )


def cabeza(canales, salidas, propia=True):
    """Una conv 3x3 propia de la salida (si propia) y una 1x1 que da los canales finales."""
    capas = [ConvBNReLU(canales, canales)] if propia else []
    return nn.Sequential(*capas, nn.Conv2d(canales, salidas, 1))


class PelotaNet(nn.Module):
    def __init__(self, frames_entrada=3, frames_salida=1, anchura=16, cabeza_compartida=False):
        super().__init__()
        c = anchura
        self.frames_entrada = frames_entrada
        self.frames_salida = frames_salida
        self.anchura = anchura
        self.cabeza_compartida = cabeza_compartida
        self.normalizar = True  # x / 255 en forward; fusionar() lo pasa a la primera convolución

        # Codificador: cada etapa reduce a la mitad con una conv de paso 2 y refina con conv de paso 1
        self.e2 = nn.Sequential(ConvBNReLU(frames_entrada, c, paso=2), ConvBNReLU(c, c))
        self.e4 = nn.Sequential(ConvBNReLU(c, 2 * c, paso=2), ConvBNReLU(2 * c, 2 * c))
        self.e8 = nn.Sequential(ConvBNReLU(2 * c, 4 * c, paso=2), ConvBNReLU(4 * c, 4 * c), ConvBNReLU(4 * c, 4 * c))
        self.e16 = nn.Sequential(ConvBNReLU(4 * c, 4 * c, paso=2), ConvBNReLU(4 * c, 4 * c), ConvBNReLU(4 * c, 4 * c))

        # Decodificador: subir y sumar el atajo del codificador
        self.subir = nn.Upsample(scale_factor=2, mode="nearest")
        self.d8 = ConvBNReLU(4 * c, 4 * c)
        self.reducir = ConvBNReLU(4 * c, 2 * c, nucleo=1)
        self.d4 = ConvBNReLU(2 * c, 2 * c)

        # Cabezas a 1/4 de resolución (con una conv 3x3 común antes si se comparte)
        self.comun = ConvBNReLU(2 * c, 2 * c) if cabeza_compartida else nn.Identity()
        propia = not cabeza_compartida
        self.calor = cabeza(2 * c, frames_salida, propia)
        self.desplazamiento = cabeza(2 * c, 2 * frames_salida, propia)
        self.diametro = cabeza(2 * c, frames_salida, propia)
        # Sesgo inicial de CenterNet: sigmoide(-2,19) = 0,1, el mapa empieza casi vacío y el
        # entrenamiento no arranca inundado de falsos positivos (casi todas las celdas son fondo)
        nn.init.constant_(self.calor[-1].bias, -2.19)

    def forward(self, x):
        if self.normalizar:
            x = x / 255.0
        f4 = self.e4(self.e2(x))
        f8 = self.e8(f4)
        f16 = self.e16(f8)
        p8 = self.d8(self.subir(f16) + f8)
        p4 = self.comun(self.d4(self.subir(self.reducir(p8)) + f4))
        return self.calor(p4), self.desplazamiento(p4), self.diametro(p4)


@torch.no_grad()
def fusionar(modelo):
    """Copia lista para desplegar: cada BatchNorm se integra en su convolución y la división
    por 255 en los pesos de la primera. El resultado es matemáticamente el mismo.

    BatchNorm en inferencia es lineal:  y = gamma * (z - media) / sqrt(var + eps) + beta,
    y z = W * x es la convolución, así que  y = (W * f) x + (beta - media * f)
    con f = gamma / sqrt(var + eps): una sola convolución con pesos W*f y sesgo beta - media*f.
    """
    m = copy.deepcopy(modelo).eval()
    for bloque in m.modules():
        if isinstance(bloque, ConvBNReLU) and isinstance(bloque[1], nn.BatchNorm2d):
            conv, bn = bloque[0], bloque[1]
            f = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            nueva = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                              conv.padding, bias=True)
            nueva.weight.copy_(conv.weight * f[:, None, None, None])
            nueva.bias.copy_(bn.bias - bn.running_mean * f)
            bloque[0], bloque[1] = nueva, nn.Identity()
    # conv(x / 255) = (W / 255) x + b: la normalización pasa a los pesos de la primera convolución
    m.e2[0][0].weight.div_(255.0)
    m.normalizar = False
    return m


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


@torch.no_grad()
def contar_macs(modelo, alto, ancho):
    """Multiplicaciones-acumulaciones de las convoluciones para una entrada (las de BN, ReLU,
    sumas y Upsample son despreciables). Por cada conv: elementos de salida x (canales de
    entrada / grupos) x tamaño del núcleo."""
    total = 0

    def sumar(capa, entrada, salida):
        nonlocal total
        total += salida.numel() * capa.in_channels // capa.groups * capa.kernel_size[0] * capa.kernel_size[1]

    ganchos = [c.register_forward_hook(sumar) for c in modelo.modules() if isinstance(c, nn.Conv2d)]
    modelo.eval()(torch.zeros(1, modelo.frames_entrada, alto, ancho))
    for gancho in ganchos:
        gancho.remove()
    return total


def campo_receptivo(modelo, alto=400, ancho=640):
    """Campo receptivo teórico (alto, ancho) en píxeles de entrada de la celda central del mapa.

    Con todos los pesos positivos, sin sesgos y BatchNorm neutra, ninguna ReLU anula nada, así que
    el gradiente de la salida central respecto a la entrada es distinto de cero exactamente en los
    píxeles que pueden influir en ella. Si toca el borde, el campo real es mayor que la imagen.
    """
    m = copy.deepcopy(modelo).eval()
    with torch.no_grad():
        for capa in m.modules():
            if isinstance(capa, nn.Conv2d):
                capa.weight.fill_(0.01)
                if capa.bias is not None:
                    capa.bias.zero_()
            elif isinstance(capa, nn.BatchNorm2d):
                capa.reset_parameters()
                capa.reset_running_stats()
    x = torch.ones(1, m.frames_entrada, alto, ancho, requires_grad=True)
    calor = m(x)[0]
    calor[0, 0, calor.shape[2] // 2, calor.shape[3] // 2].backward()
    filas, columnas = torch.nonzero(x.grad[0].abs().sum(0), as_tuple=True)
    return int(filas.max() - filas.min() + 1), int(columnas.max() - columnas.min() + 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    alto, ancho = 400, 640

    print(f"Variantes con entrada de {alto}x{ancho} (imagen mono completa):")
    print(f"{'anchura':>8} {'frames':>7} {'parámetros':>11} {'GMACs':>7}   salidas (calor / desplaz. / diámetro)")
    for anchura in (8, 16, 24, 32):
        for frames in (1, 3, 5):
            m = PelotaNet(frames_entrada=frames, frames_salida=frames, anchura=anchura)
            salidas = m.eval()(torch.zeros(1, frames, alto, ancho))
            formas = " / ".join("x".join(map(str, s.shape[1:])) for s in salidas)
            print(f"{anchura:>8} {frames:>7} {contar_parametros(m):>11,} {contar_macs(m, alto, ancho) / 1e9:>7.3f}   {formas}")

    m = PelotaNet(frames_entrada=3, frames_salida=3, anchura=16)
    print(f"\nCampo receptivo de la celda central (anchura 16, 3 frames): {campo_receptivo(m)} px (alto, ancho)")

    # Fusión: dar a las BatchNorm estadísticas no triviales (unos pasos en modo entrenamiento con
    # datos aleatorios) y comprobar que el modelo fusionado da exactamente lo mismo
    m.train()
    with torch.no_grad():
        for _ in range(5):
            m(torch.rand(2, 3, alto, ancho) * 255)
    x = torch.rand(2, 3, alto, ancho) * 255
    m.eval()
    with torch.no_grad():
        originales = m(x)
        fusionado = fusionar(m)
        fusionadas = fusionado(x)
    diferencia = max((a - b).abs().max().item() for a, b in zip(originales, fusionadas))
    print(f"Fusión BatchNorm + escala de entrada: diferencia máxima {diferencia:.2e} "
          f"(BatchNorm restantes: {sum(isinstance(c, nn.BatchNorm2d) for c in fusionado.modules())})")
