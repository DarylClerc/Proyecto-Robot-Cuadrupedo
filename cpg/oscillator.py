"""Generador de trayectoria del pie (CPG simple, task space) para caminata 2D.

En vez de una EDO tipo Hopf/van der Pol, se usa un generador paramétrico
manejado por una fase que avanza a frecuencia constante. Esto es más simple
de ajustar con CEM (ver enunciado: "se recomienda emplear osciladores
simples, ya que suelen ser más fáciles de ajustar").

Ciclo por pata (fase phi en [0, 1)):
  - Stance (phi < duty_factor): el pie barre en línea recta en x, de
    +stride/2 a -stride/2, a la altura nominal de stance (z0). Este barrido
    es lo que empuja el cuerpo hacia adelante (asumiendo fricción/apoyo).
  - Swing (phi >= duty_factor): el pie se levanta (arco de medio seno,
    altura ground_clearance) y vuelve de -stride/2 a +stride/2.

Marcha: trote (patas diagonales en fase). Definido por PHASE_OFFSET.
"""
import numpy as np

# Offset de fase por pata para marcha de trote: diagonales juntas.
PHASE_OFFSET = {
    "FR": 0.0,
    "RL": 0.0,
    "FL": 0.5,
    "RR": 0.5,
}


class CpgParams:
    """Parámetros del CPG, compartidos por las 4 patas (marcha simétrica)."""

    def __init__(
        self,
        frequency=1.5,       # Hz, ciclos por segundo
        duty_factor=0.5,     # fracción del ciclo en stance
        stride_length=0.10,  # m, largo total del paso (pico a pico en x)
        ground_clearance=0.05,  # m, altura máxima del pie durante swing
        stance_height=-0.30,    # m, altura nominal del pie (z) respecto a la cadera
        kp_walk=60.0,   # ganancia proporcional del PD de posición durante la caminata
        kd_walk=4.0,    # ganancia derivativa del PD de posición durante la caminata
    ):
        self.frequency = frequency
        self.duty_factor = duty_factor
        self.stride_length = stride_length
        self.ground_clearance = ground_clearance
        self.stance_height = stance_height
        self.kp_walk = kp_walk
        self.kd_walk = kd_walk

    def as_array(self):
        return np.array(
            [
                self.frequency,
                self.duty_factor,
                self.stride_length,
                self.ground_clearance,
                self.stance_height,
                self.kp_walk,
                self.kd_walk,
            ]
        )

    @staticmethod
    def from_array(arr):
        return CpgParams(*arr)


def foot_trajectory(phase_frac, params: CpgParams):
    """Offset (dx, dz) del pie respecto a la posición nominal de stance,
    para una fase phase_frac en [0, 1) (ya envuelta, ver `wrap_phase`).

    Tanto stance como swing usan un perfil coseno con velocidad CERO en
    ambos extremos (touchdown y liftoff). Una primera versión usaba stance
    lineal (velocidad constante) empalmado con swing coseno (velocidad cero
    en los bordes) -- eso deja una discontinuidad de velocidad justo en la
    transición stance/swing, una patada impulsiva en cada paso que resultó
    ser la causa de una oscilación de cabeceo/balanceo que crecía con el
    tiempo hasta frenar la caminata (ver cpg/cost.py). Iguales perfiles en
    ambas fases eliminan el salto.
    """
    duty = params.duty_factor
    half_stride = params.stride_length / 2.0

    if phase_frac < duty:
        s = phase_frac / duty  # 0..1 dentro del stance
        dx = half_stride * np.cos(np.pi * s)  # de +half_stride a -half_stride, vel=0 en bordes
        dz = 0.0
    else:
        s = (phase_frac - duty) / (1 - duty)  # 0..1 dentro del swing
        dx = -half_stride * np.cos(np.pi * s)  # de -half_stride a +half_stride, vel=0 en bordes
        dz = params.ground_clearance * np.sin(np.pi * s)  # arco, 0 en los extremos

    return dx, dz


def wrap_phase(t, leg, params: CpgParams):
    """Fase en [0,1) para una pata dada, en el tiempo t (segundos)."""
    phase = params.frequency * t + PHASE_OFFSET[leg]
    return phase % 1.0


def foot_target(t, leg, hip_offset_y, params: CpgParams):
    """Posición deseada (x, y, z) del pie, relativa al joint de abducción,
    en el tiempo t. y se mantiene fija (caminata 2D, sin componente lateral)."""
    phase_frac = wrap_phase(t, leg, params)
    dx, dz = foot_trajectory(phase_frac, params)
    x = dx
    y = hip_offset_y
    z = params.stance_height + dz
    return np.array([x, y, z])
